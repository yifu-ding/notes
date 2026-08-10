---
date: 2026-07-12
---

**Contents**

1. [[#[概念] CUDA 函数修饰符：`__device__` / `__global__` / `__host__`|[概念] CUDA 函数修饰符：`__device__` / `__global__` / `__host__`]]
2. [[#[概念] Warp / Lane / Thread 三者关系|[概念] Warp / Lane / Thread 三者关系]]
3. [[#[概念] `__shfl` 系列：Warp Shuffle 指令|[概念] `__shfl` 系列：Warp Shuffle 指令]]
4. [[#[概念] `warp_reduce_sum` 执行过程|[概念] `warp_reduce_sum` 执行过程]]
5. [[#[概念] Warp Shuffle vs Shared Memory Reduce|[概念] Warp Shuffle vs Shared Memory Reduce]]
6. [[#语法|语法]]
7. [[#基础编译命令|基础编译命令]]
8. [[#[模式] Elementwise Kernel：ReLU 前向与反向|[模式] Elementwise Kernel：ReLU 前向与反向]]
9. [[#[模式] Shared Memory Block Reduction：并行求和|[模式] Shared Memory Block Reduction：并行求和]]
10. [[#[模式] Warp Shuffle Reduction|[模式] Warp Shuffle Reduction]]
11. [[#Linear Layer 前向 + 反向 CUDA|Linear Layer 前向 + 反向 CUDA]]
12. [[#Naive CUDA GEMM|Naive CUDA GEMM]]
13. [[#Shared Memory Tiled GEMM|Shared Memory Tiled GEMM]]
14. [[#Row-wise Softmax CUDA|Row-wise Softmax CUDA]]
15. [[#Online Softmax 核心公式|Online Softmax 核心公式]]
16. [[#LayerNorm 前向 CUDA|LayerNorm 前向 CUDA]]
17. [[#Matrix Transpose CUDA|Matrix Transpose CUDA]]
18. [[#Scaled Dot-Product Attention 前向 CUDA|Scaled Dot-Product Attention 前向 CUDA]]
19. [[#RMSNorm 前向 CUDA|RMSNorm 前向 CUDA]]
20. [[#GELU / SiLU Elementwise Kernel|GELU / SiLU Elementwise Kernel]]
21. [[#SwiGLU Kernel（LLaMA FFN 核心）CUDA|SwiGLU Kernel（LLaMA FFN 核心）CUDA]]
22. [[#Cross-Entropy Loss CUDA|Cross-Entropy Loss CUDA]]
23. [[#Inclusive Prefix Sum（Scan）CUDA|Inclusive Prefix Sum（Scan）CUDA]]
24. [[#Adam Optimizer Kernel|Adam Optimizer Kernel]]
25. [[#Embedding Lookup（Gather）CUDA|Embedding Lookup（Gather）CUDA]]



> [!note] CUDA 算子岗位考察重点
> 核心不是会不会写算法，而是对 **GPU 内存层次和并行模式** 的理解。面试常见递进路径：
> Naive kernel → Shared Memory Tiling → Warp Shuffle → 分析瓶颈（计算 or 带宽 bound） → 进一步优化
>
> 常用 `__device__` 辅助函数（以下题目均依赖）：
> ###### 基础函数
> ```cpp
> __device__ float warp_reduce_sum(float val) {
>     for (int offset = 16; offset > 0; offset >>= 1)
>         val += __shfl_down_sync(0xffffffff, val, offset);
>     return val;
> }
> __device__ float warp_reduce_max(float val) {
>     for (int offset = 16; offset > 0; offset >>= 1)
>         val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
>     return val;
> }
> ```


###### [概念] CUDA 函数修饰符：`__device__` / `__global__` / `__host__`

| 修饰符 | 在哪跑 | 谁能调用 |
|--------|--------|---------|
| `__global__` | GPU | CPU（用 `<<<>>>` 启动） |
| `__device__` | GPU | GPU（被其他 `__device__` 或 `__global__` 调用） |
| `__host__` | CPU | CPU（默认，不写就是这个） |

`warp_reduce_sum` 标注 `__device__` 表示它是 GPU 辅助函数，只能在 kernel 内部调用，不能被 CPU 直接调用。

---

###### [概念] Warp / Lane / Thread 三者关系

```
Thread（线程）
  └── 全局概念，有唯一的 threadIdx

Warp（32 个 thread 的执行组）
  └── 硬件调度单位，32 个 thread 一起执行同一条指令

Lane（0–31）
  └── 一个 thread 在其所在 Warp 内的编号
  └── lane = threadIdx.x % 32
```

假设 blockDim.x = 128：
```
threadIdx.x =   0..31  → Warp 0，lane 0..31
threadIdx.x =  32..63  → Warp 1，lane 0..31
threadIdx.x =  64..95  → Warp 2，lane 0..31
threadIdx.x = 96..127  → Warp 3，lane 0..31
```

Lane 是 Warp 内的相对编号，不同 Warp 的 lane 互相独立。`__shfl` 系列指令只能在**同一 Warp 的 lane 之间** 通信，不能跨 Warp。

---

###### [概念] `__shfl` 系列：Warp Shuffle 指令

`__shfl`（shuffle，洗牌）让 Warp 内 lane 之间直接读彼此寄存器，不经过 shared memory。CUDA 提供四种变体：

| 函数 | 目标 lane | 用途 |
|------|----------|------|
| `__shfl_sync(mask, val, src)` | 固定 `src` | Broadcast：把某个 lane 的值发给所有人 |
| `__shfl_down_sync(mask, val, offset)` | `lane + offset` | Tree reduce：向右读，最终汇聚到 lane 0 |
| `__shfl_up_sync(mask, val, offset)` | `lane - offset` | Prefix sum：向左读，适合 causal 操作 |
| `__shfl_xor_sync(mask, val, laneMask)` | `lane XOR laneMask` | Butterfly AllReduce：所有 lane 得到结果 |

两个参数含义（容易混淆）：
- **第一参数 `membermask`**：哪些 lane 参与，`0xffffffff` = 全 32 个 lane
- **第三参数**：目标 lane 的计算方式（见上表）

*示例（8-lane 简化，原理相同）：*

```
初始值: [0, 1, 2, 3, 4, 5, 6, 7]  (lane 0..7)

__shfl_sync     (src=2):       [2, 2, 2, 2, 2, 2, 2, 2]   全部读 lane 2
__shfl_down_sync(offset=2):    [2, 3, 4, 5, 6, 7, ?, ?]   lane i ← lane i+2
__shfl_up_sync  (offset=2):    [?, ?, 0, 1, 2, 3, 4, 5]   lane i ← lane i-2
__shfl_xor_sync (laneMask=2):  [2, 3, 0, 1, 6, 7, 4, 5]   lane i ← lane i⊕2
```

---

###### [概念] `warp_reduce_sum` 执行过程

32 个 lane 各持有 `v0, v1, ..., v31`，`__shfl_down_sync` 每轮把右侧 lane 的值加到自己身上：

```
offset=16:  lane0 += lane16,  lane1 += lane17, ...  lane15 += lane31
offset=8:   lane0 += lane8,   lane1 += lane9,  ...
offset=4:   lane0 += lane4,   ...
offset=2:   lane0 += lane2,   ...
offset=1:   lane0 += lane1
```

最终 lane0 = v0+v1+...+v31  ← 5 轮（log₂32）后汇聚到 lane 0

`__shfl_xor_sync` 版本（Butterfly AllReduce）：laneMask 依次 1→2→4→8→16，最终**所有 lane** 都持有总和（所以叫 allreduce?），而 `down` 版本只有 lane 0 持有（reduce to lane0）。

---

###### [概念] Warp Shuffle vs Shared Memory Reduce

| | Shared Memory Reduce | Warp Shuffle Reduce |
|--|---------------------|---------------------|
| 数据路径 | 写 shared memory → `__syncthreads()` → 读 | 直接读对方寄存器 |
| 延迟 | ~20-30 cycle（含同步） | ~4-5 cycle |
| Bank conflict | 可能存在 | 无 |
| 同步方式 | `__syncthreads()`（block 级 barrier） | `_sync` 后缀确保同一 warp 内 lane 到达（warp 内天然 SIMT，无需额外 barrier） |
| 适用范围 | block 内所有 thread | 同一 warp 内 32 个 lane |


---


###### 语法
- `cudaMalloc`  ==地址有&==，d_X是刚刚初始化的无意义的值，cudaMalloc这个函数需要修改地址变量本身，让他指向真实 alloc 的地址 address，所以带&
- `cudaMemcpy(目标地址, 源地址, 拷贝大小(字节数), 拷贝方向);` ==地址没有 &==，因为不需要修改 d_X 或 h_X 这些指针变量本身，只需要知道指针变量存储的地址值即可读写地址上的数据
- `cudaDeviceSynchronize()` 插在2 个 stream kernel 之间可能破坏stream并发
- `cudaMemset` cudaMemset 是按字节填充，多用于初始化。例如 cudaMemset(d_data, 0, n * sizeof(float)); 初始化长度为 n 的数组；例如cudaMemset(d_int, 1, sizeof(int)); 是把 int 的 4 个 byte 都写成 0x01，也就是 01 01 01 01  


> [!tip] 面试追问：blockDim 为什么选 128 或 256，不选 1024？
> 每个 SM 的 register 数量有限（A100 为 65536 个 32-bit register），block 越大每个 thread 能分到的 register 越少，可能触发 register spilling（溢出到显存）。128/256 通常是 occupancy 和 register 压力的平衡点。用 `nvcc --ptxas-options=-v` 可查看 register 用量。


###### 基础编译命令

```sh
nvidia-smi --query-gpu=compute_cap --format=csv       # 查看当前 GPU 架构
nvidia-smi                                            # 查看 GPU 实时状态（显存、温度、利用率）

nvcc -O2 -o out main.cu && ./out                      # 基础编译 + 运行
nvcc -O2 -arch=sm_75 -o out main.cu                   # 指定 GPU 架构（T4 = sm_75，A100 = sm_80，H100 = sm_90）

compute-sanitizer ./out                               # 用 CUDA Memcheck 查内存越界 / 非法访问
nvcc -O2 --ptxas-options=-v -o out main.cu            # 查看 register 用量（lmem > 0 表示发生了 register spilling）
nvcc -O2 -ptx -o out.ptx main.cu                      # 查看生成的 PTX 汇编（可读的 GPU 中间表示）

# ── Nsight Compute（ncu）：单 kernel 深度分析 ──────────────────
ncu ./out                                             # 基础 profile（输出关键指标摘要）
ncu --kernel-name gemm_naive ./out                    # 只 profile 指定 kernel（避免 profile 所有 kernel，更快）

ncu --set basic ./out                                 # 预设 metric 集合 快速
ncu --set full  ./out                                 # 预设 metric 集合 全量

ncu -o report ./out                     # → 生成 report.ncu-rep 保存报告文件（下载到本地用 Nsight Compute GUI 打开）

                                        # 指定查看某几个关键 metric
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,l1tex__t_bytes.sum ./out

# ── Nsight Systems（nsys）：全局时间线，看 CPU/GPU 协作 ─────────
nsys profile --output report ./out      # → 生成 report.nsys-rep
```

*Nsight Compute 核心 metric 速查：*

| Metric | 含义 | 低了说明什么 |
|--------|------|------------|
| SM Throughput % | SM 计算利用率 | memory bound，计算没跑满 |
| DRAM Throughput % | HBM 带宽利用率 | 访存不连续或 cache 命中率高 |
| L1/L2 Hit Rate | Cache 命中率 | 数据复用差，考虑 shared memory tiling |
| Occupancy % | 活跃 warp / 最大 warp | register/smem 占太多，减小 block size |
| Warp Stall - Memory | warp 在等访存 | 典型 memory bound 症状 |
| Registers/Thread | 每 thread 寄存器用量 | 高 → 可能 spilling |

*典型 workflow：先用 `nsys` 找哪个 kernel 最耗时 → 再用 `ncu` 深挖那个 kernel。*


---


###### [模式] Elementwise Kernel：ReLU 前向与反向

*Question* 用 CUDA 实现 ReLU 的前向和反向。输入 X、输出 Y，形状均为 $(N,)$ 的一维 float 数组。

*原理* Elementwise 是最简单的 GPU 并行模式：每个 thread 处理一个元素，完全并行无依赖。核心是正确计算全局 index：`i = blockIdx.x * blockDim.x + threadIdx.x`，并做越界保护。

$$
\text{ReLU}(x) = \max(0, x), \qquad \frac{\partial L}{\partial x} = \begin{cases} \frac{\partial L}{\partial y} & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

*Solution*

```cpp
__global__ void relu_forward(const float* X, float* Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = fmaxf(X[i], 0.0f);
}

__global__ void relu_backward(const float* dY, const float* X, float* dX, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dX[i] = X[i] > 0.0f ? dY[i] : 0.0f;
}
// int block = 256; int grid = (n + block - 1) / block;
// relu_forward<<<grid, block>>>(X, Y, n);
```


**fmaxf:**
- max 通常用于整数类型。
- fmax 用于 double (双精度浮点数)。
- fmaxf 用于 float (单精度浮点数)。
- fmaxl 用于 long double (长双精度浮点数)。


> [!tip] 面试追问：这个 kernel 是 compute bound 还是 memory bandwidth bound？
> Memory bandwidth bound。每个 thread 读一个 float、写一个 float，几乎没有计算。
> 优化方向：
> 1. 向量化访存（`float4`），一个 thread 读写 4 个元素，减少 kernel launch 和 index 计算开销，提高 HBM 带宽利用率。
> 2. kernel-fusion，和前后其他计算融合一块儿


[[main 函数验证的完整 code block#带 main 函数的 relu_forward|完整代码（含 main）→]]

---



###### [模式] Shared Memory Block Reduction：并行求和

*Question* 对长度为 $N$ 的 float 数组求和，在一个 kernel 内用 shared memory 实现 block 内 tree reduction，多个 block 的结果用 `atomicAdd` 汇总。

*原理* Parallel reduction 是 softmax sum/max、LayerNorm mean/var 的基础。Tree reduction 每轮将 stride 减半：

```
Thread: 0  1  2  3  4  5  6  7
Round1: [0+4][1+5][2+6][3+7] .  .  .  .
Round2: [0+2][1+3] .  .  .  .  .  .
Round3: [0+1] .  .  .  .  .  .  .
Result: sdata[0]
```

*Solution*

```cpp
__global__ void reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];  // 动态 shared memory，启动时指定大小，所以需要在启动 kernel 的时候加上第三个参数
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? input[i] : 0.0f;  // 越界填 0，不影响求和
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();                      // 每轮都必须同步
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}
// int block = 256;  int grid  = (n + block - 1) / block;
// reduce_sum<<<grid, block, block * sizeof(float)>>>(d_in, d_out, n);
```

> [!bug] 常见错误
> - `__syncthreads()` 漏写：相邻 thread 读到未更新的值，结果不确定。
> - `stride` 初始为 `blockDim.x` 而不是 `blockDim.x / 2`：第一轮 `tid + stride` 会越界 shared memory。
> - 忘记对越界 thread 填 0：`i >= n` 时 sdata 应为 0 而不是未初始化内存。

> [!warning] Bank Conflict 问题
> 当 stride 为 1 时，相邻 thread 访问 `sdata[tid]` 和 `sdata[tid+1]`，连续访问无 conflict。但某些 stride 下可能出现多个 thread 访问同一 bank。更优的做法是改为 sequential addressing，或直接用 warp shuffle（见下题）。

[[main 函数验证的完整 code block#带 main 函数的 reduce_sum（基础版）|完整代码（含 main）→]]

---

###### [模式] Warp Shuffle Reduction

*Question* 用 `__shfl_down_sync` 实现 warp 内 reduce，不依赖 shared memory，再扩展到 block-level reduce sum/max。

*原理* Warp 内 32 个 thread 可以直接读对方寄存器，延迟低于 shared memory，且无 bank conflict。`__shfl_down_sync(mask, val, offset)` 让每个 lane 收到 `lane + offset` 的值：

```
offset=16: lane0←lane16, lane1←lane17, ... lane15←lane31
offset=8:  lane0←lane8,  lane1←lane9,  ...
...
offset=1:  lane0←lane1
最终 lane 0 = warp sum（其他 lane 为中间值）
```

*Solution*

```cpp
// ── warp-level reduce（复用以上 device 函数） ──
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);  // lane 之间互相读寄存器，无需 sm 
    return val;
}

// block-level reduce sum（多 warp 场景）
__global__ void block_reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float smem[];           // 每个 warp 存一个部分和，需 blockDim.x/32 个 float
    int tid  = threadIdx.x;
    int i    = blockIdx.x * blockDim.x + tid;
    int lane = tid % 32;
    int wid  = tid / 32;

    float val = (i < n) ? input[i] : 0.0f;
    val = warp_reduce_sum(val);  // warp 内归约
    if (lane == 0) smem[wid] = val;  // 只有 lane 0 保存完整的 warp 总和, 每个 warp 
    __syncthreads();                 // 的 lane 0 写入 shared memory

    int num_warps = blockDim.x / 32; 
    val = (tid < num_warps) ? smem[tid] : 0.0f; 
    if (wid == 0) val = warp_reduce_sum(val);   // 对所有 warp 结果再归约 只需第一个 warp 执行 reduce sum

    if (tid == 0) atomicAdd(output, val);  // 累加到全局输出
}
// block_reduce_sum<<<grid, block, (block/32)*sizeof(float)>>>(input, output, n);
```


*优化版本* 

每个 thread 读取两个元素（`i` 和 `i + blockDim.x`），提高内存吞吐；shared memory 将 block 内规约至 32 个元素后，最后一步用 warp shuffle 完成，避免最后几轮 shared memory 的同步开销。Host 端通过多轮 kernel 调用完成全局规约。

```cpp
// ---- device 工具函数（同 warp shuffle 章节） ----
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset); // lane 之间互相读寄存器，无需 sm
    return val;  // 只有 lane 0 持有最终和
}


// ---- kernel：每个 thread 读两个元素，hybrid shared mem + warp shuffle ----
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];  // 大小 = blockDim.x / 32 个 float（每 warp 一个槽）

    int tid  = threadIdx.x;
    int i    = blockIdx.x * blockDim.x * 2 + tid;  // 每个 block 覆盖 2*blockDim 个元素

    // 每个 thread 读两个元素，越界补 0
    float val = 0.0f;
    if (i < n)              val += input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];

    // warp 内规约（32 → 1），lane 0 写入 shared memory
    val = warp_reduce_sum(val);
	int lane = tid % 32;
	int wid = tid / 32;
	
	if (lane == 0) sdata[wid] = val;
	__syncthreads();

    // 最多 32 个 warp，用第一个 warp 再做一次 warp reduce
    int num_warps = blockDim.x / 32;
    val = (tid < num_warps) ? sdata[tid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    if (tid == 0) atomicAdd(output, val);
}


// -------------- host 调用函数，在 cpu 上 --------------
int reduce_sum(const float *d_input, float *d_output, int n) {
  const int block = 256;
  const int warpSize = 32; 
  float *d_temp1, *d_temp2;
  cudaMalloc(&d_temp1, n * sizeof(float));
  cudaMalloc(&d_temp2, n * sizeof(float));

  cudaMemcpy(d_temp1, d_input, n * sizeof(float), cudaMemcpyDeviceToDevice);
  int num_warps = block / warpSize; // 每个 block 有多少个 warp
  int cur_n = n;
  while (cur_n > 1) {  // 反复把数组长度压缩，直到只剩一个值
    int grid = (cur_n + block * 2 - 1) / (block * 2);  // 每个 block 每一轮最多处理  2*blockDim.x 个元素
    int smem = num_warps * sizeof(float);
    cudaMemset(d_temp2, 0, grid * sizeof(float));
    reduce_sum_kernel<<<grid, block, smem>>>(d_temp1, d_temp2, cur_n);
    cudaGetLastError();
    cudaDeviceSynchronize();
    float *tmp = d_temp1;
    d_temp1 = d_temp2;
    d_temp2 = tmp;
    cur_n = grid; // 下一轮输入 = 本轮 block 数
  }

  cudaMemcpy(d_output, d_temp1, sizeof(float), cudaMemcpyDeviceToDevice);
  cudaFree(d_temp1);
  cudaFree(d_temp2);
  return 0;
}
```

> [!tip] 优化要点
> - **每 thread 读 2 元素**：减少 block 数量，每个 thread 做更多工作，提升内存带宽利用率。
> - **Shared mem 大小 = blockDim/32**：不再需要整个 block 大小的 shared mem，只需每 warp 一个槽。
> - **最后 32→1 用 warp shuffle**：省去最后几轮 `__syncthreads()`，延迟更低。
> - **Host 多轮循环**：每轮将问题规模缩小 `block*2` 倍，O(log N) 轮后收敛为一个值。

[[main 函数验证的完整 code block#带 main 函数的 reduce_sum（优化版）|完整代码（含 main）→]]

---

###### Linear Layer 前向 + 反向 CUDA

*Question* 实现全连接层的前向和反向，输入 $X$（$B \times K$），权重 $W$（$N \times K$），偏置 $b$（$N$），输出 $Y$（$B \times N$）：

$$
Y = X W^\top + b, \qquad \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W, \qquad \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y}^\top X, \qquad \frac{\partial L}{\partial b} = \sum_{i=0}^{B-1} \frac{\partial L}{\partial Y_{i,:}}
$$

*原理* 三个操作各自对应一次 GEMM，$db$ 是一次行向规约。关键要记清矩阵形状和转置方向：

| 操作 | 矩阵乘法形式 | 输出形状 |
|------|-------------|---------|
| 前向 $Y$ | $(B \times K) \times (K \times N)$ → $X W^\top$ | $B \times N$ |
| 反向 $dX$ | $(B \times N) \times (N \times K)$ → $dY \cdot W$ | $B \times K$ |
| 反向 $dW$ | $(N \times B) \times (B \times K)$ → $dY^\top \cdot X$ | $N \times K$ |

*Solution — 前向（GEMM + 融合 bias）*

```cpp
// Y = X @ W^T + b
// X: [B, K], W: [N, K], b: [N], Y: [B, N]
// 每个 thread 计算 Y[row][col] 一个元素
__global__ void linear_forward(const float* X, const float* W, const float* b,
                                float* Y, int B, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // batch 维度 (0..B-1)
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // out 维度  (0..N-1)
    if (row >= B || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += X[row * K + k] * W[col * K + k];   // W 按行存 [N, K]，W[col][k] = W[col*K+k]
    Y[row * N + col] = sum + b[col];               // 融合 bias 加法，省掉单独的 bias kernel
}
// dim3 block(32, 32); dim3 grid((N+31)/32, (B+31)/32);
// linear_forward<<<grid, block>>>(X, W, b, Y, B, K, N);
```

> [!tip] 为什么 W 存成 `[N, K]` 而不是 `[K, N]`？
> PyTorch `nn.Linear` 的 `weight` 形状是 `[out_features, in_features]` = `[N, K]`。
> 这样 `W[col][k]` 在内存是 `W[col*K + k]`，同一 col 内沿 k 连续访问 → 读 W 时该 thread 是连续的。
> 而同一 warp 内不同 col 访问同一 k 时，warp 的 threadIdx.x（col 方向）连续 → 每次读 W 的一行是 coalesced ✅

*Solution — 反向（dX / dW / db）*

```cpp
// ── 1. dX = dY @ W，形状 [B, K] ──
// dX[b][k] = sum_n dY[b][n] * W[n][k]
__global__ void linear_backward_dx(const float* dY, const float* W,
                                    float* dX, int B, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // batch
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // in_feat k
    if (row >= B || col >= K) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++)
        sum += dY[row * N + n] * W[n * K + col];
    dX[row * K + col] = sum;
}

// ── 2. dW = dY^T @ X，形状 [N, K] ──
// dW[n][k] = sum_b dY[b][n] * X[b][k]
__global__ void linear_backward_dw(const float* dY, const float* X,
                                    float* dW, int B, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // out_feat n
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // in_feat k
    if (row >= N || col >= K) return;

    float sum = 0.0f;
    for (int b = 0; b < B; b++)
        sum += dY[b * N + row] * X[b * K + col];
    dW[row * K + col] = sum;
}

// ── 3. db = sum_b dY[b]，形状 [N]，每个 thread 处理一个 out_feat ──
__global__ void linear_backward_db(const float* dY, float* db, int B, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float sum = 0.0f;
    for (int b = 0; b < B; b++)
        sum += dY[b * N + n];
    db[n] = sum;
}
```

> [!bug] 常见错误：dW 的 GEMM 维度搞错
> - dW 的 "M" 是 N（out_features），"N" 是 K（in_features），"K" 是 B（batch）。
> - 把 B 和 N 搞反会导致形状错误，建议面试时先写出矩阵形状再对应 row/col。

*融合优化版：GEMM + bias + ReLU/SiLU 一个 kernel*

```cpp
// 在 tiled GEMM 基础上，写完 C[row][col] 前，融合 bias 和激活函数
// 省去单独的 bias kernel launch 和 activation kernel launch
__global__ void linear_bias_relu_forward(const float* X, const float* W,
                                          const float* b, float* Y,
                                          int B, int K, int N) {
    __shared__ float sX[32][32];
    __shared__ float sW[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + 31) / 32; t++) {
        int xCol = t * 32 + threadIdx.x;
        int wCol = t * 32 + threadIdx.y;   // W 是 [N, K]，沿 K 方向分 tile
        sX[threadIdx.y][threadIdx.x] = (row < B && xCol < K) ? X[row * K + xCol] : 0.0f;
        sW[threadIdx.y][threadIdx.x] = (col < N && wCol < K) ? W[col * K + wCol] : 0.0f;
        __syncthreads();
        for (int k = 0; k < 32; k++) sum += sX[threadIdx.y][k] * sW[threadIdx.x][k];
        __syncthreads();
    }

    if (row < B && col < N) {
        float out = sum + b[col];
        Y[row * N + col] = fmaxf(out, 0.0f);   // 融合 ReLU，换成 SiLU 改这一行即可
    }
}
```

> [!note] 实际工程中怎么做？
> 生产代码不会手写 GEMM，而是：
> 1. **cuBLAS `cublasSgemm`**：调用 NVIDIA 优化的 BLAS，峰值算力接近理论上限。
> 2. **cuBLASLt**：更灵活，支持 FP16/BF16、行/列 layout、epilogue（bias + 激活融合）。
> 3. **CUTLASS / cuDNN**：更高级的模板库，支持 Tensor Core。
>
> 面试中手写 kernel 的价值是展示对 **coalesced access、shared memory tiling、bank conflict** 的理解，而不是替代 cuBLAS。

---

###### Naive CUDA GEMM

*Question* 用 CUDA 实现矩阵乘法 $C = A \times B$（$M \times K$ 乘 $K \times N$），每个 thread 负责输出矩阵 $C$ 的一个元素。

*原理* 2D thread block 自然对应矩阵行列：`row = blockIdx.y * blockDim.y + threadIdx.y`，`col = blockIdx.x * blockDim.x + threadIdx.x`，每个 thread 沿 $K$ 做点积。

*Solution*

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C,
                            int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
// dim3 block(32, 32);
// dim3 grid((N + 31) / 32, (M + 31) / 32);
// gemm_naive<<<grid, block>>>(A, B, C, M, K, N);
```

> [!bug] 性能瓶颈
> 访问 `B[k * N + col]` 时，同一 warp 内 32 个 thread（同一行，不同列）访问 `B` 的不同行 —— stride = $N$，相邻 thread 地址相差 $N$ 个 float，**无法 coalesced access**，HBM 带宽利用率极低。这正是 tiling 要解决的问题。


> [!tip] 为什么 col 要映射到 threadIdx.x？
> ```cpp
> dim3 block(32, 32);
> // 展平后：threadIdx.x + threadIdx.y * 32
> // Warp 0 = threadIdx.y=0, threadIdx.x=0..31  → 矩阵第0行的32个列
> // Warp 1 = threadIdx.y=1, threadIdx.x=0..31  → 矩阵第1行的32个列
> ```
> 同一 Warp（同一行，x 连续）访问 `C[row][col]`，col 是 x 方向，地址连续 → **coalesced access** ✅
>
> 反过来，若用 `threadIdx.y` 作列方向，同一 Warp 访问跨行地址 → 不连续 → **性能暴跌** 。


[[main 函数验证的完整 code block#带 main 函数的 gemm_naive|完整代码（含 main）→]]

---

###### Shared Memory Tiled GEMM

*Question* 在 naive GEMM 基础上，用 shared memory 做分块。每次把 $A$ 和 $B$ 当前 tile（大小 $\text{TS} \times \text{TS}$）搬入 `__shared__`，让 block 内所有 thread 复用同一份数据，减少 global memory 读取次数。

*原理* 沿 $K$ 维分 $\lceil K/\text{TS} \rceil$ 个 tile，每次迭代：① block 内所有 thread **协作** 加载 A 的列 tile 和 B 的行 tile 到 shared memory；② 同步；③ 在 shared memory 内做 tile 内点积累加；④ 同步后进入下一 tile。

Global memory 访问从每次迭代 $O(K)$ 降到 $O(K/\text{TS})$ 次，理论加速比为 TS。

*Solution*

```cpp
#define TS 32

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                            int M, int K, int N) {
    __shared__ float sA[TS][TS];
    __shared__ float sB[TS][TS];
	
	// target: 一个 block 计算一个 (TS,TS) 的 C[row][col] tile
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        // 协作加载 tile（越界填 0），按照 K 维度来划分 tile，t 表示当前正在处理第几个 tile
        int aCol = t * TS + threadIdx.x;  // A[row][aCol]
        int bRow = t * TS + threadIdx.y;  // B[bRow][col]
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();              // 等所有 thread 加载完毕

        for (int k = 0; k < TS; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];  // 每个 thread 负责算一个位置，所有 thread 累加
        __syncthreads();              // 等所有 thread 用完 tile，再覆盖 shared memory
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

> [!tip] 两个 `__syncthreads()` 的作用
> - **加载后**：防止某些 thread 还没写完 sA/sB，另一些 thread 就开始读，导致读到旧数据。
> - **计算后**：防止下一轮迭代覆盖 sA/sB 时，某些 thread 还在读当前 tile。两个缺一不可。

> [!warning] sA 的 Bank Conflict
> `sA[threadIdx.y][k]`：同一 warp 内 32 个 thread（threadIdx.y 不同，k 固定）访问同一列 → 32 个 thread 落在同一 bank → **32-way bank conflict** 。
> 解决方法：Padding，将 `float sA[TS][TS+1]`，每行多一个 padding float，使各行起始地址错开，消除 conflict。

---

###### Row-wise Softmax CUDA

*Question* 对形状 $(M, N)$ 的矩阵，每行独立做 softmax，每个 block 负责一行，用 warp shuffle 实现 block 内 max/sum 归约，要求数值稳定。

*原理* 每行需三遍扫描：① 找最大值（warp reduce max）→ ② 计算 exp 累加 sum（warp reduce sum）→ ③ 除以 sum 归一化。每遍扫描的 block-level 归约模式与 Reduction 题相同。

*Solution*

```cpp
// 一个 block 处理一行，blockDim.x 必须为 32 的倍数
__global__ void softmax_forward(const float* X, float* Y, int N) {
    extern __shared__ float smem[];   // 大小：(blockDim.x / 32) 个 float
    int row       = blockIdx.x;
    int tid       = threadIdx.x;
    int num_warps = blockDim.x / 32;

    // ── Pass 1: find row max ──
    float max_val = -1e38f;
    for (int j = tid; j < N; j += blockDim.x)
        max_val = fmaxf(max_val, X[row * N + j]);
    max_val = warp_reduce_max(max_val);
    if (tid % 32 == 0) smem[tid / 32] = max_val;
    __syncthreads();
    max_val = (tid < num_warps) ? smem[tid] : -1e38f;
    if (tid < 32) max_val = warp_reduce_max(max_val);
    if (tid == 0) smem[0] = max_val;
    __syncthreads();
    max_val = smem[0];

    // ── Pass 2: compute exp, accumulate sum ──
    float sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float e = expf(X[row * N + j] - max_val);
        Y[row * N + j] = e;
        sum += e;
    }
    sum = warp_reduce_sum(sum);
    if (tid % 32 == 0) smem[tid / 32] = sum;
    __syncthreads();
    sum = (tid < num_warps) ? smem[tid] : 0.0f;
    if (tid < 32) sum = warp_reduce_sum(sum);
    if (tid == 0) smem[0] = sum;
    __syncthreads();
    sum = smem[0];

    // ── Pass 3: normalize ──
    for (int j = tid; j < N; j += blockDim.x)
        Y[row * N + j] /= sum;
}

// int block = 256; size_t smem_size = (block / 32) * sizeof(float);
// softmax_forward<<<M, block, smem_size>>>(X, Y, N);
```

> [!tip] 面试追问：三遍扫描能否合并成一遍？
> 可以，用 **Online Softmax**（Milakov & Gimelshein 2018）：一遍扫描同时维护当前最大值 $m$ 和修正后的累加 sum $d$，当最大值更新时对 $d$ 乘以修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$。Flash Attention 的 online softmax 正是这一思想，可以减少 global memory 读写次数。详见 [[实习面试准备/coding-手写深度学习/C++手写算子.md#Online Softmax 核心公式|Online Softmax 核心公式]]。

[[main 函数验证的完整 code block#带 main 函数和 cpu 验证的 softmax|完整代码（含注释解释和 main）→]]

---


###### Online Softmax 核心公式

普通稳定 softmax 通常是三遍：

1. 扫描得到 $m = \max_i x_i$
2. 扫描得到 $d = \sum_i e^{x_i-m}$
3. 输出 $y_i = e^{x_i-m}/d$

Online softmax 把前两步合并：扫描元素/一个 tile 时，同时维护“到目前为止的最大值” $m$ 和“按当前最大值缩放后的分母” $d$。

单个元素版本：

$$
m_{new} = \max(m_{old}, x), \qquad
d_{new} = d_{old} \cdot e^{m_{old}-m_{new}} + e^{x-m_{new}}
$$

为什么要乘 $e^{m_{old}-m_{new}}$？因为旧的 $d_{old}$ 是按旧最大值 $m_{old}$ 缩放的：

$$
d_{old} = \sum_{i<j} e^{x_i - m_{old}}
$$

如果发现了更大的 $m_{new}$，旧分母必须整体改成以 $m_{new}$ 为基准：

$$
\sum_{i<j} e^{x_i - m_{new}}
= e^{m_{old}-m_{new}} \sum_{i<j} e^{x_i-m_{old}}
$$

最后计算 softmax 

$$
y_j = \frac{e^{x_j - m}}{\sum_{k=0}^{N-1} e^{x_k - m}}, \qquad m = \max_{0 \le k < N} x_k
$$

$m$ 对应代码里的 `row_m`，分母 `row_d` 由各 lane 的局部分母换算后求和得到：

$$
d_{\text{scaled}} = d \cdot \exp(m - row\_m)
$$

每个 lane 原来维护的 $d = \sum_{j \in S} e^{x_j - m}$（以本地最大 $m$ 为基准），换到全局基准 $row\_m$ 后再汇总：

$$
row\_d = \sum_{\text{lane}} d_{\text{scaled,lane}} = \sum_j e^{x_j - row\_m}
$$

对应代码：`float d_scaled = d * expf(m - row_m);`，随后 `warp_allreduce_sum` 对所有 lane 求和。


*完整 CUDA 写法：Online Softmax row-wise kernel*  
这个版本是“每个 warp 处理一行”。第一遍用 online softmax 合并 max 和 denominator；第二遍写出结果。注意：如果任务要求输出完整 softmax，最后归一化这遍通常省不掉。Online softmax 主要省的是“单独找 max + 单独求 sum”这两遍。 

```cpp
// warp 内 all-reduce：每个 lane 最后都拿到 sum
__device__ float warp_allreduce_sum(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

// warp 内 all-reduce max：每个 lane 最后都拿到 max
__device__ float warp_allreduce_max(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}

// 合并两个 online softmax state：
// state A: (m1, d1) 表示 d1 = sum exp(x - m1)
// state B: (m2, d2) 表示 d2 = sum exp(x - m2)
__device__ void merge_online_state(float m2, float d2, float& m1, float& d1) {
    float m_new = fmaxf(m1, m2);
    d1 = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
    m1 = m_new;
}

// X/Y shape: [M, N]
// 一个 warp 处理一行；blockDim.x 可以是 128/256，即一个 block 里有多个 warp、处理多行
__global__ void online_softmax_forward(const float* X, 
								  float* Y, int M, int N) {
    int tid       = threadIdx.x;
    int lane      = tid & 31;
    int warp_id   = tid >> 5;
    int num_warps = blockDim.x >> 5;

    int row = blockIdx.x * num_warps + warp_id;  // 每个 warp 处理一行（一个 blockIdx.x 有 num_warps）
    if (row >= M) return;

    // Step 1: 每个 lane 扫自己负责的列，维护本 lane 的 online state
    float m = -INFINITY;
    float d = 0.0f;

    for (int j = lane; j < N; j += 32) {   // 一个 warp 有 32 个 lane，每个 lane 处理一列
        float x = X[row * N + j];  
        float m_new = fmaxf(m, x);
        d = d * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }

    // Step 2: 把 32 个 lane 的 (m, d) 合并成整行的 (row_m, row_d)
    // 先求全局 max
    float row_m = warp_allreduce_max(m);

    // 每个 lane 的 d 原本基于自己的 m，要统一缩放到 row_m 基准
    float d_scaled = d * expf(m - row_m);
    float row_d = warp_allreduce_sum(d_scaled);

    // Step 3: 写出 softmax。这里需要再读一遍 X，因为最终 y_j 依赖全局 row_m/row_d
    for (int j = lane; j < N; j += 32) {
        Y[row * N + j] = expf(X[row * N + j] - row_m) / row_d;
    }
}
// int block = 128;                         // 4 warps/block
// int rows_per_block = block / 32;         // 计算一个 block 可以处理多少行
// int grid = (M + rows_per_block - 1) / rows_per_block;  // 计算需要多少个 block 来处理所有行
// online_softmax_forward<<<grid, block>>>(d_X, d_Y, M, N);
```

> [!warning] 注意：Row-wise softmax 不一定能真正“一遍输出”
> Online softmax 可以把“找 max + 求 sum”从两遍合成一遍；但如果最后要把每个 $y_i$ 写出来，通常还需要再扫一遍输入来做 `expf(X[i] - m) / d`。
> 真正厉害的是 Flash Attention：它不需要显式写出整行 softmax，而是边扫 K/V tile，边维护 $(m, \ell, O)$，所以可以避免实例化完整 attention 矩阵。

*Flash Attention 里的完整最小 CUDA 写法（教学版）*  
下面这个 kernel 是单头 attention 的 online-softmax 版本：一个 warp 处理一个 `(batch b, query i)`，不存 `scores[T]`，也不存完整 attention 矩阵。为简单清晰，假设 `d <= 32`，每个 lane 负责一个 hidden dimension。

```cpp
// Q, K, V, O shape: [B, T, d]
// grid = (T, B)，一个 warp/block 处理一个 query token
// 约束：blockDim.x = 32, d <= 32
__global__ void flash_attention_forward_warp(const float* Q,
                                             const float* K,
                                             const float* V,
                                             float* O,
                                             int B, int T, int d,
                                             int causal) {
    int lane = threadIdx.x;   // 0..31
    int i    = blockIdx.x;    // query position
    int b    = blockIdx.y;    // batch id

    if (lane >= 32 || b >= B || i >= T) return;

    float scale = rsqrtf((float)d);

    // 每个 lane 负责输出向量的一个维度 r=lane
    float q_r = (lane < d) ? Q[b * T * d + i * d + lane] : 0.0f;
    float acc = 0.0f;         // 未归一化 accumulator: sum exp(score - m) * V[j, r]
    float m   = -INFINITY;    // 当前已看过 key 的最大 score
    float l   = 0.0f;         // 当前分母: sum exp(score - m)

    for (int j = 0; j < T; j++) {
        if (causal && j > i) break;

        // Step 1: 计算 score_j = dot(Q_i, K_j) / sqrt(d)
        float k_r = (lane < d) ? K[b * T * d + j * d + lane] : 0.0f;
        float partial = q_r * k_r;
        float dot = warp_allreduce_sum(partial);   // 所有 lane 都得到 dot
        float score = dot * scale;

        // Step 2: online softmax 更新 m 和 l
        float m_new = fmaxf(m, score);
        float old_scale = expf(m - m_new);
        float p = expf(score - m_new);

        // Step 3: 同步更新输出 accumulator
        // 旧 acc 原本基于旧 m，需要乘 old_scale；新 token 权重是 p
        float v_r = (lane < d) ? V[b * T * d + j * d + lane] : 0.0f;
        acc = acc * old_scale + p * v_r;
        l = l * old_scale + p;
        m = m_new;
    }

    // Step 4: 最后归一化并写回 HBM。没有写过 scores/attention matrix。
    if (lane < d) {
        O[b * T * d + i * d + lane] = acc / l;
    }
}
// dim3 grid(T, B); int block = 32;
// flash_attention_forward_warp<<<grid, block>>>(Q, K, V, O, B, T, d, /*causal=*/0);
```

这个教学版和真实 Flash Attention 的关系：

- 相同点：都用 online softmax 维护 $m, \ell, O$，不显式存 $T \times T$ attention matrix。
- 简化点：这里是一个 warp 处理一个 query，且 `d <= 32`；真实实现会用 Q/K/V tile、shared memory、多个 warp/CTA 协作、向量化加载、处理更大的 head dim。
- 面试重点：公式和 IO 思路比手写工业级 kernel 更重要，即 **计算量仍是 $O(T^2d)$，但 HBM 读写从 materialize attention matrix 的 $O(T^2)$ 降下来** 。

---

###### LayerNorm 前向 CUDA

*Question* 对形状 $(M, N)$ 的输入，每行做 LayerNorm：计算均值和方差，归一化后仿射变换 $y = \gamma \hat{x} + \beta$。每个 block 处理一行。

$$
\mu = \frac{1}{N}\sum_{j=0}^{N-1} x_j, \qquad
\sigma^2 = \frac{1}{N}\sum_{j=0}^{N-1}(x_j - \mu)^2
$$

$$
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad
y_j = \gamma_j \hat{x}_j + \beta_j
$$

代码里保存 `rstd` $= 1/\sqrt{\sigma^2+\varepsilon}$ 供反向复用，避免重算开方。

*原理* 需要两遍 block-level 归约：① 算均值；② 算方差。模式与 Softmax 完全相同，只需把 max 换成 sum。保存 `rstd`（即 $1/\sqrt{\text{var}+\varepsilon}$）供反向使用。

*Solution*

```cpp
__global__ void layernorm_forward(const float* X, const float* gamma, const float* beta,
                                   float* Y, float* mean_out, float* rstd_out,
                                   int N, float eps) {
    extern __shared__ float smem[];
    int row       = blockIdx.x;
    int tid       = threadIdx.x;
    int num_warps = blockDim.x / 32;

    // ── Pass 1: mean ──
    float sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) sum += X[row * N + j];  
    sum = warp_reduce_sum(sum);
    if (tid % 32 == 0) smem[tid / 32] = sum;  // 每个 warp 取第一个线程（lane0）读 sum
    __syncthreads();
    sum = (tid < num_warps) ? smem[tid] : 0.0f;
    if (tid < 32) sum = warp_reduce_sum(sum);   // 只需要第一个 warp 来算 smem 上面的 sum
    if (tid == 0) smem[0] = sum / N;
    __syncthreads();
    float mean = smem[0];
    if (tid == 0 && mean_out) mean_out[row] = mean;

    // ── Pass 2: variance ──
    float var = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float d = X[row * N + j] - mean;
        var += d * d;
    }
    var = warp_reduce_sum(var);
    if (tid % 32 == 0) smem[tid / 32] = var;
    __syncthreads();
    var = (tid < num_warps) ? smem[tid] : 0.0f;
    if (tid < 32) var = warp_reduce_sum(var);
    if (tid == 0) smem[0] = rsqrtf(var / N + eps);
    __syncthreads();
    float rstd = smem[0];
    if (tid == 0 && rstd_out) rstd_out[row] = rstd;

    // ── Pass 3: normalize + affine ──
    for (int j = tid; j < N; j += blockDim.x) {
        float xhat = (X[row * N + j] - mean) * rstd;
        Y[row * N + j] = gamma[j] * xhat + beta[j];
    }
}
// int block = 256;  size_t smem = (block / 32) * sizeof(float);
// layernorm_forward<<<M, block, smem>>>(X, gamma, beta, Y, nullptr, nullptr, N, 1e-5f);
```

> [!tip] 为什么保存 rstd 而不是 var？
> 反向传播需要 $1/\sqrt{\text{var}+\varepsilon}$，直接保存 rstd 避免反向再做一次 `rsqrtf`，这是 PyTorch `F.layer_norm` 实现策略。

[[main 函数验证的完整 code block#带 main 函数的 layernorm 的 cpu 验证 | 完整函数 →]]

---

###### Matrix Transpose CUDA

*Question* 用 CUDA 实现矩阵转置 $B = A^\top$，$A$ 形状 $(M, N)$，要求 coalesced memory access（读 A 合并，写 B 也合并）。

*原理* Naive 转置读 A 合并但写 B 非合并（跨行跳跃），用 shared memory 作中转可两端都合并：① 从 A 合并读一个 tile 到 shared memory；② 从 shared memory 转置读出，合并写入 B。

*Solution*

```cpp
#define TS 32

__global__ void transpose(const float* A, float* B, int M, int N) {
    // padding +1 消除 bank conflict（每行写时相邻 thread 访问不同 bank）
    __shared__ float tile[TS][TS + 1];

    // 从 A 合并读入 tile（threadIdx.x 变化 → A 的列地址连续）
    int aRow = blockIdx.y * TS + threadIdx.y;
    int aCol = blockIdx.x * TS + threadIdx.x;
    if (aRow < M && aCol < N)
        tile[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
    __syncthreads();

    // 转置后写入 B（交换 blockIdx 是 block 级别的转置坐标交换）
    // threadIdx.x 变化 → B 的列地址连续，这个列是 x 是为了保证读写内存连续
    int bRow = blockIdx.x * TS + threadIdx.y;  
    int bCol = blockIdx.y * TS + threadIdx.x;
    if (bRow < N && bCol < M)
        B[bRow * M + bCol] = tile[threadIdx.x][threadIdx.y];  // 实际元素级别的转置靠 tile 读取时交换下标
}
// dim3 block(TS, TS); dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS);
// transpose<<<grid, block>>>(A, B, M, N);
```

> [!warning] 为什么需要 padding `[TS][TS+1]`？
>
> **Shared memory bank 基础**
> Shared memory 被切成 32 个 bank，每个 bank 宽 4 字节（1 个 float）。一个 float 的 bank 编号 = `(该 float 的 shared mem 偏移量 / 4) % 32`。同一 warp 的 32 个 thread **同时** 访问 shared mem，如果多个 thread 落在同一 bank → 串行化 → bank conflict。
>
> **写入时（无 conflict）**
> 写 `tile[threadIdx.y][threadIdx.x]`，同一 warp：threadIdx.y 相同，threadIdx.x = 0..31。
> 访问的是同一行的 32 个连续 float：`tile[y][0], tile[y][1], ..., tile[y][31]`
> 行内偏移依次 +1 → bank 编号依次 +1 → 32 个 thread 落在 32 个不同 bank ✅
>
> **读出时（有 conflict）**
> 读 `tile[threadIdx.x][threadIdx.y]`，同一 warp：threadIdx.y 相同（设为 `c`），threadIdx.x = 0..31。
> 访问的是同一**列** 的 32 个 float：`tile[0][c], tile[1][c], ..., tile[31][c]`
> 在 row-major 布局下，`tile[row][col]` 的偏移 = `row * TS + col`
> 所以这 32 次访问的偏移为 `0*TS+c, 1*TS+c, ..., 31*TS+c`
> Bank 编号 = `(row * TS + c) % 32`，当 TS = 32 时：
> `(row * 32 + c) % 32 = c % 32` —— 对所有 row 都相同！
> → 32 个 thread 全部落在 bank `c` → **32-way conflict** ❌
>
> **padding 后（stride 不再是 32 的倍数）**
> 改为 `tile[TS][TS+1]`，每行占 `TS+1 = 33` 个 float。
> 偏移变为 `row * 33 + c`，bank = `(row * 33 + c) % 32`。
> 因为 `33 % 32 = 1`，所以 bank = `(row * 1 + c) % 32 = (row + c) % 32`
> row = 0..31 → bank 编号依次为 `c, c+1, ..., c+31`（mod 32）→ 32 个不同 bank ✅
> 多的那 1 列纯粹是"占位"，不存任何有效数据，只为错开 bank。

---

###### Scaled Dot-Product Attention 前向 CUDA

*Question* 实现单头 Scaled Dot-Product Attention 的 CUDA 前向，$Q, K, V$ 形状均为 $(B, T, d)$，每个 block 负责一个 $(b, i)$ 的查询向量，对所有 $T$ 个 key 做 softmax 后与 $V$ 加权求和。

*原理*

$$
\text{score}_j = \frac{Q[b,i,:] \cdot K[b,j,:]}{\sqrt{d}}, \quad \alpha = \text{softmax}(\text{score}), \quad \text{out}[b,i,:] = \sum_j \alpha_j V[b,j,:]
$$

*Solution*

```cpp
// 一个 block 处理一个 (batch b, query i)
// blockDim.x = 32（单 warp），适用于 T <= 32 的小序列
__global__ void attention_forward(const float* Q, const float* K, const float* V,
                                   float* out, int T, int d) {
    extern __shared__ float scores[];   // 动态 shared mem，存 T 个 score，大小由第三个启动参数决定

    int b   = blockIdx.y;               // batch 维度
    int i   = blockIdx.x;               // query 位置（第 i 个 token）
    int tid = threadIdx.x;              // warp 内 lane，每个 thread 负责部分 k/j
    float scale = rsqrtf((float)d);     // 1/sqrt(d)，即 scaled dot-product 的缩放因子

    // ── Step 1: 计算 attention scores ────────────────────────────────────────
    // score[j] = Q[b,i,:] · K[b,j,:] / sqrt(d)
    // 每个 thread 负责一部分 j，步长 blockDim.x，循环覆盖所有 T 个 key
    for (int j = tid; j < T; j += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++)                         // 沿 head_dim 做点积
            dot += Q[b*T*d + i*d + k] * K[b*T*d + j*d + k];
        scores[j] = dot * scale;                             // 写入 shared mem，后续 softmax 用
    }
    __syncthreads();    // 等所有 thread 写完 scores，再读

    // ── Step 2: Softmax（数值稳定版） ────────────────────────────────────────
    // 先找全局最大值，再做 exp(x - max)，防止 exp 上溢
    float max_val = -1e38f;
    for (int j = tid; j < T; j += blockDim.x) max_val = fmaxf(max_val, scores[j]);
    max_val = warp_reduce_max(max_val);                      // warp 内 all-reduce，每个 lane 都拿到全局 max

    float sum = 0.0f;
    for (int j = tid; j < T; j += blockDim.x) {
        scores[j] = expf(scores[j] - max_val);              // 减 max 后 exp，结果写回 shared mem
        sum += scores[j];
    }
    sum = warp_reduce_sum(sum);                              // warp 内 all-reduce 求分母
    for (int j = tid; j < T; j += blockDim.x) scores[j] /= sum;   // 归一化：得到 softmax 权重
    __syncthreads();    // 等归一化写完，Step 3 读时保证数据一致

    // ── Step 3: 加权求和 V，写出结果 ─────────────────────────────────────────
    // out[b,i,k] = sum_j softmax(score[j]) * V[b,j,k]
    // 每个 thread 负责输出向量的一部分维度 k
    for (int k = tid; k < d; k += blockDim.x) {
        float val = 0.0f;
        for (int j = 0; j < T; j++)                         // 所有 key 位置加权累加
            val += scores[j] * V[b*T*d + j*d + k];
        out[b*T*d + i*d + k] = val;                         // 写回 global mem
    }
}
// dim3 grid(T, B);      // x 轴 = query 位置，y 轴 = batch
// int block = 32;       // 单 warp
// size_t smem = T * sizeof(float);
// attention_forward<<<grid, block, smem>>>(Q, K, V, out, T, d);
```

> [!warning] 上面实现的问题：$O(T^2)$ 显存
> 完整 scores 矩阵大小为 $(B, T, T)$。T 很大时（如 4096）会撑爆 shared memory 或 global memory。

> [!tip] Flash Attention 核心思想（面试必答扩展）
> **分块计算 softmax，无需实例化完整 attention 矩阵**，显存从 $O(T^2)$ 降至 $O(T)$。这里本质考点就是 [[实习面试准备/coding-手写深度学习/C++手写算子.md#Online Softmax 核心公式|Online Softmax]]；完整教学版 kernel 见同一节里的 `flash_attention_forward_warp`。
> 1. 将 $Q, K, V$ 沿 $T$ 维切成 tile。
> 2. 对每个 Q-tile 遍历所有 K/V-tile，用 **online softmax** 维护当前最大值 $m$、修正后的 sum $\ell$ 和输出 accumulator $O$，增量更新输出。
> 3. 全部 tile 遍历完后，output 已在寄存器中累积完成，一次写回 HBM。
> 计算量与 naive attention 相同，但 HBM 读写次数从 $O(T^2)$ 降到 $O(T)$，是 IO-bound 问题的典型优化。

> [!note] Causal Mask
> Step 1 之后、softmax 之前，对 $j > i$ 的位置设 `scores[j] = -1e9f`，softmax 后对应权重趋近 0，模型看不到未来 token。

---


###### RMSNorm 前向 CUDA

*Question* 实现 RMSNorm（LLaMA 用的归一化），每行除以 RMS 后仿射：

$$
\text{RMS}(x) = \sqrt{\frac{1}{N}\sum_{j=0}^{N-1} x_j^2 + \varepsilon}, \qquad y_j = \frac{x_j}{\text{RMS}(x)} \cdot \gamma_j
$$

*原理* 比 LayerNorm 少一步均值归约，只需一遍 $\sum x^2$ 的 block reduce，再归一化。

*Solution*

```cpp
__global__ void rmsnorm_forward(const float* X, const float* gamma,
                                 float* Y, int N, float eps) {
    extern __shared__ float smem[];
    int row       = blockIdx.x;
    int tid       = threadIdx.x;
    int num_warps = blockDim.x / 32;

    // ── Pass 1: 计算 sum(x^2) ──
    float ss = 0.0f;
    for (int j = tid; j < N; j += blockDim.x)
        ss += X[row * N + j] * X[row * N + j];
    ss = warp_reduce_sum(ss);
    if (tid % 32 == 0) smem[tid / 32] = ss;
    __syncthreads();
    ss = (tid < num_warps) ? smem[tid] : 0.0f;
    if (tid < 32) ss = warp_reduce_sum(ss);
    if (tid == 0) smem[0] = rsqrtf(ss / N + eps);   // 1 / RMS
    __syncthreads();
    float rrms = smem[0];

    // ── Pass 2: 归一化 + 仿射 ──
    for (int j = tid; j < N; j += blockDim.x)
        Y[row * N + j] = X[row * N + j] * rrms * gamma[j];
}
// int block = 256; size_t smem = (block / 32) * sizeof(float);
// rmsnorm_forward<<<M, block, smem>>>(X, gamma, Y, N, 1e-6f);
```

> [!tip] RMSNorm vs LayerNorm 区别
> | | LayerNorm | RMSNorm |
> |--|-----------|---------|
> | 均值归约 | ✅ 两遍（mean + var） | ❌ 省略 |
> | 计算 | $\hat x = (x-\mu)/\sigma$ | $\hat x = x / \text{RMS}$ |
> | 性能 | 稍慢（多一次 reduce） | 更快 |
> | 使用 | BERT, GPT-2 | LLaMA, Mistral, Gemma |
> 
> RMSNorm 假设"均值归零"对性能无关紧要，省去均值计算。

---

###### GELU / SiLU Elementwise Kernel

*Question* 实现 GELU 和 SiLU（Swish）激活函数的 CUDA kernel：

$$
\text{GELU}(x) \approx x \cdot \sigma(1.702 \cdot x), \qquad \text{SiLU}(x) = x \cdot \sigma(x), \quad \sigma(x) = \frac{1}{1+e^{-x}}
$$

*Solution*

```cpp
// ── GELU（tanh 近似版，GPT-2 用的） ──
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__global__ void gelu_forward(const float* X, float* Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = X[i];
    const float c = 0.7978845608f;            // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    Y[i] = 0.5f * x * (1.0f + tanhf(inner));
}

// ── SiLU / Swish（LLaMA FFN 用的） ──
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
__global__ void silu_forward(const float* X, float* Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = X[i];
    Y[i] = x / (1.0f + expf(-x));
}

// ── SiLU 反向 ──
// d(SiLU)/dx = sigma(x) + x * sigma(x) * (1 - sigma(x))
//            = sigma(x) * (1 + x * (1 - sigma(x)))
__global__ void silu_backward(const float* dY, const float* X, float* dX, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = X[i];
    float sig = 1.0f / (1.0f + expf(-x));
    dX[i] = dY[i] * sig * (1.0f + x * (1.0f - sig));
}
// int block = 256; int grid = (n + block - 1) / block;
// gelu_forward<<<grid, block>>>(X, Y, n);
```

> [!tip] 常见激活函数速查
> | 激活函数 | 公式 | 用在 |
> |---------|------|------|
> | ReLU | $\max(0,x)$ | ResNet, ViT |
> | GELU | $x\cdot\sigma(1.702x)$ | BERT, GPT |
> | SiLU/Swish | $x \cdot \sigma(x)$ | LLaMA FFN |
> | SwiGLU | $\text{SiLU}(xW_1) \odot (xW_2)$ | LLaMA, PaLM |
>
> SwiGLU = element-wise multiply 两条线性投影，其中一条过 SiLU。面试如果问 LLaMA FFN，要知道它用 SwiGLU 而不是普通 GELU。

---

###### SwiGLU Kernel（LLaMA FFN 核心）CUDA

*Question* LLaMA 的 FFN 由三个线性层 + SwiGLU 组成，请实现 SwiGLU 这一步的 CUDA kernel：

$$
\text{SwiGLU}(\text{gate}, \text{up}) = \text{SiLU}(\text{gate}) \odot \text{up}
$$

其中 gate 和 up 都是前两个线性层的输出，形状均为 $(B \times T, H)$（`H` 为 intermediate hidden size），$\odot$ 表示逐元素乘法。完整 LLaMA FFN 如下：

```
x_in → [W_gate → gate,   W_up → up]  (两个线性层并行)
      → SiLU(gate) ⊙ up              ← 这一步就是下面的 kernel
      → W_down → x_out
```

*原理* 纯 elementwise 操作，每个 thread 处理一个位置，读 gate 和 up、写 out，是 Memory Bound kernel。

*Solution — 前向*

```cpp
// gate, up, out 形状均为 [n]（已展平为一维）
// out[i] = SiLU(gate[i]) * up[i]
__global__ void swiglu_forward(const float* gate, const float* up,
                                float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g   = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));   // sigmoid(gate)
    out[i] = g * sig * up[i];               // SiLU(gate) * up
}
// int block = 256; int grid = (n + block - 1) / block;
// swiglu_forward<<<grid, block>>>(gate, up, out, B * T * H);
```

*Solution — 反向*

设 $s = \sigma(g)$（sigmoid），$\text{SiLU}(g) = g \cdot s$，则：

$$
\frac{\partial L}{\partial \text{gate}} = \frac{\partial L}{\partial \text{out}} \cdot u \cdot \frac{d\,\text{SiLU}(g)}{dg} = \frac{\partial L}{\partial \text{out}} \cdot u \cdot s(1 + g(1-s))
$$

$$
\frac{\partial L}{\partial \text{up}} = \frac{\partial L}{\partial \text{out}} \cdot \text{SiLU}(g) = \frac{\partial L}{\partial \text{out}} \cdot g \cdot s
$$

```cpp
// dout: [n] 上游梯度；gate, up: [n] 前向保存的值；dgate, dup: [n] 输出梯度
__global__ void swiglu_backward(const float* dout, const float* gate, const float* up,
                                 float* dgate, float* dup, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g   = gate[i];
    float u   = up[i];
    float sig = 1.0f / (1.0f + expf(-g));    // σ(gate)
    float silu_g = g * sig;                   // SiLU(gate)

    // d_silu/d_gate = σ(g) * (1 + g * (1 - σ(g)))
    float d_silu_dg = sig * (1.0f + g * (1.0f - sig));

    dgate[i] = dout[i] * u * d_silu_dg;      // chain rule through SiLU
    dup[i]   = dout[i] * silu_g;             // SiLU(gate) 直接乘过来
}
```

> [!tip] 向量化优化：用 `float4` 读写
> SwiGLU 是纯 elementwise，严重 Memory Bound。用 `float4` 一次读/写 4 个 float，减少 4× 的 transaction 数量：
> ```cpp
> __global__ void swiglu_forward_vec4(const float4* gate, const float4* up,
>                                      float4* out, int n4) {  // n4 = n / 4
>     int i = blockIdx.x * blockDim.x + threadIdx.x;
>     if (i >= n4) return;
>     float4 g4 = gate[i], u4 = up[i], o4;
>     // 对 4 个分量逐一做 SiLU * up
>     auto silu = [](float x){ return x / (1.0f + expf(-x)); };
>     o4.x = silu(g4.x) * u4.x;  o4.y = silu(g4.y) * u4.y;
>     o4.z = silu(g4.z) * u4.z;  o4.w = silu(g4.w) * u4.w;
>     out[i] = o4;
> }
> // 要求 n 是 4 的倍数；gate/up/out 地址 16-byte 对齐（cudaMalloc 保证）
> ```

> [!note] LLaMA FFN 完整结构
> ```
> // 前向（伪代码，每层 FFN）
> gate = linear(x, W_gate)    // [B*T, H] → [B*T, FFN_dim]
> up   = linear(x, W_up)      // [B*T, H] → [B*T, FFN_dim]
> hidden = swiglu(gate, up)   // [B*T, FFN_dim]，本 kernel
> out  = linear(hidden, W_down) // [B*T, FFN_dim] → [B*T, H]
> ```
> LLaMA 3（8B）中 H=4096，FFN_dim=14336，每个 FFN 层的 SwiGLU 处理 $B \times T \times 14336$ 个元素。

---

###### Cross-Entropy Loss CUDA

*Question* 对 logits 矩阵 $(B, C)$（B 个样本，C 个类），给定标签 `labels[B]`，计算每个样本的 cross-entropy loss 并输出均值。每个 block 处理一行（一个样本）。

$$
\text{loss}_b = -\log\!\left(\frac{e^{x_{b,y_b}}}{\sum_{c=0}^{C-1} e^{x_{b,c} - \max_c x_{b,c}}}\right) = \log\sum_c e^{x_{b,c}-m_b} - (x_{b,y_b} - m_b)
$$

*Solution*

```cpp
__global__ void cross_entropy_forward(const float* logits,   // [B, C]
                                       const int*   labels,   // [B]
                                       float*       losses,   // [B]
                                       int B, int C) {
    extern __shared__ float smem[];
    int b         = blockIdx.x;
    int tid       = threadIdx.x;
    int num_warps = blockDim.x / 32;
    if (b >= B) return;

    const float* row = logits + b * C;

    // ── Pass 1: 找最大值（数值稳定） ──
    float max_val = -1e38f;
    for (int c = tid; c < C; c += blockDim.x)
        max_val = fmaxf(max_val, row[c]);
    max_val = warp_reduce_max(max_val);
    if (tid % 32 == 0) smem[tid / 32] = max_val;
    __syncthreads();
    max_val = (tid < num_warps) ? smem[tid] : -1e38f;
    if (tid < 32) max_val = warp_reduce_max(max_val);
    if (tid == 0) smem[0] = max_val;
    __syncthreads();
    max_val = smem[0];

    // ── Pass 2: 计算 log-sum-exp ──
    float sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x)
        sum += expf(row[c] - max_val);
    sum = warp_reduce_sum(sum);
    if (tid % 32 == 0) smem[tid / 32] = sum;
    __syncthreads();
    sum = (tid < num_warps) ? smem[tid] : 0.0f;
    if (tid < 32) sum = warp_reduce_sum(sum);
    if (tid == 0) {
        float log_sum_exp = logf(smem[0]) + max_val;          // 还原真实 log(sum(exp))
        float gt_logit    = row[labels[b]];                   // 正确类的 logit
        losses[b] = log_sum_exp - gt_logit;                   // NLL loss
    }
}

// 在 host 端对 losses[B] 做 mean（可再加一个 reduce kernel，或 host 端 CPU 算）
// int block = 256; size_t smem = (block / 32) * sizeof(float);
// cross_entropy_forward<<<B, block, smem>>>(logits, labels, losses, B, C);
```

> [!bug] 常见错误
> - 不做 `max` 减法：当 logit 很大时 `expf` 直接 overflow 到 inf，最终 loss 为 nan。
> - `log_sum_exp` 忘记加回 `max_val`：`logf(sum)` 只是 $\log\sum e^{x-m}$，真实值是 $\log\sum e^x = \log\sum e^{x-m} + m$。

---

###### Inclusive Prefix Sum（Scan）CUDA

*Question* 对长度 $N$ 的数组做 inclusive prefix sum（$y_i = \sum_{k=0}^{i} x_k$）。用 shared memory 实现 block 内 Hillis-Steele scan，多 block 用两轮 kernel 拼接。

*原理* Hillis-Steele 是 work-efficient scan 的经典变体：每一轮 stride 翻倍，每个 thread 读 `i-stride` 的值并累加到自己，经过 $\log_2 N$ 轮后每个位置都持有前缀和。

```
初始: [1, 2, 3, 4, 5, 6, 7, 8]
stride=1: [1, 3, 5, 7, 9, 11, 13, 15]   (i >= 1: y[i] += y[i-1])
stride=2: [1, 3, 6, 10, 14, 18, 22, 26] (i >= 2: y[i] += y[i-2])
stride=4: [1, 3, 6, 10, 15, 21, 28, 36] (i >= 4: y[i] += y[i-4])
```

*Solution*

```cpp
// ── Block-level inclusive scan（处理单 block，N <= blockDim.x） ──
__global__ void scan_block(const float* input, float* output, float* block_sums, int n) {
    extern __shared__ float sdata[];   // 大小 = blockDim.x
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Hillis-Steele scan
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float val = (tid >= stride) ? sdata[tid - stride] : 0.0f;
        __syncthreads();          // 等所有 thread 读完旧值再写
        sdata[tid] += val;
        __syncthreads();
    }

    if (i < n) output[i] = sdata[tid];
    // 记录每个 block 的总和，供第二轮 kernel 用
    if (block_sums && tid == blockDim.x - 1)
        block_sums[blockIdx.x] = sdata[tid];
}

// ── 第二轮：把上一 block 的总和加到下一 block 的每个元素 ──
__global__ void add_block_sums(float* output, const float* block_sums, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < n)
        output[i] += block_sums[blockIdx.x - 1];
}

// Host 调用：
// scan_block<<<grid, block, block*sizeof(float)>>>(input, output, block_sums, n);
// // 对 block_sums 本身再做一次 scan（递归）
// scan_block<<<1, grid, grid*sizeof(float)>>>(block_sums, block_sums, nullptr, grid);
// add_block_sums<<<grid, block>>>(output, block_sums, n);
```

> [!tip] 面试追问：Prefix Sum 有什么应用？
> 1. **Stream Compaction**：过滤满足条件的元素（先 prefix sum 算出各元素的目标写位置，再 scatter）。
> 2. **Radix Sort**：每位的计数排序需要前缀和确定各桶的起始偏移。
> 3. **Variable-length 数据的 Offset 计算**：如 sparse attention 中计算每个 query 的 KV 起始位置。

---

###### Adam Optimizer Kernel

*Question* 实现 Adam 参数更新 kernel，逐元素处理：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g^2
$$
$$
\hat m = m_t / (1 - \beta_1^t), \quad \hat v = v_t / (1 - \beta_2^t), \quad \theta \mathrel{-}= \alpha \cdot \hat m / (\sqrt{\hat v} + \varepsilon)
$$

*Solution*

```cpp
__global__ void adam_update(float* params,    // 参数，in-place 更新
                             float* grads,    // 梯度
                             float* m,        // 一阶矩 (momentum)
                             float* v,        // 二阶矩 (velocity)
                             int n,
                             float lr,
                             float beta1, float beta2, float eps,
                             float beta1_t,   // beta1^t（当前时间步的累乘）
                             float beta2_t) { // beta2^t
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g  = grads[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi / (1.0f - beta1_t);
    float v_hat = vi / (1.0f - beta2_t);
    params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}
// adam_update<<<grid, 256>>>(params, grads, m, v, n,
//     lr=1e-3f, beta1=0.9f, beta2=0.999f, eps=1e-8f, beta1_t, beta2_t);
```

> [!tip] AdamW 区别
> AdamW 在更新前先做 weight decay：`params[i] *= (1 - lr * weight_decay)`，然后再减去 adam 更新量。不把 weight decay 加进梯度，而是直接对参数做 L2 收缩。

---

###### Embedding Lookup（Gather）CUDA

*Question* 给定 embedding 表 `W[V, d]`（词表大小 V，维度 d）和 token ids `ids[B, T]`，输出 `out[B, T, d]`，即对每个 token 取对应的 embedding 行。

*Solution*

```cpp
// 每个 thread 负责一个 (b, t) 对应的 embedding 的一个维度
__global__ void embedding_forward(const float* W,    // [V, d]
                                   const int*   ids,  // [B, T]
                                   float*       out,  // [B, T, d]
                                   int B, int T, int V, int d) {
    // 把 (b, t, k) 映射到一维 thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * d;
    if (idx >= total) return;

    int k   = idx % d;          // embedding 维度
    int bt  = idx / d;          // 在 B*T 中的位置
    int b   = bt / T;
    int t   = bt % T;

    int token_id = ids[b * T + t];
    out[idx] = W[token_id * d + k];
}
// int block = 256; int grid = (B*T*d + block - 1) / block;
// embedding_forward<<<grid, block>>>(W, ids, out, B, T, V, d);
```

> [!bug] 性能分析：Embedding Lookup 为什么 Memory Bound？
> - 每个 token 的 embedding 行完全随机（取决于 token id），cache 命中率极低。
> - 当词表 V 很大（如 32000+），embedding 表本身就几百 MB，L2 cache 装不下。
> - 优化方向：① 合并同 batch 里的相同 token id（去重 gather）；② 用 BF16/FP16 存储减少带宽。

