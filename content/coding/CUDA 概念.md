---
date: 2026-07-13
---

**Contents**

1. [[#Grid / Block / Thread 三级层次|Grid / Block / Thread 三级层次]]
	1. [[#Grid / Block / Thread 三级层次#[Q1] 三级结构是什么？层级关系怎么理解？|[Q1] 三级结构是什么？层级关系怎么理解？]]
	2. [[#Grid / Block / Thread 三级层次#[Q2] Block 和 Thread 的数量怎么定？|[Q2] Block 和 Thread 的数量怎么定？]]
	3. [[#Grid / Block / Thread 三级层次#[Q2.1] `<<<>>>` 完整四个参数是什么？|[Q2.1] `<<<>>>` 完整四个参数是什么？]]
	4. [[#Grid / Block / Thread 三级层次#[Q3] Block 内的 thread 有什么特殊能力？Block 之间为什么不行？|[Q3] Block 内的 thread 有什么特殊能力？Block 之间为什么不行？]]
	5. [[#Grid / Block / Thread 三级层次#[Q4] 硬件上 Grid / Block / Thread 分别对应什么？|[Q4] 硬件上 Grid / Block / Thread 分别对应什么？]]
	6. [[#Grid / Block / Thread 三级层次#[Q5] block size 怎么选？为什么常见 128 / 256？|[Q5] block size 怎么选？为什么常见 128 / 256？]]
	7. [[#Grid / Block / Thread 三级层次#[Q6] 越界怎么处理？为什么 kernel 里要写 `if (i < n)`？|[Q6] 越界怎么处理？为什么 kernel 里要写 `if (i < n)`？]]
2. [[#dim3 的 x/y/z 与硬件含义|dim3 的 x/y/z 与硬件含义]]
	1. [[#dim3 的 x/y/z 与硬件含义#[Q7] x/y/z 是什么？影响硬件调度吗？|[Q7] x/y/z 是什么？影响硬件调度吗？]]
	2. [[#dim3 的 x/y/z 与硬件含义#[Q8] x 维有什么特殊的物理含义？|[Q8] x 维有什么特殊的物理含义？]]
	3. [[#dim3 的 x/y/z 与硬件含义#[Q9] 为什么 GEMM 里 col 要映射到 threadIdx.x？|[Q9] 为什么 GEMM 里 col 要映射到 threadIdx.x？]]
3. [[#Row-major vs Column-major|Row-major vs Column-major]]
	1. [[#Row-major vs Column-major#[Q10] 为什么 CUDA 用 row-major，cuBLAS 用 column-major？|[Q10] 为什么 CUDA 用 row-major，cuBLAS 用 column-major？]]
	2. [[#Row-major vs Column-major#[Q11] cuBLAS 的 column-major 在实际使用中如何处理？|[Q11] cuBLAS 的 column-major 在实际使用中如何处理？]]
	3. [[#Row-major vs Column-major#[Q11.1] CUTLASS Device API|[Q11.1] CUTLASS Device API]]
	4. [[#Row-major vs Column-major#[Q11.2] CuTE（底层，CUTLASS 内部实现基础）|[Q11.2] CuTE（底层，CUTLASS 内部实现基础）]]
	5. [[#Row-major vs Column-major#[Q12] CuTE 是什么？和 cuBLAS / 手写 kernel 有什么关系？|[Q12] CuTE 是什么？和 cuBLAS / 手写 kernel 有什么关系？]]
	6. [[#Row-major vs Column-major#[Q13] PyTorch tensor 是 row-major 吗？什么时候会变成非 contiguous？|[Q13] PyTorch tensor 是 row-major 吗？什么时候会变成非 contiguous？]]
	7. [[#Row-major vs Column-major#[Q14] NVCC 自带编译优化|[Q14] NVCC 自带编译优化]]

---

### Grid / Block / Thread 三级层次

#### [Q1] 三级结构是什么？层级关系怎么理解？

CUDA 启动一个 kernel 时，所有 thread 被组织成三层：

```
Grid（网格）
└── Block（线程块）× 多个
    └── Thread（线程）× 多个
```

类比：
- **Thread** = 一个工人，执行 kernel 函数体
- **Block** = 一个小组，组内工人可以共享黑板（shared memory）、互相等待（`__syncthreads()`）
- **Grid** = 整个工厂，所有小组同时开工，小组之间无法直接通信

---

#### [Q2] Block 和 Thread 的数量怎么定？

通过 `<<<grid, block>>>` 指定，两者都是 `dim3` 类型（最多三维）：

```cpp
dim3 block(256);              // 每个 block 256 个 thread（一维）
dim3 grid((n+255)/256);       // grid 大小 = 向上取整，保证覆盖所有数据
my_kernel<<<grid, block>>>(...);
```

kernel 内部用内置变量定位自己的全局位置：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
//      ↑ 第几个block  ↑ block大小    ↑ block内第几个thread
```

---

#### [Q2.1] `<<<>>>` 完整四个参数是什么？

```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, streamID>>>(...);
```

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `gridDim` | `dim3` / `int` | Grid 的形状（有多少个 block） | 必填 |
| `blockDim` | `dim3` / `int` | Block 的形状（每个 block 多少 thread） | 必填 |
| `sharedMemBytes` | `size_t` | 每个 block 动态分配的 shared memory 字节数 | `0` |
| `streamID` | `cudaStream_t` | 在哪个执行队列里运行，用于异步 / 并发 | `0`（默认 stream） |

**sharedMemBytes** 对应 kernel 内 `extern __shared__ float sdata[]` 的实际大小，在启动时确定：

```cpp
int block = 256;
// reduce_sum 用 block 个 float 的 shared memory
reduce_sum<<<grid, block, block * sizeof(float)>>>(input, output, n);
```

**streamID** 用于让多个 kernel 在不同 stream 上并发执行：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
reduce_sum<<<grid, block, block * sizeof(float), stream>>>(input, output, n);
cudaStreamSynchronize(stream);   // 只等这个 stream，不阻塞其他 stream
cudaStreamDestroy(stream);
```

注意：`cudaDeviceSynchronize()` 会等待**所有 stream** 完成，插在两个 stream kernel 之间会破坏并发。

---

#### [Q3] Block 内的 thread 有什么特殊能力？Block 之间为什么不行？

| 能力 | Block 内 Thread | 不同 Block 的 Thread |
|------|----------------|---------------------|
| Shared Memory | ✅ 共享同一块片上 SRAM | ❌ 各自独立，互不可见 |
| `__syncthreads()` | ✅ 可以做 barrier 同步 | ❌ 无法跨 block 同步 |
| 直接通信 | ✅（通过 shared memory） | ❌（只能通过 global memory + atomics） |

原因：**一个 Block 必须跑在同一个 SM（Streaming Multiprocessor）上**，shared memory 是 SM 上的物理资源。不同 Block 可能在不同 SM 上，无法访问对方的 shared memory。

---

#### [Q4] 硬件上 Grid / Block / Thread 分别对应什么？

- Thread   →  一个 CUDA Core（标量运算单元）
- Warp     →  32 个 Thread，SM 实际调度的最小单位（SIMT 执行）
- Block    →  若干 Warp，调度到同一个 SM 执行
- Grid     →  所有 Block，分发到 GPU 上所有 SM


Block 内的 thread 数量决定 warp 数量：`num_warps = blockDim.x / 32`。一个 SM 可以同时驻留多个 Block，上限由 shared memory 和 register 用量决定。

---

#### [Q5] block size 怎么选？为什么常见 128 / 256？

- **必须是 32 的倍数**：不足 32 会浪费 warp 中的 thread slot（tail effect）。
- **128 / 256** 是 occupancy 和 register 压力的平衡点：block 越大每个 thread 能分到的 register 越少，可能触发 register spilling（寄存器溢出到显存）。
	- 每个 SM 有固定数量寄存器，例如A100 是 65536 个 32-bit 寄存器/SM
	- 寄存器在 SM 中的所有 thread 共享，比如 block size = 256（256 thread），每个 thread 用 32 个 register -> 每个 block 有 256x32=8192 个寄存器 -> 每个 SM 最多 65536 / 8192 = 8 个 block
- **什么是 spilling**：当一个 kernel 函数体很复杂，编译器计算出每个 thread 需要的寄存器数超过配额时，多余的变量会被溢出（spill）到 L2 cache 或 global memory（显存）。
- **1024** 是硬件上限，但通常不选，register 压力大，灵活性差。
- **命令：** 用 `nvcc --ptxas-options=-v` 查 register 用量，用 CUDA Occupancy Calculator 算最优 block size。

---

#### [Q6] 越界怎么处理？为什么 kernel 里要写 `if (i < n)`？

grid 大小向上取整后，最后一个 block 可能有多余的 thread，index 超出数据范围。不加越界保护会读写非法内存，行为未定义。


---

### dim3 的 x/y/z 与硬件含义

#### [Q7] x/y/z 是什么？影响硬件调度吗？

x/y/z 是软件坐标系，**硬件只认线性 ID**。GPU 执行时会把 dim3 坐标展平：

```
线性 thread ID = threadIdx.x
               + threadIdx.y * blockDim.x
               + threadIdx.z * blockDim.x * blockDim.y
```

类比 C 语言多维数组 `arr[z][y][x]`：x 是最内层，变化最快。x/y/z 只是便利坐标系，不影响硬件调度。

---

#### [Q8] x 维有什么特殊的物理含义？

Warp 按 x 维连续打包：

```
threadIdx.x = 0..31   → Warp 0
threadIdx.x = 32..63  → Warp 1
...
```

threadIdx.y / z 的变化在 x 轴排满后才开始。因此：

- 同一 Warp 的 32 个 thread，threadIdx.x 连续
- 它们同时执行同一条指令（SIMT）
- 访问的内存地址若沿 x 方向连续 → **coalesced access**，打满 HBM 带宽

这是所有 kernel 都把「列」映射到 `threadIdx.x` 的原因。

---

#### [Q9] 为什么 GEMM 里 col 要映射到 threadIdx.x？

C/CUDA 默认 row-major，同一行连续存储，col 是最内层：

```
A[row][col] 内存布局：
A[0][0], A[0][1], A[0][2], ...   ← 第0行连续
A[1][0], A[1][1], A[1][2], ...   ← 第1行连续
```

访问公式 `A[row * N + col]`，col +1 → 地址 +1 → 天然 coalesced。令 col = threadIdx.x，同一 Warp 访问同一行相邻元素 = 连续地址 ✅。

若反过来令 row = threadIdx.x，同一 Warp 访问跨行地址，间隔 N 个 float，完全不连续 ❌。

---

### Row-major vs Column-major

#### [Q10] 为什么 CUDA 用 row-major，cuBLAS 用 column-major？

历史原因：

- **Row-major** 来自 C 语言（1970s）：`int A[M][N]` 天然按行存储，C 系语言全部继承。
- **Column-major** 来自 Fortran（1950s）：科学计算第一门高级语言，矩阵按列存储。
- **BLAS** 用 Fortran 写成，是线性代数库的事实标准接口。
- **cuBLAS** 是 GPU 上的 BLAS 实现，为兼容 CPU BLAS 接口沿用 column-major。
- **CUDA kernel** 是 C 扩展，自然是 row-major。

两套系统并存，没有对错，只是历史路径不同。

---

#### [Q11] cuBLAS 的 column-major 在实际使用中如何处理？

利用转置恒等式绕开，**不需要真的转置数据**：

$$C = A \times B \iff C^\top = B^\top \times A^\top$$

row-major 存储的矩阵 $A$，在 cuBLAS（column-major 视角）里看到的正好是 $A^\top$。因此把参数顺序对调，cuBLAS 就能算出正确的 row-major 结果：

```cpp
// 目标：C = A * B（A: M×K, B: K×N, C: M×N，全部 row-major）
// cuBLAS 视角：C^T = B^T * A^T
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,      // 注意：先 N（B的列）再 M（A的行）
    &alpha,
    B, N,         // 先传 B
    A, K,         // 再传 A
    &beta,
    C, N);
```

这是 cuBLAS 最常见的坑，背下来。


#### [Q11.1] CUTLASS Device API

 CUTLASS 把布局提升到模板参数，直接表达 row-major，不需要转置技巧：


```cpp
#include <cutlass/gemm/device/gemm.h>

// 模板参数声明布局，编译期确定
using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,   // A: float, row-major
    float, cutlass::layout::RowMajor,   // B: float, row-major
    float, cutlass::layout::RowMajor    // C: float, row-major
>;

Gemm gemm_op;
Gemm::Arguments args(
    {M, N, K},      // Problem shape，直接写 M/N/K，不用反转
    {A, K},         // A 的指针 + leading dim（row-major 下是 K）
    {B, N},         // B 的指针 + leading dim（N）
    {C, N},         // C（输入，用于 beta 缩放）
    {C, N},         // D（输出）
    {alpha, beta}
);
gemm_op(args);      // 内部自动选最优 kernel

```


#### [Q11.2] CuTE（底层，CUTLASS 内部实现基础）

CuTE 用 (shape, stride) 对构造 Tensor，完全手动控制 tiling。这是写自定义 kernel 的方式：

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void cute_gemm(
    float const* A, float const* B, float* C,
    int M, int N, int K, float alpha, float beta)
{
    // Step 1: 用 (shape, stride) 定义内存布局
    //   row-major stride = (行跨度, 列跨度) = (K, 1) 和 (N, 1)
    auto layoutA = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
    auto layoutB = make_layout(make_shape(K, N), make_stride(N, Int<1>{}));
    auto layoutC = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

    // Step 2: 把裸指针包装成 CuTE Tensor（带类型和布局）
    Tensor gA = make_tensor(make_gmem_ptr(A), layoutA);
    Tensor gB = make_tensor(make_gmem_ptr(B), layoutB);
    Tensor gC = make_tensor(make_gmem_ptr(C), layoutC);

    // Step 3: 每个 thread block 负责一个 output tile
    //   用 local_tile 切出当前 block 的子矩阵
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    constexpr int BM = 128, BN = 128, BK = 32;
    Tensor blkA = local_tile(gA, make_shape(Int<BM>{}, Int<BK>{}), cta_coord);
    Tensor blkB = local_tile(gB, make_shape(Int<BK>{}, Int<BN>{}), cta_coord);
    Tensor blkC = local_tile(gC, make_shape(Int<BM>{}, Int<BN>{}), cta_coord);

    // Step 4: 在 K 维分 tile 迭代，调用 cute::gemm 累加
    auto tiled_mma = make_tiled_mma(SM80_16x8x8_F32F16F16F32_TN{});
    // ... 加载到 shared memory、调 tiled_copy、调 tiled_mma ...
    // 完整实现约 100 行，CUTLASS examples/cute/tutorial/ 有标准模板
}

```

**CuTE 核心函数速查：**

- `make_shape(M, K)` — 构造形状对象，描述张量各维度大小，如 `(M, K)` 表示 M 行 K 列。可以混用运行时值（int）和编译期常量（`Int<32>{}`）。

- `make_stride(K, 1)` — 构造步长对象，描述各维度相邻元素的内存地址间隔。row-major M×K 矩阵：行步长 = K（跳到下一行要跨 K 个元素），列步长 = 1（相邻列地址连续）。

- `make_layout(shape, stride)` — 把 shape 和 stride 组合成 Layout 对象，完整描述一种内存排布方式。Layout 是纯元数据，不涉及实际数据。

- `make_gmem_ptr(ptr)` — 把裸 C++ 指针包装成 CuTE 能识别的 global memory 指针类型，让类型系统知道数据在哪块内存（global / shared / register）。

- `make_tensor(ptr, layout)` — 把指针和布局组合成 Tensor 对象。Tensor = 数据位置 + 内存排布，是 CuTE 的核心数据结构，之后所有操作都在 Tensor 上进行。

- `local_tile(tensor, tile_shape, coord)` — 从大 Tensor 中按 `tile_shape` 切出当前 block 负责的子张量（返回一个 view，不拷贝数据）。`coord` 由 `blockIdx` 决定，相当于矩阵分块的坐标索引。

- `make_tiled_mma(mma_atom)` — 以一个 MMA 原子指令（如 `SM80_16x8x8_F32F16F16F32_TN{}`，即 Tensor Core 的 16×8×8 矩阵乘指令）为基础，构造 Tiled MMA 对象，描述整个 thread block 如何协作完成一个大的矩阵乘 tile。

---

#### [Q12] CuTE 是什么？和 cuBLAS / 手写 kernel 有什么关系？

CuTE（CUTLASS Tensor Extension）是 NVIDIA CUTLASS 库的底层抽象层，不绑定任何 major 顺序。核心思想：用 **(shape, stride)** 二元组描述任意内存布局：

```
row-major M×N：shape=(M,N), stride=(N,1)   // 行跨度=N，列跨度=1
col-major M×N：shape=(M,N), stride=(1,M)   // 行跨度=1，列跨度=M
```

布局信息编码在类型里，kernel 内部统一用 `make_tensor` + `layout` 访问，不写死 `row*N+col`。

| 层级 | 代表 | Major | 你控制什么 |
|------|------|-------|-----------|
| 手写 CUDA kernel | `A[row*N+col]` | row-major | 全部手动 |
| cuBLAS | `cublasSgemm` | column-major | 参数顺序 |
| CUTLASS / CuTE | `make_tensor` | layout-agnostic | shape + stride |
| PyTorch | `torch.mm` | row-major（C-contiguous） | 不用管 |
| cuDNN | `cudnnConvolutionForward` | NCHW 或 NHWC 可选 | 指定 format |

---

#### [Q13] PyTorch tensor 是 row-major 吗？什么时候会变成非 contiguous？

默认 row-major（C-contiguous）。以下操作会产生非连续 tensor：

```python
x = torch.randn(3, 4)    # row-major, strides=(4,1)
y = x.t()                # 转置，strides=(1,3)  ← column-major 视角
y.is_contiguous()         # False
z = y.contiguous()        # 重新分配内存，回到 row-major
```

PyTorch 底层调 cuBLAS 时会自动处理 strides，非 contiguous tensor 通常也能正确运行（内部做转换），但可能有性能损失。

---

**速记**

- **C / CUDA / NumPy / PyTorch** → row-major，col 变化最快
- **Fortran / MATLAB / cuBLAS / LAPACK** → column-major，row 变化最快
- **CuTE / CUTLASS** → layout-agnostic，stride 说了算


---

#### [Q14] NVCC 自带编译优化

| 等级    | 常见优化                                      | 适用场景             |
| ----- | ----------------------------------------- | ---------------- |
| `-O0` | 基本不优化，尽量保留源码结构和变量                         | 调试               |
| `-O1` | 删除无用代码、常量传播、简化分支、简单函数内联                   | 需要较快编译，同时希望有基础优化 |
| `-O2` | 包含 `-O1`，增加公共子表达式消除、指令调度、更多函数内联、循环优化、向量化等 | 一般发布和性能测试，最常用    |
| `-O3` | 包含 `-O2`，增加更激进的循环展开、循环交换、循环拆分、更积极的向量化和内联  | 计算密集型程序，需要实测     |

1. -O3 不一定比 -O2 快。激进内联和循环展开可能增加代码大小、寄存器压力和指令缓存压力。
2. 优化后调试较困难，变量可能被删除、合并或移动，断点执行顺序可能与源码不一致。
3. 不同编译器、版本和硬件在每个等级启用的具体优化并不完全相同。GCC 将 -O2 定义为启用大部分不涉及明显空间与速度权衡的优化，而 -O3 会额外启用更激进的循环变换。
4. CUDA device kernel 默认已经启用设备代码优化，使用 -G 等选项关闭它（例如 debug 模式的时候需要-G, -g）
---

**[Q15] shared memory**

1️⃣ **shared memory 数组**两种定义方式：

```cpp
// 静态：编译时确定大小，写死在代码里
__shared__ float smem[256];

// 动态：运行时由 <<<grid, block, smem_bytes>>> 第三个参数决定
extern __shared__ float smem[];
```


2️⃣ **多个动态 shared memory 数组：**

不能直接声明两个 extern __shared__：

```cpp
// 错误写法
extern __shared__ float sA[];
extern __shared__ int   sB[];  // sA 和 sB 会指向同一地址！

```
正确做法是声明一个 char[]，手动算偏移：
```cpp
extern __shared__ char smem_raw[];
float* sA = (float*)smem_raw;
int*   sB = (int*)(smem_raw + block * sizeof(float));
// 启动时 smem_bytes = block * sizeof(float) + block * sizeof(int)
```

---

###### [概念] Warp Divergence：分支导致的串行化

**定义**：同一 Warp 内的 32 个 thread 执行同一条指令（SIMT），若遇到 `if/else`，硬件会**两个分支都执行**，不走某分支的 thread 被 mask 掉（不写结果）。这叫 Warp Divergence，会将并行变为串行。

```cpp
// ❌ 典型 divergence：warp 内奇偶 thread 走不同分支
if (threadIdx.x % 2 == 0) {
    do_A();   // 偶数 thread 执行，奇数 mask
} else {
    do_B();   // 奇数 thread 执行，偶数 mask
}
// 实际耗时 = A + B，而不是 max(A, B)

// ✅ 消除方式一：让同一 warp 内的 thread 走同一分支
// 按 warp 粒度分支（32 的倍数对齐）
if (threadIdx.x / 32 == 0) { ... }   // 整个 warp 0 走同一分支

// ✅ 消除方式二：用数学运算替代分支（branchless）
float val = (threadIdx.x % 2 == 0) ? a : b;
// 改写为：
float val = a * (1 - threadIdx.x % 2) + b * (threadIdx.x % 2);
```

| 场景 | 是否 Divergence |
|------|----------------|
| `if (i < n)` 越界保护 | ✅ 只有最后一个 block 的最后几个 thread 触发，影响极小 |
| `if (threadIdx.x % 2 == 0)` | ❌ 每个 warp 内都 diverge，严重 |
| `if (warpId == 0)` | ✅ 整个 warp 同时进入或不进入，无 divergence |
| 树形 reduce 中 `if (tid < stride)` | ⚠️ stride 较小时最后几轮 diverge，但影响有限 |

> [!tip] 面试追问：Tree Reduction 里 `if (tid < stride)` 会 diverge 吗？
> 会，但影响可控。stride 从 `blockDim/2` 递减，当 stride < 32 时，同一 warp 内部分 thread 不满足条件 → diverge。但此时 warp 数已经很少，整体损失可以接受。Warp Shuffle 版本完全规避了这个问题（32 个 lane 都参与计算，无分支）。

---

###### [概念] CUDA Memory Hierarchy 速查

```
寄存器 (Register)
  容量：每 SM ~65536 个 32-bit register（A100）
  延迟：0 cycle（最快，就在 SM 里）
  特点：每个 thread 私有，不共享；用多了会 spilling 到 local memory（慢）

Shared Memory / L1 Cache（片上，同一 SM）
  容量：每 SM 最大 164 KB（A100，可动态分配 shared/L1 比例）
  延迟：~20-30 cycle（无 bank conflict 时）
  带宽：~19 TB/s（片内，远超 HBM）
  特点：Block 内 thread 共享，__syncthreads() 确保一致性

L2 Cache（片上，全 GPU）
  容量：40 MB（A100）
  延迟：~200 cycle
  带宽：~5 TB/s

HBM（Global Memory，片外）
  容量：80 GB（A100）
  延迟：~500-800 cycle
  带宽：2 TB/s（A100），3.35 TB/s（H100 SXM5）
  特点：所有 SM 共享；访问延迟最高，是性能瓶颈所在
```

| 内存类型 | 作用域 | 生命周期 | 访问速度 |
|---------|--------|---------|---------|
| Register | Thread 私有 | Thread | 最快 |
| Local Memory | Thread 私有（register spill） | Thread | 慢（在 global mem） |
| Shared Memory | Block 内共享 | Block | 快（片上） |
| Global Memory (HBM) | 全 GPU | 整个程序 | 慢 |
| Constant Memory | 只读，全 GPU | 整个程序 | 快（有专用 cache） |
| Texture Memory | 只读，全 GPU，2D 空间局部性 | 整个程序 | 中（有 cache） |

> [!tip] 面试追问：Compute Bound vs Memory Bound 怎么判断？
> 计算**算术强度**（Arithmetic Intensity）= FLOP / Bytes：
> - AI 高（如 GEMM，矩阵足够大时 AI ≈ N/2）→ Compute Bound
> - AI 低（如 Elementwise，每次读写只做 1 次乘加）→ Memory Bound
>
> A100 FP32 峰值：~19.5 TFLOP/s，HBM 带宽：~2 TB/s
> Ridge Point = 19.5T / 2T ≈ 9.75 FLOP/Byte
> AI < 9.75 → Memory Bound；AI > 9.75 → Compute Bound

---

###### [概念] Occupancy：如何分析 SM 利用率

**Occupancy** = 活跃 Warp 数 / SM 最大 Warp 数（A100 最大 64 warp/SM）

限制 Occupancy 的三个因素（取最严格的那个）：

```
1. Register：每 thread 用 32 个 register，block=256 thread
   → 每 block 用 32*256 = 8192 registers
   → 每 SM 65536 / 8192 = 8 个 block = 8*8 = 64 warp → 100% occupancy

2. Shared Memory：每 block 用 32 KB smem，SM 上限 164 KB
   → 每 SM 最多 5 个 block → 5*8 = 40 warp → ~62% occupancy ← 瓶颈

3. Block 数量上限：每 SM 最多 32 block（H100）
```

> [!tip] Occupancy 不是越高越好
> 高 Occupancy 可以用更多 Warp 掩盖内存延迟（Latency Hiding）。
> 但对于 Compute Bound 的 kernel，每个 thread 需要更多 register 来保留中间结果，适当降低 Occupancy 反而更快（寄存器够用，不 spill）。
> 原则：**先保证没有 register spill，再看 Occupancy。**

---


###### CUDA Streams 与 Async 流水线

*概念* CUDA Stream 是命令队列，同一 stream 内的操作顺序执行；不同 stream 可以**并发**，实现 compute 与 memory transfer 的 overlap。

```cpp
// ── 不使用 Stream：串行，H2D 传输阻塞计算 ──
cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);   // 等传完
kernel<<<grid, block>>>(d_A, d_B);                     // 再算

// ── 使用 Stream + PinnedMemory：流水线 overlap ──
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Stream 1：传前半段数据 + 算前半段
cudaMemcpyAsync(d_A, h_A, half_bytes, cudaMemcpyHostToDevice, s1);
kernel<<<grid, block, 0, s1>>>(d_A, d_out,  half_n);

// Stream 2：传后半段 + 算后半段（与 Stream 1 并发）
cudaMemcpyAsync(d_A + half_n, h_A + half_n, half_bytes, cudaMemcpyHostToDevice, s2);
kernel<<<grid, block, 0, s2>>>(d_A + half_n, d_out + half_n, half_n);

cudaStreamSynchronize(s1);
cudaStreamSynchronize(s2);
cudaStreamDestroy(s1); cudaStreamDestroy(s2);
```

**必须用 Pinned Memory 才能 overlap**：`cudaMallocHost(&h_A, bytes)` 分配页锁定内存，DMA 引擎可在 GPU 计算时同时传输。普通 `malloc` 的堆内存无法 overlap。

```cpp
// ── CUDA Events：精确计时 ──
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
kernel<<<grid, block>>>(d_in, d_out, n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
cudaEventDestroy(start); cudaEventDestroy(stop);
```

| API | 作用 |
|-----|------|
| `cudaMemcpyAsync` | 异步拷贝（需 pinned memory + stream） |
| `cudaStreamSynchronize(s)` | 等待某个 stream 完成 |
| `cudaDeviceSynchronize()` | 等待所有 stream 完成（粒度粗，避免放在两个 stream 之间） |
| `cudaEventRecord(e, s)` | 在 stream s 中插入事件（时间戳标记） |
| `cudaStreamWaitEvent(s, e)` | stream s 等待事件 e 完成后才继续（跨 stream 依赖） |

> [!warning] `cudaDeviceSynchronize()` 插在两个并发 stream 之间会销毁并发性
> 两个 stream 原本可以 overlap，插入 `cudaDeviceSynchronize()` 后 GPU 等待所有 stream 完成才继续 → 变回串行。应用 `cudaStreamSynchronize(s1)` 只等特定 stream。

---

###### [概念] Atomics：atomicAdd / atomicMax 使用场景

**原子操作**保证同一内存地址的 read-modify-write 不被其他 thread 中断，但会**串行化竞争**。

```cpp
// atomicAdd：多个 thread 同时写同一地址时的正确求和
atomicAdd(&d_output[0], val);   // 全局求和（多 block 归约的最后一步）
atomicAdd(&histogram[bucket], 1);  // 直方图

// atomicMax（int only）：找全局最大
atomicMax(&d_max[0], __float_as_int(val));  // 对 float 用 reinterpret trick
// 注意：只对非负 float 有效（IEEE 754 正数的位表示保持大小顺序）

// CAS（Compare And Swap）：实现自定义 atomicMax for float
__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
```

> [!tip] 减少 atomic 竞争的技巧
> 1. **Block-level reduce 先行**：block 内用 shared memory 算出 block sum，只有 thread 0 做 `atomicAdd`，竞争线程数从 N 降到 block 数。
> 2. **分桶 / warp-level atomics**：对 histogram，先在 shared memory 做 warp 内统计，再 atomicAdd 到 global。
> 3. **`atomicAdd` on L2**（Ampere+）：对 global memory 的 atomic 可能走 L2 cache，比 HBM 快很多。
> 