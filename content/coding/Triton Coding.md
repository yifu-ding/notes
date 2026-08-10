
> **`kernel[grid](args)` — Triton 的启动语法**，对应 CUDA 的 `kernel<<<grid, block>>>(args)`。`[(hidden_size,)]` 是 grid，即启动多少个 program（线程块）。

| | CUDA | Triton |
|---|---|---|
| 启动语法 | `kernel<<<grid, block>>>(args)` | `kernel[grid](args)` |
| 1D grid | `<<<N, 128>>>` | `[(N,)]` |
| 2D grid | `<<<dim3(M,N), 128>>>` | `[(M, N)]` 或简写 `[M, N]` |
| 2D grid 维度索引 | `blockIdx.x` / `blockIdx.y` | `program_id(0)` / `program_id(1)` |
| block 配置 | 第二个参数 | `num_warps=` 关键字参数 |

###### 写一个 per-token 的 int8 量化 kernel

> **关键思路**：一个 Triton program 负责一行（一个 Token）。先将该 Token 的 hidden dimension 整体加载到寄存器，做 absmax reduction 得到动态 scale，然后直接在同一个 kernel 内完成除法、舍入、截断和 INT8 写回。这样避免单独启动 reduction kernel 和 quantization kernel，也不需要在全局内存中保存中间结果。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def per_token_int8_quant_kernel(
    x_ptr,          # [num_tokens, hidden_size]
    q_ptr,          # [num_tokens, hidden_size], int8
    scale_ptr,      # [num_tokens], float32
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一个 token（一行），天然隔离，无需跨 program 同步
    token_id = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    row_start = token_id * hidden_size
    # 连续行读取，容易合并访存（coalesced memory access）
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    # 寄存器内 reduction，不写回全局内存
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(absmax / 127.0, 1.0e-12)  # 全零 token 避免除零
    scaled = x / scale
    # 显式 round-to-nearest，避免直接 float -> int 的截断语义，int8 量化的标准是round half away from zero
    rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    quantized = tl.clamp(rounded, -127.0, 127.0).to(tl.int8)
    tl.store(q_ptr + row_start + offsets, quantized, mask=mask)  # 写回 INT8（连续）
    tl.store(scale_ptr + token_id, scale)                        # 一个 token 一个 scale


def per_token_int8_quant(x: torch.Tensor):
    """
    Args:
        x: CUDA tensor，形状 [..., hidden_size]，dtype 为 fp16/bf16/fp32。
    Returns:
        q:     int8 tensor，形状与 x 相同
        scale: float32 tensor，形状为 x.shape[:-1]
    Dequant:
        x_hat = q.float() * scale.unsqueeze(-1)
    """
    if not x.is_cuda: raise ValueError("x must be a CUDA tensor")
    if x.ndim < 2: raise ValueError("x must have at least 2 dimensions")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype: {x.dtype}")
    x = x.contiguous()  # 内核假设最后一维连续
    hidden_size = x.shape[-1]
    num_tokens = x.numel() // hidden_size
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(num_tokens, device=x.device, dtype=torch.float32)
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 65536:
        raise ValueError(f"hidden_size={hidden_size} is too large for this single-program implementation")
    num_warps = 4 if block_size <= 1024 else (8 if block_size <= 4096 else 16)
    per_token_int8_quant_kernel[(num_tokens,)](
        x, q, scale, hidden_size=hidden_size, BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return q, scale.view(x.shape[:-1])
```

> **为什么不用 `tl.round`？** `tl.math.round`（底层映射 LLVM/CUDA 的 `round()`）在不同版本/平台上行为不完全一致，部分实现遵循 IEEE 754 的 **banker's rounding（round half to even）**，而 INT8 量化的标准约定是 **round half away from zero**（对称舍入）。
>
> | 值 | round half to even | round half away from zero |
> |---|---|---|
> | 0.5 | → 0（偶数） | → 1 |
> | 1.5 | → 2（偶数） | → 2 |
> | 2.5 | → 2（偶数） | → 3 |
> | -0.5 | → 0 | → **-1** |
>
> 用 banker's rounding 会导致正负值在 x.5 处处理不对称，引入系统性偏差。显式写法与 TensorRT、vLLM、bitsandbytes 等框架保持一致。




###### per-channel 的 int8 量化（面试版）

- 一个 Triton program 负责一个 channel，沿 num_tokens 维做归约。

> **注意**：此版本适合说明算法逻辑或面试 coding，不是高性能实现。
> 
> 实际优化时通常使用 **二维 tile + 两阶段 reduction**：
> - 一个 program 同时处理一组 `[BLOCK_T, BLOCK_C]` 的二维数据块
> - 沿 Token 维对每个 channel 做局部 absmax，输出 partial max 到全局内存
> - 第二个 kernel 完成全局 reduction 得到 scale
> - 第三个 kernel（或规模允许时与第二步融合）完成量化写回

> **访存差异**：
> - Per-token：按行读取，连续且容易合并访存（coalesced）
> - Per-channel：按列归约，朴素实现是跨步读取（strided），通常需要 tiled reduction 或两阶段 reduction 才能达到较高性能

```python
import torch
import triton
import triton.language as tl

@triton.jit
def per_channel_int8_quant_kernel(
    x_ptr,              # [num_tokens, hidden_size]
    q_ptr,              # [num_tokens, hidden_size], int8
    scale_ptr,          # [hidden_size], float32
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # 一个 program 处理一个 channel（列），跨步读取所有 token 行
    channel_id = tl.program_id(axis=0)
    token_offsets = tl.arange(0, BLOCK_T)
    mask = token_offsets < num_tokens
    offsets = token_offsets * hidden_size + channel_id  # x[token_id, channel_id]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # 沿 token 维统计该 channel 的 absmax（跨步访存，性能瓶颈所在）
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1.0e-12)
    scaled = x / scale
    # Round half away from zero
    rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    q = tl.clamp(rounded, -127.0, 127.0).to(tl.int8)
    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(scale_ptr + channel_id, scale)


def per_channel_int8_quant(x: torch.Tensor):
    """
    对最后一个维度做 per-channel INT8 对称量化。
    Args:
        x: CUDA tensor，形状为 [num_tokens, hidden_size] 或 [..., hidden_size]。
    Returns:
        q:     INT8 tensor，形状与 x 相同。
        scale: FP32 tensor，形状为 [hidden_size]。
    Dequantization:
        x_hat = q.float() * scale
    """
    if not x.is_cuda: raise ValueError("x must be a CUDA tensor")
    if x.ndim < 2: raise ValueError("x must have at least 2 dimensions")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype: {x.dtype}")
    x = x.contiguous()
    hidden_size = x.shape[-1]
    num_tokens = x.numel() // hidden_size
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(hidden_size, device=x.device, dtype=torch.float32)
    block_t = triton.next_power_of_2(num_tokens)
    # 单 program 归约版本，适合 num_tokens 不太大的情况
    if block_t > 65536:
        raise ValueError(f"num_tokens={num_tokens} is too large for the single-program reduction implementation")
    num_warps = 4 if block_t <= 1024 else (8 if block_t <= 4096 else 16)
    per_channel_int8_quant_kernel[(hidden_size,)](
        x, q, scale, num_tokens=num_tokens, hidden_size=hidden_size, BLOCK_T=block_t, num_warps=num_warps,
    )
    return q, scale
```


###### 高性能 per-channel 量化（三阶段 tiled 版本）

> **思路**：将 `[num_tokens, hidden_size]` 的矩阵切成 `[BLOCK_T, BLOCK_C]` 的二维 tile。
> - **Phase 1**：每个 program 处理一个 tile，对 BLOCK_T 行做局部 absmax，输出 `partial_absmax[token_block, channel]`，访存连续。
> - **Phase 2**：对每个 channel 跨 token_block 做最终 reduce，得到全局 scale。
> - **Phase 3**：用 scale 对 tile 做量化写回。
>
> Phase 1 和 Phase 3 都是连续 tile 访存，彻底避免了朴素版本的跨步列读取。

```python
import math
import torch
import triton
import triton.language as tl

# ── Phase 1：每个 tile 输出局部 absmax ─────────────────────────────────────
@triton.jit
def _absmax_partial_kernel(
    x_ptr,          # [num_tokens, hidden_size]
    partial_ptr,    # [num_token_blocks, hidden_size]
    num_tokens, hidden_size,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_t, pid_c = tl.program_id(0), tl.program_id(1)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask, c_mask = t_off < num_tokens, c_off < hidden_size
    # 二维 tile 读取：行连续，合并访存
    x = tl.load(x_ptr + t_off[:, None] * hidden_size + c_off[None, :],
                 mask=t_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    local_max = tl.max(tl.abs(x), axis=0)  # 沿 token 维 reduce -> [BLOCK_C] 局部 absmax
    tl.store(partial_ptr + pid_t * hidden_size + c_off, local_max, mask=c_mask)


# ── Phase 2：跨 token_blocks 做最终 reduce，得到每个 channel 的 scale ──────
@triton.jit
def _scale_finalize_kernel(
    partial_ptr,    # [num_token_blocks, hidden_size]
    scale_ptr,      # [hidden_size]
    num_token_blocks, hidden_size,
    BLOCK_TB: tl.constexpr,   # >= num_token_blocks 的 2 的幂
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_off < hidden_size
    tb_off = tl.arange(0, BLOCK_TB)
    tb_mask = tb_off < num_token_blocks
    partials = tl.load(partial_ptr + tb_off[:, None] * hidden_size + c_off[None, :],
                       mask=tb_mask[:, None] & c_mask[None, :], other=0.0)
    scale = tl.maximum(tl.max(partials, axis=0) / 127.0, 1.0e-12)  # [BLOCK_C] 全局 scale
    tl.store(scale_ptr + c_off, scale, mask=c_mask)


# ── Phase 3：用 scale 做 tile 级量化写回 ──────────────────────────────────
@triton.jit
def _quant_kernel(
    x_ptr,          # [num_tokens, hidden_size]
    q_ptr,          # [num_tokens, hidden_size], int8
    scale_ptr,      # [hidden_size]
    num_tokens, hidden_size,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_t, pid_c = tl.program_id(0), tl.program_id(1)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask2d = (t_off < num_tokens)[:, None] & (c_off < hidden_size)[None, :]
    x = tl.load(x_ptr + t_off[:, None] * hidden_size + c_off[None, :],
                 mask=mask2d, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + c_off, mask=c_off < hidden_size, other=1.0)  # [BLOCK_C]
    scaled = x / scale[None, :]                                               # broadcast
    rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    q = tl.clamp(rounded, -127.0, 127.0).to(tl.int8)
    tl.store(q_ptr + t_off[:, None] * hidden_size + c_off[None, :], q, mask=mask2d)


# ── Python 入口 ────────────────────────────────────────────────────────────
def per_channel_int8_quant_fast(
    x: torch.Tensor, BLOCK_T: int = 128, BLOCK_C: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    高性能 per-channel INT8 量化，三阶段 tiled 实现。
    Args:
        x:       CUDA FP16/BF16/FP32 tensor，形状 [num_tokens, hidden_size]。
        BLOCK_T: token tile 大小，建议 64~256。
        BLOCK_C: channel tile 大小，建议 32~128（需为 2 的幂）。
    Returns:
        q:     INT8 tensor，形状与 x 相同。
        scale: FP32 tensor，形状 [hidden_size]。
    Dequantization:
        x_hat = q.float() * scale
    """
    if not x.is_cuda: raise ValueError("x must be a CUDA tensor")
    if x.ndim != 2: raise ValueError("x must be 2D [num_tokens, hidden_size]")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype: {x.dtype}")
    x = x.contiguous()
    num_tokens, hidden_size = x.shape
    num_token_blocks = math.ceil(num_tokens / BLOCK_T)
    num_channel_blocks = math.ceil(hidden_size / BLOCK_C)
    partial = torch.empty(num_token_blocks, hidden_size, device=x.device, dtype=torch.float32)
    scale = torch.empty(hidden_size, device=x.device, dtype=torch.float32)
    q = torch.empty_like(x, dtype=torch.int8)

    # Phase 1
    _absmax_partial_kernel[num_token_blocks, num_channel_blocks](
        x, partial, num_tokens, hidden_size, BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C,
    )
    # Phase 2：BLOCK_TB 需覆盖全部 token_blocks
    BLOCK_TB = triton.next_power_of_2(num_token_blocks)
    _scale_finalize_kernel[(num_channel_blocks,)](
        partial, scale, num_token_blocks, hidden_size, BLOCK_TB=BLOCK_TB, BLOCK_C=BLOCK_C,
        num_warps=4 if BLOCK_TB <= 1024 else (8 if BLOCK_TB <= 4096 else 16),
    )
    # Phase 3
    _quant_kernel[num_token_blocks, num_channel_blocks](
        x, q, scale, num_tokens, hidden_size, BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C,
    )
    return q, scale
```


###### 反量化+乘法 Kernel Fusion （weight-only int8）

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_dequant_matmul_kernel(
    x_ptr, wq_ptr, scale_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Per-output-channel scale，只加载一次
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_ids = k_start + offs_k
        x = tl.load(x_ptr + offs_m[:, None] * K + k_ids[None, :],
                     mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        wq = tl.load(wq_ptr + k_ids[:, None] * N + offs_n[None, :],
                     mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0)
        # INT8 权重在寄存器中反量化，不写回中间张量
        acc += tl.dot(x, (wq.to(tl.float32) * scale[None, :]).to(tl.float16))

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=mask)
```


###### INT4 Pack / Unpack Kernel

```python
import triton
import triton.language as tl

@triton.jit
def int4_pack_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pair = pid * BLOCK + tl.arange(0, BLOCK)
    i0, i1 = pair * 2, pair * 2 + 1
    x0 = tl.load(x_ptr + i0, mask=i0 < N, other=0).to(tl.int32)
    x1 = tl.load(x_ptr + i1, mask=i1 < N, other=0).to(tl.int32)
    # 取低四位，负数自动转换为补码表示
    packed = (x0 & 0xF) | ((x1 & 0xF) << 4)
    tl.store(out_ptr + pair, packed.to(tl.uint8), mask=i0 < N)


@triton.jit
def int4_unpack_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pair = pid * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(x_ptr + pair, mask=pair < tl.cdiv(N, 2), other=0).to(tl.int32) # cdiv: ceiling division（向上取整除法），防止//把奇数的最后一个漏算了
    low  = packed & 0xF
    high = (packed >> 4) & 0xF
    # 恢复有符号 INT4（>=8 表示负数）
    low  = tl.where(low  >= 8, low  - 16, low)
    high = tl.where(high >= 8, high - 16, high)
    i0, i1 = pair * 2, pair * 2 + 1
    tl.store(out_ptr + i0, low.to(tl.int8),  mask=i0 < N)
    tl.store(out_ptr + i1, high.to(tl.int8), mask=i1 < N)
```


###### Top-2 Router Kernel

```python
import triton
import triton.language as tl

@triton.jit
def top2_kernel(
    logits_ptr, value_ptr, index_ptr,
    E: tl.constexpr, BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    x = tl.load(logits_ptr + token * E + offs, mask=offs < E, other=-float("inf")).to(tl.float32)
    # Top-1
    idx1 = tl.argmax(x, axis=0)
    val1 = tl.max(x, axis=0)
    # 屏蔽 Top-1，再选 Top-2
    x2   = tl.where(offs == idx1, -float("inf"), x)
    idx2 = tl.argmax(x2, axis=0)
    val2 = tl.max(x2, axis=0)
    tl.store(value_ptr + token * 2,     val1)
    tl.store(value_ptr + token * 2 + 1, val2)
    tl.store(index_ptr + token * 2,     idx1)
    tl.store(index_ptr + token * 2 + 1, idx2)
```


###### Top-K Router Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def topk_router_kernel(
    logits_ptr,     # [T, E]
    values_ptr,     # [T, TOPK]
    indices_ptr,    # [T, TOPK]
    E: tl.constexpr, TOPK: tl.constexpr, BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    x = tl.load(logits_ptr + token * E + offs_e, mask=offs_e < E, other=-float("inf")).to(tl.float32)
    for k in tl.static_range(0, TOPK):
        idx = tl.argmax(x, axis=0)
        val = tl.max(x, axis=0)
        tl.store(values_ptr  + token * TOPK + k, val)
        tl.store(indices_ptr + token * TOPK + k, idx)
        # 屏蔽当前最大值，进入下一轮
        x = tl.where(offs_e == idx, -float("inf"), x)
```


###### MoE Token Dispatch

> **为什么拆成两个 kernel？** "分配目标位置"和"复制 hidden vector"并行粒度不同：前者每个 pair 只需一次 atomic，后者需要沿 hidden dim 做二维 tile 复制。合并会导致冲突，拆开更自然。
>
> - **prefix sum** 给出每个 expert 的区间起点 `expert_offsets[e]`
> - **atomic_add** 给出 expert 内部的 local slot，两者相加得到每个 pair 的写入目标行

**Kernel 1 — 计算 dispatch slot，只负责计算 token 索引表，不负责读写真正的 token hidden，所以计算量小** （grid: `num_pairs`，每个 program 算一个 pair 的目标行号）

```python
import triton
import triton.language as tl

# expert_offsets 是 Kernel 1 之前 就算好的
# 对所有 expert 的 token 数做 prefix sum，得到每个 expert 在输出 buffer 里的起点
@triton.jit
def compute_dispatch_slot_kernel(
    expert_ids_ptr, expert_offsets_ptr, counters_ptr, dst_ptr, 
    N_PAIRS: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_PAIRS
    expert = tl.load(expert_ids_ptr + offs, mask=mask, other=0)
    # atomic_add 返回 add 前的旧值，即本 pair 在该 expert 内的 local slot
    local = tl.atomic_add(counters_ptr + expert, 1, mask=mask)
    base  = tl.load(expert_offsets_ptr + expert, mask=mask)
    tl.store(dst_ptr + offs, base + local, mask=mask)  # 从 base + local 取出，写入 dst[num_pairs]，是每个 token-expert pair 的目标行号，只负责计算索引
```

**Kernel 2 — 按二维 tile （以及 kernel 1 给出的索引表）复制真实的 token hidden vector，只负责复制，不负责分配**（grid: `(num_pairs, H // BLOCK_H)`）

```python
@triton.jit
def dispatch_copy_kernel(
    x_ptr, dst_ptr, out_ptr,
    H: tl.constexpr, TOPK: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pair, block_h = tl.program_id(0), tl.program_id(1)
    token     = pair // TOPK   # 源 token 行
    dst_token = tl.load(dst_ptr + pair)   # 目标行（Kernel 1 写入）
    offs_h = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask   = offs_h < H
    x = tl.load(x_ptr + token * H + offs_h, mask=mask)
    tl.store(out_ptr + dst_token * H + offs_h, x, mask=mask)
```


---


###### Paged KV Cache Store Kernel

> **地址映射**：`slot = block_id * block_size + offset`，`slot < 0` 跳过。每个 program 处理一个 token，将其 `[num_kv_heads, head_dim]` 的 KV 写入 `[num_blocks, block_size, num_kv_heads, head_dim]` 的 paged cache。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def store_kv_kernel(
    key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
    num_kv_heads: tl.constexpr, head_dim: tl.constexpr,
    block_size: tl.constexpr, hidden_size: tl.constexpr, BLOCK: tl.constexpr,
):
    token_id     = tl.program_id(0)
    slot         = tl.load(slot_mapping_ptr + token_id)
    valid        = slot >= 0
    block_id     = slot // block_size
    block_offset = slot %  block_size

    offsets = tl.arange(0, BLOCK)
    mask = (offsets < hidden_size) & valid
    src = token_id * hidden_size + offsets
    dst = (block_id * block_size + block_offset) * hidden_size + offsets

    k = tl.load(key_ptr   + src, mask=mask, other=0.0)
    v = tl.load(value_ptr + src, mask=mask, other=0.0)
    tl.store(key_cache_ptr   + dst, k, mask=mask)
    tl.store(value_cache_ptr + dst, v, mask=mask)


def store_kv(
    key: torch.Tensor, value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    slot_mapping: torch.Tensor, block_size: int,
) -> None:
    num_tokens, num_kv_heads, head_dim = key.shape
    hidden_size = num_kv_heads * head_dim
    BLOCK = triton.next_power_of_2(hidden_size)
    store_kv_kernel[(num_tokens,)](
        key, value, key_cache, value_cache, slot_mapping,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=block_size, hidden_size=hidden_size, BLOCK=BLOCK,
    )
```

###### Paged KV Cache Gather Kernel

> **地址映射**：逻辑 token `t` → 物理地址
> ```
> logical_block  = t // block_size
> offset         = t %  block_size
> physical_block = block_table[seq, logical_block]
> ```
> 输入形状：`kv_cache [num_blocks, block_size, num_heads, head_dim]`，`block_table [num_sequences, max_blocks]`

```python
import triton
import triton.language as tl

@triton.jit
def paged_kv_gather_kernel(
    cache_ptr, block_table_ptr, out_ptr,
    SEQ_LEN: tl.constexpr, NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr, BLOCK_D: tl.constexpr,
):
    seq, token, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    logical_block    = token // BLOCK_SIZE
    offset_in_block  = token %  BLOCK_SIZE
    physical_block   = tl.load(block_table_ptr + seq * MAX_BLOCKS + logical_block)

    offs_d = tl.arange(0, BLOCK_D)
    mask   = offs_d < HEAD_DIM

    cache_offset = (physical_block * BLOCK_SIZE * NUM_HEADS * HEAD_DIM
                    + offset_in_block * NUM_HEADS * HEAD_DIM
                    + head * HEAD_DIM + offs_d)
    out_offset   = (seq   * SEQ_LEN  * NUM_HEADS * HEAD_DIM
                    + token * NUM_HEADS * HEAD_DIM
                    + head * HEAD_DIM + offs_d)

    tl.store(out_ptr + out_offset, tl.load(cache_ptr + cache_offset, mask=mask, other=0.0), mask=mask)
```


###### Flash Attention Forward

> **核心思路**：Online softmax——每次只加载一个 `[BLOCK_M, BLOCK_N]` 的 QK tile，在寄存器中维护每行的 running max `m` 和 running sum `l`，无需全局 softmax 归一化再写回。避免了将完整 attention score 矩阵 `[SEQ, SEQ]` 写入 HBM。
>
> **在线 softmax 递推**（每个 KV block 更新一次）：
> ```
> m_new  = max(m_old, block_max)          # 更新 running max
> alpha  = exp(m_old - m_new)             # 旧累积的缩放因子
> p      = exp(scores - m_new[:, None])   # 当前块的 softmax numerator
> l      = l * alpha + sum(p, axis=1)     # 更新 running sum
> acc    = acc * alpha[:, None] + p @ V   # 更新输出 numerator
> ```
> 最终 `output = acc / l[:, None]`

**伪代码**

```python
# 每个 program 负责一个 query block × batch_head
q_block, batch_head = tl.program_id(0), tl.program_id(1)

q   = load_q_block(q_block)                          # [BLOCK_M, HEAD_DIM]
m   = full([BLOCK_M], -inf, fp32)                    # running max
l   = zeros([BLOCK_M], fp32)                         # running sum
acc = zeros([BLOCK_M, HEAD_DIM], fp32)               # output numerator

for start_n in range(0, SEQ_LEN, BLOCK_N):
    k, v = load_k_block(start_n), load_v_block(start_n)   # [BLOCK_N, HEAD_DIM]
    # scores = dot(q, transpose(k)) * softmax_scale  # [BLOCK_M, BLOCK_N]
    # QK 点积 + causal mask
	qk = tl.dot(q, tl.trans(k)) * scale   # [BLOCK_M, BLOCK_N]
	if causal:  # 此处 causal 需为 const，否则可能分支发散
		qk = tl.where(causal_mask, qk, -float("inf"))

    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.exp(m_i - m_new)                        # 修正旧累积的缩放
    p     = tl.exp(qk - m_new[:, None])          # 当前块 numerator
    l_i = l_i * alpha + tl.sum(p, axis=1)
	acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
	m_i = m_new

acc = acc / l_i[:, None] # 循环结束后归一化写回
tl.store(o_ptrs, acc)
```
