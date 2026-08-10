---
date: 2026-06-23
---

==可以 DeepML 里练==


###### Softmax 激活函数实现

*Question* 编写一个 Python 函数，计算给定分数列表的 softmax 激活值。函数需处理数值稳定性问题，防止对较大值取指数时发生溢出。返回浮点数列表。

*E.g.* 输入 `scores = [1, 2, 3]`，输出 `[0.0900, 0.2447, 0.6652]`

*原理* Softmax 将一组数值转换为概率分布，每个元素的概率正比于其指数值除以所有元素指数值之和：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

*数值稳定性* 直接计算 $e^{x_i}$ 在 $x_i$ 很大时会溢出。常见技巧：减去最大值 $\max(x)$ 再取指数，结果不变但避免溢出：

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

*Solution*

```python
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    max_scores = np.max(scores, axis=-1, keepdims=True)
    shifted = scores - max_scores
    res = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    return res
```

---

###### 单神经元（Sigmoid 二分类）

*Question* 模拟一个带 sigmoid 激活函数的单神经元，用于二分类。函数接收特征向量列表（每个向量包含多个特征）、真实二值标签、权重向量和偏置，返回 sigmoid 激活后的预测概率列表，以及预测概率与真实标签之间的均方误差（MSE），均保留四位小数。

*E.g.* 输入 `features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]`，`labels = [0, 1, 0]`，`weights = [0.7, -0.4]`，`bias = -0.1`，输出 `([0.4626, 0.4134, 0.6682], 0.3349)`

*原理*

1. 对每个输入向量计算加权求和：$z = \mathbf{w} \cdot \mathbf{x} + b$
2. 经 sigmoid 激活得到预测概率：$\hat{y} = \sigma(z) = \dfrac{1}{1 + e^{-z}}$
3. 计算 MSE：$\text{MSE} = \dfrac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$

*Solution*

```python
import numpy as np

def single_neuron_model(features, labels, weights, bias):
    features = np.array(features)   # (n, m)
    labels = np.array(labels)       # (n,)
    weights = np.array(weights)     # (m,)

    z = features @ weights + bias   # (n,)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    logits = sigmoid(z)             # (n,)
    mse = np.mean((logits - labels) ** 2)

    probabilities = [round(p, 4) for p in logits.tolist()]  # 或者 probabilities = list(logits) 
    return probabilities, round(float(mse), 4)
```

> [!bug] 常见错误
> - `np.ndarray(data)` ❌ → `np.array(data)` ✅（前者接收 shape 整数，后者接收数据）
> - `weights * features` ❌ → `features @ weights` ✅（需要点积，不是逐元素乘）
> - `np.sum(np.mean(...))` ❌ → `np.mean(...)` ✅（1D 数组上 mean 已是标量）

---

###### Log-Softmax 函数实现

*Question* 给定一个 1D numpy 数组，计算其 log-softmax 值。

*E.g.* 输入 `[1, 2, 3]`，输出 `[-2.4076, -1.4076, -0.4076]`

*原理* Log-softmax 是 softmax 取对数，数值上更稳定：

$$\log\text{softmax}(x_i) = x_i - \log\sum_j e^{x_j}$$

减去 $\max(x)$ 后等价形式（防溢出）：

$$= (x_i - \max x) - \log\sum_j e^{x_j - \max x}$$

*Solution 1（直接法，更稳定）*

```python
import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    return shifted - np.log(np.sum(np.exp(shifted)))
```

*Solution 2（先算 softmax 再取 log）*

```python
import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.array(scores)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    shifted = scores - scores_max
    sm = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    return np.log(sm)
```

> [!tip] 两种写法的区别
> Solution 2 结果正确，但先算 softmax 再取 log，当某个 softmax 值极小时 `log(~0)` 会有精度损失。Solution 1 跳过中间的 softmax，直接用减法得到 log-softmax，数值更稳定，也是工业实践（如 PyTorch `F.log_softmax`）的做法。

---

###### ReLU 激活函数实现

*Question* 实现 ReLU（Rectified Linear Unit）激活函数：输入大于 0 时返回原值，否则返回 0。

*E.g.* `relu(0) → 0.0`，`relu(1) → 1.0`，`relu(-1) → 0.0`

*原理*

$$\text{ReLU}(x) = \max(0, x)$$

*Solution*

```python
def relu(x: float) -> float:
    return float(max(0, x))
```

---

###### Leaky ReLU 激活函数实现

*Question* 实现 Leaky ReLU 激活函数：输入大于 0 时返回原值，否则返回 `alpha * x`（默认 `alpha=0.01`）。

*E.g.* `leaky_relu(0) → 0`，`leaky_relu(1) → 1`，`leaky_relu(-1) → -0.01`，`leaky_relu(-2, alpha=0.1) → -0.2`

*原理*

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$$

相比 ReLU，负值区域保留一个小斜率 $\alpha$，缓解 **神经元死亡（dying ReLU）** 问题。

*Solution*

```python
def leaky_relu(z: float, alpha: float = 0.01) -> float:
    return z if z > 0 else alpha * z
```

---

###### 两个正态分布的 KL 散度

*Question* 计算两个正态分布 $P \sim \mathcal{N}(\mu_p, \sigma_p^2)$ 和 $Q \sim \mathcal{N}(\mu_q, \sigma_q^2)$ 之间的 KL 散度，返回浮点数。

*E.g.* `kl_divergence_normal(0.0, 1.0, 1.0, 1.0) → 0.5`

*原理* KL 散度衡量分布 $P$ 相对于参考分布 $Q$ 的差异，对两个正态分布有解析公式：

$$\text{KL}(P \| Q) = \log\frac{\sigma_q}{\sigma_p} + \frac{\sigma_p^2 + (\mu_p - \mu_q)^2}{2\sigma_q^2} - \frac{1}{2}$$

==知道 KL 散度公式就行==

验证示例：$\log\frac{1}{1} + \frac{1 + (0-1)^2}{2 \cdot 1} - \frac{1}{2} = 0 + 1 - 0.5 = 0.5$ ✓

*Solution*

```python
import math

def kl_divergence_normal(mu_p: float, sigma_p: float, mu_q: float, sigma_q: float) -> float:
    return (math.log(sigma_q / sigma_p)
            + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
            - 0.5)
```


----


###### RoPE

公式：[[06 旋转位置编码#[Q3] RoPE 的数学公式是什么？]]

$$
\theta_i = 10000^{-2i/d}, \qquad i = 0, 1, \ldots, \frac{d}{2}-1
$$

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    [x0, x1, x2, x3, ...]
    -> [-x1, x0, -x3, x2, ...]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:

    dim = x.size(-1)
    assert dim % 2 == 0

    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, dim, 2, device=x.device, dtype=torch.float32)
            / dim
        )
    )  # 10000*(-2i/d)

    # [B, S, D/2]
    angles = position_ids.float().unsqueeze(-1) * inv_freq

    # Repeat each angle twice:
    # [B, S, D]
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)

    cos = angles.cos().unsqueeze(1).to(x.dtype)
    sin = angles.sin().unsqueeze(1).to(x.dtype)

    return x * cos + rotate_half(x) * sin
```

----

###### FP8 Per-Tensor Dynamic Quantization

*原理* FP8 是 8-bit 浮点量化格式，有两种变体：
- `e4m3fn`：4 位 exponent + 3 位 mantissa，精度高，动态范围小
- `e5m2`：5 位 exponent + 2 位 mantissa，精度低，动态范围大

Per-tensor 动态量化步骤：
1. 计算 $\text{amax} = \max|x|$
2. 推导缩放因子 $s = \text{amax} / \text{FP8\_MAX}$，clamp 防止除零
3. $x / s$ 缩放至 FP8 可表示范围，clamp 后转换为 FP8

反量化：$\hat{x} = q \cdot s$

*Solution*

```python
import torch
from typing import Tuple

def fp8_quantize(
    x: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
    x_fp32 = x.float()
    fp8_max = torch.finfo(fp8_dtype).max 
    scale = torch.clamp(x_fp32.abs().max() / fp8_max, min=eps)
    x_scaled = torch.clamp(x_fp32 / scale, -fp8_max, fp8_max)
    return x_scaled.to(fp8_dtype), scale


def fp8_dequantize(
    q: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return (q.float() * scale).to(output_dtype)
```

----

###### MLA

```python
def forward(
    self,
    x: torch.Tensor,               # [B, S, hidden_size]
    position_ids: torch.Tensor,    # [B, S]
    past_kv_latent: torch.Tensor | None = None,  # [B, S_past, kv_latent_dim]
    past_k_rope: torch.Tensor | None = None,     # [B, 1, S_past, qk_rope_dim]
):
    B, S, _ = x.shape

    # 1. Query projection
    q = self.q_proj(x).view(B, S, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
    q_rope = apply_rope(q_rope, position_ids)  # q rope

    # 2. KV (no cache) down_proj
    kv_latent, k_rope = torch.split(
        self.kv_down_proj(x), [self.kv_latent_dim, self.qk_rope_dim], dim=-1
    )
    k_rope = apply_rope(k_rope.unsqueeze(1), position_ids) # k rope  # [B, 1, S, qk_rope_dim]

    # 3. Append KV cache (存 down_proj 后的 compressed kv)
    if past_kv_latent is not None:
        kv_latent = torch.cat([past_kv_latent, kv_latent], dim=1)
    if past_k_rope is not None:
        k_rope = torch.cat([past_k_rope, k_rope], dim=2)
    total_kv_len = kv_latent.size(1)

    # 4. kv (+cache) up_proj
    kv = self.kv_up_proj(kv_latent).view(
        B, total_kv_len, self.num_heads, self.qk_nope_dim + self.v_head_dim
    ).transpose(1, 2)
    k_nope, v = torch.split(kv, [self.qk_nope_dim, self.v_head_dim], dim=-1)

    # 5. Concatenate content and positional parts
    q_full = torch.cat([q_nope, q_rope], dim=-1)
    k_full = torch.cat([k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], dim=-1)

    # 6. Attention with causal mask
    scores = torch.matmul(q_full, k_full.transpose(-2, -1)) / math.sqrt(self.q_head_dim)
    q_pos = torch.arange(total_kv_len - S, total_kv_len, device=x.device)
    k_pos = torch.arange(total_kv_len, device=x.device)
    causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    scores = scores.masked_fill(causal_mask[None, None], float("-inf"))

    out = torch.matmul(F.softmax(scores, dim=-1), v)
    out = self.out_proj(out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.v_head_dim))

    return out, {"kv_latent": kv_latent, "k_rope": k_rope}
```

----

###### SmoothQuant + INT8 量化

*原理* 激活值存在 per-channel outlier，难以直接量化；权重分布更平滑。SmoothQuant 将量化难度从激活迁移到权重：

$$XW = \underbrace{(X / s)}_{\text{smooth activation}} \cdot \underbrace{(W \cdot s)^T}_{\text{absorb into weight}}$$

其中 per-input-channel 平滑尺度：

$$s_i = \frac{\max|x_i|^\alpha}{\max|w_i|^{1-\alpha}}$$

- $\alpha \to 1$：更多难度迁移到权重；$\alpha \to 0$：保持激活原始分布
- 量化方案：激活用 **per-token**（每行一个 scale），权重用 **per-output-channel**（每行一个 scale）

*Solution*

```python
import torch
import torch.nn.functional as F
from typing import Tuple


@torch.no_grad()
def compute_smoothquant_scale(
    activation: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Per-input-channel scale: s_i = amax(x_i)^alpha / amax(w_i)^(1-alpha)."""
    x, w = activation.float(), weight.float()
    reduce_dims = tuple(range(x.ndim - 1))
    act_amax = x.abs().amax(dim=reduce_dims).clamp_min(eps)   # [in_features]
    w_amax   = w.abs().amax(dim=0).clamp_min(eps)             # [in_features]
    return act_amax.pow(alpha) / w_amax.pow(1.0 - alpha)


@torch.no_grad()
def apply_smoothquant(
    activation: torch.Tensor,
    weight: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """XW = (X/s)(Ws)^T."""
    # s_x = smooth_scale.to(device=activation.device, dtype=activation.dtype)
    # s_w = smooth_scale.to(device=weight.device, dtype=weight.dtype)
    # return activation / s_x, weight * s_w.unsqueeze(0)
	记忆版：
	return activation / s, weight * s.unsqueeze(0) 

@torch.no_grad()
def quantize_int8_per_token(
    x: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric INT8. scale: [..., 1]."""
    scale = (x.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(eps)  # amax 只返回 value 不返回 index，如果 max 的话需要 .value 才等价
    q = torch.clamp(torch.round(x.float() / scale), -127, 127).to(torch.int8)
    return q, scale


@torch.no_grad()
def quantize_int8_per_output_channel(
    weight: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT8. scale: [out_features, 1]."""
    scale = (weight.float().abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(eps)  
    q = torch.clamp(torch.round(weight.float() / scale), -127, 127).to(torch.int8)
    return q, scale


@torch.no_grad()
def smoothquant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """SmoothQuant INT8 linear（simulation：dequant before matmul）。"""
    scale = compute_smoothquant_scale(x, weight, alpha)
    x_s, w_s = apply_smoothquant(x, weight, scale)
    q_x, sx = quantize_int8_per_token(x_s)
    q_w, sw = quantize_int8_per_output_channel(w_s)
    return F.linear(q_x.float() * sx, q_w.float() * sw, bias)
```

> [!tip] 注意
> - `amax(dim=reduce_dims)` 对除最后一维外的所有维度取最大值，得到 per-input-channel 统计。如果 max 的话需要 .value 
> - weight 的 `amax(dim=0)` 沿 out_features 方向聚合，保留 in_features 维度
> - 真实内核会融合 scale 和 INT8 GEMM，这里 dequant 后再 matmul 仅用于模拟验证

---



###### Paged KV Cache Store / Gather（Python）

> **Prefill vs Decode**：store 函数完全相同。区别在输入规模：prefill 时 `num_tokens` = 整个 prompt 的 token 数；decode 时 `num_tokens` = batch_size（每条序列只新增 1 个 token）。`slot_mapping` 由调度器负责计算，两阶段共用同一函数。

*生成 slot_mapping*：对这一 batch 里每个待写入的 token，调度器查它对应的逻辑块 → 物理块，算出 slot

```python
# 伪代码
for seq in batch:
    for new_token_pos in seq.new_token_positions:
        logical_block = new_token_pos // block_size
        offset        = new_token_pos %  block_size
        phys_block    = seq.block_table[logical_block]
        slot = phys_block * block_size + offset
        slot_mapping.append(slot)
```

*Store — 写入 KV cache*

```python
def store_key_value(
    key: torch.Tensor,           # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor,
    key_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]; slot = block_id * block_size + offset; <0 → skip
    block_size: int,
) -> None:
    valid         = slot_mapping >= 0
    token_ids     = torch.arange(slot_mapping.shape[0], device=slot_mapping.device)[valid]  # token 在输入里的行号
    slots         = slot_mapping[valid]   # 找到物理的 slot 的编号
    block_ids     = slots // block_size   # 找到物理 block id 和 offset 用来索引 cache 
    block_offsets = slots %  block_size
    key_cache[block_ids, block_offsets]   = key[token_ids]
    value_cache[block_ids, block_offsets] = value[token_ids]
```

*Gather — 从 KV cache 读出历史 KV（decode 时用于 attention）*

> 每条序列的逻辑 token `t` → `logical_block = t // block_size`，`offset = t % block_size`，再经 `block_table` 查出物理块号，用 advanced indexing 一次取出。

```python
def gather_key_value(
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    block_table: torch.Tensor,  # [batch_size, max_blocks]  逻辑块 → 物理块
    context_lens: torch.Tensor, # [batch_size]  每条序列的实际长度
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 key/value: [batch_size, max_seq_len, num_kv_heads, head_dim]"""
    max_seq_len = int(context_lens.max().item())
    t = torch.arange(max_seq_len, device=key_cache.device)  # [max_seq_len]
    # 逻辑位置 → 物理块号 + 块内 offset
    logical_blocks = t // block_size                          # [max_seq_len]
    offsets        = t %  block_size                          # [max_seq_len]
    phys_blocks    = block_table[:, logical_blocks]           # [B, max_seq_len]
    # Advanced indexing: [B, max_seq_len, num_kv_heads, head_dim]
    keys   = key_cache[phys_blocks, offsets]
    values = value_cache[phys_blocks, offsets]
    # 超出 context_len 的位置置零
    mask = (t[None, :] < context_lens[:, None])              # [B, max_seq_len]
    keys   = keys   * mask[..., None, None]
    values = values * mask[..., None, None]
    return keys, values
```

*Naive Decode Attention — online softmax（支持 GQA/MQA）*

> **Online softmax**：逐 token 维护 running max `m`、running sum `l`、累积输出 `acc`。每步用 `alpha = exp(m_old - m_new)` 修正旧值，无需先收集所有 scores。
> ```
> m_new = max(m, score)
> alpha = exp(m - m_new)           # 修正旧累积
> l     = l * alpha + exp(score - m_new)
> acc   = acc * alpha + exp(score - m_new) * v
> output = acc / l                 # 最终归一化
> ```

```python
def naive_decode_attention(
    query: torch.Tensor,       # [B, Hq, D]
    key_cache: torch.Tensor,   # [num_blocks, block_size, Hkv, D]
    value_cache: torch.Tensor,
    block_table: torch.Tensor, # [B, max_blocks]
    seq_lens: torch.Tensor,    # [B]
) -> torch.Tensor:             # [B, Hq, D]
    B, num_q_heads, D = query.shape
    _, block_size, num_kv_heads, _ = key_cache.shape
    queries_per_kv = num_q_heads // num_kv_heads
    scale  = 1.0 / math.sqrt(D)
    output = torch.empty_like(query)

    for b in range(B):
        seq_len = int(seq_lens[b])
        for qh in range(num_q_heads):
            kvh = qh // queries_per_kv
            q   = query[b, qh]                                    # [D]
            m, l = -float("inf"), 0.0
            acc  = torch.zeros(D, device=q.device, dtype=q.dtype)
            for t in range(seq_len):
                phys  = int(block_table[b, t // block_size])
                k     = key_cache[phys, t % block_size, kvh]
                v     = value_cache[phys, t % block_size, kvh]
                score = (torch.dot(q, k) * scale).item()
                m_new = max(m, score)
                alpha = math.exp(m - m_new)                       # 修正旧累积的缩放
                l     = l * alpha + math.exp(score - m_new)
                acc   = acc * alpha + math.exp(score - m_new) * v
                m     = m_new
            output[b, qh] = acc / l                               # 归一化

    return output
```


---

###### Dynamic KV Cache（非 Paged，逐层 concat）

```python
import torch

class DynamicKVCache:
    """每层维护一个 [B, Hkv, total_len, D] 的 tensor，decode 时逐步 concat。"""

    def __init__(self):
        self.key_cache:   list[torch.Tensor | None] = []
        self.value_cache: list[torch.Tensor | None] = []

    def update(
        self,
        new_key: torch.Tensor,    # [B, Hkv, new_len, D]
        new_value: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """追加新 KV，返回该层完整 KV: [B, Hkv, total_len, D]。"""
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx]   = new_key
            self.value_cache[layer_idx] = new_value
        else:
            self.key_cache[layer_idx]   = torch.cat([self.key_cache[layer_idx],   new_key],   dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], new_value], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[2]
```



###### MoE torch 代码

```python
class SimpleMoE(nn.Module):
    def __init__(self, d, n_experts, top_k=2):
        super().__init__()
        self.k = top_k
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)) for _ in range(n_experts)])

    def forward(self, x):                 # x: [T, D]
        score, idx = self.router(x).topk(self.k, dim=-1)
        weight = score.softmax(dim=-1)    # [T, K]
        output = torch.zeros_like(x)
        for k in range(self.k):
            expert_ids = indices[:, k]   
            for e, expert in enumerate(self.experts):
	            mask = expert_ids == e
                if mask.any():
	                tokens = x[mask] 
	                out = expert(tokens)
                    output[mask] += (weights[mask, k:k+1] * out)
        return output
```


