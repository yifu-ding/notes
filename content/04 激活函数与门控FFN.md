**Contents**

1. [[#标准 FFN 与激活函数|标准 FFN 与激活函数]]
	1. [[#标准 FFN 与激活函数#[Q1] FFN 块的计算公式|[Q1] FFN 块的计算公式]]
	2. [[#标准 FFN 与激活函数#[Q2] GeLU 的计算公式|[Q2] GeLU 的计算公式]]
	3. [[#标准 FFN 与激活函数#[Q3] Swish 的计算公式|[Q3] Swish 的计算公式]]
2. [[#门控 FFN|门控 FFN]]
	1. [[#门控 FFN#[Q4] 使用 GLU 线性门控单元的 FFN 块计算公式|[Q4] 使用 GLU 线性门控单元的 FFN 块计算公式]]
	2. [[#门控 FFN#[Q5] 使用 GeLU 的 GLU 块计算公式|[Q5] 使用 GeLU 的 GLU 块计算公式]]
	3. [[#门控 FFN#[Q6] 使用 Swish 的 GLU 块计算公式|[Q6] 使用 Swish 的 GLU 块计算公式]]
3. [[#各模型选择与对比|各模型选择与对比]]
	1. [[#各模型选择与对比#[Q7] 各 LLM 使用哪种激活函数？|[Q7] 各 LLM 使用哪种激活函数？]]
	2. [[#各模型选择与对比#[Q8] 为什么现代 LLM 更多使用 SwiGLU，而不是 GeGLU？|[Q8] 为什么现代 LLM 更多使用 SwiGLU，而不是 GeGLU？]]


### 标准 FFN 与激活函数

#### [Q1] FFN 块的计算公式

标准 Transformer FFN 由两个线性变换和一个非线性激活函数组成：

$$
\operatorname{FFN}(x)=f(xW_1+b_1)W_2+b_2
$$

其中，$f$ 是激活函数；$W_1$ 将隐藏维度从 $h$ 扩展到中间维度，$W_2$ 再将其投影回 $h$。

标准 FFN 只有 $W_1$ 和 $W_2$ 两个可训练权重矩阵。忽略偏置项时，若中间维度为 $d_{\mathrm{ff}}$，其参数量约为：

$$
h d_{\mathrm{ff}}+d_{\mathrm{ff}}h=2h d_{\mathrm{ff}}
$$

原始 Transformer 通常令 $d_{\mathrm{ff}}=4h$，因此标准 FFN 的参数量约为 $8h^2$。

---

#### [Q2] GeLU 的计算公式

GeLU（Gaussian Error Linear Unit）的精确形式为：

$$
\operatorname{GeLU}(x)
=x\Phi(x)
=\frac{x}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

常用的近似计算公式为：

$$
\operatorname{GeLU}(x)
\approx
\frac{x}{2}\left(
1+\tanh\left[
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right]
\right)
$$

其中，$\operatorname{erf}$ 是误差函数：

$$
\operatorname{erf}(z)=\frac{2}{\sqrt{\pi}}\int_0^z e^{-t^2}\,\mathrm{d}t
$$

这里的“误差”不是指神经网络的预测误差，而是来自概率论和测量误差理论。早期研究测量误差时，通常假设大量独立的小误差叠加后服从高斯分布；计算“误差落在某个范围内的概率”需要积分高斯函数 $e^{-t^2}$，所以这个积分被称为误差函数。

若随机变量 $Z\sim\mathcal N(0,1)$，则标准正态分布的累积分布函数为：

$$
\Phi(x)=\frac{1}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

因此：

$$
\Phi(x)=P(Z\leq x)
$$

GeLU 使用 Gaussian 的原因是，它可以被理解为一种与输入大小有关的平滑随机门控。假设使用服从伯努利分布的随机门 $m$，且保留输入的概率为 $\Phi(x)$：

$$
m\sim\operatorname{Bernoulli}(\Phi(x))
$$

则随机门控输出 $mx$ 的期望为：

$$
\mathbb E[mx]=x\Phi(x)=\operatorname{GeLU}(x)
$$

所以 GeLU 不像 ReLU 那样用阈值直接丢弃负数，而是根据输入在标准高斯分布中的位置，对输入进行平滑加权。

精确 GeLU 需要计算 $\operatorname{erf}$，其计算通常比近似式复杂。因此可以使用上面的 $\tanh$ 近似式降低计算成本，同时保持很小的近似误差。

训练大模型时，并不存在“模型超过多大就必须使用近似 GeLU”的固定分界线。选择精确形式还是近似形式主要取决于：

1. **模型原始定义与兼容性。** 必须尽量与预训练配置和 checkpoint 使用的 GeLU 版本一致。例如，一些 GPT 风格模型使用 $\tanh$ 近似形式；其他模型可能使用精确形式。
2. **底层算子效率。** 如果框架和硬件为精确 GeLU 提供了高效 fused kernel，即使训练大模型也完全可以使用精确形式；如果近似形式的 kernel 明显更快，则可能选择近似形式。
3. **性能收益。** GeLU 在整个 Transformer 计算量中的占比远低于矩阵乘法，因此不能仅根据模型参数规模判断近似形式是否值得使用，应当实际 benchmark。

PyTorch 的 `GELU` 默认使用精确形式 `approximate="none"`，也支持 `approximate="tanh"`。现代 LLM 还常直接使用 SwiGLU，而不再需要选择 GeLU 的精确或近似形式。

---

#### [Q3] Swish 的计算公式

$$
\operatorname{Swish}_{\beta}(x)=x\cdot\sigma(\beta x)
$$

其中：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

当 $\beta=1$ 时，Swish 也称为 SiLU：

$$
\operatorname{SiLU}(x)=x\cdot\sigma(x)
$$

原始 Swish 中，$\beta$ 可以是固定超参数，也可以作为可学习参数。因此，$\beta$ **可以是 learnable parameter**；训练时可通过反向传播更新它。实际使用时通常直接固定 $\beta=1$；现代 LLM 中所说的 SwiGLU，通常就是使用 $\operatorname{SiLU}(x)=x\sigma(x)$，而不会额外学习 $\beta$。

---

### 门控 FFN

#### [Q4] 使用 GLU 线性门控单元的 FFN 块计算公式

GLU（Gated Linear Unit）通过一条分支生成门控值，并与另一条线性分支逐元素相乘。使用 $W_1$、$W_2$ 表示两个输入投影矩阵：

$$
\operatorname{GLU}(x)
=\sigma(xW_1+b_1)\odot(xW_2+b_2)
$$

其中，$\sigma$ 是 Sigmoid 函数，它将门控分支的值压缩到 $(0,1)$：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

完整的门控 FFN 还需要使用第三个矩阵 $W_3$，将门控后的中间表示投影回隐藏维度：

$$
\operatorname{FFN}_{\mathrm{GLU}}(x)
=\left(\sigma(xW_1+b_1)\odot(xW_2+b_2)\right)W_3+b_3
$$

其中，$\odot$ 表示逐元素乘法。

是否包含偏置项是架构设计选择，而不是 GLU 的硬性要求：

- 原始 GLU 定义通常包含偏置项。
- 部分模型会保留偏置，以增加线性层的平移能力。
- LLaMA 等现代 LLM 通常去掉 FFN 中的偏置，以减少少量参数和计算、简化实现；由于大模型中偏置带来的收益通常有限，去掉后一般不会显著影响效果。

---

#### [Q5] 使用 GeLU 的 GLU 块计算公式

$$
\operatorname{GeGLU}(x)
=\operatorname{GeLU}(xW_1)\odot xW_2
$$

完整的 GeGLU FFN 为：

$$
\operatorname{FFN}_{\mathrm{GeGLU}}(x)
=\left(\operatorname{GeLU}(xW_1)\odot xW_2\right)W_3
$$

GeGLU FFN 共有 **3 个可训练权重矩阵**：$W_1$ 用于激活分支，$W_2$ 用于线性门控分支，$W_3$ 用于输出投影。若模型使用偏置，对应偏置项同样也是可学习参数。

因为 GeGLU 与 SwiGLU 一样具有三个权重矩阵，所以在与标准 FFN 进行参数量公平对比时，**GeGLU 的中间维度同样应该缩小到约 $\frac{8}{3}h$**。是否缩小不是由 GeLU 或 Swish 激活函数决定的，而是由是否需要控制总参数量和计算量决定的。

---

#### [Q6] 使用 Swish 的 GLU 块计算公式

$$
\operatorname{SwiGLU}(x)
=\operatorname{Swish}_{\beta}(xW_1)\odot xW_2
$$

完整的 SwiGLU FFN 为：

$$
\operatorname{FFN}_{\mathrm{SwiGLU}}(x)
=\left(\operatorname{Swish}_{\beta}(xW_1)\odot xW_2\right)W_3
$$

SwiGLU FFN 含有 **3 个可训练权重矩阵**。设门控 FFN 的中间维度为 $d_{\mathrm{ff}}'$，忽略偏置项时，其参数量约为：

$$
hd_{\mathrm{ff}}'+hd_{\mathrm{ff}}'+d_{\mathrm{ff}}'h
=3hd_{\mathrm{ff}}'
$$

标准 FFN 使用两个矩阵，中间维度为 $4h$ 时，参数量约为：

$$
2h(4h)=8h^2
$$

为了让门控 FFN 与标准 FFN 的参数量大致相同，令：

$$
3hd_{\mathrm{ff}}'=8h^2
\quad\Rightarrow\quad
d_{\mathrm{ff}}'=\frac{8}{3}h
=\frac{2}{3}\times4h
$$

因此，缩小中间维度的原因是 SwiGLU、GeGLU 等门控 FFN 比标准 GeLU FFN 多一个输入投影矩阵：

- **标准 GeLU FFN** 只有两个矩阵，因此使用 $4h$ 时已经是目标参数预算，不需要缩小。
- **GeGLU 与 SwiGLU FFN** 都有三个矩阵；若要与标准 FFN 公平比较，二者都应缩小到约 $\frac{8}{3}h$。
- 如果不要求参数量或计算量相同，也可以不缩小门控 FFN 的中间维度，但这会让模型拥有更多参数和更高计算成本，性能提升便不能只归因于激活函数或门控结构。

---

### 各模型选择与对比

#### [Q7] 各 LLM 使用哪种激活函数？

| 模型 | 激活函数 |
| --- | --- |
| GPT-3 | GeLU |
| LLaMA | SwiGLU |
| LLaMA 2 | SwiGLU |
| Baichuan | SwiGLU |
| ChatGLM-6B | GeLU |
| ChatGLM2-6B | SwiGLU |
| BLOOM | GeLU |
| Falcon | GeLU |

选择激活函数通常是模型质量、训练稳定性、计算效率和已有实验结果之间的权衡：

1. **GeLU 平滑且经过充分验证。** 它可以看作根据输入大小进行平滑的概率门控，在 BERT、GPT-3 等模型中表现稳定，因此许多较早或沿用经典 Transformer FFN 的模型使用 GeLU。
2. **SwiGLU 通常能提高模型质量。** SwiGLU 同时包含非线性激活分支和线性门控分支，表达能力通常强于单一 GeLU FFN；相关实验表明，在相近参数量和计算量下，门控 FFN 往往能获得更好的效果。
3. **SwiGLU 的代价是多一个权重矩阵。** 为了与标准 FFN 的参数量大致相当，SwiGLU 通常把中间维度从 $4h$ 缩小到约 $\frac{8}{3}h$。
4. **模型通常会继承所属架构族的设计。** 例如 LLaMA 系列采用 SwiGLU，后续参考 LLaMA 架构的模型往往继续使用 SwiGLU；沿用 GPT、BERT 等经典 FFN 设计的模型则更常使用 GeLU。

---

#### [Q8] 为什么现代 LLM 更多使用 SwiGLU，而不是 GeGLU？

首先，需要区分 **GeLU FFN** 和 **GeGLU FFN**：

- GeLU FFN 是经典的两矩阵 FFN。
- GeGLU 和 SwiGLU 都是三矩阵门控 FFN，二者仅门控分支的激活函数不同。

现代 LLM 更常选择 SwiGLU，主要原因是早期对 GLU 变体的对比实验表明：在相近参数量和计算预算下，SwiGLU 的效果通常非常好，并且经 PaLM、LLaMA 等模型大规模验证后，逐渐成为一种可靠的默认选择。后续模型往往直接继承这套已验证的设计，而不是重新承担选择 GeGLU 的训练实验成本。

这**不主要是因为 SwiGLU 固定 $\beta=1$ 后计算量更小**：

1. GeGLU 和 SwiGLU 的主要计算量都来自三个大型矩阵乘法；激活函数本身通常只占 FFN 总计算量的一小部分。
2. 精确 GeLU 需要计算 $\operatorname{erf}$，近似 GeLU 需要计算 $\tanh$；SiLU 需要计算 Sigmoid，而 Sigmoid 内部需要指数运算。三者都不是简单的加法或乘法。
3. 在实际硬件上，它们的速度取决于 fused kernel、数据类型和编译器优化。不能只根据公式断定 SiLU 一定明显快于 GeLU。
4. $\beta$ 是否可学习只增加一个标量或少量参数，其参数量和主要计算开销几乎可以忽略；现代 SwiGLU 固定 $\beta=1$ 更多是为了简单和采用经过验证的 SiLU 定义，而不是为了显著节省计算。

因此，更准确的结论是：**SwiGLU 的流行主要来自较好的实证效果、训练稳定性以及 PaLM/LLaMA 等成功架构带来的路径依赖；潜在的激活函数计算差异只是次要工程因素。** GeGLU 同样平滑且有效，在特定模型和硬件上也完全可能是更好的选择，最终应通过同等参数量、同等计算预算下的消融实验决定。
