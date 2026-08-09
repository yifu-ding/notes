---
tags:
  - 八股
  - 生成模型
  - Diffusion
  - FlowMatching
---

# DDPM 和 Flow-Matching 的训练目标分别是如何构造的

> [!abstract] 一句话总结
> 两者都可以写成"先构造中间状态，再让网络回归一个监督目标"，区别主要在**监督目标是什么**。

---

## DDPM / Score Matching

先从真实样本 $x_0$ 构造带噪样本：

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

最常见的是让模型**预测加入的噪声**：

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0, t, \epsilon} \left[ \left\| \epsilon_\theta(x_t, t) - \epsilon \right\|^2 \right]
$$

也可以等价地**预测 score**：

$$
s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)
$$

对应目标为：

$$
\mathcal{L}_{\text{score}} = \mathbb{E} \left[ \left\| s_\theta(x_t, t) + \frac{\epsilon}{\sigma_t} \right\|^2 \right]
$$

> [!tip] 直觉
> DDPM 直接监督模型识别"当前样本里有什么噪声"，或"往高概率区域应该朝哪个梯度方向走"。

---

## Flow Matching

先采样真实数据 $x_0$ 和噪声 $x_1 = \epsilon$，再人为定义一条连接路径，例如线性路径：

$$
x_t = (1 - t)\, x_0 + t\, \epsilon
$$

这条路径的真实速度为：

$$
u_t = \frac{dx_t}{dt} = \epsilon - x_0
$$

然后直接让模型**预测速度**：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \left\| v_\theta(x_t, t) - (\epsilon - x_0) \right\|^2 \right]
$$

> [!note]
> 如果时间方向定义成从噪声到数据，目标写成 $(x_0 - \epsilon)$，本质一样。

---

## 核心区别

| | DDPM | Flow Matching |
|---|---|---|
| 监督目标 | 噪声 $\epsilon$ 或 score | 路径上的速度向量 $v$ |
| 推理 | score 换算得到速度场 | 直接积分速度场 |
| 损失形式 | MSE | MSE |

> [!warning] 关键认知
> 不只是"预测头"不同——真正的差异在于**前向概率路径和训练目标的构造方式**。
>
> 在特定噪声路径和参数化下，$\epsilon$、score、$x_0$ 和 velocity 可以相互转换。
