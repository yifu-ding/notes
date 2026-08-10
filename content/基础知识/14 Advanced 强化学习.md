### 算法全景与分类

#### [Q1] RLHF 后训练中有哪些主要算法路线？

RLHF 之后，LLM 对齐领域发展出多条相对独立的算法路线。理解这些路线的**继承与独立关系** 比逐个记住公式更重要。

**从训练方式看，可以粗分为两大类：**

- **在线策略优化路线**：训练时不断让当前策略生成新回答，再根据奖励更新策略。代表方法包括 **PPO、PPO-PTX、GRPO、DAPO、Dr. GRPO**；
- **离线偏好优化路线**：训练时直接读取固定的 chosen/rejected 偏好对，不需要 on-policy rollout。代表方法包括 **DPO、IPO、ORPO** 。

**从历史脉络看：**

- PPO 来自传统策略梯度家族中的 TRPO → PPO 演化；
- GRPO / DAPO / Dr. GRPO 是面向 LLM 场景对 PPO 的轻量化或去偏改造；
- DPO / IPO / ORPO 则不是 PPO 的后代，而是从**偏好数据的直接优化** 出发的另一条路线。

> [!tip]
> 直觉上，这两条路线的本质区别是：
> 
> - **PPO 系路线** 把问题看成“模型先去尝试生成，再根据结果好坏回头改策略”——本质上是**在线试错**；
> - **DPO 系路线** 把问题看成“我们手里已经有 chosen / rejected 偏好对，直接让模型学会更偏向 chosen”——本质上是**离线比较学习** 。
> 
> 换句话说：
> 
> - **PPO / GRPO** 更像“** 边做边学**”：当前策略先 rollout，拿到奖励后，再决定哪些行为该强化、哪些该压低；
> - **DPO / IPO / ORPO** 更像“** 看例子学偏好**”：不要求模型当场探索新答案，而是直接从现成偏好数据中学习“哪种回答更值得提高概率”。
> 
> 所以两者最深层的差异，不只是“一个要不要 Critic”或“一个稳不稳定”，而是：
> 
> - **前者优化的是与环境交互后的回报**，强调探索能力；
> - **后者优化的是固定偏好数据中的相对概率关系**，强调训练简单和稳定。

> [!NOTE] **代表性工作：**
> 
> - **PPO 系列** 最经典的代表是 **InstructGPT**，其论文中的主模型就是 **PPO-PTX**；
> - **PPO / 拒绝采样混合路线** 的代表是 **Llama 2-Chat**，Meta 明确写过它结合了 **Rejection Sampling 和 PPO**；
> - **GRPO 路线** 的代表是 **DeepSeekMath 7B**，GRPO 就是在这篇工作里被系统提出；
> - **更激进的 GRPO 式 RL 路线**，一个高知名度代表是 **DeepSeek-R1 / R1-Zero**，论文明确说明其 RL 框架建立在 **GRPO** 上；
> - **DAPO 路线** 目前更像“最新一代 open RL recipe”的代表，典型例子是 **ByteDance Seed 基于 Qwen2.5-32B base 的 DAPO 系统**；
> - **DPO 系列** 里最知名的公开代表之一是 **Zephyr-7B**，其论文使用的是 **dDPO / DPO 路线**；
> - **Llama 3 的官方后训练** 很有代表性：Meta 在 **2024 年 4 月 18 日** 的官方博客里把它概括为 **SFT、Rejection Sampling、PPO、DPO** 的组合；而在 **2024 年 7 月 23 日** 的技术论文里，则更详细地展开了其主干配方是 **Reward Model + Rejection Sampling + SFT + DPO**，并说明他们也探索过 PPO。
> 
> 

> [!NOTE] **除了 PPO 系和 DPO 系之外，还有哪些被广泛认可的路线？**
> 
> 目前比较公认的还有三类：
> 
> 1. **SFT 路线**：先用高质量指令数据把“会不会答、怎么答”打牢。它不是偏好优化，但几乎是所有后续路线的基础；
> 2. **拒绝采样 / RAFT / RSFT 路线**：先采样多个候选，再筛出最好的当作 SFT 标签。这条路线工程上很常见，因为比在线 RL 稳；
> 3. **混合式后训练路线**：工业界越来越常见的做法不是只选 PPO 或只选 DPO，而是把 **SFT + 采样筛选 + 偏好优化 + 一小段 RL** 组合起来，用不同阶段解决不同问题。
> 
> 直觉上可以把当前主流实践理解成：
> 
> - **SFT** 负责把模型教到“能用”；
> - **RS / RAFT** 负责把“可验证的好答案”筛出来；
> - **DPO** 负责稳定地吸收偏好数据；
> - **PPO / GRPO 类 RL** 负责在可验证任务上继续探索和冲高上限。
> 

**选择路线时最常见的考量**

| 考量 | 更适合的方法 |
| --- | --- |
| 奖励可自动验证（数学、代码、规则） | GRPO / DAPO |
| 已有大量高质量偏好对 | DPO / IPO / ORPO |
| 需要继续探索新解法 | PPO / GRPO |
| 基础设施更接近 SFT | DPO / ORPO |
| 担心能力退化 | PPO-PTX |

进一步参见：[[基础强化学习#[Q22] 什么是 PPO 的 clipping？|PPO 的 clipping]] | [[基础强化学习#[Q23] TRPO 如何通过 KL 信任区域限制更新？|TRPO 信任区域]]

---

### PPO-PTX

#### [Q2] PPO-PTX 是什么？InstructGPT 如何防止能力退化？

**背景**

InstructGPT（OpenAI, 2022）是将 RLHF 应用于大规模语言模型的代表性工作。实践中发现：如果 PPO 只围绕指令奖励优化，模型可能会逐渐遗忘预训练阶段学到的通用能力，例如语言流畅性、知识覆盖和泛化表现下降。

**PPO-PTX 的核心做法**

PPO-PTX 在标准 PPO 目标之外，再混入一项**预训练语言建模损失（Pretraining Loss, PTX）**：

$$
\mathcal{L}_{\mathrm{PPO\text{-}PTX}}(\theta)
=
\mathcal{L}_{\mathrm{PPO}}(\theta)
-
\lambda\,\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{pretrain}}}
\left[\log\pi_\theta(x)\right]
$$

其中 $\lambda$ 控制 PPO 与预训练损失之间的权重。

**为什么这样有效？**

KL 惩罚只是限制当前策略不要离参考模型太远，但它仍然是**间接约束** 。PTX 则更直接：它要求模型在预训练分布上继续保持较好的语言建模能力。于是，模型一边朝着更符合偏好的方向更新，一边又不会太快丢掉通用语言能力。

> [!note] PTX vs KL 惩罚
> KL 惩罚回答的是“不要偏太远”；PTX 回答的是“原来的语言建模能力也要继续保住”。两者可以同时存在。

**什么时候会考虑 PPO-PTX？**

- 模型规模大，预训练成本高，不希望后训练破坏已有能力；
- 指令数据比较窄，担心 PPO 只会在局部风格上过拟合；
- 已观察到 RLHF 后模型在通用 benchmark 上回退。

---

### GRPO

#### [Q3] 什么是 GRPO？它如何去掉独立 Critic？

GRPO（Group Relative Policy Optimization）可以看成是**面向 LLM 推理任务的 PPO 简化版** 。它保留了“在线 rollout + 策略梯度 + 受控更新”这条主线，但去掉了 PPO 中单独训练的 Critic / Value Model，转而用**同一题下多条回答之间的相对好坏** 来构造 Advantage。

**先回顾 PPO 为什么需要 Critic**

在标准 PPO 中，Actor 负责生成回答，Critic 负责估计当前前缀状态的期望回报 $V(s_t)$。训练时会用：

$$
A_t = G_t - V(s_t)
$$

其中 $G_t$ 是实际得到的回报，$V(s_t)$ 是基准线。这个基准线的作用不是定义奖励，而是降低策略梯度的方差。

但在 LLM 场景里，Critic 往往和策略模型同量级，代价很高：

- 需要额外显存保存一套大模型；
- 需要单独训练 value head 或 value model；
- Critic 自身不准时，还会把噪声传给 Actor。

**GRPO 的核心想法**

GRPO 不再问”这个前缀理论上值多少分”，而是改问：**同一个 prompt 下，这个回答比同组其他回答更好吗？** 具体做法是：对同一个 prompt $x$，一次采样 $G$ 个回答 $y_1,\dots,y_G$，再为它们分别计算奖励 $r_1,\dots,r_G$。随后，用组内均值和标准差构造相对 Advantage：

$$
\hat{A}_i
=
\frac{r_i-\mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

这意味着：

- 某个回答如果高于组内平均水平，$\hat{A}_i > 0$，就提高它的概率；
- 如果低于组内平均水平，$\hat{A}_i < 0$，就降低它的概率；
- 组内均值本身就扮演了 baseline 的角色，因此不再需要独立 Critic。

**GRPO 的训练流程可以直观理解为四步**

1. 用当前策略对同一个 prompt 生成一组候选回答；
2. 用 Reward Model、规则或可验证结果给每条回答打分；
3. 在组内做相对比较，得到每条回答的 Advantage；
4. 用 PPO 风格的 clipped objective 更新策略。

因此，GRPO 并不是“没有 PPO 的思想”，而是**保留 PPO 的更新框架，只把 Advantage 的来源从 Critic 改成组内相对奖励** 。

**为什么 GRPO 特别适合数学、代码这类任务？**

因为这类任务往往有相对清晰的自动评分方式：

- 数学题可以看最终答案是否正确；
- 代码题可以跑单元测试；
- 格式任务可以用规则验证。

一旦奖励足够可靠，GRPO 就能通过“同题多答、优中选优”的方式稳定提供比较信号，而不必再额外训练一个 Value Model。

**GRPO 与 PPO 的关键区别**

| 对比项 | PPO | GRPO |
| --- | --- | --- |
| Advantage 来源 | Critic 估计的 $V(s_t)$ | 组内相对奖励 |
| 训练模型数 | 常见为 4 个（Actor/Critic/Ref/RM） | 常见为 3 个（无独立 Critic） |
| 更新依据 | “比预期好多少” | “比同组其他回答好多少” |
| 更适合 | 通用在线 RL | 可自动评分的 LLM 推理任务 |

**GRPO 的优点**

- **省显存、省训练链路**：去掉了独立 Critic；
- **更贴近推理任务的相对比较本质**：很多问题更容易判断“谁更好”，而不是精确估计一个 value；
- **实现上更接近 PPO 的简化版**：易于复用已有在线 RL 基础设施。

**GRPO 的局限**

它省掉了 Critic，但并没有消灭不稳定性，只是把敏感点换了地方：

- 若组大小 $G$ 太小，组内均值和方差估计会很噪；
- 若一组回答全对或全错，组内方差接近 0，训练信号会退化；
- 它学到的是“相对更好”，因此非常依赖组内样本覆盖；
- 若奖励本身不可靠，组内比较也会把噪声放大。

> [!warning] 一个容易误解的点
> GRPO 去掉的是**独立 Critic**，不是去掉 baseline。它只是把 baseline 从“学出来的 Value”换成了“同组回答的平均表现”。

**什么时候适合用 GRPO？**

- 奖励能够自动计算或近似自动计算；
- 希望继续让模型探索新答案，而不是只学习固定偏好对；
- PPO 的 Critic 成本过高，希望保留在线 RL 但简化链路。

进一步参见：[[13 基础强化学习#[Q6] PPO 阶段为什么通常同时存在四个模型？|PPO 四模型设置]] | [[13 基础强化学习#[Q22] 什么是 PPO 的 clipping？|PPO clipping]] | [[#[Q4] DAPO 对 GRPO 做了哪些改进？|DAPO]]

#### [Q3.1] GRPO clip 之后的梯度是多少？

GRPO 对每个 token 应用 PPO 风格的 clipped objective，令概率比为：

$$
r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_\text{old}(y_t \mid x, y_{<t})}
$$

同一 response 内各 token 共享序列级优势 $A_t = A$，单 token 目标为：

$$
L_t(\theta) = \min\bigl(r_t A,\ \operatorname{clip}(r_t,\ 1-\epsilon,\ 1+\epsilon)\, A\bigr)
$$

**当 $A > 0$（希望提高该 token 概率）**

一旦 $r_t > 1+\epsilon$，min 选常数分支，梯度归零：

$$
\nabla_\theta L_t = \begin{cases} A \cdot r_t \cdot \nabla_\theta \log\pi_\theta(y_t), & r_t \le 1+\epsilon \\ 0, & r_t > 1+\epsilon \end{cases}
$$

**当 $A < 0$（希望降低该 token 概率）**

一旦 $r_t < 1-\epsilon$，min 选常数分支，梯度归零：

$$
\nabla_\theta L_t = \begin{cases} 0, & r_t < 1-\epsilon \\ A \cdot r_t \cdot \nabla_\theta \log\pi_\theta(y_t), & r_t \ge 1-\epsilon \end{cases}
$$

> [!note] 核心结论
> 梯度为 0 的条件是**"在正确方向上已经走得足够远"**——正优势 token 的概率比超过上界，或负优势 token 的概率比低于下界。
>
> 若 loss 中含有 KL 惩罚 $\beta D_\text{KL}$，即使 policy-gradient 部分梯度为 0，KL 项仍可能产生非零梯度。

#### [Q3.2] 为什么 clip 完之后 GRPO 还不稳定？

clip 约束的是"单步概率比不超过边界"，并不能消除以下几类不稳定源：

**1. 大量 token 被 clip → 有效梯度幅度缩小**

被 clip 的 token 不贡献梯度，但仍占分母：

$$
\nabla_\theta L = \frac{1}{T} \sum_{t \in \text{unclipped}} \nabla_\theta L_t \quad (\text{而非除以 } T - K)
$$

clip 比例越高，有效梯度幅度越小，等效降低了当前 batch 的更新力度。

**2. 少数未 clip token 主导更新方向 → 梯度更噪声**

若 100 个 token 中 80 个被 clip，剩余 20 个决定整个 batch 的更新方向，梯度方向对少量 token 的噪声更加敏感。

**3. 被 clip token 的概率仍会变化**

clip 只阻断该 token 自身的 policy-gradient 信号。其他 token 更新了共享 Transformer 参数后，该 token 的概率仍会连带改变——"梯度为 0"不等于"概率被冻结"。

**4. 其他 loss 项不受 clip 约束**

若总 loss 含 KL 或 entropy 正则：

$$
L_\text{total} = L_\text{GRPO} + \beta L_\text{KL} + \lambda L_\text{aux}
$$

即使某 token 的 GRPO 梯度为 0，KL 项仍可能产生非零梯度并持续推动参数更新。

> [!note] 理解 clip 的作用边界
> clip 的设计目标是防止单步更新过大，不是保证训练稳定。clip 比例过高、组内全对/全错、奖励噪声大时训练仍会退化——这些正是 DAPO 进一步修补的动机。

---

### DAPO

#### [Q4] DAPO 对 GRPO 做了哪些改进？

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）可以理解为：**接受 GRPO 的总体框架，但把它补成一套真正可大规模跑通的 long-CoT RL recipe。**

一个很有代表性的 DAPO 工作，是 **ByteDance Seed 在 Qwen2.5-32B base model 上做的大规模 RL 系统** 。这项工作的背景很直接：作者先用较朴素的 GRPO 去复现推理 RL，结果在 AIME 2024 上只做到大约 **30 分**；而他们最终的 DAPO 系统把同一个 base model 推到了 **50 分**，并在论文中写到其结果超过了 **DeepSeek-R1-Zero-Qwen-32B 的 47 分**，且使用了约 **50% 的训练步数** 。

所以，DAPO 不只是”又一个 GRPO 小变体”，更像是**一套面向长推理任务的开源 RL 工程配方** 。它的目标不是单纯改一个公式，而是同时解决 long-CoT 场景里最常见的几类训练问题：

- **entropy collapse**：策略过早变得单一，探索不够；
- **无区分度样本过多**：整组回答全对或全错，几乎没有训练信号；
- **长回答吃亏**：长推理链在 loss 里被隐式弱化；
- **超长输出带来的奖励噪声**：回答被截断后，错误惩罚会污染训练。

ByteDance Seed 的论文把 DAPO 的关键做法总结为四项，这四项比“单看 DAPO 这个名字”更重要。

**第一，Clip-Higher（非对称 clipping）**

标准 PPO / GRPO 通常对概率比使用对称 clip。DAPO 认为，对正 Advantage 的好动作可以适当放宽上界，让模型更积极地强化明显更优的回答；而对负 Advantage 的差动作，仍保持相对保守的压制。直觉上，发现真正更好的推理路径时，就应该允许模型更大胆地朝它靠拢。作者把这一步看成是在解决**探索不够、熵塌缩过快** 的问题。

**第二，动态采样（Dynamic Sampling）**

如果某个 prompt 采样出的所有回答都同样正确，或者同样错误，那么组内比较没有信息量。DAPO 会尽量过滤这类“无区分度样本”，只保留那些组内确实有好坏差异的 prompt 参与训练，从而减少无效梯度步骤。

直觉上，这相当于不让训练算力浪费在”这组样本根本分不出高下”的题上。

**第三，token 级策略梯度（Token-level Policy Gradient Loss）**

标准 GRPO 更偏向在回答级别聚合 log-prob。这样做时，长回答常常因为归一化方式而被隐式削弱。DAPO 转而强调 token 级别的贡献，希望避免模型只因为“短回答更容易拿到更大单位梯度”而倾向于缩短推理链。

这一步对 long-CoT 特别关键，因为长推理任务里，模型往往恰恰需要**更长、更完整的思考过程** 。

**第四，Overlong Reward Shaping（超长输出奖励塑形）**

在长推理训练中，模型常常会生成特别长的回答，甚至超过长度上限。若一个回答只是因为**被截断** 而拿到坏奖励，那么这个惩罚里混入了大量“长度造成的噪声”，不完全代表推理本身有问题。

DAPO 因此加入了 overlong reward shaping：核心思想是**不要把“超长被截断”与“真正推理错误”混为一谈**，从而降低奖励噪声、稳定训练。

**关于 KL 惩罚，还要补一句**

DAPO 这套 recipe 也倾向于**不再依赖显式 KL 惩罚**，而更依靠 clipping、本身的采样过滤和规则奖励来控制训练。这不是它名字里的主改动，但确实体现了它和传统 PPO-RLHF 的风格差异：在长推理任务里，它更愿意给探索留空间。

> [!summary] DAPO 的核心改动
> | 改动 | 主要解决的问题 |
> | --- | --- |
> | Clip-Higher | 好回答被过早截断，探索不足、熵塌缩 |
> | Dynamic Sampling | 组内无差异样本导致无效训练 |
> | Token-level Loss | 长回答被隐式低权重 |
> | Overlong Reward Shaping | 截断带来的奖励噪声 |

> [!note] 如何记住 DAPO？
> 可以把它记成：**“GRPO + 一整套 long-CoT 工程修补包”** 。它的重点不只是 Advantage 怎么算，而是如何把长推理 RL 真正训稳、训强、训到可复现。

**一句话概括**

DAPO 不是另起炉灶的新路线，而是==在 GRPO 框架内，为长推理任务补齐大规模可复现训练的关键工程细节==。

---

### Dr. GRPO

#### [Q5] Dr. GRPO 解决了什么问题？

Dr. GRPO 可以理解为对 GRPO 的**统计去偏版本** 。它关注的不是“要不要 Critic”，也不是“clip 要不要放宽”，而是更基础的问题：**GRPO 里用组内相对奖励构造的 Advantage，本身会不会带系统性偏差？** 答案是：会，尤其体现在两个方面。

**第一类偏差：问题难度偏差**

GRPO 对每个 prompt 都在组内做标准化。如果一道题本身很简单，那么这一组回答的奖励可能普遍偏高，且彼此差异较小；如果一道题很难，则奖励普遍偏低，波动模式也不同。

结果是，不同难度题目的 Advantage 尺度并不天然可比。训练时，简单题往往更容易持续提供稳定的正向信号，困难题反而可能因为噪声更大而贡献较小更新。模型就会更偏向强化“本来就容易做对的题”。

**第二类偏差：回答长度偏差**

如果训练在回答级别聚合 log-prob，长回答的单位 token 更新常常被平均得更小。对于需要长推理链的问题，这会形成一种隐式惩罚：

- 不是因为长回答更差；
- 而是因为它在损失里被归一化得更弱。

**Dr. GRPO 的核心修正**

Dr. GRPO 试图让不同题目、不同长度回答的梯度尺度更公平，主要思路包括：

1. **在问题级别做去偏**：避免简单题因为组内统计特性而获得系统性更大更新；
2. **在长度维度做去偏**：让长回答不会仅仅因为 token 更多就被平均掉梯度。

因此，Dr. GRPO 的关键词不是”更轻量”或”更激进”，而是==让 GRPO 的 Advantage 更接近无偏、可比较的训练信号==。

> [!note] Dr. GRPO vs DAPO
> DAPO 主要在训练机制上改：clip、采样、token-level loss、KL 约束；Dr. GRPO 主要在统计估计上改：减少难度偏差和长度偏差。两者可以视为互补方向。

---

### DPO

#### [Q6] 什么是 DPO？它的偏好对从哪里来？

DPO（Direct Preference Optimization）是最典型的**离线偏好优化** 方法。它不再像 PPO / GRPO 那样边训练边 rollout，而是直接读取一个已经准备好的固定偏好数据集：

$$
(x, y_w, y_l)
$$

其中 $y_w$ 是 chosen，$y_l$ 是 rejected。训练目标不是生成新回答再打分，而是让模型在已有偏好对上学会：**相对于 rejected，更倾向 chosen** 。

**这些偏好对从哪里来？**

通常流程是：

1. 先为同一个 prompt 准备多个候选回答；
2. 再由人工、AI Judge、Reward Model 或可验证规则挑出较好和较差的回答；
3. 最终把它们整理成固定的 $(x, y_w, y_l)$ 数据集。

因此，DPO 虽然叫“离线”，但并不意味着候选回答从来没有被模型生成过。更准确地说：

- **数据制作阶段** 可以使用历史模型或 SFT 模型生成候选；
- **真正优化 DPO 时**，不再要求当前策略进行 on-policy rollout。

> [!important] “DPO 不需要 rollout”的准确含义
> 不是说偏好对不能来自模型采样，而是说 **DPO 训练期间不需要边生成、边打分、边更新** 。它优化的是固定数据，而不是当前策略实时产生的新数据。

**DPO 训练时具体在做什么？**

训练时，模型会在 teacher forcing 模式下，分别读取 chosen 和 rejected 这两条固定回答，计算它们的序列 log-prob：

$$
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x, y_{<t})
$$

这里的 teacher forcing 只是为了评估：

- 当前模型给 chosen 分配了多大概率；
- 当前模型给 rejected 分配了多大概率。

它并不要求这些回答是标准答案，也不要求模型训练时重新把它们生成一遍。

**DPO 的核心数学思想**

DPO 的关键观察是：对 KL 正则化的奖励最大化目标，最优策略可以写成参考策略乘上一个与奖励相关的指数项。因此，可以把原本隐含的奖励函数，改写为“策略相对于参考模型的概率比”。

从这个关系出发，再结合 Bradley–Terry 偏好建模（参见[[基础强化学习#[Q4] 奖励模型是如何训练的？|Bradley–Terry 模型]]），就得到 DPO 的损失：

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
-\log\sigma\!\left(
\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right]
\right)
$$

直觉上，它优化的是让 chosen 相对于参考模型的提升幅度，大于 rejected 相对于参考模型的提升幅度。所以 DPO 不是简单地“对 chosen 做 SFT”，而是在做**相对偏好优化** 。

**DPO 的优点**

- 不需要在线 rollout；
- 不需要独立 Reward Model 和 Critic 参与训练闭环；
- 可以复用普通 SFT 的离线训练基础设施；
- 通常比在线 RL 更稳定、更便宜。

**DPO 的局限**

- 模型无法主动探索训练集外的新回答；
- 很依赖偏好对的数据质量与覆盖范围；
- 若 chosen / rejected 的长度与格式差异太大，模型可能学到表面风格而非真实质量；
- 数据一旦落后于当前模型，训练信号会逐渐失真。

> [!tip] 怎样缓解 DPO 的分布失配？
> - 优先用当前 SFT 模型或近期 checkpoint 生成候选；
> - 混合外部强模型与人工答案，避免文风单一；
> - 必要时先做少量 chosen-SFT，再做 DPO；
> - 定期刷新偏好对，形成迭代式 DPO。

进一步参见：[[基础强化学习#[Q4] 奖励模型是如何训练的？|偏好建模基础]] | [[#IPO#[Q7] IPO 解决了 DPO 的什么问题？|IPO]] | [[#ORPO#[Q8] ORPO 和 DPO 有什么区别？|ORPO]]

---

### IPO

#### [Q7] IPO 解决了 DPO 的什么问题？

IPO（Identity Policy Optimization）主要针对 DPO 的一个问题：**log-sigmoid 形式的偏好损失在数据较少时容易过拟合，且容易过早饱和。**

在 DPO 中，只要模型把 chosen 和 rejected 分得“足够开”，损失就会快速接近 0。问题在于，这种“分开”不一定真的意味着模型学会了更好的回答，也可能只是：

- 过度放大了某些表面模式；
- 对少量训练样本过度自信；
- 在概率空间上走得过远。

IPO 的做法是把 DPO 那种易饱和的偏好损失，改成更像**平方误差** 的形式，让模型不是一味把两者拉得无限远，而是逼近一个更合理的目标间隔。直觉上，与其追求”chosen 一定压倒 rejected”，不如追求”chosen 比 rejected 好到一个合适的幅度”。因此，IPO 可以看作 DPO 的**防过拟合版本** 。当偏好数据规模较小，或者担心模型在少量偏好对上学得过猛时，IPO 往往更稳。

---

### ORPO

#### [Q8] ORPO 和 DPO 有什么区别？

ORPO（Odds Ratio Policy Optimization）的核心目标是进一步简化流程，==把 SFT 和偏好优化合并成一个阶段==。DPO 通常默认：先有一个 SFT 模型，然后再在偏好对上做第二阶段优化。ORPO 认为，如果训练数据本来就同时包含 chosen / rejected 信息，那么可以把“学习 chosen 的正确表达”和“压低 rejected 的相对概率”放到同一个目标里一起做。

它与 DPO 的最大区别有两点：

1. **不依赖单独的参考模型**；
2. **把 SFT 与偏好优化合并为单阶段训练** 。

这带来的直接好处是工程更轻：

- 少一次独立偏好优化阶段；
- 少一个参考模型参与前向计算；
- 对只有 SFT 微调基础设施的团队更友好。

当然，代价是它的理论出发点与 DPO 不完全相同，也未必总能像 DPO 那样清晰地保留“相对参考策略”的解释。

**一句话区分**

- **DPO**：有参考模型，二阶段，强调 chosen/rejected 的相对概率差；
- **ORPO**：无参考模型，单阶段，把 SFT 与偏好优化合并。

---

### 混合式工业路线

#### [Q9] Llama 3 的官方后训练是怎么把多条路线组合起来的？

Llama 3 值得单独记一下，因为它体现了一个工业界越来越普遍的后训练思路：不押注单一算法，而是让不同方法各负责自己更擅长的那一段。

在深入步骤之前，有一个细节值得先说清楚。Meta 在 2024 年 4 月 18 日的官方博客里，用一句概括性的话描述 Llama 3 的后训练为”SFT + Rejection Sampling + PPO + DPO 的组合”。但同年 7 月 23 日发布的论文《The Llama 3 Herd of Models》里，真正详细展开的主干流程是 Reward Model → Rejection Sampling → SFT → DPO，并额外说明他们探索过 PPO，但在 Llama 3 这套大规模训练里，DPO 更省算力，在 instruction-following 上表现也更好。因此，说”Llama 3 是一个标准 PPO 模型”并不准确，更贴切的理解是：以 RM + RS + SFT + DPO 为主干、同时吸收了 PPO 经验的混合式后训练配方。

**第一步：训练 Reward Model**

Meta 先用人类偏好数据训练奖励模型。这个 RM 不直接部署，而是作为后续流程的评分工具，负责给多个候选回答打分、判断 chosen / rejected，并支撑后续的 rejection sampling 和 DPO。

**第二步：Rejection Sampling**

对每个人工标注的 prompt，Llama 3 从最新策略里采样通常 10 到 30 个候选回答，再由 RM 选出最好的那个。这比直接拿原始模型输出去 SFT 更稳——先用”生成 + 筛选”把数据质量抬高，只保留同一组候选里的最佳样本，再拿这批数据做监督微调。

**第三步：SFT**

有了 rejection-sampled 的高质量回答，再结合人工和合成数据，Meta 对预训练模型做标准的 SFT。这一阶段的目的是把模型整体推到”会当助手”的区域：按指令输出、保持语气和格式、处理代码、推理和工具调用等专项能力。SFT 在这里相当于先把助手的基本行为定住。

**第四步：DPO 偏好对齐**

SFT 之后，Meta 继续用 DPO 做偏好优化。论文里提到，每一轮 post-training 都会做 SFT followed by DPO，DPO 主要使用最近几轮、最贴近当前策略分布的偏好数据。SFT 负责教会模型怎么回答，DPO 负责在多个可行回答里进一步偏向人类更喜欢的那个，所以 Llama 3 不是停在 SFT 就结束了。

**PPO 在 Llama 3 里的地位**

按 2024 年 4 月 18 日的官方博客，PPO 属于 Meta 总结其 post-training 方法时列出的组成部分之一。按 7 月 23 日的论文，Meta 明确说探索过 on-policy 算法如 PPO，但认为 DPO 对大模型更省计算，在 instruction following 上表现也更好，因此论文详细展开的主流程是 RS + SFT + DPO。面试时被问到”Llama 3 是 PPO 还是 DPO”，更准确的回答是：Meta 的官方口径吸收了 PPO 这条路线的经验，但论文真正详细披露的主干流程是 RM + RS + SFT + DPO。

**为什么这套路线有代表性**

它说明工业界越来越少只押单一算法。RM 学会打分，RS 筛出高质量生成，SFT 把助手行为整体拉正，DPO 把偏好进一步压进模型分布，PPO / RL 经验则作为备选路线，用来处理更依赖在线探索的能力。这套混合配方在实践中比”选一个最优算法”更常见，也更灵活。

**官方链接**

- Meta 官方博客（2024-04-18）：[Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- Meta 官方论文页（2024-07-23）：[The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- arXiv 论文链接：[https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

---

### 面试速览

#### [Q10] 如何在面试中快速区分这些算法？

**先问三个问题：**

1. **训练时有没有在线 rollout？**
   - 有：PPO / PPO-PTX / GRPO / DAPO / Dr. GRPO
   - 没有：DPO / IPO / ORPO

2. **有没有独立 Critic？**
   - 有：PPO / PPO-PTX
   - 没有但仍在线：GRPO / DAPO / Dr. GRPO
   - 没有且离线：DPO / IPO / ORPO

3. **训练信号主要来自哪里？**
   - Advantage + 概率比：PPO 系列
   - chosen / rejected 相对偏好：DPO / IPO / ORPO

**一句话总结**

| 算法 | 一句话总结 |
| --- | --- |
| PPO-PTX | PPO 混入预训练损失，防止能力退化 |
| GRPO | 用组内相对奖励代替独立 Critic |
| DAPO | 在 GRPO 上改 clip、采样、token 级梯度与 KL 约束 |
| Dr. GRPO | 对 GRPO 的难度偏差和长度偏差做去偏 |
| DPO | 直接用固定偏好对优化 chosen 相对 rejected 的概率 |
| IPO | DPO 的防过拟合、抗饱和版本 |
| ORPO | 不用参考模型，把 SFT 与偏好优化合并成单阶段 |

进一步参见：[[基础强化学习#[Q26] 这些方法都是在 PPO 基础上改进的吗？PPO 是鼻祖？|算法继承关系]] | [[基础强化学习#[Q22] 什么是 PPO 的 clipping？|PPO clipping]]
