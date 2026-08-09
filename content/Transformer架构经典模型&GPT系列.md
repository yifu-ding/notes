### 4.1.1 T5（2019 年 10 月发布）

**T5（Text-To-Text Transfer Transformer）**提出了一个统一的模型框架，将各种 NLP 任务都视为 Text-to-Text 任务，也就是输入为 Text，输出也为 Text 的任务，如下图所示。这样的好处是可以把所有的问题都套进去一个统一的范式，从而可以采用同样的模型架构、同样的训练策略、同样的损失函数、同样的解码手段。由此可以方便地评估在阅读理解、摘要生成、文本分类等一系列 NLP 任务上，不同的模型结构、预训练目标函数、无标签数据集等的影响。

![[assets/Untitled 3.jpg|600]]

> [!example] T5 的统一 Text-to-Text 范式
> - `translate English to German: That is good.` → `Das ist gut.`
> - `cola sentence: The course is jumping well.` → `not acceptable`
> - `stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.` → `3.8`
> - `summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught of severe weather in mississippi...` → `six people hospitalized after a storm in attala county.`

#### 数据

作者从 Common Crawl（一个公开的网页存档数据集，每个月大概抓取 20TB 文本数据）里清出了 750 GB 的训练数据，然后取名为 **C4（Colossal Clean Crawled Corpus）**，超大型干净爬取数据。跟 **GPT-2、GPT-3** 的训练数据来源一样，是从网上爬的文本数据，由于是爬的数据，所以数据量足够大，而且类型丰富，缺点是数据质量差，需要过滤，过滤手段包括：

> [!note]
> 1. 保留以终点符号（如句号、感叹号、问号等）结尾的行
> 2. 过滤掉内容过少的网页和字数太少的行
> 3. 过滤掉出现“脏字”的网站
> 4. 过滤掉带 Javascript 的行
> 5. 过滤任何包含“lorem ipsum（用于排版测试）”的页面
> 6. 过滤包含编程语言中常用大括号的页面
> 7. 删除一些重复出现的数据
> 8. 过滤非英文的网页，用 langdetect 判断网页究竟是否是英文网页

实验结果表明，**用更专的数据来做预训练，对下游任务的提升越明显，或者说预训练的语料跟任务语料 domain 越接近，效果越好。**但一开始就限定语料的坏处在于，**它不能适应多领域的任务。**所以应该**在丰富的数据上进行预训练，然后再在领域相关、任务相关的语料上继续预训练，最后再 fine-tuning。**

由于 C4 数据集很大，模型预训练完也不能覆盖所有数据，即每个样本模型只见过一次，因此 T5 缩小数据规模，让样本能被重复训练，效果会怎么样。实验表明，**数据量太少，很容易让模型过拟合，进而影响下游任务的性能。**推荐的做法是**数据越多越好，即使预训练不能覆盖完。**

#### 模型

首先作者们先对预训练模型中的多种模型架构（Transformer）进行了比对，最主要的模型架构可以分成下面三种：

![[assets/Untitled.jpg|720]]

- 编码器-解码器
- 语言模型
- Prefix 语言模型

上面这些模型架构都是 Transformer 构成，主要是对其中注意力机制的 Mask 操作进行变化。

为了公平，只有两个模型有同样数量的参数或者同样的计算效率才能进行比较，所以这里提出 5 个 model：

> [!note]
> **模型1**：编码层和解码层各有 L 层，这个模型的参数量是 2P，计算量是 M。
>
> **模型2**：编码层和解码层各有 L 层，但它们参数共享，即模型的参数量是 P，计算量是 M。
>
> **模型3**：编码层和解码层各有 L/2 层，即模型的参数量是 P，计算量是 M/2。
>
> **模型4**：只有 L 层解码层，采用语言模型的形式，模型的参数量是 P，计算量是 M。
>
> **模型5**：只有 L 层解码层，但采用 Prefix 语言模型的形式，参数量是 P，计算量是 M。

通过实验作者们发现，**在提出的这个 Text-to-Text 架构中，Encoder-Decoder 模型效果最好。**而且编码层和解码层共享参数，能减少接近一半的参数量，且模型效果损失不明显，是较优的模型架构方案。于是就把它定为 T5 模型，因此所谓的 **T5 模型其实就是个 Transformer 的 Encoder-Decoder 模型**。

但这里有几点和 Transformer 不同：

1. **使用简化版本的 Layer Normalization**，其中激活只是重新调整，没有附加偏差：常规 LN 为 $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$；T5 中为 $y = w \cdot \frac{x}{\sqrt{\sigma^2 + \epsilon}}$，其中 $\mu$ 为均值，$\sigma^2 + \epsilon$ 为方差，$w$ 参数初始化时全为 1。
2. **在 LN 之后再进行残差连接**。
3. **位置编码使用分桶思想的相对位置编码**（详见位置编码章节），而不是三角式位置编码。

#### 预训练

T5 对预训练目标进行了大范围探索，主要进行了 4 个层次的探索，如右图所示：

![[assets/Untitled 2.jpg|433]]

**第一个层次**，高层次方法（自监督的预训练方法）对比，总共三种方式。

1. **语言模型式**：从左到右预测；
2. **BERT 式**：将文本一部分破坏，然后还原；
3. **顺序还原式**：将文本打乱，然后还原。

> [!example] 示例
> - Prefix language modeling：`Thank you for inviting` → `me to your party last week.`
> - BERT-style：`Thank you <M> <M> me to your party apple week.` → *(original text)*
> - Deshuffling：`party me for you to . last fun you inviting week Thank` → *(original text)*

其中发现 **BERT-style 最好**，进入下一轮。

**第二个层次**，对文本一部分进行破坏时的策略，也分三种方法：

1. **Mask 法**：将被破坏 token 换成特殊符如 `[M]`。
2. **小段替换法**：可以把它当作是把上面 Mask 法中相邻 `[M]` 都合成了一个特殊符，每一小段替换一个特殊符，提高计算效率。
3. **丢弃法**：没有替换操作，直接随机丢弃一些字符。

> [!example] 示例
> - noise, mask tokens：`Thank you <M> <M> me to your party <M> week.` → *(original text)*
> - noise, replace spans：`Thank you <X> me to your party <Y> week.` → `<X> for inviting <Y> last <Z>`
> - noise, drop tokens：`Thank you me to your party week.` → `for inviting last`

实验表明，这三种变体跟 **BERT-style** 效果差不多，但是后面两种方法不用把原始句子预测出来，只用预测被挑选的 **15% token**，训练速度更快，而 **Replace spans** 比 **Drop tokens** 效果稍微好一点，所以 **T5 采用 Replace spans 作为预训练目标**，进入下一轮。

**第三个层次**，对文本百分之多少进行破坏进行探索，挑了 4 个值，`10%`、`15%`、`25%`、`50%`，最后发现 **BERT 的 15% 效果最好**。

**第四个层次**，因为 **Replace Span** 需要决定对大概多长的小段进行破坏，于是对不同长度进行探索，`2`、`3`、`5`、`10` 这四个值，最后发现 **3 结果最好**。

#### 微调

作者使用了两种微调方法：

1. **adapted layers**：adapted layers 接在编码器和解码器的每一个 block 的全连接层后面，在 fine-tuning 的时候只更新它们。adapted layers 有一个内部维度 d 作为超参。
2. **gradual unfreezing**：一开始离任务层近的参数先更新，其它保持不动，随着训练的进行，逐渐放开其它层的参数。

实验表明，**所有参数一起更新效果是最好的，但是缺点就是慢。**adapter layers 效果也还可以，**假如任务数据量小的话，d 取小一些，任务数据量大的话，d 取大一些。**

那么加入更多的任务是否能提高模型的性能？得益于 text-to-text 范式，可以在预训练的时候把有监督的训练也加进来，一起做预训练，而无需采用不同的损失函数。那么现在有**无监督任务、有监督任务1、有监督任务2、有监督任务3**，一共 4 个任务，一起做训练的时候，作者对于怎么采样数据进行了实验：

1. **Examples-proportional mixing**：设任务的数据集大小为 $e_n, n \in 1, \cdots, N$，采样时，采样自第 $m$ 个任务数据的概率为 $r_m = \frac{min(e_m, K)}{\sum min(e_n, K)}$，这里 $K$ 是提前设定的参数。
2. **Temperature-scaled mixing**：在上面 **examples-proportional mixing** 的基础上，再做一些软化，具体的，对上面求得的 $r_m$ 再求 $1/T$ 方根，当 $T=1$，即 **Examples-proportional mixing** 时，T 越大，各个任务数据集采样越均衡。
3. **Equal mixing**：各个任务数据采样概率相同。

实验结果表明**效果都较差**，不过这里是**多个任务一起做训练，相当于把 pre-training 和 fine-tuning 两个合并了，而不会对单个任务进行 fine-tuning，所以效果不好也合理。**

那么能否把这两者结合起来，先用多个任务进行预训练，再对具体任务进行微调：

1. **实验一**：用多个任务进行预训练，再对具体任务进行 fine tuning，其中 fine tuning 的任务也是预训练任务的其中之一。
2. **实验二**：与第一种方法类似，只不过 fine tuning 的任务不在预训练中，作者把这个称为 **leave-one-out** 多任务学习，这种更贴合实际场景。
3. **实验三**：同实验一类似，只不过在预训练时，把无监督目标剔除。

![[assets/Untitled 4.jpg|760]]

> [!summary]
> 1. **实验一：多任务预训练 + 微调** 的效果可以和标准的**无监督预训练 + 微调范式**差不多。
> 2. **实验三：有监督多任务预训练**的效果很差，说明**无监督预训练确实很有用**。

由于**多任务预训练 + 微调的效果和标准的无监督预训练 + 微调范式的效果差不多，但是前者在预训练过程还能够监控下游任务的性能**，因此 T5 采用**多任务预训练 + 微调**的方法。

最后，T5 把之前的实验都结合起来，训练了 5 个规模的模型：

| model_name | layers | D_model | D_feed-forward | Muti-head | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T5-small | 6 | 512 | 2028 | 8 | 60M |
| T5-base | 12 | 768 | 3072 | 12 | 220M |
| T5-large | 24 | 1024 | 4,096 | 16 | 770M |
| T5-3B | 24 | 1024 | 16,384 | 32 | 3B |
| T5-11B | 24 | 2028 | 65,536 | 128 | 11B |

### 4.1.2 BART（2019 年 10 月发布）

**BART（Bidirectional and Auto-Regressive Transformers）**是一种非常适用于生成式任务的模型，当然它也能完成判别式任务，而且效果也很好。它主要**结合了 BERT 和 GPT 两种模型思路，使得它不仅具有双向编码的优势，也具有单向自回归编码的优势。**即 **BART 不仅在生成任务上（NLG）特别有效，同时在自然语言理解任务上（NLU）表现的也很出色。**

**BART** 沿用了标准的 **Transformer** 结构，也就是 **Encoder-Decoder** 的 Transformer。**BART 模型的实现本质上仍然是 Sequence-to-Sequence 范式，实质就是 Encoder-Decoder 架构，Encoder 属于双向编码，对被破坏的文本进行编码；Decoder 属于单向解码，也就是自回归解码，解码出想要的目标文本。**优化目标是负的对数似然函数。

**BART** 的预训练从两个方面来做：

> [!note]
> 1. 通过随机噪声函数（说白了就是能够制造破坏文档结构的任何方法）来破坏文章结构。
> 2. 通过模型能够学会将结构已经被破坏了的文章进行重构，使文章变回原来的样子。

![[assets/Untitled 6.jpg|349]]

对于破坏文章结构，**BART** 通过评估不同方法后发现采用下列方法效果最好，**一个是随机打乱原文句子的顺序，一个是随机将文中的连续小片段（连续几个字或词）用一个 `[MASK]` 代替**，BERT 是一个词用一个 `[MASK]` 替换，**BART 是连续多个词用一个 `[MASK]` 替换。**这里片段的长度是随机选取的，用泊松分布的方法来采样不同长度的片段（包括长度为 0）。作者还仿照其它的预训练模型的方式，对 **BART** 进行了消融实验，把 **BERT、XLNet** 等预训练方法移植到 **BART** 中，以此更好的验证哪一些模型的哪一些因素对最终任务的性能影响最大。

**BART** 的模型结构如下所示：

![[assets/Untitled 5.jpg|760]]

在 **BART** 中，它的输入输出并不需要严格的保持长度一致，即 Encoder 的输入与 Decoder 的输出不需要对齐。从上述结构图中可以看出，**左边部分 Encoder 接收结构破坏后的文章，它经过 Encoder 双向编码，类似于 BERT，右边部分接收来自 Encoder 的输出后，经过自回归解码输出预测结果，类似于 GPT。**自回归解码依据的是最大似然概率，这一点和 **N-Gram** 词袋模型或者 **NNLM** 模型是相似的。对于微调，**BART 的 Encoder 和 Decoder 接收同样的输入，且此时的输入文章是没有被破坏了的**，这样一来，Decoder 最后一维的输出被拿来作为文章的向量表示，类似

![[assets/Untitled 7.jpg|400]]

![[assets/Untitled 8.jpg|400]]

#### BART 预训练与微调补充

BART 的预训练目标就是对于被破坏的文章优化一个重构损失函数，实际上就是针对 Decoder 的预测结果和目标文章 label 的交叉熵损失，它允许接受任意破坏形式的文章。BART 有几种具体的打破文章结构的方式：

> [!note]
> 1. **单 Token 级别的掩码（Token Masking）**：和 BERT 一样，按一定比例随机采样 Token 进行 `[MASK]`。
> 2. **Token 级别的丢失（Token Deletion）**：与 MASK 不同，MASK 模型知道原 Token 在哪，而 deletion 模型不仅需要知道原 Token 是什么，还需要知道原 Token 都在哪些位置，这提高了模型的预测能力。
> 3. **片段级别的 Token MASK（Text Infilling）**：通过均值 $\lambda=3$ 的泊松分布对 span text 进行采样，比如长度为 0、长度为 3、长度为 5 的片段，分别只用一个 MASK 替换。text-infilling 为了让模型学习到一个 MASK 中原本有多少个 Token，以及 Token 是什么。
> 4. **句序打乱（Sentence Permutation）**：将完整句子顺序打乱，让模型学习原来应该在什么位置。
> 5. **文章旋转（Document Rotation）**：随机在文章中取一个词，并以该词为基点将文章旋转，然后再让模型学习原文的起始 Token 是什么。

**分类任务**：Encoder 和 Decoder 接受相同的输入，然后取 Decoder 输出的最后一个 Token 的 hidden state 接上一个分类层进行微调。BERT 在输入序列的第一个位置添加 `[CLS]` Token；BART 在最后一个位置添加 `[CLS]`，所以最后取得也是最后一个 Token。

**机器翻译**：将 Encoder 和 Decoder 看成一个整体，作为新的 Decoder；将原来 Encoder 的词嵌入层替换成新的 Encoder。微调时先只更新随机初始化的新 Encoder、BART 的位置嵌入向量和原编码器第一层的自注意输入投影矩阵，其余参数固定住，然后少量迭代全部参数。

**Token 级别分类任务**有实体识别、阅读理解、问答等，均为抽取式；使用 Decoder 输出的每一个 Token 的 hidden state，对每个 Token 进行分类。**文本生成任务**如抽象问答和摘要任务：给模型一个原始文本和目标文本，编码器接收原始文本，解码器解码出预测文本，并和目标文本求损失。

> [!summary]
> - 预训练效果和预训练任务、任务数据相关。
> - Document Rotation 或 Sentence Permutation 单独使用时表现不佳；Token Deletion 或 Token MASK 效果较好，在生成任务上删除大体优于掩码。
> - 自回归机制对于生成任务能有效提升性能；双向编码机制在 SQuAD 任务上非常关键。
> - 除 ELI5 外，有 Text-filling 方式参与时 BART 表现都很好，反映了 Text-filling 的有效性。

## 4.2 BERT 家族

### 4.2.1 BERT（2018 年 10 月发布）

**BERT（Bidirectional Encoder Representations from Transformers）**是一个语言表示模型。它的主要模型结构是 Trasnformer 的 Encoder 堆叠而成，它其实是一个 2 阶段的框架，分别是 pretraining，以及在各个具体任务上进行 finetuning。BERT 模型可以作为公认的里程碑式的模型，但是它最大的优点不是创新，而是集大成者，并且这个集大成者有了各项突破，从大量无标记数据集中训练得到的深度模型，可以显著提高各项自然语言处理任务的准确率。

**BERT** 参考了 **ELMo** 模型的双向编码思想、借鉴了 **GPT** 用 Transformer 作为特征提取器的思路、采用了 **Word2Vec** 所使用的 **CBOW** 方法。具体的，**GPT 使用 Transformer Decoder 作为特征提取器、具有良好的文本生成能力，然而当前词的语义只能由其前序词决定，并且在语义理解上不足，而 BERT 使用了 Transformer Encoder 作为特征提取器，并使用了掩码训练方法。虽然使用双向编码让 BERT 不再具有文本生成能力，但是 BERT 的语义信息提取能力更强，**这 3 种模型结构如下所示：

![[assets/Untitled 17.jpg|760]]

> [!note]
> - **ELMo** 使用自左向右编码和自右向左编码的两个 LSTM 网络，分别以 $P(w_i|w_1,\cdots,w_{i-1})$ 和 $P(w_i|w_{i+1},\cdots,w_n)$ 为目标函数独立训练，将训练得到的特征向量以拼接的形式实现双向编码，本质上还是单向编码，只不过是两个方向上的单向编码的拼接而成的双向编码。
> - **GPT** 使用 Transformer Decoder 作为 Transformer Block，以 $P(w_i|w_1,\cdots,w_{i-1})$ 为目标函数进行训练，用 Transformer Block 取代 LSTM 作为特征提取器，实现了单向编码，是一个标准的预训练语言模型，使用 Fine-Tuning 模式解决下游任务。
> - **BERT** 也是一个标准的预训练语言模型，它以 $P(w_i|w_1,\cdots,w_{i-1},w_{i+1},\cdots,w_n)$ 为目标函数进行训练，BERT 使用的编码器属于双向编码器。BERT 和 ELMo 的区别在于使用 Transformer Block 作为特征提取器，加强了语义特征提取的能力。BERT 和 GPT 的区别在于使用 Transformer Encoder 作为 Transformer Block，并且将 GPT 的单向编码改成双向编码，BERT 舍弃了文本生成能力，换来了更强的语义理解能力。

BERT 模型就是 Transformer Encoder 的堆叠：

![[assets/Untitled 16.jpg|600]]

> [!note]
> - $\mathrm{BERT}_{\mathrm{BASE}}: L=12,H=768,A=12$
> - $\mathrm{BERT}_{\mathrm{LARGE}}: L=24,H=1024,A=16$

其中 $L$ 是 Transformer Block 层数，$H$ 是特征向量维数，$A$ 是 Self-Attention 头数。BERT$_{BASE}$ 参数量 **110M**，BERT$_{LARGE}$ 参数量 **340M**。

#### BERT 的输入表示

BERT 的输入表示如图下图所示。比如输入的是两个句子 `my dog is cute`，`he likes playing`。这里采用类似 GPT 的两个句子的表示方法，首先会在第一个句子的开头增加一个特殊的 Token `[CLS]`，在 `cute` 的后面增加一个 `[SEP]` 表示第一个句子结束，在 `##ing` 后面也会增加一个 `[SEP]`。这里的分词会把 `playing` 分成 `play` 和 `##ing` 两个 Token，这是把词分成更细粒度的 WordPiece 方法一种解决未登录词的常见办法。

接着对每个 Token 进行 3 个 Embedding：**词的 Embedding、位置的 Embedding 和 Segment 的 Embedding**。词的 Embedding 和位置的 Embedding 之前都进行了详细的介绍。Segment 只有两个，要么是属于第一个句子 Segment 要么属于第二个句子，不管那个句子，它都对应一个 Embedding 向量。**同一个句子的 Segment Embedding 是共享的，这样它能够学习到属于不同 Segment 的信息。**对于情感分类这样的任务，只有一个句子，因此 Segment id 总是 0；而对于 Entailment 任务，输入是两个句子，因此 Segment 是 0 或者 1。

![[assets/Untitled 15.jpg|760]]

#### BERT 的预训练

![[assets/Untitled 10.jpg|420]]

BERT 采用二段式训练方法：**第一阶段：使用易获取的大规模无标签语料，来训练基础语言模型；第二阶段：根据指定任务的少量带标签训练数据进行微调训练。**不同于 GPT 等标准语言模型使用 $P(w_i|w_1,\cdots,w_{i-1})$ 为目标函数进行训练，能看到全局信息的 BERT 使用 $P(w_i|w_1,\cdots,w_{i-1},w_{i+1},\cdots,w_n)$ 为目标函数进行训练。并且 **BERT 用语言掩码模型 MLM 方法训练词的语义理解能力；用下句预测 NSP 方法训练句子之间的理解能力，从而更好地支持下游任务。**BERT 在预训练阶段使用了前文所述的两种训练方法，在真实训练中一般是两种方法混合使用。

##### 语言掩码模型 MLM

BERT 认为使用自左向右编码和自右向左编码的单向编码器拼接而成的双向编码器，在性能、参数规模和效率等方面，都不如直接使用深度双向编码器强，这也是为什么 BERT 使用 Transformer Encoder 作为特征提取器，而不使用自左向右编码和自右向左编码的两个 Transformer Decoder 作为特征提取器。由于无法使用标准语言模型的训练模式，**BERT 借鉴完形填空任务和 CBOW 的思想，使用语言掩码模型 MLM 方法训练模型。**

MLM 方法也就是随机去掉句子中的部分 Token，然后模型来预测被去掉的 Token 是什么。这其实是一个分类问题，根据这个时刻的 hidden state 来预测这个时刻的 Token 应该是什么。随机去掉的 Token 被称作掩码词，在训练中，掩码词将以 15% 的概率被替换成 `[MASK]`，也就是说随机 mask 语料中 15% 的 Token，这个操作则称为掩码操作（在 CBOW 模型中，每个词都会被预测一遍）。

但是这样设计 MLM 的训练方法会引入弊端：**在模型微调训练阶段或模型推理阶段，输入的文本中将没有 `[MASK]`，进而导致产生由训练和预测数据偏差导致的性能损失。**基于此，**BERT 并没有总用 `[MASK]` 替换掩码词，而是按照一定比例选取替换词。**在选择 15% 的词作为掩码词后这些掩码词有三类替换选项：

> [!note]
> - **80%** 练样本中将选中的词用 `[MASK]` 来代替。
> - **10%** 的训练样本中选中的词不发生变化，为了缓解训练文本和预测文本的偏差带来的性能损失。
> - **10%** 的训练样本中将选中的词用任意的词来进行代替，为了让 BERT 学会根据上下文信息自动纠错。

这样做编码器不知道哪些词需要预测的，哪些词是错误的，因此被迫需要学习每一个 Token 的表示向量，另外双向编码器比单项编码器训练要慢，进而导致 BERT 的训练效率低了很多，但是实验也证明 **MLM 训练方法可以让 BERT 获得超出同期所有预训练语言模型的语义理解能力，牺牲训练效率是值得的。**

##### 下句预测 NSP

在很多自然语言处理的下游任务中，如问答和自然语言推断，都基于两个句子做逻辑推理，而**语言模型并不具备直接捕获句子之间的语义联系的能力，或者可以说成单词预测粒度的训练到不了句子关系这个层级，为了学会捕捉句子之间的语义联系，BERT 采用了下句预测 NSP 作为无监督预训练的一部分。**

NSP 的具体做法是，BERT 输入的语句将由两个句子构成，其中，50% 的概率将语义连贯的两个连续句子作为训练文本，另外 50% 的概率将完全随机抽取两个句子作为训练文本。

例：相关 `[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]`；不相关 `[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]`。

其中 `[SEP]` 标签表示分隔符。`[CLS]` 表示标签用于类别预测，结果为 1，表示输入为连续句对；结果为 0，表示输入为随机句对。通过训练 `[CLS]` 编码后的输出标签，BERT 可以学会捕捉两个输入句对的文本语义。

#### BERT 下游任务微调

BERT 支持**句对分类、单句分类、文本问答和单句标注**四类任务。

![[assets/Untitled 11.jpg|420]]

句对分类中，两个句子用 `[SEP]` 拼接，在句首加入 `[CLS]`，将其输出作为分类标签；多分类任务在 `[CLS]` 输出后接全连接层与 Softmax 层。

![[assets/Untitled 12.jpg|420]]

单句分类在句首加入 `[CLS]`，同样以其输出作为分类标签。

## 4.3 GPT 系列

OpenAI 是一家致力于推动人工智能前沿研究的公司，尤其在自然语言处理领域取得了重大突破。其开发的 **GPT（Generative Pre-trained Transformer）**系列模型，包括 **GPT-1、GPT-2、GPT-3、ChatGPT 和 GPT-4**，逐步提升了语言模型的能力和应用范围。GPT-1 和 GPT-2 奠定了预训练模型的基础，而 GPT-3 拥有 1750 亿个参数，能够执行少量示例学习任务。ChatGPT 在 GPT-3 的基础上优化了对话生成；GPT-4 具备更强推理能力、更高精确度，并支持多模态输入。

GPT 和 BERT 本质都基于 Transformer，均先在无标签数据上学习预训练语言模型，再针对特定任务微调。**GPT 使用 Transformer 中去掉中间 Encoder-Decoder Attention 层的 Decoder，将 Multi-Head Attention 换成 Masked Multi-Head Attention，即 Masked Self Attention，是单向语言模型，给定前几个词预测下一个词，更适合自然语言生成任务；BERT 使用 Transformer 的 Encoder，即 Self Attention，是双向语言模型，更适合自然语言理解任务。**

### 4.3.1 GPT-1（2018 年 6 月发布）

2018 年 6 月，OpenAI 发布 GPT-1。它基于 Transformer 的 Decoder 结构进行单向语言模型训练。**GPT-1 的核心思想是先通过无标签文本训练生成语言模型，再根据具体 NLP 任务（如文本蕴涵、QA、文本分类等）用有标签数据微调。**

![[assets/Untitled 18.jpg|760]]

#### 预训练

第一阶段目标是预训练语言模型，使用 **BooksCorpus** 数据集，包含超过 7000 本来自各种类型的未出版书籍。它包含一长段连续文本，使得生成模型能够学习长距离信息。另一个数据集是 1B Word Benchmark，与 ELMo 使用方法大致相同，但是在句子层面被打乱，破坏了长距离结构。

给定无监督 token 语料 $\mathcal U=\{u_1,\ldots,u_n\}$，最大化：

$$L_1(\mathcal U)=\sum_i\log P(u_i\mid u_{i-k},\ldots,u_{i-1};\Theta)$$

其中 $k$ 是上下文窗口大小，条件概率 $P$ 由参数 $\Theta$ 的神经网络建模。GPT-1 使用多层 Transformer 解码器，即 Multi-Head Self-Attention，之后增加前馈网络层。

$$h_0=UW_e+W_p$$
$$h_l=\mathrm{Transformer\_block}(h_{l-1}),\ \forall l\in[1,n]$$
$$P(u)=\mathrm{Softmax}(h_nW_e^T)$$

其中 $U=(u_{-k},\ldots,u_{-1})$ 是 token 上下文向量，$n$ 是层数，$W_e$ 是 token 嵌入矩阵，$W_p$ 是位置嵌入矩阵。训练使用 12 层 Transformer、768 维词编码、12 个注意力头，Adam 优化器，最大学习率 $2.5\mathrm e{-4}$，批量大小 64，epoch 为 100；使用 BPE、dropout 0.1、GELU 和可学习位置编码。

#### 微调

对于有标签训练集 $\mathcal C$，给定输入序列 $x^1,\ldots,x^m$ 和标签 $y$：

$$P(y\mid x^1,\ldots,x^m)=\mathrm{softmax}(h_l^mW_y)$$
$$L_2(\mathcal C)=\sum_{(x,y)}\log P(y\mid x^1,\ldots,x^m)$$

GPT-1 发现将语言模型作为辅助对象参与微调可提升监督模型的泛化性能并加速收敛，使用：

$$L_3(\mathcal C)=L_2(\mathcal C)+\lambda L_1(\mathcal C)$$

> [!note]
> 1. **分类**：文本最后一个词的向量作为微调输入，得到分类结果。
> 2. **推理**：输入为先验、分隔符和假设，最后一个词向量用于二分类。
> 3. **相似性**：两个句子相互颠倒，最后一个词的向量相加后进行分类。
> 4. **问答**：上下文、问题与多个回答由分隔符分隔，对每个回答最后一个词的向量分类，Softmax 取概率最大者。

微调时分类器加入 dropout（比例 0.1）；多数任务学习率为 $6.25\mathrm e{-5}$、批量大小 32，训练 3 个 epoch，训练前 0.2% warmup，$\lambda=0.5$。

GPT-1 在 12 个有监督基准任务中的 9 个超过当时最佳模型；Zero-shot 任务也表现出稳定性。但其性能通常低于微调后的有监督任务，仍是强大的领域专家，而非通用语言学家。

> [!summary] GPT-1 总结
> **优点**：预训练与微调结合；基于 Transformer 架构，擅长捕捉长距离依赖；具备迁移学习能力。
>
> **缺点**：单向语言模型，双向上下文理解受限；参数规模相对较小（1.17 亿）；对特定下游任务依赖微调。

### 4.3.2 GPT-2（2019 年 2 月发布）

**GPT-2 的核心思想**：当模型容量非常大且数据量足够丰富时，仅靠语言模型学习即可完成其它有监督学习任务，无需在下游任务微调。GPT-2 沿用 GPT-1 的单向 Transformer 模式，但使用更多参数和更大数据集。


> [!note]
> 1. 使用 **Zero-shot**，而 GPT-1 为预训练加微调。
> 2. 字典从 40,000 增加到 50,257。
> 3. Transformer 堆叠至 48 层，隐层维度 1600，参数量达 15 亿（GPT-1 约 1 亿）。
> 4. 数据集为 Reddit 高赞文章构成的 **WebText**，约 800 万篇、40G；移除涉及 Wikipedia 的文章以避免测试集冲突。GPT-1 数据仅 5GB。
> 5. batch size 从 64 增加到 512，上下文窗口从 512 增加到 1024。

GPT-2 去掉 fine-tuning 层，不再为不同任务分别建模，而让模型根据输入内容判断任务；输入也会加入提示词 prompt。它将 Layer Normalization 放在每个 sub-block 之前，把 post-norm 改为 pre-norm，最后一个 Self Attention 后再加 Layer Normalization；残差层初始化值按 $1/\sqrt N$ 缩放。

> [!note]
> GPT-2 认为，当一个语言模型的容量足够大时，它就足以覆盖所有的有监督任务，也就是说所有的有监督学习都是无监督语言模型的一个子集。例如当模型训练完 `Micheal Jordan is the best basketball player in the history` 语料的语言模型之后，便也学会了 `question: "who is the best basketball player in the history ?", answer: "Micheal Jordan"` 的 Q&A 任务。GPT-2 的核心思想概括为：**任何有监督任务都是语言模型的一个子集，当模型的容量非常大且数据量足够丰富时，仅仅靠训练语言模型的学习便可以完成其它有监督学习的任务。**

> [!summary]
> GPT-2 认为所有有监督学习都是无监督语言模型的一个子集。例如，学习完 `Micheal Jordan is the best basketball player in the history` 后，也可完成 `question: "who is the best basketball player in the history?", answer: "Micheal Jordan"` 的 Q&A 任务。

### 4.3.3 GPT-3（2020 年 5 月发布）

Zero-shot 被 GPT-2 证明可行后，GPT-3 沿用通用模型思路，将模型扩大到 **175B 参数**，同时升级为 **Sparse Transformer**。GPT-3 可通过少量样本学习（**Few-Shot Learner**）；因模型庞大，fine-tune 成本很高，因此采用**上下文学习（In-context Learning）**：不进行梯度更新或 fine-tune，直接在上下文中学习。

GPT-3 的出色性能在很大程度上归功于其采用的 **In-context Learning** 方法。为了理解 In-context Learning，先来探讨一下元学习 Meta-learning 的概念。**元学习的核心思想是通过学习如何学习，来找到一种有效的学习策略或初始化参数，使得模型能够在新的、未见过的任务上快速适应并取得良好的性能。**

**In-context Learning** 是元学习思想的一种具体实现，它允许模型在给定一些示例的情况下，直接通过这些示例来学习并完成任务，而无需显式地更新模型参数。在 GPT-3 中，这种学习方式被应用于各种 NLP 任务中。具体来说，**当给定一个新的任务时，可以向 GPT-3 提供少量的示例输入和对应的输出，即上下文，然后让 GPT-3 根据这些示例来推断并生成针对新输入的输出。**通过这种方式，GPT-3 能够在不依赖大量有标签训练数据的情况下，快速适应并完成各种 NLP 任务。

GPT-3 的 In-context Learning 能力得益于其巨大的参数量和训练数据集。**通过在大规模无监督文本数据上进行预训练，GPT-3 已经学习到了丰富的语言知识和模式。**这使得它能够在给定少量示例的情况下，快速理解并应用这些知识来完成新任务。同时，**GPT-3 的巨大参数量也使其具备了强大的表征能力，能够捕捉并表达复杂的语言现象和语义关系。**

![[assets/Untitled 20.jpg|760]]

#### 上下文学习

GPT-3 的出色性能很大程度归功于 In-context Learning。元学习的核心是“学习如何学习”，寻找有效学习策略或初始化参数，使模型在新的、未见任务上快速适应。In-context Learning 是其具体实现：给定少量示例输入与输出作为上下文，GPT-3 依据示例推断并生成对新输入的输出。

#### 预训练

GPT-3 的预训练类似 GPT-2，但扩大模型、数据集、训练长度与多样性：

> [!note]
> 1. 96 层多头自注意力，96 个头。
> 2. 词向量长度 12888。
> 3. 上下文窗口提升至 2048。

训练数据包括过滤后的 C4、WebText2、Books1/Books2 和英文 Wikipedia。训练语料的 60% 来自 C4、22% 来自 WebText2、16% 来自 Books、3% 来自 Wikipedia。数据按质量赋予不同权重，权重越高越易抽样。

实验表明：模型越大、测试案例越多，最终效果越好；案例很多时 Prompt 不那么重要。大模型上下文学习曲线表明，从上下文学习任务的能力提升，模型越大，上下文信息使用效率越高。Zero-shot、One-shot、Few-shot 的性能差距通常随模型容量增加而增大，说明更大模型更擅长元学习。

> [!summary] GPT-3
> GPT-3 是 OpenAI 于 2020 年发布的自回归语言模型，拥有 1750 亿参数。优点是强大的语言生成与少样本学习能力；缺点是训练/部署计算资源需求高，且可能生成包含偏见或不适当的内容。

### 4.3.4 GPT-3.5（2022 年 11 月发布）

GPT-3 零样本理解强，但对话表现不佳：缺少正向价值观、回答不够圆滑、回答风格与人类预期有偏差，以及编造事实。其根本原因是 GPT 对人类偏好不够了解，因此需要继续训练；完全依赖人工标注不现实，于是借助强化学习让模型自我学习。

#### 强化学习与 RLHF

强化学习通常将机器看作策略函数 $\pi$：根据当前状态 $s_t$ 预测动作 $a$ 的概率，采样动作 $a_t$ 执行，环境转入 $s_{t+1}$。$r_t$ 是当前步骤价值，$u_t=r_t+\lambda r_{t+1}+\lambda^2r_{t+2}+\cdots$ 是全局价值。

> [!note]
> 1. 收集训练数据 $(s_t,a_t,u_t)$，其中 $a_t$ 由 $\pi$ 采样，$u_t$ 为系统反馈。
> 2. 基于目标函数用梯度更新 $\pi$。
> 3. 重复至结束条件。

RLHF 分三步：**SFT（监督微调）**，由人工标注问答数据训练 GPT；**RM（奖励模型）**，模型生成多个答案，人工排序并训练奖励模型；**PPO**，以 RM 为回报函数训练策略模型。

> [!summary] GPT-3.5
> 优点：真实性与价值观对齐更好、响应更自然、无害性提升。缺点：通用 NLP 性能可能下降、仍可能生成荒谬内容、对指令敏感且可能误解简单概念。

### 4.3.5 GPT-4（2023 年 3 月发布）


GPT-4 是 OpenAI 发布的最新 GPT 系列模型。它是一个大规模多模态模型，相比 GPT-3.5，**GPT-4 可以接受图像和文本两种形式的输入，产生文本输出。**输出依旧是一个自回归的单词预测任务。技术上，**GPT-4 采用了专家混合技术，进一步增强模型的能力。**整体来说，**GPT-4** 在各种专业和学术基准上表现出了人类的水平，对于生成式的幻觉、安全问题均有较大的改善。

#### 模型结构

GPT-4 的体系结构由 16 个不同的专家模型组成，每个模型都有 111B 个参数，总计约 1.76 万亿个参数。除了更大的参数规模外，另一个重要的细节是 GPT-4 使用了专家混合 **MoE（Mixture of Experts）**架构，这意味着模型中的不同组件协同工作，每个组件都有助于最终输出。在 GPT-4 中 Attention 的参数量有 55B，MoE 的参数量是 $111B*16$，一共 120 层 Transformer。每个 token 会通过一个路由算法选择两个 MLP 进行计算，参数 seq_len 为 8k，每个 MLP 分到 1k 个 token。

![[assets/Untitled 21.jpg|600]]

**GPT-4 的模型宽度、深度基本和 GPT-3（175B）差不多，区别在于 MLP 的数量要多 16 倍。**

#### 并行策略

GPT-4 训练采用的并行策略是：

> [!note]
> - **张量并行**：8
> - **流水并行**：16
> - **数据并行**：196

总计使用约 3125 台机器（25000 张 A100）进行训练。其中 batch size 为 60M token，seq_len 为 8k。

张量并行和流水并行包含了 GPT-4 完整的模型参数，其结构如右图所示。其中，张量并行通讯耗时占比小于 15%，PipeDream 流水线气泡占比 28% 左右，Interleaved 1F1B 流水线气泡占比 16% 左右。

> [!note] GPT-4 能力
> 1. 图像理解能力与多模态输入。
> 2. 更长上下文窗口（图中为 8k）。
> 3. 复杂任务处理能力提升。
> 4. 幻觉与安全问题有所改善。

### GPT-4o（2024 年 5 月发布）

GPT-4o 中的 `o` 指 **omni**：将文本、图像、音频等模态统一放入一个模型能力框架。此前 Voice Mode 类似“音频转文本 → GPT 处理文本 → 文本转语音”的三段流水线，语气、停顿、背景声等信息会在第一步被压扁；GPT-4o 改为端到端处理多模态信号。

图中记录：GPT-4o 于 **2024-05-13** 进入旗舰模型序列，后续于 **2024-08-08** 前后补充训练来源、能力边界与部署约束说明，归档时间为 **2024-10-25**。其音频响应最快 232 ms、平均约 320 ms；旧 Voice Mode 在 GPT-3.5/GPT-4 上平均约 2.8 s/5.4 s。

#### o1（2024 年 9 月发布）：把推理训练成模型能力

##### 1. 大规模 RL 训练的是中间过程

o1 的重点在于训练目标变了。它不是只学习“给出最终答案”，还要在内部推理里学会尝试不同策略、发现前面步骤的问题、把路线拉回来。**这个训练方向的收益很直接：复杂题不再完全依赖一次性生成，模型可以在内部多走几步再交答案。**

o1-preview 刚出现时，数学和代码题提升特别明显。AIME、Codeforces、GPQA 这类任务不靠套模板，靠的是多步约束能不能保持住。o1 在 Codeforces 上到第 **89** 百分位，AIME 2024 pass@1 接近 **74%**，这些数字背后的含义很直接：模型开始把“多想几步”变成可测的能力。

##### 2. 隐藏 CoT 不是黑箱借口

很多人第一次用 o1 会觉得奇怪：模型明明在推理，却不把完整推理过程吐出来。这里要分清两件事。用户需要的是可检查的结论、必要的推导和能落地的解释；**把 raw CoT 直接暴露出来，会把内部策略变成可诱导、可复制、可攻击的接口。**

“隐藏 CoT”更像产品边界，不是神秘设定。模型可以在内部写草稿，最终答案只暴露必要信息。这样既保留了推理收益，也减少了把内部策略直接交给攻击者的风险。

> [!example]
> 让模型解一道竞赛题时，最终答案可以给关键方程、边界条件和验证步骤；不需要把每一个失败尝试、犹豫分支和内部打分都摊开。更有价值的是“可验证解释”，不是 raw CoT。

##### 3. deliberative alignment 的直觉

deliberative alignment 可以翻译成“让模型先读规则再做判断”。**传统安全策略容易变成表层拒答：看见敏感词就拒，或者被换一种说法绕过。**推理模型多了一层空间，可以先把用户意图、上下文、政策边界放在内部推理里过一遍，再决定如何回答。

这对安全很有价值，也有新的压力。**推理越强，模型越会计划，越可能在有害任务上表现出更高执行力。**o1 之后的安全评估不能只看“会不会拒答”，还要看越狱、幻觉、危险能力、生物化学、网络安全、自我改进等维度有没有被推高。

理解 o1 可以抓三个锚点：推理阶段为什么值得花更多算力，raw CoT 为什么不能当作普通解释展示，能力提升为什么会把安全评估从内容过滤推到任务级风险。
