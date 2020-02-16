---
layout: article
title: 图神经网络：用于多任务学习的图自编码器（MTGAE）
date: 2020-01-06 00:21:00 +0800
tags: [Graph, Link Prediction, Node Classification, GNN]
categories: blog
pageview: true
key: Multi-Task-Graph-Autoencoders
---



------

论文链接： [https://arxiv.org/pdf/1811.02798.pdf](https://arxiv.org/pdf/1811.02798.pdf)

github链接：[https://github.com/vuptran/graph-representation-learning](https://github.com/vuptran/graph-representation-learning)



## Challenges

首先阐述了图上预测任务的难点：

- 极端的类标不平衡问题：链路预测中，已知存在的边（正链）明显少于已知缺失的边（负链），难以从稀疏的样本中学习到有效的信息；
- 图结构的复杂性：边的有向/无向，加权/无权，高度稀疏，边类型多；
- 合并边信息:节点(或者边)有时由一组称为边信息的特征来描述，边信息可以对输入图的拓扑特征进行编码。节点和边缘上的这种显式数据并不总是可用的，并且被认为是可选的。一个有用的模型应该能够结合关于节点和/或边的可选边信息，以尽可能提高预测性能;
- 效率和可扩展性:真实世界的图形数据集包含大量的节点和/或边。为了在实际应用中实现实用，模型必须具有内存和计算效率。



## Autoencoder Architecture for Link Prediction and Node Classification

论文介绍了一个基于多任务的图自动编码器框架（MTGAE），能够从图的局部拓扑结构和显式的节点特征中学习到节点的多任务共享嵌入表示。

创新点：

- 模型简单、有效、通用性强；
- 能够在单个阶段对无监督链路预测和半监督节点分类的多任务学习训练一个联合同步的端到端模型；
- 在五个具有挑战性的benchmark图数据集上进行实验评估，对比三个专门为LPNC设计的强baseline，实现了性能的提升；



![76a58ce0f85f054aafdea8b569c285c.png](http://ww1.sinaimg.cn/large/005NduT8ly1gan6iv5hlij30q70c678c.jpg)

以上是多任务图自动编码器(MTGAE)体系结构的示意图：

- 左：两个节点之间有正链(实线)和负链(虚线)的部分可观测图；尚未连接的节点对具有未知的链接状态；
- 中：一个对称、连接稀疏的自动编码器，参数共享，端到端训练，从邻接矩阵学习节点的嵌入表示；
- 右：用于链接预测和节点分类的多任务范例输出；
  - 最后一层为无监督链路预测重构领域；
  - 倒数第二层对节点标签进行解码，用于半监督分类；



### Problem Formulation and Notation

MTGAE模型的输入：图$$
\mathcal{G}=(\mathcal{V}, \mathcal{E})
$$，包含节点数$$N=|\mathcal{V}|$$。邻接矩阵$$\mathbf{A} \in \mathbb{R}^{N \times N}$$。

对于一个部分可观测图，$$\mathbf{A} \in \{1,0, \mathsf{UNK} \}^{N \times N}$$，1表示正链，0表示负链，$$\mathsf{UNK}$$表示未知状态的边（缺失边或未观察到的边）。$$a_i\in \mathbb{R}^N$$表示邻接向量

有向/无向，有权/无权，二分图 均可作为输入。

节点特征矩阵$$
\mathbf{X} \in \mathbb{R}^{N \times F}
$$。

MTGAE模型$$
h(\mathbf{A}, \mathbf{X})
$$的目标是学习节点的低维（$D$）潜在表示$$\mathbf{Z} \in \mathbb{R}^{N \times D}$$，产生一个重构的输出$$
\hat{\mathbf{A}}
$$，使得$$
\mathbf{A}
$$与$$
\hat{\mathbf{A}}
$$之间的误差最小化，因此能够保留图的全局结构。



### Unsupervised Link Prediction

MTGAE框架包含了一系列针对$$\mathbf{a}_{i}$$的非线性转换，由两部分组成：

- 编码器：$$g\left(\mathbf{a}_{i}\right): \mathbb{R}^{N} \rightarrow \mathbb{R}^{D}$$；

- 解码器：$$f\left(g\left(\mathbf{a}_{i}\right)\right): \mathbb{R}^{D} \rightarrow \mathbb{R}^{N}$$；

我们堆叠编码器的两层，生成节点$i$的$D$维潜在特征表示$$\mathbf{z}_i \in \mathbb{R}^{ D}$$，堆叠两层解码器，获取一个精确的重构输出$$
\hat{\mathbf{a}}_{i} \in \mathbb{R}^{N}
$$，形成了四层的自编码器架构。

**注意：$$\mathbf{a}_{i}$$很稀疏，因为实验中，输入图的高达90%的边随机删除；重构的$$\hat{\mathbf{a}}_{i}$$是稠密的，包含了预测出来的缺失边。**

编码解码的隐藏层部分计算如下：


$$
\begin{aligned}
\text { Encoder } &  \quad \mathbf{z}_{i}=g\left(\mathbf{a}_{i}\right)=\operatorname{ReLU}\left(\mathbf{W} \cdot \operatorname{ReLU}\left(\mathbf{V} \mathbf{a}_{i}+\mathbf{b}^{(1)}\right)+\mathbf{b}^{(2)}\right) \\
\text { Decoder } & \quad \hat{\mathbf{a}}_{i}=f\left(\mathbf{z}_{i}\right)=\mathbf{V}^{\mathrm{T}} \cdot \operatorname{ReLU}\left(\mathbf{W}^{\mathrm{T}} \mathbf{z}_{i}+\mathbf{b}^{(3)}\right)+\mathbf{b}^{(4)} \\
\text { Autoencoder } & \quad \mathbf{\hat { a }}_{i}=h\left(\mathbf{a}_{i}\right)=f\left(g\left(\mathbf{a}_{i}\right)\right)
\end{aligned}
$$


激活函数选择$$
\operatorname{ReLU}(\mathbf{x})=\max(0,\mathbf{x})
$$。最后的解码层利用线性变换给缺失边打分，作为重构的一部分。

MTGAE架构限制为对称，编码解码部分的$$\{\mathbf{W,V}\}$$分别共享参数，相比无约束框架参数减少了两倍。参数共享能够提升多任务学习的学习能力和泛化能力。偏置单元$$\mathbf{b}$$不共享参数，$$
\left\{\mathbf{W}^{\mathrm{T}}, \mathbf{V}^{\mathrm{T}}\right\}
$$是$$\{\mathbf{W,V}\}$$的转置。为了简化符号，需要学习的参数统一为：



$$
\theta=\left\{\mathbf{W}, \mathbf{V}, \mathbf{b}^{(k)}\right\}, k=1, \ldots, 4
$$



因为该自编码器从图的邻域结构中学习节点嵌入，所以称之为 Local Neighborhood Graph Autoencoder ($\mathsf{LoNGAE}$)。

如果节点特征矩阵$$
\mathbf{X} \in \mathbb{R}^{N \times F}
$$可用，将$$(\mathbf{A,X})$$连接获得一个增广邻接矩阵$$
\overline{\mathbf{A}} \in \mathbb{R}^{N \times(N+F)}
$$，然后在$$\bar{\mathbf{a}}_i$$上进行编码解码转换来实现无监督链路预测。此变体称之为 $\alpha \mathsf{LoNGAE}$，

> 为什么要这么做的原因：将邻接矩阵和节点特征矩阵绑定，可以在整个自编码转换过程中支持图连接和节点特征的共享表示。



#### Train

- 前向传播过程：以邻接向量$$\mathbf{a_i}$$作为输入，为无监督链路预测计算重构输出$$\hat{\mathbf{a}}_{i}=h\left(\mathbf{a}_{i}\right)$$；

- 反向传播过程：通过最小化 Masked Balanced Cross-Entropy (MBCE) loss，该损失函数只考虑和可观测边关联的参数的贡献。

  对于链路预测中的类不平衡问题，定义了一个权重因子$$\zeta \in[0,1]$$作为交叉熵损失函数中正样本的乘数。
  
  $$\mathsf{MBCE}$$损失函数定义如下：
  
  
  $$
  \begin{aligned}
  \mathcal{L}_{\mathrm{BCE}}=-\mathbf{a}_{i} \log \left(\sigma\left(\hat{\mathbf{a}}_{i}\right)\right) & \cdot \zeta-\left(1-\mathbf{a}_{i}\right) \log \left(1-\sigma\left(\hat{\mathbf{a}}_{i}\right)\right) \\
  \mathcal{L}_{\mathrm{MBCE}} &=\frac{\mathbf{m}_{i} \odot \mathcal{L}_{\mathrm{BCE}}}{\sum \mathbf{m}_{i}}
  \end{aligned}
  $$



​	其中$$\mathcal{L}_{\mathrm{BCE}}$$是平衡的交叉熵损失函数，带有权重 $$\zeta=1-\frac{\# \text { present links }}{\# \text { absent links }}$$，$\sigma(\cdot)$是$$\mathsf{sigmoid}$$函数，$\odot$是$$\mathsf{Hadamard}$$乘法（元素乘法），$$\mathbf{m_i}$$是布尔方	程：$$\mathbf{m_i}=1 \  \text{if} \ \mathbf{a_i} \neq \text{UNK, else} \  \mathbf{m_i = 0}$$，也就是说只考虑可观测边的损失函数。

同样的自编码器框架可以用于增广邻接矩阵$\bar{\mathbf{A}}$的行向量$$
\overline{\mathbf{a}}_{i} \in \mathbb{R}^{N+F}
$$。

在最后的解码层，重构的$$h(\bar{\mathbf{a}}_i)$$分为两部分：

- $$\hat{\mathbf{a}}_i \in \mathbb{R}^N$$：对应于原始邻接矩阵的重构；
- $$\hat{\mathbf{x}}_i \in \mathbb{R}^F$$：对应于原始节点特征矩阵的重构；

训练时候，在连接图拓扑结构和节点特征$$
\left(\mathbf{a}_{i}, \mathbf{x}_{i}\right)
$$上优化$\theta$，但是在计算重构输出$$
\left(\hat{\mathbf{a}}_{i}, \hat{\mathbf{x}}_{i}\right)
$$的损失函数的时候用不同的损失函数方程。这是为了保证处理不同输入时的灵活性。实验中，输入$$
\left(\mathbf{a}_{i}, \mathbf{x}_{i}\right)
$$被约束为$[0,1]$之间。

增广的$$\mathsf{\alpha MBCE}$$损失函数定义为：


$$
\begin{aligned}
\mathcal{L}_{\alpha \mathrm{MBCE}}=\mathcal{L}_{\mathrm{MBCE}\left(\mathbf{a}_{i}, \mathbf{\hat { a }}_{i}\right)}+\mathcal{L}_{\mathrm{CE}\left(\mathbf{x}_{i}, \hat{\mathbf{x}}_{i}\right)} \\
\mathcal{L}_{\mathrm{CE}\left(\mathrm{x}_{i}, \hat{\mathrm{x}}_{i}\right)}=-\mathrm{x}_{i} \log \left(\sigma\left(\hat{\mathrm{x}}_{i}\right)\right)-\left(1-\mathrm{x}_{i}\right) \log \left(1-\sigma\left(\hat{\mathrm{x}}_{i}\right)\right)
\end{aligned}
$$


其中 $$\mathcal{L}_{\mathrm{CE}\left(\mathrm{x}_{i}, \hat{\mathrm{x}}_{i}\right)}$$ 是标准的交叉熵损失函数。在预测的时候，我们只利用重构输出$$\hat{\mathbf{a}}_i$$进行链路预测，忽略输出$$\hat{\mathrm{x}}_i$$。



### Semi-Supervised Node Classification

对于给定的增广邻接矩阵$$\bar{\mathbf{a}}_i$$，自编码器学习对应的节点嵌入向量$$\mathbf{z}_i$$来获取优化的重构表示。直观上来讲，$$\mathbf{z}_i$$编码了图结构和节点特征，能够用于预测节点$i$的标签。对于多分类问题，使用$$\text{softmax}$$激活函数解码来学习标签的概率分布。


$$
\begin{aligned}
\hat{\mathbf{y}}_{i}=\operatorname{softmax}\left(\tilde{\mathbf{z}}_{i}\right)=\frac{1}{\mathcal{Z}} \exp \left(\tilde{\mathbf{z}}_{i}\right) = \frac{1}{\sum \exp(\tilde{\mathbf{z}}_i)} \exp \left(\tilde{\mathbf{z}}_{i}\right)  \\
\tilde{\mathbf{z}}_{i}=\mathbf{U} \cdot \operatorname{ReLU} \left(\mathbf{W}^{\mathrm{T}} \mathbf{z}_{i}+\mathbf{b}^{(3)}\right)+\mathbf{b}^{(5)}
\end{aligned}
$$


最后在节点分类中，通过一个带掩码的$$\text{softmax}$$分类器来学习节点标签。对于多任务，最小化它们的联合损失函数：


$$
\mathcal{L}_{\mathrm{MULTI}-\mathrm{TASK}}=-\mathrm{MASK}_{i} \sum_{c \in C} \mathbf{y}_{i c} \log \left(\hat{\mathbf{y}}_{i c}\right)+\mathcal{L}_{\mathrm{MBCE}}
$$

其中，$C$是节点标签的集合，当节点$i$的标签为$c$时$$\mathbf{y}_{ic}=1$$，$$
\hat{\mathbf{y}}_{i c}
$$是节点$i$的标签为$c$的$$\text{softmax}$$概率，$$\mathcal{L}_{\mathrm{MBCE}}$$是链路预测的损失函数。如果节点$i$有标签，那么$$\text{MASK}_i=1$$，否则为0。





## Experiment

### Datasets and Baselines

- 数据集

![c564cf78ae6da88c04653538092a50c.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaobk9fsltj30f309o3zp.jpg)

​	表中$$
\left|O^{+}\right|:\left|O^{-}\right|
$$表示正负链路比值。Label Rate表示训练用的标注数据占总数据的比例。

- baseline

  ![f433405d7980ef6baddf5b287045c13.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaobssfkjdj30f606kdgg.jpg)

  - SDNE：用于对比在网络重构任务中的表示能力；$$\{\mathrm{Arxiv-GRQC, BlogCatalog}\}$$ 

  - MF：用于链路预测

    - $$\{\mathrm{Protein,Metabolic, Conflict}\}$$ ：10%的可观测边作为训练集；
    - $$\{\mathrm{PowerGrid}\}$$：90%可观测边作为训练集，剩下的10%作为测试集；

  - VGAE：用于链路预测；$$\{\mathrm{~ C o r a , ~ C i t e s e e r , ~ P u b m e d \} ~}$$；

  - GCN：用于节点分类；$$\{\mathrm{~ C o r a , ~ C i t e s e e r , ~ P u b m e d \} ~}$$；



### Implementation Details

- 构建自环：邻接矩阵对角线设置为1；
- 缺失边和未知边设置为0，在训练时对掩码损失函数没有贡献；通过交叉验证发现设置均值为0最好；



### Results and Analysis

#### Reconstruction

重构效果的评价指标为precision@k，来评估模型检索已知存在的边(正链)作为重建的一部分的能力。

![1a4be420e6f2f7ddc4eb12ed9e75af2.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaocwzj4atj30ub0c9go1.jpg)

通过比较：

- 实验中的LoNGAE模型性能优于SDNE；
- 在$$\mathrm{Arxiv-GRQC}$$数据集上，当数据集缺失边超过40%时，LoNGAE模型性能劣于SDNE；
- 在$$\mathrm{BlogCatalog}$$数据集上，当$k$值较大时，LoNGAE模型的性能普遍优于SDNE；



#### Link Prediction

链路预测的目标是恢复输入图中缺失或未知链接的状态。在实验设计中，假设随机选择一组边缺失，将之作为验证集。我们的任务是训练自编码器，产生一组预测，和真实标签比较。

![6124a99eb1d069ed7c3299b03728c86.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaod7thj3gj30te05wdh9.jpg)

实验结果：

- 在不考虑特征的情况下，数据集$$\{\mathrm{Protein,Metabolic, Conflict}\}$$中，LoNGAE模型性能略优于MF；

  在$$\{\mathrm{PowerGrid}\}$$数据集上，LoNGAE模型性能明显优于MF；

- 考虑特征的情况下，链路预测性能得到了明显的提升；

  $$\{\mathrm{Metabolic, Conflict}\}$$数据集除了节点特征之外，还有边特征，MF方法有效的利用了边特征，但是LoNGAE没有，未来考虑利用边特征；



![616b8d930ea8ffe52feb62c7dbb13b7.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaodscvqvuj30so06v767.jpg)

上表展示了，LoNGAE模型和其他相关图嵌入模型在链路预测任务上的对比结果。

- 类似于MF，能够利用额外信息（节点、边）的图嵌入模型总是能取得性能的提升。



#### Node Classification

![d2777c967490ca889491fb08094c5d7.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaoe50edioj30ew07caas.jpg)





#### Multi-task Learning

![cf5c6e85bce0953cc8c4d12ec9920d1.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaoe5pgfxlj30ey0a90u3.jpg)

