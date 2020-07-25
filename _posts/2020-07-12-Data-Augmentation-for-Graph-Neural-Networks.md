---
layout: article
title: 图神经网络：Data Augmentation for Graph Neural Networks
date: 2020-07-12 00:21:00 +0800
tags: [GNN, Graph, Node Classification, Data Augmentation]
categories: blog
pageview: true
key: Data-Augmentation-for-Graph-Neural-Networks
---

------

- Paper: [Data Augmentation for Graph Neural Networks](http://arxiv.org/abs/2006.06830)
- Code: [https://github.com/GAugAuthors/GAug](https://github.com/GAugAuthors/GAug)



## 一、概述

- 研究半监督节点分类上的data augmentation；
- 展示了神经链路预测可以有效的编码class-homophilic结构，来促进类内边，降低类间边的作用；
- 提出了GAUG图数据增强框架，利用链路预测提高基于gnn的节点分类的性能；



## 二、Related work

- **[ICLR2020] **Dropedge: Towards deep graph convolutional networks on node classification.
  - DropEdge在每个训练epoch随机地从输入图中删除一定数量的边，类似于一个数据增强器，或者一个消息传递减少器；
  - 从理论上证明了DropEdge可以降低过平滑的收敛速度，或者减轻过平滑引起的信息损失；
  - DropEdge是一种通用方法，可以与许多其他gnn模型(例如GCN、ResGCN、GraphSAGE和JKNet)一起配置，以增强性能；

- **[AAAI2020]** Measuring and relieving the over-smoothing problem for graph neural networks from the topological view.
  - 对GNN的过度平滑问题进行了系统和定量的研究。引入两个定量指标MAD和MADGap，分别测量图节点表示的平滑度和过平滑度。然后，验证了平滑是GNN的本质，导致平滑度过高的关键因素是节点接收到的消息的低信噪比，这部分由图拓扑确定
  - 提出了两种从拓扑学角度缓解过平滑问题的方法：
    - MADReg，它在训练目标中添加了基于MADGap的正则化器；
    - AdaEdge根据模型预测优化图形拓扑。



## 三、方法

### 3.1 定义

**图的基本概念**

图$$\mathcal{G}=(\mathcal{V}, \mathcal{E})$$，节点集$$\mathcal{V}$$，边集$$\mathcal{E}$$，节点数$$N=\mid \mathcal{V} \mid$$，邻接矩阵$$\mathbf{A}$$， 特征矩阵$$\mathbf{X} \in \mathbb{R}^{N \times F}$$。

**图神经网络的基本框架**

![image-20200713100519284](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200713113040.png)

### 3.2 动机

现实场景中的图可能含有噪声，噪声产生的原因有很多：

- 对抗性噪声：故意生成的用于污染交互空间；
- 实际观察的误区：推荐系统从不推荐一些对象，导致链路闭塞；
- 预处理：删除自环、孤立节点、边权重等操作引入噪声；
- 人为错误：例如引文网络中文献的错误引用；

这些噪声会使得下游任务得到的“可观测图”和“理想图”之间产生差距。

可以通过边操作来扩充数据。在最好的情况下，我们可以生成一个图$G_i$(理想连通性)，其中添加了假定的(但缺少的)链接，并删除了不相关的/不重要的(但已存在的)链接。

![image-20200713104716098](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200713113002.png)

图1显示了在ZKC图中实现的这种好处: 在同一组(类内)的节点之间有策略地添加边，并删除不同组(类间)的节点之间的边，显著提高了节点分类测试性能。直观地说，这个过程促进了同类节点嵌入的平滑性，并区分了其他类节点嵌入，提高了区别性。



**理论**

在标签全知的情况下，使用GNN训练时，通过有策略的边操作来提升类内边和降级类间边，使得类的区分变得微不足道。

考虑一个极端场景，其中所有可能的类内边都存在，而类间边都不存在，图可以看作是$k$个完全连通的组件，其中$k$为类的数量，每个组件中的所有节点都有相同的标签。GNNs可以很容易地在不同的类之间生成不同的节点表示，并为其中的节点生成相同的表示。

**Theorem 1**

![image-20200713112948173](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200713112948.png)



对于一个理想的图$G_i$而言，训练过程中类标的差异性已经不太重要了。然而，这并不意味着在测试中会出现这样的结果，因为测试阶段节点连接性可能反映的是$G$而不是$G_i$。作者认为，如果训练中的修改太过刻意，就会有过度拟合理想图$G_i$的风险，而且由于训练测试的差距太大，在$G$上的表现就会很差。



### 3.3 方法设置
图像里面，data augmentation分两阶段进行：

- 利用一种变换扩充训练集：$$f: \mathcal{S} \rightarrow \mathcal{T}$$；
- 合并原训练集和扩充数据，用于训练模型：$$\mathcal{S} \cup \mathcal{T}$$；

在图里面，特别是节点分类里，这样的方式并不适合，因为训练集的规模$$\mid\mathcal{S}\mid=1$$；

本文提出了两种策略：

- **Modified-graph setting**: 利用单个图的变换生成修改的图：$$f: \mathcal{G} \rightarrow \mathcal{G}_m$$，然后用$$\mathcal{G}_m$$取代$$\mathcal{G}$$，用于训练和推理；
- **Original-graph setting**: 利用多图变换：$$f_i: \mathcal{G} \rightarrow \mathcal{G}_m^i \ \ \text{for}\ \  i=1,\ldots, N$$，然后用$$\mathcal{G} \cup\left\{\mathcal{G}_{m}^{i}\right\}_{i=1}^{N}$$进行训练，用$$\mathcal{G}$$进行推理；

这两种策略适用于不同的场景（类似于直推式和归纳式）：

- 当给定图$$\mathcal{G}$$不变时，也就是直推式场景中，Modified-graph setting更适合。对于原图生成一个修改的图$$\mathcal{G}_m$$，使用$$\mathcal{G}_m$$进行训练和测试；
- 当给定的图$$\mathcal{G}$$是动态的（大规模的），也就是归纳式场景中，Original-graph setting是更合适的。因为这个时候用$$\mathcal{G}_m$$来校准动态图的连通性是不太灵活的，扩充数据，提升泛化性更合适。



## 四、具体方法

### 4.1 GAUG-M for the Modified-Graph Augmentation Setting
GAUG-M方法适用于Modified-graph setting，包括两个阶段：

- **使用一个链路预测函数来获得$$\mathcal{G}$$中所有可能的和存在的边的概率。**链路预测函数可以用任何合适的方法代替；

  链路预测函数可以定义为具有如下形式的模型：$$f_{e p}: \mathbf{A}, \mathbf{X} \rightarrow \mathbf{M}$$。输入图，输出边概率矩阵$$\mathbf{M}$$。

  论文使用GAE作为链路预测器，主要模型架构如下：

  $$
  \mathbf{M}=\sigma\left(\mathbf{Z} \mathbf{Z}^{T}\right), \quad \text { where }\ \  \mathbf{Z}=f_{G C L}^{(1)}\left(\mathbf{A}, f_{G C L}^{(0)}(\mathbf{A}, \mathbf{X})\right)
  $$

  两层GCN作为编码器，一个内积层作为解码器，整个GAE实现图的重构，输出边概率矩阵。

- **利用预测的边概率，确定地添加(删除)新的(现有的)边来创建一个修改的图$$\mathcal{G}_m$$，它被用作GNN节点分类器的输入；**

  令$$\mid\mathcal{E}\mid$$表示$$\mathcal{G}$$的边数。利用输出的边概率矩阵，添加top $$i \mid\mathcal{E}\mid$$ 条最高概率的虚边，删除 $$j \mid\mathcal{E}\mid$$ 条最小概率的实边。其中$$i, j \in[0,1]$$。



![image-20200713132929041](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200713132929.png)

- (a): 根据链路预测指导，随着增边比例上调，类内边涨幅远大于类间边，F1整体趋势小幅度上升；
- (b): 随机增边的情况下，随着增边比例上调，类内边涨幅远小于类间边，F1大幅度下降；

以上说明，链路预测可以帮助指导类内边的补充，也就是加强同类节点的交互；而随机修改，由于类间虚边数量明显大于类内虚边，增加的边中类间边的比例也会明显大于类内边；

- (c): 根据链路预测指导，随着删边比例上调，类内边和类间边的降幅相近，F1整体趋势小幅度下降；
- (d): 随机删边的情况下，随着删边比例上调，类内边降幅远大于类间边，F1大幅度下降；

以上说明，随机修改，由于类内实边数量明显大于类间实边，删除的边中类内边的比例也会明显大于类间边；而链路预测可以缓解这个情况，因为类内边和类间边中都有低置信度边，删除的时候平衡两者的量。



### 4.2 GAUG-O for the Original-Graph Augmentation Setting

GAUG-O不需要对边进行添加/删除，具有端到端可训练性，同时利用链路预测和节点分类损失来迭代提高链路预测器的增强能力和节点分类器GNN的分类能力。

![image-20200713142135045](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200713142135.png)

GAUG-O由三个主要部分组成:

- 产生边缘概率估计的可微链路预测器；
- 产生稀疏图变量的插值和采样步骤；
- 利用这些变量学习节点分类的嵌入的GNN。

该体系结构训练端到端的分类和链路预测损失。

区别于GAUG-M的离散化图修改方式，GAUG-O支持端到端可训练的数据增强过程。

GAUG-O中的链路预测依然使用GAE，为了防止链路预测任意偏离原始图的邻接矩阵，我们用原始$$\mathbf{A}$$对预测的$$\mathbf{M}$$进行插值，从而得到邻接$$\mathbf{P}$$。在边抽样阶段，我们通过边上的伯努利抽样来稀疏$$\mathbf{P}$$，得到图的变体邻接矩阵$$\mathbf{A'}$$。

为了训练目的，论文使用(软的，可微的)松弛伯努利抽样程序作为伯努利近似。这种松弛是Gumbel-Softmax重新参数化技巧的二进制特例。使用松弛采样，应用直通式(ST)梯度估计器，它在前传中对松弛采样进行轮询，从而稀疏邻接。在向后传递中，梯度直接传递给松弛样本，而不是四舍五入的值，从而可以进行训练。在形式上,
$$
\mathbf{A}_{i j}^{\prime}=\left\lfloor\frac{1}{1+e^{-\left(\log \mathbf{P}_{i j}+G\right) / \tau}} +\frac{1}{2}\right\rfloor, \quad \text { where } \quad \mathbf{P}_{i j}=\alpha \mathbf{M}_{i j}+(1-\alpha) \mathbf{A}_{i j}
$$
其中$$\mathbf{A'}$$是采样的邻接矩阵，$$\tau$$是Gumbel-Softmax分布的温度系数，$$G \sim \operatorname{Gumbel}(0,1)$$是一个Gumbel随机变体，$$\alpha$$是调整链路预测器对原始图影响的超参数。

最后，邻接变体$$\mathbf{A'}$$和特征矩阵$$\mathbf{X}$$被一起传入GNN节点分类器。训练优化的损失函数为分类损失和链路预测（重构）损失的联合：
$$
\mathcal{L}=\mathcal{L}_{n c}+\beta \mathcal{L}_{e p}, \quad \text { where } \quad \mathcal{L}_{n c}=C E(\hat{\mathbf{y}}, \mathbf{y}) \text { and } \mathcal{L}_{e p}=B C E\left(\sigma\left(f_{e p}(\mathbf{A}, \mathbf{X})\right), \mathbf{A}\right)
$$
其中，$$\beta$$是用于权衡重构损失的超参数，BCE/CE表示标准(二元)交叉熵损失。训练联合损失的目的在于控制边缘预测性能中潜在的过度漂移。





## 五、实验

### 5.1 设置

- 6个benchmark数据集
- 4个流行的节点分类gnn模型
- 2个baseline

![image-20200713215314591](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200713215834.png)