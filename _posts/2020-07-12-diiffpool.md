---
layout: article
title: 图神经网络：Hierarchical graph representation learning with differentiable pooling (DiffPool)
date: 2020-07-12 00:10:00 +0800
tags: [GNN, Graph, Graph Classification]
categories: blog
pageview: true
key: GNN-diffpool
---

------

- Paper: [Hierarchical graph representation learning with differentiable pooling](https://arxiv.org/abs/1806.08804)
- Code: [https://github.com/RexYing/diffpool](https://github.com/RexYing/diffpool)
- pytorch-geometric pooling层实现：[link](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.diff_pool.dense_diff_pool)



## 概述

当前的GNN图分类方法本质上是平面（flat）的，不能学习图形的层次表示。文中提出了DIFFPOOL模型，这是一个可微的图pooling模块，它可以生成图的层次表示，并可以以端到端的方式与各种图神经网络架构相结合。DIFFPOOL为深度GNN的每一层的节点学习可微分的cluster assignment，将节点映射到一组cluster，然后形成下一个GNN层的粗化输入（coarsened input）。



### 相关概念

- hard assignment：每个数据点都归到一个类别(i.e. cluster)。例如，样本1就是类型1，样本2就是类型2。
- soft assignment：把数据点归到不同的类，分到每个类有不同的置信度。例如，样本1有30%的可能是类型1，70%的可能是类型2
- assignment：文中涉及的assignment就是把节点分类、归类的意思。



### 背景

当前GNNs进行图分类的主要问题在于，它们本质上是平面的（flat），它们在图的边上传播信息，无法以层次化（hierarchical）的方式推断和聚合信息。**目前将GNNs应用于图分类时，标准的方法是为图中的所有节点生成embedding，然后对所有节点的embedding进行全局pooling操作**，例如，使用简单的求和操作或神经网络。这种全局pooling方法忽略了图中可能存在的层次结构，并且不利于研究人员为整个图上的预测任务构建有效的GNN模型。简单来讲，它不希望先得到所有结点的embedding，然后再一次性得到图的表示，这种方式比较低效，而是希望**通过一个逐渐压缩信息的过程就可以得到图的表示**。



### 相关工作

在图分类的任务中，应用GNNs的一个主要挑战是从节点embedding到整个图的表示。解决这一问题的常见方法有：

- 简单地在最后一层中将所有节点embedding求和（Convolutional networks on graphs for learning molecular fingerprints，NIPS 2015）；
- 引入一个连接到图中的所有节点的“虚拟节点”来求平均值（Gated graph sequence neural networks，ICLR 2016）；
- 使用深度学习架构来聚合节点embedding（Neural message passing for quantum chemistry，ICML 2017）。

然而，所有这些方法都有其局限性，即**它们不能学习层次结构表示**(即，所有节点的embedding都在一个单层中被pool)，因此无法捕获许多真实世界中的图的结构。

最近的一些方法也提出了将CNN架构应用于**拼接（concatenate）所有节点的embedding**：

- [PATCHY-SAN] Learning convolutional neural networks for graphs, ICML 2016；

- [DGCNN] An end-to-end deep learning architecture for graph classification, AAAI 2018

但这些方法需要学习节点的排序，这通常是非常困难的，相当于求解图同构。

最后，有一些工作通过**结合GNNs和确定性图聚类算法来学习层次图表示**：

- Convolutional neural networks on graphs with fast localized spectral filtering，NIPS 2016；
- SplineCNN: Fast geometric deep learning with continuous B-spline kernels，CVPR 2018；
- Dynamic edge-conditioned filters in convolutional neural networks on graphs，CVPR 2017

与这些方法不同，DiffPool寻求以端到端的方式学习图的层次结构，而不依赖于确定性的图聚类程序。



## 模型介绍

### 简介

![image-20200711235626019](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200712005029.png)

上图是DiffPool结构示意图

- 在每一层，运行一个GNN模型获得节点的embeddings，再用这些embeddings将相似的节点进行聚类得到下一层的粗化输入，然后在这些粗化的图上运行另一个GNN层。整个过程重复$L$层，然后使用最后的输出表示进行图分类任务。
- 如图所示，层次化的GNN加入了pooling层，能够捕捉图的层次信息，扩大了感受野。但是，这里的Pooled network训练十分困难，需要两个Loss保证网络能够收敛，Pooled network的1,2,3层各不相同，需要分别训练。导致整个网络参数量巨大。



### 问题定义

- 定义一个图$$G=(A,F)$$，其中$$A∈{0,1}^{n×n}$$是一个邻接矩阵，$$F \in \mathbb{R}^{n \times d}$$是节点的特征矩阵;

- 假设一个有标签的图数据集为$$\mathcal{D}=\left\{\left(G_{1}, y_{1}\right),\left(G_{2}, y_{2}\right), \ldots\right\}$$，其中$$y_{i} \in \mathcal{Y}$$是对应图$$G_{i} \in \mathcal{G}$$的标签；

- 图分类任务的目标是学习一个映射函数：$$f: \mathcal{G} \rightarrow \mathcal{Y}$$。这个函数将一个图映射到一个标签。

- 与标准的有监督机器学习相比，面临的挑战是需要一种从这些输入图中提取有用特征向量的方法。也就是说，为了应用标准的机器学习方法进行分类（例如神经网络），需要一个程序来将每个图转换成一个低维的向量。

- 文中并没有考虑边的特征，作者说使用论文（Dynamic edge-conditioned filters in convolutional neural networks on graphs，CVPR 2017）中的方法可以很容易的扩展文中的算法以支持边的特征。

  

#### 图神经网络

在图神经网络的基础上，以一种端到端的方式为图分类学习一种有用表示。考虑采用以下通用的“消息传递”框架：

![image-20200712005017468](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200712005017.png)

通用的“消息传递”的GNNs：

$$
H^{(k)}=M\left(A, H^{(k-1)} ; \theta^{(k)}\right)
$$

使用GCN实现的传播函数

$$
H^{(k)}=M\left(A, H^{(k-1)} ; W^{(k)}\right)=\operatorname{ReLU}\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(k-1)} W^{(k-1)}\right)
$$

文中提出的pooling模型可以使用任意一个能实现公式（1）的GNN模型，DiffPool不需要了解$M$的实现方式，可以将任意实现K次消息传递的GNN模块抽象为$$Z=\operatorname{GNN}(A, X)$$。



#### 堆叠GNNs和Pooling层

文中的目标是定义一种通用，端到端，可微的策略，使得可以用一种层次化的方式堆叠多层GNN模型。

给定一个GNN模块的输出$$Z=\operatorname{GNN}(A, X)$$和一个邻接矩阵$$A \in \mathbb{R}^{n \times n}$$，目标是寻找一种方式可以得到一个新的包含$$m<n$$个节点的粗化图，新的粗化图的邻接矩阵为$$A^{\prime} \in \mathbb{R}^{m \times m}$$，节点embedding矩阵为$$Z^{\prime} \in \mathbb{R}^{m \times d}$$。

这个新的粗化图（coarsened graph）作为下一层GNN的输入，重复$$L$$次就可到了一系列的粗化图。因此，目标就是需要学习一种pooling策略，这种策略可以**泛化到具有不同节点、边的图中，并且能够在推理过程中能适应不同的图结构**。

实际上，代码中将不同图的节点数进行统一

### Differentiable Pooling via Learned Assignments
DIFFPOOL方法通过使用GNN模型的输出学习节点上的cluster assignment matrix来解决上述问题。

#### Pooling with an assignment matrix

定义第$l$层学到的cluster assignment matrix为$$S^{(l)} \in \mathbb{R}^{n_{l} \times n_{l+1}}$$，$$n_l$$表示在第$l$层的节点数，$$n_{l+1}$$表示在第$l+1$层的节点数。assignment matrix表示第$l$层的每一个节点到第$l+1$层的每一个节点（或cluster）的概率。（**其实就相当于一个编码的过程，粗化相当于降维**）。

假设$$S^{(l)}$$预计算好了，给定邻接矩阵$$A^{(l)}$$和节点嵌入$$Z^{(l)}$$的情况下，DiffPool层的运算为$$\left(A^{(l+1)}, X^{(l+1)}\right)=\text { DIFFPOOL }\left(A^{(l)}, Z^{(l)}\right)$$，表示图的粗化过程，生成一个新的邻接矩阵$$A^{(l+1)}$$和节点嵌入向量$$Z^{(l+1)}$$。

具体过程如下：

$$
X^{(l+1)}=S^{(l)^{T}} Z^{(l)} \in \mathbb{R}^{n_{l+1} \times d} \\
$$

- $$X_{i j}^{(l+1)}$$表示第$$l+1$$层cluster $i$ 的embedding；

$$
A^{(l+1)}=S^{(l)^{T}} A^{(l)} S^{(l)} \in \mathbb{R}^{n_{l+1} \times n_{l+1}}
$$

- $$A_{i j}^{(l+1)}$$表示第$$l+1$$层cluster $i$ 与cluster $j$ 之间的连接强度；

#### Learning the assignment matrix
下面介绍DiffPool如何用公式（3）和（4）生成assignment矩阵$$S^{(l)}$$和embedding矩阵$$Z^{(l)}$$:

$$
Z^{(l)}=\mathrm{GNN}_{l, \mathrm{embed}}\left(A^{(l)}, X^{(l)}\right)
$$

$$
S^{(l)}=\operatorname{softmax}\left(\mathrm{GNN}_{l, \mathrm{pool}}\left(A^{(l)}, X^{(l)}\right)\right)
$$

这两个GNN使用相同的输入数据，但具有不同的参数，并扮演不同的角色:

- embedding GNN 为这一层的输入节点生成新的embedding;
- pooling GNN 生成从输入节点到$$n_{l+1}$$个cluster的概率

在倒数第二层，即$$L−1$$层时令assignment矩阵是一个全为1的向量，也就是说所有在最后的$L$层的节点都会被分到一个cluster，生成对应于整图的一个embedding向量。这个embedding向量可以作为可微分类器（如softmax层）的特征输入，整个系统可以使用随机梯度下降进行端到端的训练。

![image-20200712111131004](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200712111131.png)



#### 关于优化

关于优化训练过程，设计损失函数，有几个注意点：

- 第一部分为分类损失，常用交叉熵损失函数；

- 在实践中，仅使用来自图分类任务的梯度信号很难训练pooling GNN(公式4)。这是一个非凸优化问题，并且在训练中pooling GNN很容易陷入局部极小值。为了缓解这个问题，使用一个辅助的链路预测目标来训练pooling GNN。

  在每一层$l$上，最小化如下损失函数
  
  $$
  L_{\mathrm{LP}}= \mid \mid A^{(l)}, S^{(l)} S^{(l)^{T}}\mid\mid_{F}
  $$
  
  其中$$\mid\mid \cdot\mid\mid _F$$表示Frobenius范数。

  > 分配矩阵$$S^{(l)}$$相当于一个编码操作，$$S^{(l)}S^{(l)^{T}}$$类似于一个自编码器的重构过程，这部分辅助损失函数的设计相当于一个用图重构来解决链路预测问题，保证优化后的分配矩阵$$S^{(l)}$$在粗化的过程中能更好的提取$$A^{(l)}$$的潜在特征（节点与周围节点的链接信息，起到更好的cluster作用）。

- 深层次上的邻接矩阵$$A^{(l)}$$是低层assignment矩阵的函数，在训练过程中会发生变化。pooling GNN的另一个重要特性(公式4)是，为每个节点的输出cluster通常都应该接近one-hot向量，以便清楚地定义每个cluster或子图的成员关系（避免重叠）。

  因此通过最小化如下损失函数来正则化cluster assignment的熵
  
  $$
  L_{\mathrm{E}}=\frac{1}{n} \sum_{i=1}^{n} H\left(S_{i}\right)
  $$
  
  其中$H$为熵函数，$S_i$为$S$的第$i$行。

  > 自信息是熵的基础，**自信息表示某一事件发生时所带来的信息量的多少**，当事件发生的概率越大，则自信息越小。
  >
  > 对于这个损失函数的理解：$S_i$其实相当于一个概率分布，表示当前节点（cluster）$i$被分配到下一阶段的各个clusters的概率。$S_i$越接近onehot向量，也就是说越接近硬分配，所带的信息量越少，$$H(S_{i})$$越小。最小化这个损失函数的过程相当于约束粗化的过程，保证分配尽可能接近硬分配。



在训练期间，每一层的$$L_E$$和$$L_{LP}$$都加入分类损失中，相当于一个多任务损失函数。在实践中，观察到带有side objective的训练需要更长的时间收敛，但是仍然可以获得更好的性能和更多可解释的cluster assignment。



## 实验



### 设置

- 在实验中，用于DIFFPOOL的GNN模型是建立在GraphSAGE架构之上的
- 使用GraphSAGE的“mean”聚合函数，并在**每两个GraphSAGE层之后应用一个DIFFPOOL层**
- 在数据集上总共使用了2个DIFFPOOL层
- 对于ENZYMES和COLLAB这样的小数据集，1个DIFFPOOL层可以达到类似的性能。每个DIFFPOOL层之后，在下一个DIFFPOOL层或readout层之前执行**3层图卷积层**。
- 分别用两种不同的稀疏的GraphSAGE模型计算embedding矩阵和assignment矩阵。
- **在2个DIFFPOOL层的架构中，cluster数量设置为应用DIFFPOOL之前节点数量的25%，而在1个DIFFPOOL层的架构中，cluster数量设置为节点数的10%。**
- Batch normalization在GraphSAGE的每一层之后应用。
- 实验还发现，在每一层的节点embedding中添加一个$$l_2$$正则化可以使训练更加稳定。
- 所有的模型都经过了3000个epoch的训练，当验证损失开始减少时，会提前停止。
- 执行10-fold交叉验证来评估模型性能

两个简化版的DIFFPOOL:

- DIFFPOOL-DET是一个使用确定性图聚类算法(Weighted graph cuts without eigenvectors a multilevel approach,2007)生成assignment matrix的DIFFPOOL模型。
- DIFFPOOL-NOLP是DIFFPOOL的一个变体，没使用链接预测。







## Reference

- https://blog.csdn.net/yyl424525/article/details/103307795