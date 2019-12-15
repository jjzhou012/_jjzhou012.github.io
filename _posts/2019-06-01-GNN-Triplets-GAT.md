---
layout: article
title: GNN 教程：GAT
key: GNN_Triplets_GAT
tags: GNN
category: blog
pageview: true
date: 2019-06-01 23:00:00 +08:00
---
**此为原创文章，转载务必保留[出处](https://archwalker.github.io)**

## 引言
在前两篇博文中，我们讨论了基本的图神经网络算法GCN, 使用采样和聚合构建的inductive learning框架GraphSAGE, 然而图结构数据常常含有噪声，意味着节点与节点之间的边有时不是那么可靠，邻居的相对重要性也有差异，解决这个问题的方式是在图算法中引入“注意力”机制(attention mechanism), 通过计算当前节点与邻居的“注意力系数”(attention coefficient), 在聚合邻居embedding的时候进行加权，使得图神经网络能够更加关注重要的节点，以减少边噪声带来的影响。

## 图注意力机制的类型

目前主要有三种注意力机制算法，它们分别是：学习注意力权重(Learn attention weights)，基于相似性的注意力(Similarity-based attention)，注意力引导的随机游走(Attention-guided walk)。这三种注意力机制都可以用来生成邻居的相对重要性，下文会阐述他们之间的差异。

首先我们对“图注意力机制”做一个数学上的定义：

>定义（图注意力机制）：给定一个图中节点$v_0$ 和$v_0$的邻居节点 $$\left\{v_{1}, \cdots, v_{\vert\Gamma_{v_{0}}\vert}\right\} \in \Gamma_{v_{0}}$$  (这里的 $$\Gamma_{v_{0}}$$ 和GraphSAGE博文中的 $$\mathcal{N}(v_0)$$ 表示一个意思)。注意力机制被定义为将$$\Gamma_{v_{0}}$$中每个节点映射到相关性得分(relevance score)的函数$$f^{\prime} :\left\{v_{0}\right\} \times \Gamma_{v_{0}} \rightarrow[0,1]$$，相关性得分表示该邻居节点的相对重要性。满足：
>
>$$\sum_{i=1}^{\vert\Gamma_{v_{0}}\vert} f^{\prime}\left(v_{0}, v_{i}\right)=1$$

下面再来看看这三种不同的图注意力机制的具体细节

### 1、学习注意力权重
学习注意力权重的方法来自于[Velickovic et al. 2018](https://arxiv.org/abs/1710.10903) 其核心思想是利用参数矩阵学习节点和邻居之间的相对重要性。

给定节点$v_{0}, v_{1}, \cdots, v_{\vert\Gamma_{x_{0}}\vert}$相应的特征(embedding) $$\mathbf{x}_{0}, \mathbf{x}_{1}, \cdots, \mathbf{x}_{\vert\Gamma_{o^{*}}\vert}$$ ，节点$v_0$和节点$v_j$注意力权重$\alpha_{0, j}$可以通过以下公式计算：

$$
\alpha_{0, j}=\frac{e_{0, j}}{\sum_{k \in \Gamma_{v_{0}}} e_{0, k}}
$$

其中，$e_{0, j}$ 表示节点$v_j$对节点$v_0$的相对重要性。在实践中，可以利用节点的属性结合softmax函数来计算$e_{0, j}$间的相关性。比如，GAT 中是这样计算的：

$$
\alpha_{0, j}=\frac{\exp \left(\text { LeakyReLU }\left(\mathbf{a}\left[\mathbf{Wx}_{0} \| \mathbf{Wx}_{j}\right]\right)\right)}{\sum_{k \in \Gamma_{v_{0}}} \exp \left(\text { LeakyReLU }\left(\mathbf{a}\left[\mathbf{W} \mathbf{x}_{0} \| \mathbf{W} \mathbf{x}_{k}\right]\right)\right)}
$$

其中，$\mathbf{a}$ 表示一个可训练的参数向量, 用来学习节点和邻居之间的相对重要性，$\mathbf{W}$ 也是一个可训练的参数矩阵，用来对输入特征做线性变换，$\vert\vert$表示向量拼接(concate)。

![](http://ww1.sinaimg.cn/large/006tNc79ly1g3odbgkqd1j30bq06bjrs.jpg)


> 如上图，对于一个目标对象$v_0$，$a_{0,i}$ 表示它和邻居$v_i$的相对重要性权重。$a_{0, i}$可以根据 $v_0$ 和 $v_i$ 的 embedding $x_0$ 和 $x_i$ 计算，比如图中$\alpha_{0, 4}$ 是由 $x_0, x_4, \mathbf{W}, \mathbf{a}$ 共同计算得到的。

### 2 基于相似性的注意力

上面这种方法使用一个参数向量$\mathbf{a}$学习节点和邻居的相对重要性，其实另一个容易想到的点是：既然我们有节点$v$的特征表示$x$，假设和节点自身相像的邻居节点更加重要，那么可以通过直接计算$x$之间相似性的方法得到节点的相对重要性。这种方法称为基于相似性的注意力机制，比如说论文 [TheKumparampil et al. 2018](http://arxiv.org/abs/1803.03735) 是这样计算的：


$$
\alpha_{0, j}=\frac{\exp \left(\beta \cdot \cos \left(\mathbf{W} \mathbf{x}_{0}, \mathbf{W} \mathbf{x}_{j}\right)\right)}{\sum_{k \in \Gamma_{v_{0}}} \exp \left(\beta \cdot \cos \left(\mathbf{W} \mathbf{x}_{0}, \mathbf{W} \mathbf{x}_{k}\right)\right)}
$$

其中，$\beta$ 表示可训练偏差(bias)，$\cos$函数用来计算余弦相似度，和上一个方法类似，$\mathbf{W}$ 是一个可训练的参数矩阵，用来对输入特征做线性变换。

这个方法和上一个方法的区别在于，这个方法显示地使用$\cos$函数计算节点之间的相似性作为相对重要性权重，而上一个方法使用可学习的参数$\mathbf{a}$学习节点之间的相对重要性。

### 3 注意力引导的游走法

前两种注意力方法主要关注于选择相关的邻居信息，并将这些信息聚合到节点的embedding中。第三种注意力的方法的目的不同，我们以[Lee et al. 2018](http://ryanrossi.com/pubs/KDD18-graph-attention-model.pdf) 作为例子：

GAM方法在输入图进行一系列的随机游走，并且通过RNN对已访问节点进行编码，构建子图embedding。时间$t$的RNN隐藏状态 $$\mathbf{h}_{t} \in \mathbb{R}^{h}$$ 编码了随机游走中 $1, \cdots, t$ 步访问到的节点。然后，注意力机制被定义为函数 $f^{\prime} : \mathbb{R}^{h} \rightarrow \mathbb{R}^{k}$，用于将输入的隐向量$$f'\left(\mathbf{h}_{t}\right)=\mathbf{r}_{t+1}$$映射到一个$k$维向量中，可以通过比较这$k$维向量每一维的数值确定下一步需要优先游走到哪种类型的节点(假设一共有$k$种节点类型)。下图做了形象的阐述：

![](http://ww1.sinaimg.cn/large/006tNc79ly1g3odbha88dj309906j0t2.jpg)


> 如上图，$h_3$聚合了长度$L=3$的随机游走得到的信息$\left(x_{1}, x_2, x_{3}\right)$，我们将该信息输入到排序函数中，以确定各个邻居节点的重要性并用于影响下一步游走。
>

## 后话

至此，图注意力机制就讲完了，还有一些细节没有涉及，比如在 [GAT论文](https://arxiv.org/abs/1710.10903) 中讨论了对一个节点使用多个注意力机制(multi-head attention), 在[AGNN论文](http://arxiv.org/abs/1803.03735)中分析了注意力机制是否真的有效，详细的可以参考原论文。

## Reference
[Attention Models in Graphs: A Survey](http://arxiv.org/abs/1807.07984)

[Graph Attention Networks](http://arxiv.org/abs/1710.10903)

[Attention-based Graph Neural Network for Semi-supervised Learning](http://arxiv.org/abs/1803.03735)

[Graph Classification using Structural Attention](http://ryanrossi.com/pubs/KDD18-graph-attention-model.pdf)

