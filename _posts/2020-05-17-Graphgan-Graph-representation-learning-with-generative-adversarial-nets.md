---
layout: article
title: 图神经网络--对抗训练：图对抗生成网络（GraphGAN）
date: 2020-05-17 00:10:00 +0800
tags: [Adversarial, GNN, Link Prediction, Node Classification, Graph]
categories: blog
pageview: true
key: Graphgan-Graph-representation-learning-with-generative-adversarial-nets
---

------

- Paper: [Graphgan: Graph representation learning with generative adversarial nets](https://arxiv.org/abs/1711.08267)
- Code: [https://github.com/hwwang55/GraphGAN](https://github.com/hwwang55/GraphGAN)
- Code: [https://github.com/liutongyang/GraphGAN-pytorch](https://github.com/liutongyang/GraphGAN-pytorch)



## Introduction(!)

现有的图表示学习方法可以分为两类：

- 生成式模型： DeepWalk, Node2vec, metapath2vec 等
- 判别式模型： DNGR，SDNE，Ppne 等

它们的区别在于：

- 生成式的模型假设，在网络中，对每一个节点$v_c$，存在一个潜在的真实连接分布$p_{\text {true }}\left(v \mid v_{c}\right)$，揭示了节点在全局网络中的连接偏好，因此，图中的边可以被看作是由这些条件概率分布生成的可观测样本。而这些生成模型本质上通过最大化图中边的似然来学习节点的嵌入表示。
  - 例如，DeepWalk使用随机游走对每个节点的“上下文”节点进行采样，并尝试最大化目标节点的上下文节点的似然概率。
  - Node2vec进一步扩展了这个想法，提出了一个有偏的随机游走过程，它在为目标节点生成上下文时提供了更大的灵活性。
- 判别式模型并不把边看作是由潜在条件分布产生的，而是通过直接学习一个用来预测边是否存在的分类器。典型的判别式模型，就是将网络中的训练集上的任一节点对$v_c$和$v_j$看作特征，然后预测两点之间存在边的概率$p\left(e d g e \mid\left(v_{i}, v_{j}\right)\right)$。
  - SDNE利用网格的稀疏邻接向量作为每个节点的raw feature，并采用自编码器，监督式地来获得稠密的低维节点向量表达。
  - PPNE通过对正样本(连通节点对)和负样本(不连通节点对)的监督学习，直接学习节点的嵌入表示，并在学习过程中保持节点的本质属性。

虽然生成模型和判别模型通常是图表示学习方法的两个不相交类，但它们可以被看作是同一事物的两个方面。最近提出的生成对抗网络就是结合了生成模型和判别模型，提出了一个博弈的框架。

本文将生成对抗网络的思想用于图表示学习。

## GraphGAN框架

### 模型

- 符号定义

  $\mathcal{G}=(\mathcal{V}, \mathcal{E})$表示给定图，$\mathcal{V}=\left\{v_{1}, \ldots, v_{V}\right\}$表示节点集，$\mathcal{E}=\left\{e_{i j}\right\}_{i, j=1}^{V}$表示边集。对于给定节点$v_c$，$\mathcal{N}\left(v_{c}\right)$表示$v_c$的一阶邻居，$p_{\text {true }}\left(v \mid v_{c}\right)$表示节点$v_c$的真实连接分布，表示了节点$v_c$的连接偏好。从某个角度而言，$\mathcal{N}\left(v_{c}\right)$是基于$p_{\text {true }}\left(v \mid v_{c}\right)$采样得到的可观测样本集合。

- 生成器 $G\left(v \mid v_{c} ; \theta_{G}\right)$

  尽可能地去拟合或近似出真实的连接概率分布$p_{\text {true }}\left(v \mid v_{c}\right)$，从$\mathcal{V}$中选择出最有可能与$v_c$连接的节点；

  生成器的目标是生成与$v_c$真实的邻居节点尽可能相似的点，来欺骗判别器；

- 判别器 $D\left(v, v_{c} ; \theta_{D}\right)$

  计算节点对之间存在连接的可能性；

  判别器的目标是判断这些节点哪些是$v_c$的真实邻居，哪些是由生成器生成的节点；

模型的目标函数为：

$$
\begin{array}{l}
\min _{\theta_{G}} \max _{\theta_{D}} V(G, D)=\sum_{c=1}^{V}\left(\mathbb{E}_{v \sim p_{\text {true }}\left(\cdot \mid v_{c}\right)}\left[\log D\left(v, v_{c} ; \theta_{D}\right)\right]\right. \\
\left.\quad+\mathbb{E}_{v \sim G\left(\cdot \mid v_{c} ; \theta_{G}\right)}\left[\log \left(1-D\left(v, v_{c} ; \theta_{D}\right)\right)\right]\right)
\end{array}
$$

对目标函数的理解需要分成两部分：

- 先固定$G$训练$D$，对于判别器$D$而言，希望自身能准确预测，即对真实样本，让其概率最大，也就是让$\mathbb{E}_{v \sim p_{\text {true }}\left(\cdot \mid v_{c}\right)}[\log D(v, v_{c} ; \theta_{D})]$最大；对生成样本，让其概率最小，也就是让$(1-D\left(v, v_{c} ; \theta_{D}\right))$尽可能大，因此整体是一个max的目标；
- 固定$D$训练$G$，对于生成器$G$而言，为了欺骗判别器，也就是说要使判别器对我生成的样本无法准确区分，也就是预测生成节点与目标节点$v_c$之间存在边的概率最大化，让$(1-D\left(v, v_{c} ; \theta_{D}\right))$尽可能小；此时左边项没有生成器，为常数，所以整体是一个min目标；

将上述两个目标结合起来，就可以得到一个min max的目标函数。

模型框架示意图如下

<img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImggraphgan-farmework.png" alt="55703b0b6738605d22278d329772701" style="zoom: 80%;" />

生成器和判别器的参数是不断地交替训练进行更新的。每一次迭代：

- 判别器$D$通过来自$p_{\text {true }}\left(v \mid v_{c}\right)$的正样本（绿色）和来自$G\left(v \mid v_{c} ; \theta_{G}\right)$的负样本（蓝条纹）进行训练；
- 生成器$G$则通过$D$的指导，按照梯度策略进行更新；

训练直到判别器无法区分生成器生成的数据和真实数据分布之间的差异。



### 模型优化

#### 判别器优化

判别器被定义为一个sigmoid函数，作用于输入节点对的内积：

$$
D\left(v, v_{c}\right)=\sigma\left(\mathbf{d}_{v}^{\top} \mathbf{d}_{v_{c}}\right)=\frac{1}{1+\exp \left(-\mathbf{d}_{v}^{\top} \mathbf{d}_{v_{c}}\right)}
$$

其中，$\mathbf{d}_{v}, \mathbf{d}_{v_{c}} \in \mathbb{R}^{k}$是节点$v$和节点$v_c$在判别器中的$k$维向量表示。$\theta_D$可以看作所有$\mathbf{d}_{v}$的集合。（判别器也可以用其他方法的模型，比如SDNE）。

对于训练判别器而言，根据目标函数，需要最大化判别器输出，可以采用梯度上升法更新对应的节点向量表示$\mathbf{d}_{v}, \mathbf{d}_{v_{c}}$。

$$
\nabla_{\theta_{D}} V(G, D)=\left\{\begin{array}{l}
\nabla_{\theta_{D}} \log D\left(v, v_{c}\right), \text { if } v \sim p_{\text {true }} \\
\nabla_{\theta_{D}}\left(1-\log D\left(v, v_{c}\right)\right), \text { if } v \sim G
\end{array}\right.
$$



#### 生成器优化

对于生成器而言，目标函数是min函数，因此可以通过梯度下降法去优化更新，生成器的梯度为：

$$
\begin{aligned}
& \nabla_{\theta_{G}} V(G, D) \\
=& \nabla_{\theta_{G}} \sum_{c=1}^{V} \mathbb{E}_{v \sim G\left(\cdot \mid v_{c}\right)}\left[\log \left(1-D\left(v, v_{c}\right)\right)\right] \\
=& \sum_{c=1}^{V} \sum_{i=1}^{N} \nabla_{\theta_{G}} G\left(v_{i} \mid v_{c}\right) \log \left(1-D\left(v_{i}, v_{c}\right)\right) \\
=& \sum_{c=1}^{V} \sum_{i=1}^{N} G\left(v_{i} \mid v_{c}\right) \nabla_{\theta_{G}} \log G\left(v_{i} \mid v_{c}\right) \log \left(1-D\left(v_{i}, v_{c}\right)\right) \\
=& \sum_{c=1}^{V} \mathbb{E}_{v \sim G\left(\cdot \mid v_{c}\right)}\left[\nabla_{\theta_{G}} \log G\left(v \mid v_{c}\right) \log \left(1-D\left(v, v_{c}\right)\right)\right]
\end{aligned}
$$

值得注意的是，根据上面的梯度计算，梯度$\nabla_{\theta_{G}} V(G, D)$可以看作是权重为$\log \left(1-D\left(v, v_{c} ; \theta_{D}\right)\right)$的梯度$\nabla_{\theta_{G}} \log G\left(v \mid v_{c}\right)$的期望求和。也就是说如果一个生成节点被识别出是负样本节点，那么概率$D\left(v, v_{c} ; \theta_{D}\right)$就会很小，生成的节点的梯度对应的权重就会很大，从而使整个梯度变大。

论文中生成器被定义为softmax函数（因为生成器学习概率分布）：

$$
G\left(v \mid v_{c}\right)=\frac{\exp \left(\mathbf{g}_{v}^{\top} \mathbf{g}_{v_{c}}\right)}{\sum_{v \neq v_{c}} \exp \left(\mathbf{g}_{v}^{\top} \mathbf{g}_{v_{c}}\right)}
$$

其中，$\mathbf{g}_{v}, \mathbf{g}_{v_{c}} \in \mathbb{R}^{k}$是节点$v$和节点$v_c$在生成器中的$k$维向量表示。$\theta_G$可以看作所有$\mathbf{g}_{v}$的集合。基于这样的设定，首先就可以按照上式先计算出近似连接分布$G\left(v \mid v_{c} ; \theta_{G}\right)$，之后根据这个概率分布进行随机采样可得到样本集合$(v,v_c)$，最后用SGD更新$\theta_G$。



#### Graph Softmax for Generator

Softmax为生成器中的连通分布提供了简洁而直观的定义，但是传统的softmax不适用于图表示学习：

- softmax对网络中所有节点都等价看待，只是完成了归一化的任务，忽略了网络的结构和邻接信息；
- softmax的计算涉及网络中所有节点，复杂度高；

常见的做法有分层softmax以及负采样的方法用来缓解计算开销，但都没有考虑到图的结构信息。像DeepWalk，Node2Vec这些算法，在随机游走的过程中已经捕获了图的结构信息，因此后续的softmax时只用负采样进行加速即可；而GraphGAN目前实际上是没有考虑到网络结构的，这样的话，学习得到的网络嵌入是没有意义的，因此作者尝试利用softmax去嵌入网络结构信息，提出了Graph Softmax。

Graph Softmax仍然用于计算近似的连接分布$G\left(\cdot \mid v_{c} ; \theta_{G}\right)$，但需要满足三个条件：

- Normalized：需要归一化来满足有效的概率分布，$\sum_{v \neq v_{c}} G\left(v \mid v_{c} ; \theta_{G}\right)=1$；
- Graph-structure-aware：利用网络结构信息，简单的想法就是，对于图中的两个节点，它们的连接概率应该随着其最短距离的增加而降低；
- Computationally efficient：$G\left(v \mid v_{c} ; \theta_{G}\right)$的计算，应该仅仅涵盖图中少量节点，比如一些与$v_c$连接较为密切的节点。

为了定义这样的Graph Softmax，论文首先对原网络进行BFS搜索，将之展开成以$v_c$为根节点的树$T_c$；$\mathcal{N}\left(v_{c}\right)$表示$v_c$的一阶邻居集合（在树中为父母节点和孩子节点）。给定节点$v$和邻居$v_{i} \in \mathcal{N}_{c}(v)$，定义$v_i$相对于$v$的关联概率为：

$$
p_{c}\left(v_{i} \mid v\right)=\frac{\exp \left(\mathbf{g}_{v_{i}}^{\top} \mathbf{g}_{v}\right)}{\sum_{v_{j} \in \mathcal{N}_{c}(v)} \exp \left(\mathbf{g}_{v_{j}}^{\top} \mathbf{g}_{v}\right)}
$$

每一个节点$v$都可以由根节点$v_c$开始的唯一一条路径达到，定义这个路径为$P_{v_{c} \rightarrow v}=\left(v_{r_{0}}, v_{r_{1}}, \ldots, v_{r_{m}}\right)$，其中$v_{r_0}=v_c$，$v_{r_m}=v$，Graph Softmax可以定义为：

$$
G\left(v \mid v_{c}\right) \triangleq\left(\prod_{j=1}^{m} p_{c}\left(v_{r_{j}} \mid v_{r_{j-1}}\right)\right) \cdot p_{c}\left(v_{r_{m-1}} \mid v_{r_{m}}\right)
$$

总而言之，这里就是通过一个简单的想法：将图展成树，然后根据路径不断地连乘；这样就可以使得节点之间的连通概率随着其最短距离的增加而降低。

> graph softmax需要满足三个条件，论文中给出了三个定理证明，详细见论文。



#### 生成器采样

生成式模型$G$最后还是要采样获得生成样本，一种简单的方法就计算出所有的graph softmax值，$G\left(v \mid v_{c} ; \theta_{G}\right)$，然后根据概率值进行加权随机采样。

论文中提出了另一种在线采样策略：从树$T_c$的根节点$v_c$开始随机游走，如果当前游走到的节点$v$下一步将游走回父母节点，则$v$即为被选择采样的节点。

下图展示的就是一个生成器$G$在线采样策略的图解：

![866cbf7c3d0fc72f60957dadd504714](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg-graphgan-sample.png)

最后，采样和整个graphGAN的伪代码如下：

![6cd7372d3721bc795639b96206b9c7f](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg-graphgan-code.png)