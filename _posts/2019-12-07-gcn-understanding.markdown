---
layout: article
title: GCN理解
date: 2019-12-07 00:08:00 +0800
tags: [GNN, Graph Embedding]
categories: blog
pageview: true
---



## Introduction

[Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907)，文章中提出的模型叫Graph Convolutional Network(GCN)，个人认为可以看作是图神经网络的“开山之作”，因为GCN利用了近似的技巧推导出了一个简单而高效的模型，使得图像处理中的卷积操作能够简单得被用到图结构数据处理中来，后面各种图神经网络层出不穷，或多或少都受到这篇文章的启发。



## Definition

图卷积网络(GCN)是一种在图上操作的神经网络。给定一个图$G=(E,V)$， 一个GCN的输入如下：

- $X_{N \times F}$ : 节点的特征矩阵，$N$为节点数量，$F$为特征维度；
- $A_{N \times N}$ : 邻接矩阵，描述图结构；

每层的输出$H^l$对应为$N \times F^l$的特征矩阵，$F^l$是第$l$层输出的节点特征维度。

GCN中的隐藏层可以写成 $H^{l+1} = f(H^{l}, A)$, 其中 $H^0 = X$, $f$是传播函数。在每一层，节点特征被聚合后再利用传播规则$f$形成下一层特征。

在该框架中GCN的变体仅在传播函数的选择上有所不同。



## Layer-wise linear model

一个简单逐层传播规则如下：

$$
f(H^{(l)},A) = \sigma(AH^{(l)}W^{(l)})
$$

其中$W^{(l)}_{F^i \times F^{i+1}}$是第$l$层的权重矩阵， $\sigma$是一个非线性激活函数（如ReLu）。权重矩阵的维度$F^i \times F^{i+1}$，表示第二维决定了下一层的特征个数。权重在节点之间共享，因此相当于CNN中的滤波操作。

以上简单模型存在两个问题：

- 节点的聚合，表示的只是其邻居节点特征的集合，不包括它自己的特征（除非存在自环）；
- $A$是**没有经过归一化**的矩阵，这样与特征矩阵相乘会改变特征原本的分布，大度的节点在特征表示中会有较大的值，而小度的节点将具有较小的值。这可能导致梯度消失或爆炸；

解决方法：

- 向每个结点添加自环。在应用传播规则之前奖单位矩阵添加到邻接矩阵中：$\hat A = A+I$；
- 将特征表示归一化。通过将邻接矩阵$A$与节点的度矩阵$D$的逆相乘，可以将特征表示按节点度进行归一化。实际中，利用对称归一化$D^{-\frac{1}{2}} A D^{-\frac{1}{2}} $；

结合两点技巧，可以得出 [Kipf & Welling](http://arxiv.org/abs/1609.02907) (ICLR 2017)中的传播规则：

结合自环、归一化操作，引入权重和激活函数，得到一个完整的隐藏层：

$$
f(H^{(l)}, A) = \sigma(\hat D^{-\frac{1}{2}} \hat A \hat D^{-\frac{1}{2}} H^{(l)}W^{(l)})
$$

其中 $\hat A = A+I$， $\hat D$为$\hat A$的度矩阵。

<center>    
    <img style="border-radius: 0.3125em;    
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"       src="http://ww1.sinaimg.cn/mw690/005NduT8ly1g9ocor4t1lj30cu09ftax.jpg">    
    <br>    
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    
    display: inline-block;    color: #999;   
    padding: 2px;">Fig 1: 空间信息的聚合</div> 
</center>

很显然模型可学习的参数是$W^l$，模型在每一层共享了用于特征增强的参数变化矩阵。矩阵的$W^l$两个维度分别是 （ $H^l$ 的第二个维度，根据特征增强需要设计的维度（是超参数））。很显然，这个矩阵维度与顶点数目或者每个顶点的度无关，于是说这是一个在同层内顶点上共享的参数矩阵。

优缺点：

- 优点：

  这样的共享方式，$W^l$ 的维度是可以进行调节的，与顶点的数目无关，使得模型可以用于大规模的图数据集。另一方面这个模型可以完成图结构训练在测试上不一样的任务;

- 缺点：

  这个模型对于同阶的邻域上分配给不同的邻居的权重是完全相同的（也就是 GAT 论文里说的无法 enable specifying different weights to different nodes in a neighborhood）。这一点限制了模型对于空间信息的相关性的捕捉能力，这也是在很多任务上不如 GAT 的根本原因。

  

## Feature Aggregation

### Sum Rule

将邻近节点的特征值相加的特征聚合称为 Sum Rule.

$$
{\sf{aggregate}}(A,X)_i = A_iX = \sum_{j}^{N} A_{i,j}X_j
$$

每个节点获得的特征聚合仅仅取决于邻接矩阵定义的邻域。

### Mean Rule

使用邻居节点的标准化特征的加和作为特征聚合称为Mean Rule.

$$
{\sf{aggregate}}(A,X)_i = D^{-1}A_iX = \sum_{k=1}^{N}D_{i,k}^{-1}\sum_{j=1}^{N}A_{i,j}X_{j}
$$

### Spectral Rule

基于频谱的聚合规则，利用对称归一化：

$$
{\sf{aggregate}}(A,X)_i = D^{-0.5}A_iD^{-0.5}X \\
=\sum_{k=1}^{N} D^{-0.5}_{i,k} \sum_{j=1}^{N}A_{i,j} \sum_{l=1}^{N} D^{-0.5}_{j,l} X_{j} \\
= \sum_{j=1}^{N} D^{-0.5}_{i,i} A_{i,j} D^{-0.5}_{j,j} X_{j} \\
= \sum_{j=1}^{N} \frac{1}{D^{0.5}_{i,i}} A_{i,j} \frac{1}{D^{0.5}_{j,j}} X_{j}
$$

在计算第$i$个节点的聚合特征时，我们不仅要考虑第$i$个节点的度，还要考虑第$j$个节点的度。类似于Mean Rule，Spectral Rule对聚合进行归一化，使得聚合特征大致保持与输入特征相同的比例。当邻居节点的度很低时，spectral rule会更多地考虑将邻居节点的特征权重增大，反之亦然。这样在小度的邻居节点比大度的邻居节点提供的信息更有用时，这种规则是很有效的。


## 从Graph上的热传播模型理解GCN

问题的本质：**图中的每个结点无时无刻不因为邻居和更远的点的影响而在改变着自己的状态直到最终的平衡，关系越亲近的邻居影响越大。**

要想理解GCN以及其后面一系列工作的实质，最重要的是理解其中的精髓**Laplacian矩阵**在干什么。知道了Laplacian矩阵在干什么后，剩下的只是**解法的不同**——所谓的Fourier变换只是将问题从空域变换到频域去解，所以也有直接在空域解的（例如GraphSage）。

为了让问题简单便于理解，先让我们忘记时域、频域这些复杂的概念，从一个最简单的物理学现象————热传导出发。

### Graph上的热传播模型

众所周知，没有外接干预的情况下，热量从温度高传播到温度低的地方并且不可逆，根据著名的牛顿冷却定律（Newton Cool's Law），**热量传递的速度正比于温度梯度**，直观上也就是某个地方A温度高，另外一个B地方温度低，这两个地方接触，那么温度高的地方的热量会以正比于他们俩温度差的速度从A流向B。

#### 一维热传播模型

我们先建立一个一维的温度传播模型，假设有一个均匀的铁棒，不同位置温度不一样，现在我们刻画这个铁棒上面温度的热传播随着时间变化的关系。预先说明一下，一个连续的铁棒的热传播模型需要列**温度对时间和坐标的偏微分方程**来解决，为了简化问题，我们把**空间离散化**，假设铁棒是一个**一维链条**，链条上每一个单元拥有一致的温度，温度在相邻的不同的单元之间传播，如下图：

<center>    
    <img style="border-radius: 0.3125em;    
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"       src="http://ww1.sinaimg.cn/mw690/005NduT8ly1g9og2tyv91j30j102zjsy.jpg">    
    <br>    
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    
    display: inline-block;    color: #999;   
    padding: 2px;">一维离散链条上的热传播</div> 
</center>

对于第$i$个单元，只接受相邻单元$i-1$和$i+1$传来的热量，假设它当前的热量为$\phi_i$，则有：

$$
\frac{d\phi_i}{dt} = k(\phi_{i+1}-\phi_i) - k(\phi_{i}-\phi_{i-1}) \\
\Rightarrow \\
\frac{d\phi_i}{dt} -k[(\phi_{i+1}-\phi_i) - (\phi_{i}-\phi_{i-1})]
$$

$k$为常数。注意到第二项为两个差分的差分，在离散空间中，相邻位置的差分推广到连续空间就是导数，差分的差分就是二阶导数。

因此我们可以推出一维空间中的热传导方程为如下，其中在高维欧式空间中，一阶导数推广到**梯度**，二阶导数就是**拉普拉斯算子**：

$$
\frac{\partial \phi_i}{\partial t} - k \frac{\partial^2 \phi}{\partial x^2} = 0 \\
\Rightarrow \\
\frac{\partial \phi_i}{\partial t} - k \Delta \phi = 0
$$

其中$\Delta$的主流写法为$\nabla^2$, 代表拉普拉斯算子，是对各个坐标二阶导数的加和，如三维空间中的拉普拉斯表达形式为 $\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} +\frac{\partial^2}{\partial z^2}$。

> 拉普拉斯算子的物理意义：**其实就是针对空间标量函数的一种“操作”，即先求该标量函数的梯度场，再求梯度场的散度！**
>
> 为什么标量函数梯度场的散度这么重要？
>
> "因为标量函数的梯度往往是一种“驱动力”（或者叫“势”），而对于“驱动力”求散度就可以知道空间中“源”的分布。

综上所述，我们可以发现：

- 欧氏空间中，某个点温度升高的速度正比于该点周围温度的分布，可以用拉普拉斯算子衡量 $\frac{\partial \phi_i}{\partial t} = k \Delta \phi$。

- 拉普拉斯算子，是二阶导数对高维空间的推广。



#### 拓扑空间（Graph）的热传播模型

现在我们考虑Graph上的热传播模型。图上的每个结点是一个单元，且只与该单元相连的节点（单元），即存在边连接的单元进行热交换。

<center>    
    <img style="border-radius: 0.3125em;    
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"       src="http://ww1.sinaimg.cn/mw690/005NduT8ly1g9oh7bkiqlj309c06y758.jpg">    
    <br>    
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    
    display: inline-block;    color: #999;   
    padding: 2px;">图上的热传播</div> 
</center>

我们假设热量流动的速度依旧满足牛顿冷却定律，对任意节点$i$，其温度随时间变化可用以下公式刻画：

$$
\frac{d \phi_i}{dt} = -k \sum_j A_{ij}(\phi_i - \phi_j)
$$

在这里$A_{ij}$为图的邻接矩阵，简化问题，考虑图为无权、无向、无自环图：

$$
\frac{d \phi_i}{dt} = -k[\phi_i \sum_j A_{ij} - \sum_j A_{ij} \phi_j] \\
= -k [\phi_i \cdot  deg(i) -  \sum_j A_{ij} \phi_j]
$$

括号内第二项，可以看作邻接矩阵第$i$行与所有节点温度作内积。

对于所有节点，我们写成向量形式有：

$$
\left[\begin{array}{c}{\frac{d \phi_{1}}{d t}} \\ {\frac{d \phi_{2}}{d t}} \\ {\cdots} \\ {\frac{d \phi_{n}}{d t}}\end{array}\right]=-k\left[\begin{array}{c}{\operatorname{deg}(1) \times \phi_{1}} \\ {\operatorname{deg}(2) \times \phi_{2}} \\ {\ldots} \\ {\operatorname{deg}(n) \times \phi_{n}}\end{array}\right]+k A\left[\begin{array}{c}{\phi_{1}} \\ {\phi_{2}} \\ {\cdots} \\ {\phi_{n}}\end{array}\right]
$$


定义向量 $\phi = [\phi_1, \phi_2, \dots, \phi_n]^T$ ，则有：

$$
\frac{d \phi}{dt} = -k D \phi + kA \phi = -k(D - A) \phi
$$


其中 $D=diag(deg(1), deg(2), \ldots , deg(n))$ 为度矩阵，我们可以得到：

$$
\frac{d \phi}{dt} +kL \phi = 0
$$

其中 $L=D-A$ ，对比在连续欧式空间中的微分方程：

$$
\frac{\partial \phi}{\partial t} - k \Delta \phi = 0
$$


两者具有一样的形式。对比两者关系：

- 相同点：均刻画了空间温度分布随时间的变化，该变化满足相同形式的微分方程。
- 不同点：前者刻画拓扑空间中有限节点，用向量 $\phi$ 刻画当前状态，单位时间内状态的变化正比于线性变换算子 $L$ 作用在状态 $\phi$ 上； 后者刻画欧式空间的连续分布，用函数 $\phi (x,t)$ 刻画当前状态， 单位时间内状态的变化正比于拉普拉斯算子 $\Delta$ 作用在状态 $\phi$ 上。

不难发现这本质上是 **同一种变换、同一种关系在不同空间上的体现**。我们将这种线性变换算子$L$推广到拓扑空间，即成为图上的拉普拉斯算子。

> 其实，欧式空间中的离散链条上的热传导，如果将链条看成图，也可以用图上的热传导方程刻画，此时邻接矩阵只有主对角线上下的两条线上的值为1，即相邻两节点连接，头尾不连接。
> 
> $$
> \left[\begin{array}{llll}{0} & {1} & {0} & {0} \\ {1} & {0} & {1} & {0} \\ {0} & {1} & {0} & {1} \\ {0} & {0} & {1} & {0}\end{array}\right]
> $$



### 推广到GCN

给定一个空间，空间中某种概念可以在该空间中流动，速度正比于相邻位置之间的状态差。

此时，将概念从“热量”扩展到特征（feature）、消息（message），那么问题可以自然的被推广到GCN。

所以GCN本质上用于刻画，一张Graph network中特征和消息的传播和流动规律。这个传播的初始状态就是状态的变化正比于相应空间中拉普拉斯算子在当前状态上的作用。

Laplacian矩阵/算子表现的不仅是一种二阶导数的运算，另一方面，它也表现出了一种加和性，这个从图上热/消息传播方程最原始的形态就能一目了然：


$$
\frac{d \phi_i}{dt} = -k \sum_j A_{ij}(\phi_i - \phi_j)
$$


每个结点每一时刻的状态变化，就是所有邻居对该结点作用效果的总和，即所有的邻居将消息汇聚到该结点作aggregation。实际的建模中，aggregate不一定是sum pooling，也可以是attention、lstm等。



**两种空间下的Fourier分解/ 特征分解对比（卷积为什么能从欧氏空间推广到Graph空间）**。

首先， 在欧氏空间上，迭代求解的拉普拉斯Δ算子的特征函数是： $e^{-jwt}$ ，因为 $\Delta e^{-jwt} = w^2 e^{-jwt}$ 。

在图空间上，对应地位的拉普拉斯算子对应的不是特征函数，而是它的特征值和特征向量，分别对应上面的$W^2$和$e^{-jwt}$。唯一的区别是，$e^{-jwt}$是定义内积为函数积分的空间上的一组基，有无穷多个，当然这无穷多个是完备的。而$L$的特征空间是有限维的，因为$L$的对称性和半正定性，所有特征值都是实数且特征向量都是实向量，特征空间维度等于图中的节点数也就是$L$的阶。

注意，实对称矩阵的特征空间的所有基能够张出整个线性空间且它们两两正交，所以无论是拉普拉斯算子 $\Delta$ 还是拉普拉斯矩阵$L$，它们的特征空间是一个满秩且基两两正交的空间，所以把欧氏空间的$e^{-jwt}$推广到图空间的$L$的这一组特征向量，**正是同一个关系（Message Passing），同一种变换，在不同空间下的体现！**

> 三种形式的图拉普拉斯矩阵 [wikipedia](https://en.wikipedia.org/wiki/Laplacian_matrix)：
>
> 1.  $L=D-A$ 
> 2.  Symmetric normalized Laplacian： $L^{\mathrm{sym}}:=D^{-\frac{1}{2}} L D^{-\frac{1}{2}}=I-D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$
> 3.  Random walk normalized Laplacian： $L^{\mathrm{rw}}:=D^{-1} L=I-D^{-1} A$



### 恒温热源 到 半监督学习

之前所讨论的热传导场景是一个**孤立系统的无源**场景，每一个结点的热量都有初始值，无外界干预，那么根据**能量守恒定律**，整个系统的能量是定值，并且随着热传导，系统的热量被均匀分配到每个参与其中的结点。这正是**热力学第一定律**和**热力学第二定律**在这张图上的完美体现。这也是这张图上的“宇宙热寂说”，这个孤立的小宇宙，最终达到处处温度一样，平衡状态下没有热量的流动。

假设这个图是个连通图，每个结点到任一其他结点都有路可走，那么这样的事情肯定会发生：如果一个结点温度高于周围邻居，那么它的热量会持续流出到邻居，此间它自己的热量流失，温度就会降低，直到它和周围邻居一样，最终所有结点都达到一个平衡的、一样的温度。

如果我们在这张图上引入一个东西叫做**恒温热源**，事情就会发生改变。所谓恒温热源，就是这样一个存在，系统中有些点，它们接着系统外面的某些“供热中心”，使得它们温度能够保持不变。这样，这些结点能够源源不断的去从比它温度高的邻居接受热量，向温度比它低的邻居释放热量，每当有热量的增补和损失，“供热中心”便抽走或者补足多余或者损失的热量，使得这个结点温度不发生变化。

那么最终的平衡的结果不是无源情况下所有结点温度一样，而是整个图中，高温热源不断供热给周围的邻居，热量传向远方，低温热源不断持续吸收邻居和远方的热量，最终状态下每条边热量稳恒流动（不一定为0），流量不再变化，每个结点都达到**终态的某个固有温度**。这个温度具体是多少，取决于整张图上面**恒温热源的温度和它们的分布**。

**恒温热源的存在，正是我们跨向半监督学习的桥梁**。

我们把问题向广告和推荐场景去类比，在这个场景下，某些结点有着明确的label，例如某个广告被点击，某个广告的ctr是多少，某个item被收藏，这些带着label的结点有着相对的确定性————它们可以被看作是这个结点的温度，这些**有标签的结点正是恒温热源**。那么，这些图的其他参与者，那些等待被我们预测的结点正是这张图的**无源部分**，它们被动的接受着那些或近或远的恒温热源传递来的Feature，改变着自己的Feature，从而影响自己的label。

让我们做个类比：

- **温度** 好比 **label**， 是状态量
- **恒温热源 好比 有明确 label的结点**，它们把自己的Feature传递给邻居，让邻居的Feature与自己趋同从而让邻居有和自己相似的label
- 结点的**热能**就是**Feature**，能量表现为温度，feature表现为label。无标签的、被预测的结点的Feature被有明确label的结点的Feature传递并且影响最终改变自己。

需要说明的一点是，无论是有源还是无源，当有新的结点接入已经平衡的系统，系统的平衡被打破，新的结点最终接受已有结点和恒温热源的传递直到达到新的平衡。**所以我们可以不断的用现有的图去预测我们未见过的结点的性质，演化和扩充这个系统**。





## Reference

1. http://tkipf.github.io/graph-convolutional-networks/
2. [GNN 简介和入门资料](https://zhuanlan.zhihu.com/p/58792104)
3. 直觉理解 Graph Laplacian: [Quora](https://link.zhihu.com/?target=https%3A//www.quora.com/Whats-the-intuition-behind-a-Laplacian-matrix-Im-not-so-much-interested-in-mathematical-details-or-technical-applications-Im-trying-to-grasp-what-a-laplacian-matrix-actually-represents-and-what-aspects-of-a-graph-it-makes-accessible) 中 Muni Sreenivas Pydi 的回答
4. [**GNN 教程：GCN**](https://archwalker.github.io/blog/2019/06/01/GNN-Triplets-GCN.html)
5. [解读三种经典 GCN 中的参数共享](https://mp.weixin.qq.com/s?__biz=MzI2MDE5MTQxNg==&mid=2649698061&idx=1&sn=022d2f83fd8c574243576cfc8623c093&chksm=f276eadac50163cc10fdd0b2ccc8de67dcd36a0da188f2403f7e94fad4d2f6b1434807e7224c&mpshare=1&scene=24&srcid=&sharer_sharetime=1572324680307&sharer_shareid=70077e3eae7dee7b54d8ce0254b229dc&key=e65b857c38f0c6ecd0e022614d712fa583eb9c0311f3ab89d11bbf35593cc1b2202e7d57d039e7974c6c4ed28006e9187556b666cd1ef0c188e98bf400eb61d5d8db4a638c6882ddb10044f275a08de2&ascene=14&uin=NjkxMzE4MjM2&devicetype=Windows+10&version=62070158&lang=zh_CN&exportkey=A0%2BN9kGKf%2FkxixrwtDJDwL4%3D&pass_ticket=bQfSVXZ3i4Ve1CAy00gH0S5KBfleGu263MVJFFiLjOV%2F%2FDaMhKsEXbZGjxH62BtT)
6. [从源头探讨 GCN 的行文思路](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247490623&idx=3&sn=34738ea858f7533fc9d5b951ec59bba4&chksm=ebb424ebdcc3adfd9647881f51135ff7b067bb37ac19c323e2ad8dad626bcc48fa7e8f63d98b&mpshare=1&scene=24&srcid=&sharer_sharetime=1574732101852&sharer_shareid=70077e3eae7dee7b54d8ce0254b229dc&key=0ea499742c0fecddf1d95bcff553d7a5849f02df82205a3dadc000a625a54f276123ca839a4714cc0585bd384583b54e5980460e1a66e91f205a7d776dbc61ed0478d05c43e950c891a2012bc66f6757&ascene=14&uin=NjkxMzE4MjM2&devicetype=Windows+10&version=62070158&lang=zh_CN&exportkey=A%2FVTd850xyfPmStcNmE7J5E%3D&pass_ticket=bQfSVXZ3i4Ve1CAy00gH0S5KBfleGu263MVJFFiLjOV%2F%2FDaMhKsEXbZGjxH62BtT)
7. [GCN图卷积网络本质理解](https://luweikxy.gitbooks.io/machine-learning-notes/content/content/deep-learning/graph-neural-networks/graph-convolutional-networks/gcn-essential-understand.html)
8. [为什么空间二阶导（拉普拉斯算子）这么重要？](https://www.zhihu.com/question/26822364)
9. [拉普拉斯矩阵与拉普拉斯算子的关系](https://zhuanlan.zhihu.com/p/85287578)

