---
layout: article
title: 图神经网络：图注意力网络(GAT)
date: 2020-01-28 00:17:00 +0800
tags: [GNN, Graph, Deep Learning]
categories: blog
pageview: true
key: Graph-Attention-Networks
---



------

- 论文链接：[https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- tensorflow版本： [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)
- keras版本：[https://github.com/danielegrattarola/keras-gat](https://github.com/danielegrattarola/keras-gat)
- pytorch版本：[https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT)
- for link prediction: [https://github.com/raunakkmr/GraphSAGE-and-GAT-for-link-prediction](https://github.com/raunakkmr/GraphSAGE-and-GAT-for-link-prediction)



## 引言

图卷积发展至今，早期的进展可以归纳为谱图方法和非谱图方法，这两者都存在一些挑战性问题。

- 谱图方法：学习滤波器主要基于图的拉普拉斯特征，图的拉普拉斯取决于图结构本身，因此在特定图结构上学习到的谱图模型无法直接应用到不同结构的图中。
- 非谱图方法：对不同大小的邻域结构，像CNNs那样设计统一的卷积操作比较困难。

此外，图结构数据往往存在大量噪声，换句话说，节点之间的连接关系有时并没有特别重要，节点的不同邻居的相对重要性也有差异。

本文提出了图注意力网络（GAT），利用masked self-attention layer，通过堆叠网络层，获取每个节点的邻域特征，为邻域中的不同节点分配不同的权重。这样做的好处是不需要高成本的矩阵运算，也不用事先知道图结构信息。通过这种方式，GAT可以解决谱图方法存在的问题，同时也能应用于归纳学习和直推学习问题。

## 模型架构

假设一个图有$N$个节点，节点的$F$维特征集合可以表示为$$
\mathbf{h}=\left\{\vec{h}_{1}, \vec{h}_{2}, \ldots, \vec{h}_{N}\right\}, \vec{h}_{i} \in \mathbb{R}^{F}
$$。注意力层的目的是输出新的节点特征集合，$$
\mathbf{h}^{\prime}=\left\{\vec{h}_{1}^{\prime}, \vec{h}_{2}^{\prime}, \ldots, \vec{h}_{N}^{\prime}\right\}, \vec{h}_{i}^{\prime} \in \mathbb{R}^{F^{\prime}}
$$。

在这个过程中特征向量的维度可能会改变，即$$F \rightarrow F^{\prime}$$。为了保留足够的表达能力，将输入特征转化为高阶特征，至少需要一个可学习的线性变换。例如，对于节点$i,j$，对它们的特征$$\vec{h}_{i},\vec{h}_{j}$$应用线性变换$$
\mathbf{W} \in \mathbb{R}^{F^{\prime} \times F}
$$，从$F$维转化为$F^{\prime}$维新特征$\vec{h}_{i}^{\prime},\vec{h}_{j}^{\prime}$：


$$
e_{i j}=a\left(\mathbf{W} \vec{h}_{i}, \mathbf{W} \vec{h}_{j}\right)
$$

上式在将输入特征运用线性变换转化为高阶特征后，使用self-attention为每个节点分配注意力（权重）。其中$a$表示一个共享注意力机制：$$
\mathbb{R}^{F^{\prime}} \times \mathbb{R}^{F^{\prime}} \rightarrow \mathbb{R}
$$，用于计算注意力系数$e_{ij}$，也就是节点$i$对节点$j$的影响力系数（标量）。

上面的注意力计算考虑了图中任意两个节点，也就是说，图中每个节点对目标节点的影响都被考虑在内，这样就损失了图结构信息。论文中使用了masked attention，对于目标节点$i$来说，只计算其邻域内的节点$j\in \mathcal{N}$对目标节点的相关度$e_{ij}$（包括自身的影响）。

为了更好的在不同节点之间分配权重，我们需要将目标节点与所有邻居计算出来的相关度进行统一的归一化处理，这里用softmax归一化：


$$
\alpha_{i j}=\operatorname{softmax}_{j}\left(e_{i j}\right)=\frac{\exp \left(e_{i j}\right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left(e_{i k}\right)}
$$


关于$a$的选择，可以用向量的内积来定义一种无参形式的相关度计算$$\langle \mathbf{W} \vec{h}_{i}\ , \mathbf{W} \vec{h}_{j} \rangle$$，也可以定义成一种带参的神经网络层，只要满足$a:R^{d^{(l+1)}} \times R^{d^{(l+1)}} \rightarrow R$，即输出一个标量值表示二者的相关度即可。在论文实验中，$a$是一个单层前馈神经网络，参数为权重向量$$
\overrightarrow{\mathrm{a}} \in \mathbb{R}^{2 F^{\prime}}
$$，使用负半轴斜率为0.2的[LeakyReLU](https://blog.csdn.net/sinat_33027857/article/details/80192789)作为非线性激活函数：


$$
e_{ij} = \text { LeakyReLU }\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \Vert \mathbf{W} \vec{h}_{j}\right]\right)
$$


其中$\Vert$表示拼接操作。完整的权重系数计算公式为：


$$
\alpha_{i j}=\frac{\exp \left(\text { LeakyReLU }\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \| \mathbf{W} \vec{h}_{j}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left(\text { LeakyReLU }\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \| \mathbf{W} \vec{h}_{k}\right]\right)\right)}
$$


得到归一化注意系数后，计算其对应特征的线性组合，通过非线性激活函数后，每个节点的最终输出特征向量为：


$$
\vec{h}_{i}^{\prime}=\sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j} \mathbf{W} \vec{h}_{j}\right)
$$

<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gbdrzbl2cbj30en0g4dgj.jpg" alt="41d5df1d6897352944509c363f5235a.png" style="zoom:50%;" />

### 多头注意力机制

另外，本文使用多头注意力机制（multi-head attention）来稳定self-attention的学习过程，即对上式调用$K$组相互独立的注意力机制，然后将输出结果拼接起来：


$$
\vec{h}_{i}^{\prime}=\Vert_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \vec{h}_{j}\right)
$$


其中$\Vert$是拼接操作，$\alpha_{ij}^{k}$是第$k$组注意力机制计算出的权重系数，$W^{(k)}$是对应的输入线性变换矩阵，最终输出的节点特征向量$\vec{h}_{i}^{\prime}$包含了$KF^{\prime}$个特征。为了减少输出的特征向量的维度，也可以将拼接操作替换为平均操作。


$$
\vec{h}_{i}^{\prime}=\sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \vec{h}_{j}\right)
$$


下面是$K=3$的多头注意力机制示意图。不同颜色的箭头表示不同注意力的计算过程，每个邻居做三次注意力计算，每次attention计算就是一个普通的self-attention，输出一个$\vec{h}_{i}^{\prime}$，最后将三个不同的$\vec{h}_{i}^{\prime}$进行拼接或取平均，得到最终的$\vec{h}_{i}^{\prime}$。





<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gbdrzzxu0ij30kh0d6gn1.jpg" alt="ac8522cab32f909278c8b1957eb8c93.png" style="zoom:50%;" />



## 比较

- GAT计算高效。self-attetion层可以在所有边上并行计算，输出特征可以在所有节点上并行计算；不需要特征分解或者其他内存耗费大的矩阵操作。单个head的GAT的时间复杂度为$$O\left(\mid V\mid F F^{\prime}+\mid E\mid F^{\prime}\right)$$。
- 与GCN不同的是，GAT为同一邻域中的节点分配不同的重要性，提升了模型的性能。
- 注意力机制以共享的方式应用于图中的所有边，因此它不依赖于对全局图结构的预先访问，也不依赖于对所有节点(特征)的预先访问(这是许多先前技术的限制)。
  - 不必要无向图。如果边$i\rightarrow j$不存在，可以忽略计算$e_{ij}$;
  - 可以用于归纳学习；

## 评估

### 数据集

![f69da423fa5b0343249e3ccf1792b4f.png](http://ww1.sinaimg.cn/large/005NduT8ly1gbdsqvcbfgj30yg0bdacb.jpg)

其中前三个引文网络用于直推学习，第四个蛋白质交互网络PPI用于归纳学习。

### 实验设置

- 直推学习

  - 两层GAT模型，第一层多头注意力$K=8$，输出特征维度$F^{\prime}=8$（共64个特征），激活函数为指数线性单元（ELU）；
  - 第二层单头注意力，计算$C$个特征（$C$为分类数），接softmax激活函数；

  - 为了处理小的训练集，模型中大量采用正则化方法，具体为L2正则化；
  - dropout；

- 归纳学习：

  - 三层GAT模型，前两层多头注意力$K=4$，输出特征维度$F^{\prime}=256$（共1024个特征），激活函数为指数非线性单元（ELU）；
  - 最后一层用于多标签分类，$K=6$，每个头计算121个特征，后接logistic sigmoid激活函数；
  - 不使用正则化和dropout；
  - 使用了跨越中间注意力层的跳跃连接。
  - batch_size = 2 graph

### 实验结果

- 直推学习

    ![3fda1993bd23023403b90204936a2d3.png](http://ww1.sinaimg.cn/large/005NduT8ly1gbdsxphq5uj30ya0hlq71.jpg)



- 归纳学习

  ![99018f1ffd1351b1acf4c90cf7e1ff9.png](http://ww1.sinaimg.cn/large/005NduT8ly1gbdsz19osbj30yt0gsgp0.jpg)



GAT layer            |  t-SNE + Attention coefficients on Cora
:-------------------------:|:-------------------------:
![](http://www.cl.cam.ac.uk/~pv273/images/gat.jpg)  |  ![](http://www.cl.cam.ac.uk/~pv273/images/gat_tsne.jpg)

