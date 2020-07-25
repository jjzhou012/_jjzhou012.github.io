---
layout: article
title: 图的对抗攻击：Adversarial Attack on Community Detection by Hiding Individuals
date: 2020-07-24 00:21:00 +0800
tags: [GNN, Graph, Adversarial, Community Detection]
categories: blog
pageview: true
key: Adversarial-Attack-on-Community-Detection-by-Hiding-Individuals
---

------

- Paper: [Adversarial Attack on Community Detection by Hiding Individuals](https://arxiv.org/pdf/2001.07933)
- Code: [https://github.com/halimiqi/CD-ATTACK](https://github.com/halimiqi/CD-ATTACK)



## 一、概述

该论文研究社区检测任务的对抗攻击问题。利用黑盒攻击来隐藏目标群体，降低基于深度模型的社区检测方法的性能。

- 首次提出了针对不重叠社区检测模型的对抗攻击；
- 提出了一种基于图学习的社区检测模型，作为攻击的替代模型，可以用于解决一般的无监督、不重叠的社区检测问题；



## 二、问题描述

- 图的基本定义

  节点集合：$$V=\left\{v_{1}, v_{2}, \ldots, v_{N}\right\}$$

  邻接矩阵：$$A$$

  特征集合：$$X=\left\{x_{1}, x_{2}, \ldots, x_{N}\right\}$$

- 社区检测定义

  给定一个图$$G=(V,A,X)$$，将其划分为$$K$$个不重叠的子图$$G_{i}=\left(V_{i}, A_{i}, X_{i}\right), i=1, \ldots, K$$，其中$$V=\cup_{i=1}^{K} V_{i}$$，$$V_{i} \cap V_{j}=\emptyset$$，($$i\neq j$$)。

- 社区检测的攻击

  对于一种社区检测方法$$f$$，有一个目标群体$$C^{+} \subseteq V$$想要在社区检测算法面前隐藏。攻击旨在于定义一个攻击函数$$g$$，在图$$G=(V,A,X)$$中加入一些扰动，生成新的图$$\hat{G}=(\hat{V}, \hat{A}, \hat{X})$$，满足：
  
  $$
  \begin{array}{c}
  \max \mathcal{L}\left(f(\hat{G}), C^{+}\right)-\mathcal{L}\left(f(G), C^{+}\right) \\
  \text {s.t. } \hat{G} \leftarrow \arg \min g\left(f,\left(G, C^{+}\right)\right) \\
  Q(G, \hat{G})<\epsilon
  \end{array}
  $$
  
  其中

  - $$\mathcal{L}(\cdot, \cdot)$$用于衡量关于目标群体$$C^{+}$$的社区划分结果的质量；
  - $$Q(G, \hat{G})<\epsilon$$用于约束扰动大小；
  - 文中的扰动主要是边扰动，即边的增删，$$\hat{G}=(\hat{V}, \hat{A}, \hat{X})=(V, \hat{A}, X)$$

  整个攻击的目的在于，通过在图中加入微小的扰动，来最大程度的降低社区检测算法关于目标子集的检测性能。

![image-20200725142711056](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200725174945.png)



## 三、方法

### 3.1 框架

框架主要由两个模块构成：

- 对抗攻击模型$$g(\cdot)$$：生成攻击扰动，隐藏目标群体；
- 社区检测模型$$f(\cdot)$$：以无监督的方式对网络进行划分；

这两个模块有很强的交互性，攻击模块需要检测模块的反馈，来判断是否达到攻击要求；检测模块依赖攻击模块来增强它的鲁棒性。

对于图结构数据的离散导致梯度难以计算，作者通过利用Actor-Critic算法中的策略梯度作为两个模块间的交互信号。

对于黑盒攻击设置，意味着没有具体的社区检测算法可以利用，作者通过设计具有高泛化性和鲁棒性的替代模型来实例化攻击对象。

![image-20200725145315918](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200725174941.png)

论文设计了上图所示的可迭代框架，主要由两个模块构成。上半部分生成对抗性图，下半部分对替代检测模型的最优值进行优化。然而，当实例化这两种神经网络时，作者面临如下三个挑战：

- 替代社区检测模型。如何设计一个替代社区检测模型，使生成的对抗性图也适用于其他社区检测算法?
- 微小的扰动。应该用什么标准来确定修改的扰动大小，以至于检测模块无法察觉?
- 约束性的图生成方式。如何有效地生成满足小扰动要求的对抗性图?



### 3.2 社区检测的替代模型

在设计社区检测模型时有两个关键问题:

- 评价结果质量的距离函数，相当于神经网络中的损失函数；
- 检测方法，这个相当于神经网络结构；

#### 3.2.1 损失函数

将[normalized cut](https://blog.csdn.net/Leoch007/article/details/80206759)泛化为损失函数。归一化割衡量图划分移除的群体的边体积：


$$
\frac{1}{K} \sum_{k} \frac{\operatorname{cut}\left(V_{k}, \overline{V_{k}}\right)}{\operatorname{vol}\left(V_{k}\right)}
$$


其中$$\operatorname{vol}\left(V_{k}\right)=\sum_{i \in V_{k}} \operatorname{degree}(i)$$，$$\overline{V_{k}}=V \backslash V_{k}$$，$$\operatorname{cut}\left(V_{k}, \overline{V_{k}}\right)=\sum_{i \in V_{k}, j \in \overline{V_{k}}} A_{i j}$$，分子计算社区$$V_k$$和剩余图的边连接数，分母计算社区$$V_k$$中节点的关联边数。

令$$C \in \mathbb{R}^{N \times K}$$表示节点的社区分配矩阵，$$C_{i k}=1$$表示节点$$i$$属于$$k$$社区。由于本文考虑的是非重叠社区的检测问题，这里可以存在一个显示的约束：$$C^{\top} C$$为对角矩阵。		

根据社区分配矩阵，归一化割可以表示为：


$$
\frac{1}{K} \sum_{k} \frac{C_{:, k}^{\top} A\left(1-C_{:, k}\right)}{C_{: k}^{\top} D C_{:, k}}
$$
其中，$$D$$为对角矩阵，$$C_{:, k}$$是$$C$$的第$$k$$列。公式（3）减去1得到下列简化的目标函数：


$$
- \frac{1}{K} \sum_{k} \frac{C_{:, k}^{\top} A C_{:, k}}{C_{;, k}^{\top} D C_{:, k}}=-\frac{1}{K} \operatorname{Tr}\left(\left(C^{\top} A C\right) \oslash\left(C^{\top} D C\right)\right)
$$


其中，$$\oslash$$表示矩阵元素除法，$$\operatorname{Tr}(\cdot)$$表示给定方阵主对角线上元素的和。

由于$$C^{\top} C$$被约束为对角矩阵，我们在目标函数中引入一个惩罚项：


$$
\mathcal{L}_{u}=-\frac{1}{K} \operatorname{Tr}\left(\left(C^{\top} A C\right) \oslash\left(C^{\top} D C\right)\right)+\gamma\left\|\frac{K}{N} C^{\top} C-I_{K}\right\|_{F}^{2}
$$


其中，$$\|\cdot\|_{F}$$表示矩阵的2范数，$$\frac{K}{N}$$用于归一化割平衡聚类和空隙缩小偏差。



#### 3.2.2 网络结构

利用GNN实现社区检测，网络架构分为两个部分：

![image-20200725174930389](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200725174930.png)

- 节点嵌入：利用gnn获得节点嵌入；

  这里使用一个两层的GCN：
  $$
  H_{l}=\bar{A} \sigma\left(\bar{A} X W^{0}\right) W^{1}
  $$
  
- 社区分配：将相似的节点分配到相同社区；

  这里使用两层全连接层：
  $$
  C=\operatorname{softmax}\left(\sigma\left(H_{l} W_{c 1}\right) W_{c 2}\right)
  $$
  其中，$$W_{c 1} \in \mathbb{R}^{v \times r}$$，$$W_{c 2} \in \mathbb{R}^{r \times K}$$。



### 3.3 微小的扰动

对抗图应该在实现攻击效果的同时，尽可能得与原图相似，以保证攻击的隐蔽性。

作者在这里定义了一个扰动损失函数来衡量扰动的程度：
$$
\mathcal{L}_{\text {perturb}}=\sum_{v_{i}} K L\left(E N C\left(v_{i} \mid G\right) \| E N C\left(v_{i} \mid \hat{G}\right)\right)
$$
$$K L(\cdot \| \cdot)$$表示KL散度，$$E N C(\cdot)$$通过以无监督的方式捕获一些近似性来学习节点的嵌入表示。我们在此介绍用于编码节点表示的两种近似形式:

- Local proximity：给定节点对应关系，如果两个图中的对应节点的邻居相似，那么这对节点也相似。
- Global proximity：给定节点对应关系，如果两个图中的相关节点与其他所有节点的关系相似，那么它们就是相似的。

论文使用个性化PageRank来计算Global proximity。



### 3.4 图生成约束

利用一个潜变量模型来生成新的图：
$$
P(\hat{A} \mid A, X)=\int q_{\phi}(Z \mid A, X) p_{\theta}(\hat{A} \mid A, X, Z) d Z
$$
其中，$$q_{\phi}(Z \mid A, X)$$是编码器，$$p_{\theta}(\hat{A} \mid A, X, Z)$$是解码生成器。编码器可以利用常见的GAE模型，解码生成器由于下列问题，设计上存在难点：

- 预算约束:由于是一个混合组合连续优化问题，因此很难生成具有预算约束的有效图结构；
- 可扩展性:现有的图生成方法存在大规模问题

为了满足预算约束，我们提出使用掩码[19]机制直接加入关于图结构的先验知识，以防止在解码过程中产生某些不需要的边。

解决可伸缩性问题，作者基于输入图像大小提出了不同的解决方案:

- 对于一个大型图，设计一个高效的解码器，具有$$O (m)$$​时间复杂度，专注于删边，$$m$$是边的数量;
- 对于一个小规模的图，考虑删边和加边；

#### 3.4.1 基于GCN的编码器

论文参考VGAE使用平均场近似来定义变分族:
$$
q_{\phi}(Z \mid A, X)=\prod_{i=1}^{N} q_{\phi_{i}}\left(z_{i} \mid A, X\right)
$$
其中$$q_{\phi_{i}}\left(z_{i} \mid A, X\right)$$是预定义的先验分布，称为具有对角协方差的各向同性高斯分布。变分边界的参数通过两层GCN来计算。
$$
\mu, \sigma=G C N_{\phi}(A, X)
$$
其中$$\mu, \sigma$$是变分边界的均值和标准差向量。



#### 3.4.2 图生成约束

对于大规模图而言，为了满足预算约束$$\sum_{i<j}\left|A_{i j}-\hat{A}_{i j}\right| \leq \Delta$$，将生成器$$p_{\theta}(\hat{A} \mid A, X, Z)$$近似为：
$$
p_{\theta}(\hat{A} \mid A, X, Z)=\prod_{(i, j) \in S} \Theta\left(E_{i j}\right)
$$
其中$$E_{i j}=\left[Z_{i} \mid X_{i}\right] \odot\left[Z_{j} \mid X_{j}\right]$$，$$\odot$$表示元素乘法，$$Z_{i} \mid X_{i}$$是拼接操作，$$\Theta(\cdot)$$和$$S$$定义如下：
$$
\begin{array}{l}
b_{i j}=W_{b 1} \sigma\left(W_{b 2} E_{i j}\right) \text { if } A_{i j}=1 \\
\Theta\left(E_{i j}\right)=\frac{e^{b_{i j}}}{\sum e^{b_{i j}}} \text { if } A_{i j}=1
\end{array}
$$

其中，$$\Theta\left(E_{i j}\right)$$表示边的保持分数。根据边的保持分数不放回的采样$$m-\Delta$$条边，集合$$S$$表示那些已经被采样的边。直观上，我们希望选择原始图中存在的$$m−∆$$边，并最大化其保持分数的乘积。通过这种策略，我们也可以生成严格满足$$\sum_{i<j}\left|A_{i j}-\hat{A}_{i j}\right| \leq \Delta$$预算要求的图。

对于小规模图，我们将预算分为两部分：$$\Delta / 2$$用于删边，$$\Delta / 2$$用于增边。将生成器近似为：
$$
p_{\theta}(\hat{A} \mid A, X, Z)=\prod_{(i, j) \in S} \Theta\left(E_{i j}\right) \prod_{(i, j) \in S} \Psi\left(E_{i j}\right)
$$
$$\Theta(\cdot)$$和$$S$$的计算方式同公式（13）。$$\bar{S}$$表示被选择插入的边的集合。$$\Psi(\cdot)$$和$$\Theta(\cdot)$$的计算方式相同，但是使用不同的参数，本质的区别在于：

- 在$$A_{i j}=0$$时计算；
- 添加$$\Delta / 2$$的边到$$\bar{S}$$；

#### 3.4.3 损失函数

$$
\mathcal{L}_{g}=\mathcal{L}_{\text {prior }}+\left(\lambda_{1} \mathcal{L}_{\text {hido }}+\lambda_{2} \mathcal{L}_{\text {perturb }}\right)\left(\sum_{(i, j) \in S} \log \Theta\left(E_{i j}\right) \div \sum_{(i, j) \in \bar{S}} \log \Psi\left(E_{i j}\right)\right)
$$



其中，$$\mathcal{L}_{\text {hide}}$$用于在个体集合内分散社区分配。
$$
\mathcal{L}_{\text {hide}}=\min _{i \in C^{+}, j \in C^{+}} K L\left(C_{i,}: \| C_{j,:}\right)
$$
其中$$\mathcal{L}_{\text {hide}}$$可视为$$C^+$$内任意对的边缘损失或最小距离。$$\lambda_{1}<0$$用于最大化距离损失，保证$$C^+$$中的成员分散在社区内。

综上所述，公式（15）从代理社区检测模型中接收到错误信号$$\mathcal{L}_{\text {hide}}$$和$$\mathcal{L}_{\text {perturb}}$$，作为奖励来指导我们的图生成器的优化。

