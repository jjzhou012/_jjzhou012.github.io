---
layout: article
title: 图的对抗攻击： Adversarial Attack on Graph Structured Data
date: 2019-12-28 00:18:00 +0800
tags: [Adversarial attack, Graph]
categories: blog
pageview: true
key: Adversarial-Attack-on-Graph-2
---



------
论文链接：[https://arxiv.org/pdf/1806.02371.pdf](https://arxiv.org/pdf/1806.02371.pdf)

github链接：[https://github.com/Hanjun-Dai/graph_adversarial_attack](https://github.com/Hanjun-Dai/graph_adversarial_attack)



## Introduction

本文研究了一些列GNN模型的的对抗攻击问题，主要任务为图分类（graph classification）和节点分类（node classification）问题。提出了基于强化学习的方法RL-S2V，基于梯度的方法GradArgmax，基于遗传算法的方法GeneticAlg。攻击的图分类模型为structure2vec，攻击的节点分类模型为GCN。

在论文中考虑了几种不同的对抗攻击设置。当目标分类器中有更多的信息可以利用时，提出了一种基于梯度的方法和一种基于遗传算法的方法。这里主要关注以下三个设置:

- 白盒攻击 (WBA)：在这种情况下，攻击者可以访问目标分类器的任何信息，包括预测结果、梯度信息等。

- 实际黑盒攻击(PBA)：这种情况下，只有目标分类器的预测结果是可以利用的。当预测置信度可以利用时，将攻击表示为PBA-C；当只有离散的预测标签可以利用时，将攻击表示为PBA-D。

- 受限黑盒攻击（RBA）：这个设置比PBA更严格。在这种情况下，我们只能对一些样本进行黑盒查询，对其他样本进行对抗性修改。

考虑攻击者能够从目标分类器中获得的信息量，可以对上述攻击方式进行排序 WBA > PBA-C > PBA-D > RBA 。同时本文主要关注无目标攻击。



## Background

考虑到图分类以图为单位，一个包含$$
|\mathcal{G}|=N
$$个图的集合表示为$$
\mathcal{G}=\left\{G_{i}\right\}_{i=1}^{N}
$$，每个图$G_i=(V_i,E_i)$表示为节点$$
V_{i}=\left\{v_{j}^{(i)}\right\}_{j=1}^{\left|V_{i}\right|}
$$和边$$
E_{i}=\left\{\mathbf{e}_{j}^{(i)}\right\}_{j=1}^{\left|E_{i}\right|}
$$的集合。边表示为$$
\mathbf{e}_{j}^{(i)}=\left(\mathbf{e}_{j, 1}^{(i)}, \mathbf{e}_{j, 2}^{(i)}\right) \in V_{i} \times V_{i}
$$。

论文只考虑无向边。节点的特征表示为$$
x\left(v_{j}^{(i)}\right) \in \mathbb{R}^{D_{n o d e}}
$$，边的特征表示为$$
w\left(\mathbf{e}_{j}^{(i)}\right)=w\left(\mathbf{e}_{j, 1}^{(i)}, \mathbf{e}_{j, 2}^{(i)}\right) \in \mathbb{R}^{D_{e d g e}}
$$。

### Task

这篇论文考虑两个不同的监督学习任务：

- 基于归纳学习的图分类(Inductive Graph Classification)

  每个图$G_i$有一个标签$$y_{i} \in \mathcal{Y}=\{1,2, \ldots, Y\}$$，$Y$是标签种类数量。数据集$$\mathcal{D}^{(i n d)}=\left\{\left(G_{i}, y_{i}\right)\right\}_{i=1}^{N}$$以图实例和图标签的组合表示。
  
  学习过程为归纳学习，因为测试样本在训练过程中未知。
  
  学习过程中，分类器$$f^{(i n d)} \in \mathcal{F}^{(i n d)}: \mathcal{G} \mapsto \mathcal{Y}$$通过最小化以下交叉熵损失函数来$L(\cdot, \cdot)$进行优化：

  $$
  \mathcal{L}^{(i n d)}=\frac{1}{N} \sum_{i=1}^{N} L\left(f^{(i n d)}\left(G_{i}\right), y_{i}\right)
  $$

- 基于直推学习的节点分类(Transductive Node Classification)

  图$G_i$中的每个节点$c_i \in V_i$都有一个对应的标签$y_i \in \mathcal{Y}$。

  学习过程为直推学习，在整个数据集中只考虑单个图$G_0=(V_0,E_0)$。也就是说，$G_{i}=G_{0}, \forall G_{i} \in \mathcal{G}$。测试节点（未标注）也参与训练。

  数据集表示为$$\mathcal{D}^{(t r a)}=\left\{\left(G_{0}, c_{i}, y_{i}\right)\right\}_{i=1}^{N}$$，分类器表示为$$f^{(t r a)}\left(\cdot ; G_{0}\right) \in \mathcal{F}^{(t r a)}: V_{0} \mapsto \mathcal{Y}$$，通过最小化以下损失函数来优化模型：

  $$
  \mathcal{L}^{(t r a)}=\frac{1}{N} \sum_{i=1}^{N} L\left(f^{(t r a)}\left(c_{i} ; G_{0}\right), y_{i}\right)
  $$
  
  为了避免混淆，数据集重载为$$\mathcal{D}=\left\{\left(G_{i}, c_{i}, y_{i}\right)\right\}_{i=1}^{N}$$，$G_i$总是隐式的表示$G_0$。



### GNN family models

图神经网络在图$G=(V,E)$上定义了神经网络的一般结构。该架构通过迭代过程得到节点的向量表示：

$$
\begin{aligned}
\mu_{v}^{(k)}=& h^{(k)}\left(\left\{w(u, v), x(u), \mu_{u}^{(k-1)}\right\}_{u \in \mathcal{N}(v)}\right. ,
&\left.x(v), \mu_{v}^{(k-1)}\right), k \in\{1,2, \ldots, K\}
\end{aligned}
$$



其中，$\mathcal{N}(v)$表示节点$v\in V$的邻居节点，第$k$层聚合了节点$v$自身的特征$x(v)$，其邻居特征$x(u)$，其边特征$w(u,v)$，以及上一层的嵌入信息$\mu_v^{(k-1)}$和$\mu_u^{(k-1)}$。初始化的节点嵌入$$
\mu_{v}^{(0)} \in \mathbb{R}^{d}
$$被设置为0。为了简化起见，最终的节点嵌入表示为$\mu_v = \mu_v^{(K)}$。

**图形级嵌入通过在节点嵌入上应用全局池化来获得。**

普通GNN模型运行上述迭代直到收敛。



## Graph adversarial attack

对于一个训练过的分类器$f$，一个数据集$(G,c,y)\in \mathcal{D}$中的实例，图对抗攻击者$g(\cdot, \cdot): \mathcal{G} \times \mathcal{D} \mapsto \mathcal{G}$将图$G=(V,E)$修改为$\tilde{G}=(\tilde{V}, \tilde{E})$，优化过程如下：


$$
\begin{array}{cl}
{\max _{\tilde{G}}} & {\mathbb{I}(f(\tilde{G}, c) \neq y)} \\
{\text {s.t.}} & {\tilde{G}=g(f,(G, c, y))} \\
{} & {\mathcal{I}(G, \tilde{G}, c)=1}
\end{array}
$$

其中$$
\mathcal{I}(\cdot, \cdot, \cdot): \mathcal{G} \times \mathcal{G} \times V \mapsto\{0,1\}
$$是一个等价指示器，表示两个图$G, \tilde{G}$在分类语义下相同。在这个优化过程中，最大化修改过的图$\tilde{G}$被错误分类的置信度$$\mathbb{I}(\cdot)$$。

这篇论文主要关注对**离散结构**的修改。攻击者$g$能够通过对图$G$增删边来构建新的图。注意到，**对节点的增删也能通过对边的增删来实现**。

显然，对边的修改要比对节点的修改更困难，选择一个节点只需要$$
O(|V|)
$$的时间复杂度，而选择一条边需要$$
O(|V|^2)
$$。



**攻击者的目标是欺骗分类器，而不是修改实例的标签，**所以等价指示器需要进行如下设置：

- 不修改标签。在这种情况下，一个**黄金标准的分类器$f^{\ast}$可以绝对正确的分类样本**。等价指示器$\mathcal{I}(\cdot,\cdot,\cdot)$可以被定义为：

  $$
  \mathcal{I}(G, \tilde{G}, c)=\mathbb{I}\left(f^{*}(G, c)=f^{*}(\tilde{G}, c)\right)
  $$
  
  也就是说，对于节点$c$而言，$$f^{\ast}(G, c)=y$$且$$f^{\ast}(\tilde{G}, c)=y$$，不管是对于原图还是修改过的图都能正确分类。所以指示器返回结果1。

- 较小的修改量。在许多情况下，当显式语义未知时，我们会要求攻击者在邻域图中做出尽可能少的修改：

  $$
  \begin{aligned}
  \mathcal{I}(G, \tilde{G}, c)=& \mathbb{I}(|(E-\tilde{E}) \cup(\tilde{E}-E)|<m) \\
  &\cdot \mathbb{I}(\tilde{E} \subseteq \mathcal{N}(G, b)))
  \end{aligned}
  $$
  
  上面，$m$是最大允许修改的边数量，$$\mathcal{N}(G, b)=\left\{(u, v): u, v \in V, d^{(G)}(u, v)<=b\right\}$$是节点$v$的$b$跳邻域图。
  
  较小的修改量可以保证攻击的隐蔽性。



以下是论文提出的两种攻击方法。

### Attacking as hierarchical reinforcement learning

给定一个实例$(G,c,y)$和一个目标分类器$f$，攻击过程被建模为有限视界马尔可夫决策过程$\mathcal{M}^{(m)}(f, G, c, y)$。马尔可夫决策过程定义如下：

- **Action**:   攻击者允许对图进行增删边，因此一个在时间$t$的简单action是$a_{t} \in \mathcal{A} \subseteq V \times V$，由于对边的修改有较大的时间复杂度，论文用一种分层action来分解行为空间。

- **State**:    $t$时刻的状态$s_t$用元组$(\hat{G}_t,c)$表示，$\hat{G}_t$是一个源自$G$的部分修改的图。

- **Reward**:   攻击者的目的是欺骗目标分类器。因此，非零的奖励只在MDP结束时收到，奖励被定义为：
$$
  r((\tilde{G}, c))=\left\{\begin{array}{l}
  {1: f(\tilde{G}, c) \neq y} \\
  {-1: f(\tilde{G}, c)=y}
  \end{array}\right.
  $$
  
  在修改的中间步骤中，不会收到任何奖励。也就是说，$r\left(s_{t}, a_{t}\right)=0, \forall t=1,2, \dots, m-1$。在PBA-C攻击中，目标分类器的预测置信度可以访问，因此可以把预测的损失函数$r((\tilde{G}, c))=\mathcal{L}(f(\tilde{G}, c), y)$作为奖励，损失函数越大表示攻击效果越好。
  
- **Terminal**: 当$m$次攻击达到时，攻击过程结束。为了简化，本文关注固定长度的MDP。在修改量足够少就能实现攻击效果的情况下，我们可以简单地让代理修改虚边。

给定以上设置，一个简单的MDP轨迹可以表示为：$\left(s_{1}, a_{1}, r_{1}, \dots, s_{m}, a_{m}, r_{m}, s_{m+1}\right)$，其中$$s_1=(G,c), s_t=(\hat{G}_t,c), \forall t\in \{2, \ldots ,m \}, s_{m+1}=(\tilde{G},c)$$。最后一步会得到奖励$r_{m}=r\left(s_{m}, a_{m}\right)=r((\tilde{G}, c))$，其他的中间过程奖励为0：$$
r_{t}=0, \forall t \in\{1,2, \ldots, m-1\}
$$。

由于这是一个具有有限视界的离散优化问题，论文使用Q-learning来学习MDPs。

Q-learning是一个非策略的优化方式，通过直接拟合贝尔曼方程来优化：


$$
Q^{*}\left(s_{t}, a_{t}\right)=r\left(s_{t}, a_{t}\right)+\gamma \max _{a^{\prime}} Q^{*}\left(s_{t+1}, a^{\prime}\right)
$$



表示当前$s_t$采取动作$a_t$后的即时奖励$r$，加上折价$\gamma$后的下一时刻最大奖励。这里隐式的包含了一个贪婪的策略：

$$
\pi\left(a_{t} | s_{t} ; Q^{*}\right)=\operatorname{argmax}_{a_{t}} Q^{*}\left(s_{t}, a_{t}\right)
$$


在有限视界的情况下，$\gamma$设置为1。由于在大规模图中进行时间复杂度为$$
O(|V|^2)
$$的边修改策略代价昂贵，所以提出了一种分层策略，将行为$a_t \in V \times V$分解为 $a_{t}=\left(a_{t}^{(1)}, a_{t}^{(2)}\right)$，其中$a_{t}^{(1)}, a_{t}^{(2)} \in V$。因此一个简单的边修改行为$a_t$被分解为边两端节点的行为。按如下方式建模分层Q函数：

$$
\begin{aligned}
&Q^{1 *}\left(s_{t}, a_{t}^{(1)}\right)=\max _{a_{i}^{a}} Q^{2 *}\left(s_{t}, a_{t}^{(1)}, a_{t}^{(2)}\right)\\
&Q^{2 *}\left(s_{t}, a_{t}^{(1)}, a_{t}^{(2)}\right)=r\left(s_{t}, a_{t}=\left(a_{t}^{(1)}, a_{t}^{(2)}\right)\right)+\max _{a_{t+1}^{(a)}} Q^{1 *}\left(s_{t}, a_{t+1}^{(1)}\right)
\end{aligned}
$$


上述公式中，$Q^{1 \ast}$和$Q^{2 \ast}$是实现$Q^{\ast}$的两个函数。只有当一对$(a_{t}^{(1)}, a_{t}^{(2)})$被选择时，一个action才算完成，所以只有当$a_{t}^{(2)}$完成时奖励才有效。这样的分解跟上述公式(6)有一样的优化结构，但是只需要 $$O(2\times \mid V \mid )=O(\mid V \mid )$$ 的时间复杂度。

> $a_t^{(1)}$的最优价值动作函数$Q^{1\ast}$考虑最大化后续的动作$a_t^{(2)}$产生的最大价值；
>
> $a_t^{(2)}$动作完成时所产生的价值为及时奖励带来的收益和下一个时间片$t+1$时动作$a_{t+1}^{(1)}$产生的最大价值。

![56ce4229913fa189e56d28c9fcdeb36.png](http://ww1.sinaimg.cn/large/005NduT8ly1gadn3w3rlbj30wj076jt2.jpg)



展开上述的贝尔曼方程：


$$
\begin{aligned}
Q_{1,1}^{*}\left(s_{1}, a_{1}^{(1)}\right) &=\max _{a_{1}^{(2)}} Q_{1,2}^{*}\left(s_{1}, a_{1}^{(1)}, a_{1}^{(2)}\right) \\
Q_{1,2}^{*}\left(s_{1}, a_{1}^{(1)}, a_{1}^{(2)}\right) &=\max _{a_{2}^{(1)}} Q_{2,1}^{*}\left(s_{2}, a_{2}^{(1)}\right) \\
& \ldots \\
Q_{m, 1}^{*}\left(s_{m}, a_{m}^{(1)}\right) &=\max _{a_{m}^{(2)}} Q_{m, 2}^{*}\left(s_{m}, a_{m}^{(1)}, a_{m}^{(2)}\right) \\
Q_{m, 2}^{*}\left(s_{m}, a_{m}^{(1)}, a_{m}^{(2)}\right) &=r(\tilde{G}, c)
\end{aligned}
$$


在这里考虑一些更实用和具有挑战性的设置，当只有一个$Q^{\ast}$被学习。因此，我们要求所学习的Q函数泛化或传递至所有的MDPs:


$$
\max _{\boldsymbol{\theta}} \sum_{i=1}^{N} \mathbb{E}_{t, a=\operatorname{argmax}_{a_{t}} Q^{*}\left(a_{t} | s_{t} ; \boldsymbol{\theta}\right)}\left[r\left(\left(\tilde{G}_{i}, c_{i}\right)\right)\right]
$$


$Q^{\ast}$被$\theta$参数化。



#### PARAMETERIZATION OF $Q^{\ast}$

从上面我们可以看出，最灵活的参数化方法是实现$2×m$的时变$Q$函数。最后发现两个不同的参数化就够了，即 $$Q_{t, 1}^{*}=Q^{1 *}, Q_{t, 2}^{*}=Q^{2 *}, \forall t$$。

使用GNN模型对$Q$函数进行参数化，具体而言，$Q^{1 \ast}$参数化如下：


$$
Q^{1 *}\left(s_{t}, a_{t}^{(1)}\right)=W_{Q_{1}}^{(1)} \sigma\left(W_{Q_{1}}^{(2) \top}\left[\mu_{a_{t}^{(1)}}, \mu\left(s_{t}\right)\right]\right)
$$


其中$\mu_{a_{t}^{(1)}}$是图$$\hat{G}_t$$中节点$$a_{t}^{(1)}$$的嵌入表示，通过structure2vec(S2V)实现：


$$
\mu_{v}^{(k)}=\operatorname{relu}\left(W_{Q_{1}}^{(3)} x(v)+W_{Q_{1}}^{(4)} \sum_{u \in \mathcal{N}(v)} \mu_{u}^{(k-1)}\right)
$$


其中$\mu_{v}=\mu_{v}^{(K)}, \mu_{v}^{(0)}=0$。 $\mu(s_t)=\mu(\hat{G}_t = (\hat{V}_t, \hat{E}_t),c)$表示整个状态元组


$$
\mu\left(s_{t}\right)=\left\{\begin{array}{l}
{\sum_{v \in \hat{V}} \mu_{v}: \text { graph attack }} \\
{\left[\sum_{v \in \mathcal{N}_{\hat{G}_{t}}(c, b)} \mu_{v}, \mu_{c}\right]: \text { node attack }}
\end{array}\right.
$$

在节点攻击场景中，状态嵌入从节点$c$的$b$跳邻域中获得，表示为$$\mathcal{N}_{\hat{G}_{t}}(c, b)$$。$$Q^{1 \ast}$$的参数设置为$$
\theta_{1}=\left\{W_{Q_{1}}^{(i)}\right\}_{i=1}^{4}
$$，$$Q^{2 \ast}$$的参数设置类似采用$\theta_2$，加了额外的对节点$a_{t}^{(1)}$的考虑：
$$
Q^{2 *}\left(s_{t}, a_{t}^{(1)}, a_{t}^{(2)}\right)=W_{Q_{2}}^{(1)} \sigma\left(W_{Q_{2}}^{(2) \top}\left[\mu_{a_{t}^{(1)}}, {\mu_{a_{t}^{(2)}}, \mu(s_t)}\right]\right)
$$


上述方法命名为RL-S2V，因为它学习一个由S2V参数化的$Q$函数来执行攻击。





### Other attacking methods

RL-S2V适用于黑盒攻击。对于不同的攻击场景，其他算法或许更合适。

#### Random sampling

从图$G$中随机增删边，是最简单的攻击方式。当一个边操作行为$a_t=(u,v)$被采样时，只有当它满足语义约束$\mathcal{I}(\cdot, \cdot, \cdot)$时，我们才会接受它。它需要最少的信息进行攻击。尽管它很简单，但有时它可以获得很好的攻击效果。

#### Gradient based white box attack

梯度在修改连续结构的输入如图像等已经取得了成功，但是对于离散结构来说不太简单。回顾上面GNN模型的迭代过程，为每对节点$(u,v)\in V \times V$分配系数$\alpha_{u,v}$：


$$
\begin{aligned}
\mu_{v}^{(k)}=h^{(k)}\left(\quad\left\{\alpha_{u, v}\left[w(u, v), x(u), \mu_{u}^{(k-1)}\right]\right\}_{u \in \mathcal{N}(v)}\right.  \cup \\
\left\{\alpha_{u^{\prime}, v}\left[w\left(u^{\prime}, v\right), x\left(u^{\prime}\right), \mu_{u^{\prime}}^{(k-1)}\right]\right\}_{ u^{\prime} \notin \mathcal{N}(v)}, \\
\left.x(v), \mu_{v}^{(k-1)}\right), k \in\{1,2, \ldots, K\}
\end{aligned}
$$


令$$
\alpha_{u, v}=\mathbb{I}(u \in \mathcal{N}(v))
$$，也就是说$\alpha$是二值化的邻接矩阵，上述公式其实和上面的迭代过程有一样的效果。但是这些附加的系数给了我们关于边（不管存在与否）的梯度信息：


$$
\frac{\partial \mathcal{L}}{\partial \alpha_{u, v}}=\sum_{k=1}^{K} \frac{\partial \mathcal{L}^{\top}}{\partial \mu_{k}} \cdot \frac{\partial \mu_{k}}{\partial \alpha_{u, v}}
$$


为了攻击模型，通过进行梯度上升，即 $\alpha_{u, v} \leftarrow \alpha_{u, v}+\eta \frac{\partial \mathcal{L}}{\partial \alpha_{u, v}}$。但是攻击是在离散结构上进行的，只有$m$条边被允许增删。因此我们需要进行一个组合优化问题：


$$
\begin{aligned}
\max _{\left\{u_{t}, v_{t}\right\}_{t=1}^{m}} &  \quad \sum_{t=1}^{m}\left|\frac{\partial \mathcal{L}}{\partial \alpha_{u_{t}, v_{t}}}\right| \\
\text {s.t.} & \quad \tilde{G}=\operatorname{Modify}\left(G,\left\{\alpha_{u_{t}, v_{t}}\right\}_{t=1}^{m}\right) \\
& \quad \mathcal{I}(G, \tilde{G}, c)=1
\end{aligned}
$$


我们简单地使用贪心算法来求解上述优化问题。修改的边的系数集合为$$
\left\{\alpha_{u_{t}, v_{t}}\right\}_{t=1}^{m}
$$，通过以下方式持续的


$$
\hat{G}_{t+1}=\left\{\begin{array}{l}
{\left(\hat{V}_{t}, \hat{E}_{t} \backslash\left(u_{t}, v_{t}\right)\right): \frac{\partial \mathcal{L}}{\partial \alpha_{u_{t}, v_{t}}}<0} \\
{\left(\hat{V}_{t}, \hat{E}_{t} \cup\left\{\left(u_{t}, v_{t}\right)\right\}\right): \frac{\partial \mathcal{L}}{\partial \alpha_{u_{t}, v_{t}}}>0}
\end{array}\right.
$$

也就是说，我们修改了最可能引起目标变化的边。根据梯度的符号，我们可以添加或删除边缘。我们把它命名为GradArgmax，因为它根据梯度信息进行贪婪选择。因为该方法需要梯度信息，所以属于白盒攻击。同时需要考虑所有节点对的梯度信息，所以时间复杂度为$$
O(|V|^2)
$$。如果没有近似的简化方法，这种梯度攻击就无法扩展到大网络。



![cf4b0c0887888924c7d9f33045f82d4.png](http://ww1.sinaimg.cn/large/005NduT8ly1gadyiera5cj315a0hjdkz.jpg)



#### Genetic algorithm

论文提出了一种基于遗传算法的梯度攻击。对于给定的实例$(G,c,y)$和目标分类器$f$，算法涉及五个主要过程：

- **Population**： 种群代表候选的修改方案，表示为$$
  \mathcal{P}^{(r)}=\{\hat{G}_{j}^{(r)}\}_{j=1}^{\left|\mathcal{P}^{(r)}\right|}
  $$，其中$\hat{G}_{j}^{(r)}$是对原始图的有效修改方案，$r=1,2,\ldots,R$表示进化代索引，$R$是最大的进化代数。

- **Fitness**: 当前种群的每一个候选解决方案都需要一个适应度函数来评价其效果，本文使用目标模型的损失函数$$\mathcal{L}\left(f\left(\hat{G}_{j}^{(r)}, c\right), y\right)$$作为适应度函数。一个好的攻击方案应该令损失函数上升，因此适应度分数越大，表示该攻击方案越有效。

  但是因为适应度是一个连续的分数，不适合于PBA-D，因为它只有分类标签可以利用。

- **Selection**: 获得了当前种群的适应度函数，我们可以通过加权采样或者贪婪选择的方法，来选择“亲代”种群$\mathcal{P}^{(r)}_b$，进行下一代的繁衍。

- **Crossover**: 在选择了种群$$\mathcal{P}^{(r)}_b$$后，随机选择两个候选图$$\hat{G}_{1}, \hat{G}_{2} \in \mathcal{P}_{b}^{(r)}$$，对它们的边进行交叉混合：

$$
\hat{G}^{\prime}=\left(V,\left(\hat{E}_{1} \cap \hat{E}_{2}\right) \cup \operatorname{rp}\left(\hat{E}_{1} \backslash \hat{E}_{2}\right) \cup \operatorname{rp}\left(\hat{E}_{2} \backslash \hat{E}_{1}\right)\right)
$$

​	其中$\operatorname{rp}(\cdot )$表示采样一个子集。

- **Mutation**: 变异过程，对于一个候选攻击方案$\hat{G} \in \mathcal{P}^{(r)}$，假设修改的边是$$\delta E=\left\{\left(u_{t}, v_{t}\right)\right\}_{t=1}^{m}$$，对于每一条边$(u_t,v_t)$，以确定的变异概率替换其中的节点$\left(u_{t}, v^{\prime}\right)$或$\left(u^{\prime}, v_t\right)$。



种群大小为$$
\left|\mathcal{P}^{(r)}\right|
$$，交叉概率、变异概率、进化代数都是可以被微调的超参数。由于适应度函数的限制，该方法只能在PBA-C方法下使用。由于我们需要执行目标模型$f$来获得适应度分数，所以这种遗传算法的计算成本为$$O(|V|+|E|)$$ ，主要由GNN的计算成本构成。我们简单地将其命名为GeneticAlg，因为它是一个通用遗传算法框架的实例化。





## Experiment

- 对于 GeneticAlg ，设置种群大小为 $$
  \left|\mathcal{P}^{(r)}\right|
  $$，进化代数为$R=10$， 交叉率和变异率在$$\{0.1,\ldots,0.5\}$$之间微调。

- 对于 RL-S2V ，设置S2V的传播深度在$$K=\{1,\ldots,5\}$$之中微调。

- 对于 GradArgmax 和 RandSampling，无参数微调。

![7179c55ce18901264ab4a9eec45389a.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaehgl6zfoj30gh06974x.jpg)

### Graph-level attack

在该实验设置中，使用了合成数据，黄金分类器$f^{\ast}$可利用，因此等价指示器$$\mathcal{I}$$采用了显式语义。

我们通过Erdos-Renyi随机图模型构建了包含15000个图的数据集$$\mathcal{D}^{(ind)}$$，进行3-class的图分类任务，每个类包含5000个图。图分类器要求输出每个图包含多少个连通子图，图标签设定为$$\mathcal{Y}=\{1,2,3\}$$，也就是说每个图最多有3个连通子图。

数据集划分：test set 1:1500; test set 2:150; 每一类包含等量的实例样本。

攻击的目标模型为structure2vec。传播参数在$$K=\{2, \ldots , 5\}$$之中微调。

![e54df1086fb81c9fe594fe4fa4ecd79.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaehg3gs09j30y70dhwiw.jpg)

表2显示了不同设置下的实验结果。

- 对于test set 1而言，structure2vec在原始数据集上实现了很高的图分类精度，在大多数情况下，提高$K$能提升模型的泛化能力。
- 在PBA场景下，GeneticAlg 和 RL-S2V 能使图分类精度下降 $40\%  \thicksim 60 \%$。
- 在攻击图分类算法时，GradArgmax似乎不是很有效。可能的原因是S2V在获取图级嵌入时用到了全局池化，在反向传播的时候，池化操作会将梯度分配给其他每个节点的嵌入，这会使大多数节点对的$$\frac{\partial \mathcal{L}}{\partial \alpha}$$相差不大。

- 对于test set 2 上的RBA攻击场景，攻击者在生成对抗样本的时候无法访问目标模型。因为RL-S2V采用归纳学习的方式训练，所以能较好的泛化到test set 2，同时在test set 2上的攻击效果类似，表明目标分类器存在一致性错误。

结论：

- 对于监督图任务，存在对抗样本能降低图模型的性能；
- 即使模型的泛化能力较好，也会受到对抗攻击的影响；
- RL-S2V可以学习到转移攻击策略，去攻击新样本；



### Node-level attack

这部分研究节点分类的攻击问题。不同于图分类攻击，节点分类采用直推学习的方式训练模型，训练过程中可以学习测试集的特征。

使用的数据集使真实数据集：Citeseer, Cora, Pubmed, Finance。前三个数据集是小规模引文网络。最后一个是大规模电子商务交易网络。

![88110ce9a70a4a170e4056301c95d50.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaeijckl5mj30gn051t9a.jpg)

攻击的目标模型是GCN。

这里用了“小改动”来限制攻击者，也就是说，给定图$G$和节点$c$，对抗样本只能在目标节点的2跳邻域内进行删边。

我们可以看到，虽然删除一条边是对图的最小修改，但是对那些小规模图的攻击率仍然有10%，而在金融数据集中是4%。

**在这里进行了Exhaust攻击，这是所有算法在该攻击预算下能获得的最优的攻击效果。**

如果允许修改两条边，节点分类精度甚至能降低到60%或更低。但是，考虑到图的平均度不是很大，删除两条或多条边就违反了“小修改”的限制。我们需要小心地只创建对抗样本，而不是实际上更改该实例的真正标签。

在节点分类攻击中，GradArgmax效果很好，这与图分类攻击不同。这里的梯度对邻接矩阵$α$不再是平均，这使得它更容易区分有用的修改。因为最后没有用全局池化来平均。

在test set 2上的RBA攻击场景下，RL-S2V在未知样本上依然有较好的攻击泛化性能。虽然在真实数据集上没有黄金分类器，但是我们生成的对抗样本在很大概率上还是有效的：

- 对图结构的修改很小，在目标节点的2跳邻域内；
- 没有修改节点的特征；



![98a9a2f03dae7cfef6288eebb6ca423.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaej994p55j30gn0gf772.jpg)



### Inspection of adversarial samples

这部分对不同方法生成的对抗样本进行了可视化分析。

下图展示了用RL-S2V生成的图分类对抗样本。三个图的真实标签是1，2，3；但是目标分类器错误预测为2，1，2。在5(b)和5(c)中，我们可以看到模型控制目标节点连接了距离4跳以上的节点，说明虽然目标分类器structure2vec训练时$K=4$，但是它没有有效的捕获4跳邻域信息。5(a)也说明了，即时两个连接的节点只有2跳远，分类器也会错误分类。

![4d9cb18cb91681e94299f490b377c23.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaej9qvz7oj30gn074q3u.jpg)



图6展示了用GradArgmax进行节点分类攻击时的对抗样本。橙色节点是攻击的目标节点，蓝色边是GradArgmax建议添加的边，黑色边是建议删除的边。黑色节点和橙色目标节点有一致的标签，白色节点没有。边越粗，梯度值越大。

- 6(b)虽然删掉了一个同标签的邻居节点，但是依然与其他黑色节点相连。
- 6(c)虽然红色边没有连接两个同标签的节点，但是它连接了来自相同类的一个大的节点社区，距离为2跳。

![775ab1ad11fea797c16ffad84722d25.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaejlhj3ejj30nv0bmacm.jpg)



### Defense against attacks

将对抗样本加入训练集，进行对抗训练可以提升模型的鲁棒性。换句话说，将对抗样本加入训练集相当于扩大了训练集规模，有防御效果也是理所当然的。

论文采用了对抗训练，在训练的过程中进行edge drop，也就是说在每一次训练中，对图进行全局随机删边。其实也就是随机引入噪声。

可以看到，在原始数据集上的精度有细微波动，但是引入的噪声经过学习后，各种方法的攻击率下降了，说明这种简单的对抗训练还是能起到一定防御效果的。

![49dffbf5a0de9f2f90da116b267e8f1.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaek43fp7wj30ju08lgmx.jpg)