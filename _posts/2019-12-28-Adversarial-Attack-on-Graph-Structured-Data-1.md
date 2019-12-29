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
论文链接：https://arxiv.org/pdf/1806.02371.pdf

github链接：https://github.com/Hanjun-Dai/graph_adversarial_attack



## Introduction

本文研究了一些列GNN模型的的对抗攻击问题，主要任务为图分类（graph classification）和节点分类（node classification）问题。





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

- **Action**   攻击者允许对图进行增删边，因此一个在时间$t$的简单action是$a_{t} \in \mathcal{A} \subseteq V \times V$，由于对边的修改有较大的时间复杂度，论文用一种分层action来分解行为空间。

- **State**    $t$时刻的状态$s_t$用元组$(\hat{G}_t,c)$表示，$\hat{G}_t$是一个源自$G$的部分修改的图。

- **Reward**   攻击者的目的是欺骗目标分类器。因此，非零的奖励只在MDP结束时收到，奖励被定义为：

  $$
  r((\tilde{G}, c))=\left\{\begin{array}{l}
  {1: f(\tilde{G}, c) \neq y} \\
  {-1: f(\tilde{G}, c)=y}
  \end{array}\right.
  $$
  
  在修改的中间步骤中，不会收到任何奖励。也就是说，$r\left(s_{t}, a_{t}\right)=0, \forall t=1,2, \dots, m-1$。在PBA-C攻击中，目标分类器的预测置信度可以访问，因此可以把预测的损失函数$r((\tilde{G}, c))=\mathcal{L}(f(\tilde{G}, c), y)$作为奖励，损失函数越大表示攻击效果越好。
  
- Terminal 当$m$次攻击达到时，攻击过程结束。为了简化，本文关注固定长度的MDP。在修改量足够少就能实现攻击效果的情况下，我们可以简单地让代理修改虚边。

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



### Graph-level attack

在该实验设置中，使用了合成数据，黄金分类器$f^{\ast}$已知。等价指示器$$\mathcal{I}$$采用了显式语义。数据集$$\mathcal{D}^{(ind)}$$