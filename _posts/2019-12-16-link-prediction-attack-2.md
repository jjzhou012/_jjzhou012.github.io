---
layout: article
title: 链路预测的攻击：Attacking Similarity-Based Link Prediction in Social Networks
date: 2019-12-16 00:20:00 +0800
tags: [Adversarial attack, Link prediction]
categories: blog
pageview: true
key: link-prediction-attack-2
---





------

论文链接：[Attacking Similarity-Based Link Prediction in Social Networks](https://dl.acm.org/citation.cfm?id=3306127.3331707) [AAAMAS 2019]

链接2：[link](http://network-games-muri.engin.umich.edu/wp-content/uploads/sites/439/2019/04/attacking-aamas-2019.pdf)



## Introduction

结构相似性指标被广泛应用于预测未知的链路，它基于这样的假设：相似度越高的节点对之间存在连边的可能性越大。

然而，链路预测的许多场景中，如预测黑帮或恐怖主义网络中的链路，都是对抗性的，防守方通过操纵可观察信息来降低链路预测算法的性能，防止信息被过度挖掘。在传统的基于相似性的链路预测方法中，一个关键的假设是观测到的(子)网络是正确观测的。然而，就链路预测可能揭示的关系而言(即关联方更愿意隐藏的关系)，无论是为了保护隐私，还是为了避免被执法部门逮捕，它都引入了操纵网络测量的动机，以降低目标链接的相似性分数。

为了系统化地研究一个攻击者`adversary`操纵链路预测的能力，论文将链路预测攻击视为一个优化问题，攻击者通过从可观测网络中移除一定量的边，来最小化目标链路集合的总体加权相似性分数。该论文对基于相似性的链路预测的删边攻击进行了全面的研究，重点研究了基于局部信息和全局信息的两类攻击方法。在此基础上提出了。。。



## Problem Formulation

### Similarity Metrics

不再赘述

### Attack Model

攻击的目标是删除一些边的子集，来最小化目标链接集的相似性分数。

对于一个图$G(V,E)$，并未完全可观测，我们只能从可观测的部分获得查询的信息。对于一个查询 $(u,v)\in Q$，如果边$(u,v)\in E$则边可查询，否则不可查询。在这里，查询集相当于训练集，利用查询集$Q$中的边构造一个训练图$G_Q=(V_Q,E_Q)$，来计算未知的边$(u',v') \notin Q$的相似性分数。

攻击者想要通过删除训练边集$E_Q \equiv E \cap Q$中最多$k$条边，来实现一些敏感链路$H$的隐藏。其中一种相对自然和通用的方法是最小化$H$中链路的相似性分数加权和：


$$
\min _{E_{a} \subset E_{Q}} f_{t}\left(E_{a}\right) \equiv \sum_{(u, v) \in H} w_{u v} \operatorname{Sim}\left(u, v ; E_{a}\right), \quad \text { s.t. }\left|E_{a}\right| \leq k
$$


其中，权重$w_{uv}$代表了隐藏链路$(u,v)$的相对重要性，同时明确了相似性分数对已删除边集$E_a$的依赖关系。



## Attack Local Similarity Metrics

本文将局部相似性指标分为两个子类：

- Common Neighbor Degree (CND) 
- Weighted Common Neighbor (WCN)

定义了一些符号:

| Notation                                         | description                                                  |
| ------------------------------------------------ | ------------------------------------------------------------ |
| $$U=\{u_i \}, \quad \mid U \mid =n$$             | $H$中目标边的端节点集合，共$n$个节点；                       |
| $$W=\left\{w_{1}, w_{2}, \cdots, w_{m}\right\}$$ | 目标节点的共同邻居， $\forall w_i \in W$ 链接$U$中至少两个节点； |
| $N(u_i, u_j)$                                    | 节点$u_i,u_j$的共同邻居；                                    |
| $d(u_i)$                                         | 节点$u_i$的度；                                              |
| $$X \in\{0,1\}^{m \times n}$$                    | $W$和$U$之间的边的状态，$x_{ij}=1$表示$(w_i, u_j)$存在边；$x_{ij}=1 \rightarrow 0$表示$w_i,u_j$之间的边被删除； |



### Classification of Local Metrics

**Definition 3.1.**    CND度量的相似性矩阵依赖如下形式：


$$
f_t = \sum_{r=1}^{m} W_{r} \frac{\sum_{i, j \mid \left(u_{i}, u_{j}\right) \in H} x_{r i} \cdot x_{r j}}{f_{r}\left(S_{r}\right)}
$$


$S_r$是决策矩阵$X$的第$r$行的和，$f_r$是$S_r$的矩阵依赖增长函数，$W_r$是关联权重。$\sum_{i, j \mid \left(u_{i}, u_{j}\right) \in H} x_{r i} \cdot x_{r j}$ 遍历$H$中所有的链路，搜寻基于$w_r$的开放三角，后面用$\sum_{ij}$。

CND度量包括`AA`、`RA`、`CN`。

> 不太清楚$f_r(S_r)$计算什么？根据以上三种相似性指标的相似性矩阵的通用定义，猜测应该是 $RA: f_r(S_r) \rightarrow deg(w_r)$，$AA: f_r(S_r) \rightarrow \log deg(w_r)$。

**Definition 3.2.**    WCN度量的相似性矩阵可以定义为如下形式：


$$
\operatorname{Sim}\left(u_{i}, u_{j}\right)=\frac{\left|N\left(u_{i}, u_{j}\right)\right|}{g\left(d\left(u_{i}\right), d\left(u_{j}\right),\left|N\left(u_{i}, u_{j}\right)\right|\right)}
$$


其中，$g$为关于$d(u_i),d(u_j)$的严格递增函数，当$t,s$非负且$\mid N(u_i,u_j) \mid$有效时，有 $g(d(u_i)-t,d(u_j)-s) \leq g(d(u_i),d(u_j))$ ；$\operatorname{Sim}$ 为关于$\mid N(u_i,u_j) \mid$的严格递增函数，当$t$非负且$d(u_i),d(u_j)$有效时，有 $\operatorname{Sim}(\mid N(u_i,u_j) \mid - t)  \leq \operatorname{Sim}(\mid N(u_i,u_j) \mid)$。

WCN度量包括`jaccard`、`Salton`、`Hub Promoted`等。



根据以上定义，理性的攻击者将会选择$W,U$之间的边进行删除，因为删除其他的边会导致$d(u_i)$或$d(w_i)$下降，使得相似性分数上升；而删除$W,U$之间的边会减少共同邻居数，使得相似性分数降低。

整体相似性$f_t$由决策函数$X$决定，攻击局部相似性指标由此可以建模为一个优化问题，称 *Prob-Local*:


$$
\min _{X} f_{t}(X), \quad \text { s.t. } \operatorname{Sum}\left(X^{0}-X\right) \leq k
$$


其中，$X^0$是原始的决策矩阵，$ \operatorname{Sum}(\cdot)$ 表示元素和。



### Hardness Results

先不限制隐藏的链路集$H$，分析了一般情况下，攻击所有的局部相似性指标都是NP难问题。

**Theorem 3.3.** *Attacking local similarity metrics is NP-Hard.*

考虑攻击者能否通过删除至多$k$条边，使总的相似度$f_t$不大于一个常数$\theta$？注意到$f_t$的最小值为0，那么问题来了，决策问题$P_L$：能否通过删除$k$条边使得$f_t=0$。

:hushed::frowning::fearful: 作者在这里使用了一个`vertex cover` $V_c$的概念来简化这个问题，讨论两个问题：

- 决策问题$P_L$：能否通过删除$k$条边使得$f_t=0$；
- 节点覆盖的决策问题$P_{VC}$: 给定一个图$G$和一个整数$k$，是否存在$k$大小的节点覆盖；

> 给出了一个实例：
>
> 给定图$G=(V,E)$和整数$k$，按以下步骤构造一个新的图$Q$：
>
> - 复制$G$中的所有节点，作为$Q$的节点，但不链接；
> - $Q$中加入节点$w$，然后$w$和每个节点$v_i$链接；
> - $Q$中加入$n=\mid V \mid$个节点$u_1, \cdots , u_n$，然后链接每一对$(u_i,v_i)$；
> - 链接所有的节点对$(u_i,u_{i+1})$，$i=1,2,\cdots,n-1$；
>
> $Q$的隐藏链路目标集$$H=\{(v_i,v_j)\}$$为$G$中相应的边$(v_i,v_j)$构成。然后针对图$Q$和目标集$H$构造决策问题$P_L$。
>
> ![344d3e49cd014af7e5b5ad532627fa6.png](http://ww1.sinaimg.cn/mw690/005NduT8ly1g9zq47cmwfj30g407dgmc.jpg)
>
> 上图中，$G$中存在的边构成了目标集$$H=\{(v_1,v_2),(v_1,v_3),(v_2,v_3),(v_2,v_4),(v_2,v_5),(v_2,v_6)\}$$，`vertex cover`的概念虽然看着模糊，但结合图来看还是很清晰易懂的，数学表示就是：$\min \mid V_c \mid \quad s.t. \  \forall e \in H, \exists v_i \in V_c: v_i \in e$ ，描述起来就是，目标集中的每一条边的其中一个节点一定存在于$V_c$中。图中所示，节点覆盖$$V_c=\{v_2,v_3\}$$，此时$k=\mid V_c \mid = 2$。

上面两个决策问题是等价的，那么如何证明呢，以`CN`指标为例结合上图：

首先证明 $P_{VC} \Longrightarrow P_L$ 。

- 假设一个节点覆盖$$V_c=\{v_1,\cdots,v_k\}$$，大小$\mid V_c \mid = k$；
- 删掉$k$条边$$\{(v_1,w),\cdots, (v_k,w)\}$$可以令$f_t(H)=0$。理解一下，对于任意一条边$(v_i,v_j)\in H$，$v_i,v_j$中至少有一个节点一定存在于$V_c$。原本$H$中的任意边的两个节点都有共同邻居$w$，使得$CN(v_i,v_j)=1$，现在要删除边$$\{(v_1,w),\cdots, (v_k,w)\}$$，那么$H$中任意一条边$(v_i,v_j)$的两个节点的唯一一个共同邻居就没了，使得$CN(v_i,v_j)=0$，最终 $f_t(H)=0$。

然后证明 $P_{L} \Longrightarrow P_{VC}$ 。

- 假设我们删掉了$k$条边使得$f_t(H)=0$，那么每条被删掉的边一定是$(w,v_i)$，因为只有删掉含$w$的边才会让$f_t(H)$减小。
- 假设删掉的$k$条边是$$\{(w,v_1), \cdots ,(w,v_k)\}$$，然后$$V_c=\{v_1,\cdots , v_k\}$$构成了节点覆盖。因为$\forall (v_i,v_j) \in H, CN(v_i,v_j) \geq 0, f_t(H)=0$意味着$\forall (v_i,v_j) \in H, CN(v_i,v_j) = 0$。在初始化的时候每条目标链接$(v_i,v_j)$都有一个共同邻居$w$，现在$CN(v_i,v_j) = 0$，势必是因为至少其中一条边$(w,v_i)$被删了，那么对所有的目标边来说，删掉的边整合一下$$set(\{(w,v_1), \cdots ,(w,v_n)\}) = \{(w,v_1), \cdots ,(w,v_k)\}$$，也就得到了最终删掉的$k$条不同的边，其中除$w$外的节点构成了节点覆盖$$V_c=\{v_1,\cdots , v_k\}$$。

所以以上两个决策问题等价 $P_{VC} \Longleftrightarrow  P_L$ 。

对于其他相似性指标，证明过程类似，就是在构造$Q$的时候有一些区别，比如：

- 对于CND指标，构造的时候添加一些孤立节点和$w$相连，因为这类指标跟共同邻居的度有关；
- 对于WCN指标，构造的时候为每个$v_i$添加一些邻居节点，保证它们的度都是正的；



### Practical Attack

由于一般情况下攻击局部度量是困难的，作者用了两种方法来获得较好的结果:近似算法和受限的特殊情况。

近似算法的思路是通过将$f_t$函数的分母进行常数化扩展，得到一个上下界区间。

对于WCN指标，令$g_{ij}$为 $\operatorname{Sim}\left(u_{i}, u_{j}\right)$ 的分母，我们将之扩展为$L_{ij} \leq g_{ij} \leq U_{ij}$，其中$L_{ij}$通过删除$k$条边获得，$U_{ij}$为原始分母。以`Sørensen`指标为例，$$\operatorname{sim}\left(u_{i}, u_{j}\right)=\frac{2 \mid N\left(u_{i}, u_{j}\right)\mid}{d\left(u_{i}\right)+d\left(u_{j}\right)}$$ ，扩展分母得： $d_{i}^{0}+d_{j}^{0}-k \leq d\left(u_{i}\right)+d\left(u_{j}\right) \leq d_{i}^{0}+d_{j}^{0}$，这样，相似性指标被扩展为：

$$
\frac{\left|N\left(u_{i}, u_{j}\right)\right|}{U_{i j}} \leq \operatorname{Sim}\left(u_{i}, u_{j}\right) \leq \frac{\left|N\left(u_{i}, u_{j}\right)\right|}{L_{i j}}  \\
 \Rightarrow \\
 f_{t l}^{W C N} \leq f_{t}^{W C N} \leq f_{t u}^{W C N}
$$

对于CND指标，分母$f_r(S_r)$扩展为$f_{r}\left(S_{r}^{0}\right)-k \leq f_{r}\left(S_{r}\right) \leq f_{r}\left(S_{r}^{0}\right)$，其中$S_r^0$表示原始决策矩阵$X^0$的第$r$行，这样，相似性指标被扩展为:


$$
\sum_{r=1}^{m} W_{r} \frac{\sum i j \  x_{r i} x_{r j}}{f_{r}\left(S_{r}\right)} \leq f_t^{WCN} \leq \sum_{r=1}^{m} W_{r} \frac{\sum i j \  x_{r i} x_{r j}}{f_{r}\left(S_{r}\right)-k} \\
\Rightarrow \\
f_{t l}^{C N D} \leq f_{t}^{C N D} \leq f_{t u}^{C N D}
$$


由于$f_t^{WCN}$和$f_t^{CND}$的结构相似性，后续的分析中关注$f_t^{WCN}$，并且省略上标$WCN$。



#### Optimizing Bounding Function

考虑最小化$f_{tu}$，令$S'$表示攻击者删除的边集合，$S'$与决策矩阵$X'$关联。对于任意$S \subset S'$，有$X \geq X'$，$X$与$S$关联。定义集合函数$F(S)=f_{t u}\left(X^{0}\right)-f_{t u}(X)$，最小化$f_{tu}(X)$等价于：


$$
\max _{S \subset E_{Q}} F(S), \quad \text { s.t. }|S| \leq k
$$


**Theorem 3.4.**  $F(S)$是单调递增的次模函数。 

**Proof.**  假设$S \subset S'$，$F(S) \leq F(S') \Leftrightarrow f_{t u}(X) \geq f_{t u}\left(X^{\prime}\right)$。令$C_i$是$X$的第$i$列（表示$$\{w_1,\cdots,w_m\}$$与$v_i$之间的连接情况），$u_i,u_j$的共同邻居$$|N(u_i,u_j)|=\langle C_i,C_j\rangle$$，$$\langle C_i,C_j\rangle$$表示内积。此时$$
f_{t u}(X)=\sum_{i j} \frac{w_{i j}\left\langle C_{i}, C_{j}\right\rangle}{L_{i j}}
$$ ，其中$w_{ij},L_{ij}$表示常数，因为$X \geq X'$，有$$
\left\langle C_{i}, C_{j}\right\rangle \geq \langle C_{i}^{\prime}, C_{j}^{\prime}\rangle  \Rightarrow f_{t u}(X) \geq f_{t u}\left(X^{\prime}\right)$$。得证单调递增。

下面证明$F(S)$为次模([submodular](https://www.zhihu.com/question/34720027))函数。

令边$e\notin S'$与决策矩阵的第$p$行第$q$列元素关联，令$e\cup S$与$X^e$关联。$X$与$X^e$的区别就是$X^e$是在$X$的基础上删除了边$e$，即$x_{pq}^e =0$且$x_{pq}=1$。令$e\cup S'$与$X'^e$关联。定义  $$
\Delta\left(e | S\right)=F\left(e \cup S\right)-F\left(S\right)
$$ 和  $$\Delta\left(e | S^{\prime}\right)=F\left(e \cup S^{\prime}\right)-F\left(S^{\prime}\right)$$。 下面证明$$\Delta(e | S) \geq \Delta\left(e | S^{\prime}\right)$$。


$$
\begin{aligned} \Delta(e | S) &=f_{t u}(X)-f_{t u}\left(X^{e}\right)=\sum_{j} \frac{w_{j q}}{L_{j q}}\left\langle C_{j}, C_{q}\right\rangle-\sum_{j} \frac{w_{j q}}{L_{j q}}\langle C_{j}^{e}, C_{q}^{e}\rangle \\ &=\sum_{j} \frac{w_{j q}}{L_{j q}} x_{p j} \cdot x_{p q}-\sum_{j} \frac{w_{j q}}{L_{j q}} x_{p j}^{e} \cdot x_{p q}^{e}=\sum_{j} \frac{w_{j q}}{L_{j q}} x_{p j} \end{aligned}
$$


其中$\sum_j$计算所有索引对$(j,q)$关联的边$(u_j,u_q)\in H$，删除$e$只会改变决策矩阵的第$q$列。

同理，$$
\Delta\left(e | S\right) - \Delta\left(e | S^{\prime}\right)=\sum_{j} \frac{w_{j q}}{L_{j q}} x_{p j}^{\prime}
$$， 则 $$
\Delta\left(e | S^{\prime}\right)=\sum_{j} \frac{w_{j q}}{L_{j q}}\left(x_{p j}-x_{p j}^{\prime}\right)
$$，因为 $$(x_{p j}-x_{p j}^{\prime}) \geq 0$$，则有$$\Delta\left(e | S\right) - \Delta\left(e | S^{\prime}\right) \geq 0$$。根据定义，$F(S)$为次模函数。

第三步，需要在基数约束下最小化这个单调递增的次模函数。这类问题典型的贪心算法能够实现最大值的$(1-1/e)$近似。贪心算法每一步删除一条能使$F(S)$产生最大增长的边，直到删除$k$条边。假设贪心算法输出一个次优解$S^\ast$，对应于$X_u^\ast$。令$f_t(X_u^\ast)$作为$f_t(X^\ast)$的近似。最后称该算法为*Approx-Local*。



#### Bound Analysis

