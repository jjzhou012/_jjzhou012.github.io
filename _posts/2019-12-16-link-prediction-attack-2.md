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

