---
layout: article
title: 链路预测的攻击：How to Hide One’s Relationships from Link Prediction Algorithms
date: 2019-12-13 00:08:00 +0800
tags: [Adversarial attack, Link prediction]
categories: blog
pageview: true
key: link-prediction-attack-1
---

------

论文链接： [How to Hide One’s Relationships from Link Prediction Algorithms](https://www.researchgate.net/publication/335309694_How_to_Hide_One's_Relationships_from_Link_Prediction_Algorithms)



## Introduction

可以说，数以亿计的Facebook用户都会收到好友推荐，而许多收到此类推荐的人都想知道，Facebook是如何预测那些从未在网上公开过的关系的。这些好友推荐以链路预测算法为指导，如果恶意使用，可能会侵犯我们选择公开哪些关系的基本权利。虽然这种政治上的担忧已经在文献中被提及，但这些研究却含蓄地假设了一个掌控全局的角色，而完全忽视了社会网络中的成员自己能够引发这种威胁的可能性。作者通过提出启发式算法来填补这一空白，这种启发式算法赋予了普通大众权力，为他们提供了一种随时可用的方法来隐藏他们认为敏感的任何关系，而不需要他们了解网络邻居之外的拓扑结构。虽然作者证明了确定隐藏这种关系的最佳方式是棘手的，但经验评估证明了他们的启发法在实践中是有效的，并揭示了解除精心挑选的朋友关系比建立朋友关系更能保护隐私。他们的分析还表明，链路预测算法在较小的网络和密度较高的网络中更容易被操纵。评估链路预测算法的容错性和攻击容忍度，可以发现要修改的连接的选择是至关重要的，因为随机选择可能会适得其反，最终暴露想要隐藏的关系。



## Motivation

- it is more beneficial to focus on “unfriending” carefully-chosen individuals rather than befriending new ones.

  删边优于加边，“解除关系”比“添加关系”更能隐藏身份。

- 在链路预测攻击（隐私保护）中，与数据受托人不同的是，这样一个自私自利的人丝毫不关心整个网络的匿名化，也不关心保护其属性。相反，他唯一的目标是保护自己的隐私，而不考虑对整个网络的影响。



## Problem Formulation

对于一个无向网络 $G=(V,E)$ ， 不存在的边`non-edges`集合表示为 $\bar{E}$ 。

- `seeker` : 对 $\bar E$ 中的 `non-edges` 按相似度指标进行排序，相似度越高的边越有可能是网络中的边，或者未来会出现的边。
- `evader` : 有一些未公开的关系需要去保密。这些未公开的关系就`seeker`而言是`non-edges`，需要去建模分析的。`evader`的目标是重连网络，以最小化那些`non-edges`被`seeker`发现的可能性。请注意，如果`non-edges`在所有`non-edges`中的相似性排名中下降，它在`seeker`前的暴露就会减少。

为了量化`non-edges`的暴露程度，使用了两个常见的评价指标：

- `AUC` : $H$作为测试集，AUC衡量了$H$中需要隐藏的边的分数大于随机`non-edge`的分数的概率，越低证明隐藏效果越好；
- `AP` : average precision，值在0~1之间，1说明$H$中需要隐藏的关系被完全暴露，0说明$H$中的关系全部隐藏；

直观地说，这些指标量化了相似性指标识别网络中缺失边的能力。本文中，缺失的边是`evader`未公开的关系，因此他的目标是最小化性能度量。形式上，`evader`所面临的问题定义如下:

**Definition 1** $\mathsf{(Evading Link Prediction)}$.  对于一个网络$G$， $H \subset \bar{E}$ 表示需要被隐藏的`non-edges`，$\hat{A} \subseteq \bar{E} \backslash H$表示被增加的边集合， $\hat{R} \subseteq E$表示被删除的边集合，$b \in \mathbb{N}$表示攻击预算（最大的边修改数量，增或删），$s_G:\bar{E} \rightarrow \mathbb{R}$ 表示链路预测使用的相似性指标，$$f\in \left\{AUC, AP \right\}$$ 表示评价指标。该任务的目标是确定增边和删边的集合 $A^* \subseteq \hat{A}$ 和  $R^* \subseteq \hat{R}$ ，使结果 $E^\ast = (E\cup A^\ast ) \backslash R^\ast $ 存在于：


$$
\underset{E^{\prime} \in \{(E \cup A) \backslash R: A \subseteq \hat{A}, R \subseteq \hat{R},\mid A\mid +\mid R\mid  \leq b\}}{\arg \min } f\left(E^{\prime}, H, s_{G}\right)
$$


该问题对于一些相似性指标而言是NP难的，但不意味着这个问题没有讨论的必要。相反有一些非最优的策略可以考虑。



## Heuristic

### The CTR heuristic.

**Definition** : $\mathsf{CTR (Closed-Triad-Removal)}$. 对一个`evader` $w$，希望隐藏$H$中包含的关系，通过删除一条边$(v,w) \in E$:

$$
\exists x \in V:((v, x) \in E) \wedge((x, w) \in H)
$$

>如何选择边删除：对$x$而言，$(v,x)\in E$，$(x,w)\in H$，满足条件时，$(v,w)$可删。

如Figure 1所示，移除边$(v,w)$会破环网络中的封闭三元组。注意虽然$(v,w,x)$构成了封闭三元组，但是由于$(x,w)$未发布，所以`seeker`不清楚$(v,w,x)$是否构成了封闭三元组，在`seeker`眼里$(x,w)$是`non-edge`。

移除边$(v,w)$会降低$(x,w)$的局部相似性分数。当移除一条边能破坏更多的封闭三元组，那么算法会更有效。

如Figure 1所示，三条边$(x,w),(y,w),(z,w) \in H$，CTR算法通过检查所有的候选边$(v,w)$，寻找能影响$H$中最大数量边的$(v,w)$。图中移除$(v,w)$能破坏三个封闭三元组$(v,w,x),(v,w,y),(v,w,z)$。换句话说，如果$w$希望隐藏他和$x,y,z$的关系，那么他需要去和更多的$x,y,z$的朋友解除好友关系。

![845cf8326ccb4e6f690d383efcf7ed2.png](http://ww1.sinaimg.cn/mw690/005NduT8ly1g9v7zs7ms5j310n0id0vr.jpg)



### The OTC heuristic

Definition  : $\mathsf{OTC(Open-Triad-Creation)}$. 想要隐藏目标边$e\in H$，不仅可以通过降低其相似性分数，也可以增加其领域内其他边的相似性分数来间接降低目标边$e$的分数排名，达到隐藏目标边的效果。OTC检查了$(v, w)$的所有可能选择，并选择了一种能最大程度降低$H$中`non-edges`排列的方案，同时确保在此过程中$H$中没有其他`non-edges`暴露。

- $\exists u \in V:((w, u) \in H) \wedge((v, u) \notin E)$ ;
- $\exists x \in V:((x, v) \in E) \wedge((x, w) \in \bar{E} \backslash H)$ ;

> 虽然没看懂这式子写的什么玩意儿，但大胆猜测：
>
> 1. $(w,u)$是要隐藏的关系，在连接$(v,w)$之后，它不能在封闭三角形中，这样会增大它的相似性分数，所以$(v,u)$之间不能存在关系；
> 2. $(v,x)$对于$w$而言是连接紧密的陌生人，$w$和其中一个陌生人$v$建立关系，可以提升$x,w$的相似性分数，前提是$(x,w)$不能是要隐藏的关系，即$(x,w) \notin H = \bar{E}\backslash H$ ；

如 Figure 2 所示，对于一个`evader`$w$想要隐藏关系$(w,u)$，可以通过与陌生人$v$建立关系$(w,v)$，来提升$(x,w)$和$(y,v)$的相似性分数，从而提升它们的相似度排名，间接降低$(w,u)$的相似度排名。

在社交平台上，OTC方法可以按如下方式应用：如果$u$和$w$想要隐藏它们之间的关系，对于其中的$w$来说，他可以向一些人发送好友请求，这些人的好友和$w$也不是好友关系（越多越好）；或者说，向那些有高连接密度的陌生人发送好友申请。

![c7756d480e15bf43950bdb6bd7a88cc.png](http://ww1.sinaimg.cn/mw690/005NduT8ly1g9ybicnclcj30s00bfmyy.jpg)



## Evaluation

根据测试集计算所有节点对的相似性分数，计算测试集的AUC和AP。

### 实验一：两种启发式算法的攻击效果评估

- 目标网络：电话通讯网络，248763个节点，829725条边；
- $b=5$； 
- 对每个相似性指标$s_G$，随机选择10个`evader`，每个`evader`度不小于9，防止在删边的时候变成孤立节点；
- 对于每个`evader`，构造五个$H$，每个$H$包含随机选择的三条边；
- 共产生$5*10=50$个实验；



![1456a0059ab603809f104d3fc953279.png](http://ww1.sinaimg.cn/mw690/005NduT8ly1g9yfyly4foj30rv0ihdjg.jpg)



可以看出，OTC的影响似乎可以忽略不计，CTR似乎更有效，两种启发式混合似乎没有产生任何协同效应。这表明，为了隐藏一段关系，一个人应该主要关注选择“解除好友关系”，而不是结交新朋友。

同时也发现，OTC在小网络上会比较有效，CTR在小网络上依然很有效。



### 实验二：攻击容错率的参数分析

基于AUC和AP来评估相似性指标的攻击容错率。

分析了无标度网络的两个变量的影响：

- 网络节点数$$n \in \left\{200,400,600,800 \right\}$$

- 平均度 $$d\in \left\{4,6,8,10\right\}$$

  

具体来说，对于$n$和$d$的每个组合，我们生成50个无标度网络;对于每个这样的网络$G$，我们随机选择10个`evader`(度不小于9);对于每一个这样的`evader` $v$，我们创建了5个不同的集合$H$，每个集合$H$由$v$的3条随机边组成。$(G, v, H)$的每个实例构成一个单独的实验。

总体而言，这些相似度指标的攻击容错率随$n$的增大而增大，随$d$的增大而减小。

- 小网络上攻击成功率高；
- 平均度越大，关系越多，攻击成功率越高；

所以，在越大规模、越稀疏的网络上，越难以隐藏关系。



![a24b66f59733e282ff61f971e245da0.png](http://ww1.sinaimg.cn/large/005NduT8ly1g9ykm9m9ozj30qx0jstep.jpg)



### 实验三：重连方式对攻击效果的影响

启发式方法指导的重连获得的攻击效果，是否能通过随机重连复现？

启发式算法获取候选的边集合，然后选择最优边进行增删；随机重连在启发式算法获得的候选边集合中进行随机选择，然后增删。

通过比较，可以得出：

- 启发式的策略性重连优于随机重连；
- 随机重连可以作为baseline;
- 随机重连甚至还会暴露需要隐藏的关系，因为部分结果比值大于1；也就是说 :open_mouth: 随意行事反而会适得其反，最终暴露隐私！！！





![f945b4beffbdce5390173540f0576a2.png](http://ww1.sinaimg.cn/large/005NduT8ly1g9ym6vp4fmj30rp0f142c.jpg)



## Method

令 $$N_G(v)=\left\{w\in V: (v,w) \in E \right\}$$ 表示节点 $v$ 的邻居节点集合。令 $N_G(v,w) = N_G(v) \cap N_G(w)$ 表示节点 $v$ 和 $w$ 的共同邻居。 令 $d_G(v)= \mid N_G(v) \mid$ 表示节点 $v$ 的度。

对于每一个`non-edge`的相似性分数，$(x,w) \in \bar{E}$，取决于下列因素：

- *Factor 1: `non-edge`的共同邻居数*。 $\forall s \in S$ ， $s(x,w)$ 随着 $\mid N(x,w) \mid$ 的增大而增大。

- *Factor 2: `non-edge`两端节点的度*。 $$\forall s \in S\backslash \left\{s^{CN},s^{AA},s^{RA}  \right\}$$ ，如果 $N(x,w) \neq \emptyset$ ，$s(x,w)$ 随着 $d(x),d(w)$ 的增大而减小；否则若 $N(x,w) = \emptyset$ ， $s(x,w)$ 不受 $d(x),d(w)$ 的影响。

  对于 $$s \in \{s^{CN},s^{AA},s^{RA} \}$$ ，无论是否有共同邻居， $s(x,w)$ 都不受 $d(x),d(w)$ 的影响。

- *Factor 3: 共同邻居的度*。对于 $$s \in \{s^{AA},s^{RA} \}$$ ，$s(x,w)$随着共同邻居的度增大而减小；其他的相似性指标不受影响。

<!--增加一条边$(v,w)$，能够影响下列类型的边：-->



