---
layout: article
title: 链路预测的攻击：How to Hide One’s Relationships from Link Prediction Algorithms
date: 2019-12-13 00:08:00 +0800
tags: [Adversarial attack, Link prediction]
categories: blog
pageview: true
---

------

论文链接： [How to Hide One’s Relationships from Link Prediction Algorithms](https://www.researchgate.net/publication/335309694_How_to_Hide_One's_Relationships_from_Link_Prediction_Algorithms)



## Motivation 

- it is more beneficial to focus on “unfriending” carefully-chosen individuals rather than befriending new ones.

  删边优于加边，“解除关系”比“添加关系”更能隐藏身份。

- 在链路预测攻击（隐私保护）中，与数据受托人不同的是，这样一个自私自利的人丝毫不关心整个网络的匿名化，也不关心保护其属性。相反，他唯一的目标是保护自己的隐私，而不考虑对整个网络的影响。



## Problem Formulation

对于一个无向网络 $G=(V,E)$ ， 不存在的边`non-edges`集合表示为 $\bar{E}$ 。

- `seeker` : 对 $\bar E$ 中的 `non-edges` 按相似度指标进行排序，相似度越高的边越有可能是网络中的边，或者未来会出现的边。
- `evader` : 有一些未公开的关系需要去保密。这些未公开的关系就`seeker`而言是`non-edges`，需要去建模分析的。`evader`的目标是重连网络，以最小化那些`non-edges`被`seeker`发现的可能性。请注意，如果`non-edges`在所有`non-edges`中的相似性排名中下降，它在`seeker`前的暴露就会减少。

为了量化`non-edges`的暴露程度，使用了两个常见的评价指标：

- `AUC` : area under the ROC curve
- `AP` : average precision

直观地说，这些指标量化了相似性指标识别网络中缺失边的能力。本文中，缺失的边是`evader`未公开的关系，因此他的目标是最小化性能度量。形式上，`evader`所面临的问题定义如下:

**Definition 1** $\mathsf{(Evading Link Prediction)}$.  对于一个网络$G$， $H \subset \bar{E}$ 表示需要被隐藏的`non-edges`，$\hat{A} \subseteq \bar{E} \backslash H$表示被增加的边集合， $\hat{R} \subseteq E$表示被删除的边集合，$b \in \mathbb{N}$表示攻击预算（最大的边修改数量，增或删），$s_G:\bar{E} \rightarrow \mathbb{R}$ 表示链路预测使用的相似性指标， $f\in \{AUC, AP\}$ 表示评价指标。该任务的目标是确定增边和删边的集合 $A^* \subseteq \hat{A}$ 和  $R^* \subseteq \hat{R}$ ，使结果 $E^\ast = (E\cup A^\ast ) \backslash R^\ast $ 存在于：


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

如Figure 1所示，移除边$(v,w)$会破环网络中的封闭三元组。注意虽然$(v,w,x)$构成了封闭三元组，但是由于$(x,w)$未发布，所以`seeker`不清楚$(v,w,x)$是否构成了封闭三元组，在`seeker`眼里$(x,w)$是`non-edge`。

移除边$(v,w)$会降低$(x,w)$的局部相似性分数。当移除一条边能破坏更多的封闭三元组，那么算法会更有效。

如Figure 1所示，三条边$(x,w),(y,w),(z,w) \in H$，CTR算法通过检查所有的候选边$(v,w)$，寻找能影响$H$中最大数量边的$(v,w)$。图中移除$(v,w)$能破坏三个封闭三元组$(v,w,x),(v,w,y),(v,w,z)$。换句话说，如果$w$希望隐藏他和$x,y,z$的关系，那么他需要去和更多的$x,y,z$的朋友解除好友关系。


![845cf8326ccb4e6f690d383efcf7ed2.png](http://ww1.sinaimg.cn/mw690/005NduT8ly1g9v7zs7ms5j310n0id0vr.jpg)

