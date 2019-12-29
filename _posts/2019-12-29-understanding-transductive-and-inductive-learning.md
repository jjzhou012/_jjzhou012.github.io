---
layout: article
title: 理解“直推（transductive）学习”和“归纳（inductive）学习” 等
date: 2019-12-29 00:10:00 +0800
tags: [Machine learning]
categories: blog
pageview: true
key: understanding-transductive-and-inductive-learning
---



------



## 归纳学习(Inductivev learning)

考虑普通学习任务，训练集为$$
\mathcal{D}=\left\{\mathbf{X}_{t r}, \mathbf{y}_{t r}\right\}
$$，测试集（未标注）$\mathbf{X}_{t e}$，测试集不会出现在训练集中，这种情况就是归纳学习。

归纳学习是基于“开放世界” 的假设，可以泛化到未知的样本，对新增样本可以快速预测，无需额外训练过程。



## 半监督学习(Semi-supervised learning)

半监督学习是监督学习与无监督学习相结合的一种学习方法，它主要考虑如何利用**少量的标注样本**和**大量的未标注样本**进行训练和分类的问题。

半监督学习的情况下，训练集为$$
\mathcal{D}=\left\{\mathbf{X}_{t r}, \mathbf{y}_{t r}, \mathbf{X}_{un}\right\}
$$，测试集$$\mathbf{X}_{t e}
$$，此时$$\mathbf{X}_{un}$$$与$$$\mathbf{X}_{t e}$$都是未标注的，但是测试的$$\mathbf{X}_{t e}$$训练时未见过。



## 直推学习(Transductive learning)

直推学习情况下，训练集为$$
\mathcal{D}=\left\{\mathbf{X}_{t r}, \mathbf{y}_{t r}, \mathbf{X}_{un}\right\}
$$，测试集$\mathbf{X}_{un}$，由于在训练时我们已经见过$$\mathbf{X}_{un}$$（利用了$$\mathbf{X}_{un}$$的特征信息），这时就叫做直推学习。

很多人说用了测试数据是一种作弊行为，实则不然，因为这里只是用了**测试数据的特征**，而其标签我们是不知道的。

直推学习假设未标记的数据就是最终要用来测试的数据，学习的目的就是在这些数据上取得最佳泛化能力。

直推学习是基于“封闭世界”的假设，模型不具备对未知数据的泛化能力



## 总结

- 直推学习类似于半监督学习的一个子问题，或者说是一个特殊化的半监督学习问题，所以有时也归为半监督学习。
- 根据上条，一般认为半监督学习包含**纯半监督学习**和**直推学习**。
- 根据上条，**纯半监督学习**是一种**归纳学习。**