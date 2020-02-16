---
layout: article
title: 图神经网络：用于链路预测的GAT模型（DeepLinker）
date: 2020-02-15 00:10:00 +0800
tags: [Link Prediction, GNN]
categories: blog
pageview: true
key: Link-Prediction-via-Graph-Attention-Network
---

------

- 论文链接：[https://ui.adsabs.harvard.edu/abs/2019arXiv191004807G/abstract](https://ui.adsabs.harvard.edu/abs/2019arXiv191004807G/abstract)

- github:[https://github.com/Villafly/DeepLinker](https://github.com/Villafly/DeepLinker)



## 框架

DeepLinker结合GAT和编码-解码框架。完整的框架示意图如下所示。

1. 编码器：对目标节点采样固定数量的1阶和2阶邻居，利用GAT框架生成节点的向量表示；
2. 解码器：利用得到的节点向量生成边（节点对）的向量表示；
3. 输出：对边向量利用一个打分函数计算边存在的概率；

具体而言：

- 固定数量的邻域采样

  使用固定大小的邻域采样策略来解决内存瓶颈和小批量问题。在GAT中考虑了所有邻域节点，而DeepLinker中在每个节点的邻域中均匀采样固定数量的节点，且只采样一次，并在整个训练过程中固定邻域。

- 计算边向量

  通过计算两个节点向量的Hadamard product来得到边（节点对）的向量表示（d-维）：

  
  $$
  e(i, j)=\left(\vec{h}_{i}^{\prime} \odot \vec{h}_{j}^{\prime}\right)
  $$



- 计算边存在的概率

  通过可训练的权重向量和逻辑回归计算边的概率：

  
  $$
  p_{i j}\left(e_{i j} ; \theta\right)=\frac{1}{1+\exp \left(e_{i j}^{T} \theta\right)}
  $$



​	其中$\theta$是d维的参数向量，$e_{i j}^{T} \theta$是边向量和参数向量之间的点积。

![36af3d6bdc9aae1ffb4d0e8ec7c54a5.png](http://ww1.sinaimg.cn/large/005NduT8ly1gbxbwtkm35j30sa0eu762.jpg)



训练过程中，损失函数为交叉熵：


$$
\mathcal{L}=-\frac{1}{\left|\varepsilon \cup \varepsilon^{-}\right|} \sum_{(i, j) \in \varepsilon \cup \varepsilon^{-}} y_{i, j} \log p(i, j)+\left(1-y_{i, j}\right) \log (1-p(i, j))
$$


其中$y_{i,j}$为节点对$(i,j)$的标签，$\varepsilon$为训练集。实验中正负样本均衡采样。