---
layout: article
title: 图神经网络：变分图自动编码器(VGAE)
date: 2020-01-19 00:10:00 +0800
tags: [GNN, Graph, Link Prediction]
categories: blog
pageview: true
key: GNN-Variational-Graph-Auto-Encoders
---



------
- 论文链接：[https://arxiv.org/abs/1611.07308](https://arxiv.org/abs/1611.07308)
- github链接：[https://github.com/tkipf/gae](https://github.com/tkipf/gae)
- papers with code: [https://paperswithcode.com/paper/variational-graph-auto-encoders](https://paperswithcode.com/paper/variational-graph-auto-encoders)
- GAE tutorial: [https://github.com/CrawlScript/TensorFlow-GAE-Tutorial](https://github.com/CrawlScript/TensorFlow-GAE-Tutorial)



## 引言

本文将变分自动编码器（variational auto-encoders）引入到图领域，提出了基于图数据的无监督学习框架——变分图自动编码器，基本思路是：对已知图进行编码学习到其节点向量表示的分布，然后从分布中采样得到节点的向量表示，进行解码重构图结构。

为了便于理解GAE，可以先了解[变分自动编码器的原理](https://jjzhou012.github.io/blog/2020/01/13/understanding-vae.html)。变分自编码器分为编码器和解码器，也就是推断模型和生成模型。推断模型将真实样本编码为低维向量表示（隐变量），学习隐变量的分布；生成模型从隐变量的分布中采样得到对应真实样本的隐变量，生成尽可能接近真实样本的数据。在这个过程中，引入了高斯噪声保证了模型的生成能力，利用了重参数技巧代替采样过程，保证了模型可训练。具体的原理细节参考：[博文](https://jjzhou012.github.io/blog/2020/01/13/understanding-vae.html)。

## 变分图自编码器（VGAE）

VGAE是一个隐变量模型，学习了**无向图**的可解释的潜在表示。

### 模型定义

对于一个无向无权图$\mathcal{G=(V,E)}$，节点数为$$N=\mid \mathcal{V} \mid$$，邻接矩阵为$$\mathbf{A}$$（加自环），度矩阵为$$\mathbf{D}$$，潜变量矩阵为$$\mathbf{Z}_{N \times F}$$，节点特征矩阵为$$\mathbf{X}_{N \times D}$$。

- 推断模型（编码器）

  推断模型由两层GCN实现：

  $$
  \operatorname{GCN}(\mathbf{X}, \mathbf{A})=\tilde{\mathbf{A}} \operatorname{ReLU}\left(\tilde{\mathbf{A}} \mathbf{X} \mathbf{W}_{0}\right) \mathbf{W}_{1}
  $$

  其中$$\mathbf{W}_i$$是权重矩阵， $$\tilde{\mathbf{A}}=\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}$$是对称标准化邻接矩阵。
  推断模型需要学习节点向量的分布，通过学习均值和方差来表示。隐变量的后验概率分布为：
$$
  q(\mathbf{Z} | \mathbf{X}, \mathbf{A})=\prod_{i=1}^{N} q\left(\mathbf{z}_{i} | \mathbf{X}, \mathbf{A}\right), \text { with } q\left(\mathbf{z}_{i} | \mathbf{X}, \mathbf{A}\right)=\mathcal{N}\left(\mathbf{z}_{i} | \boldsymbol{\mu}_{i}, \operatorname{diag}\left(\boldsymbol{\sigma}_{i}^{2}\right)\right)
$$

  其中$$\boldsymbol{\mu}=\mathrm{GCN}_{\boldsymbol{\mu}}(\mathbf{X}, \mathbf{A})$$就是节点向量表示的均值$$\boldsymbol{\mu}_i$$的矩阵表示，$$\log \boldsymbol{\sigma}=\operatorname{GCN}_{\boldsymbol{\sigma}}(\mathbf{X}, \mathbf{A})$$就是节点向量表示的方差$$\boldsymbol{\sigma}_i$$的矩阵表示。

  注意$$\mathrm{GCN}_{\boldsymbol{\mu}}(\mathbf{X}, \mathbf{A})$$和$$\operatorname{GCN}_{\boldsymbol{\sigma}}(\mathbf{X}, \mathbf{A})$$共享第一层的参数$$\mathbf{W}_0$$。


- 生成模型（解码器）

  生成模型通过计算两个隐变量的内积来重构图结构：
  
  $$
  p(\mathbf{A} | \mathbf{Z})=\prod_{i=1}^{N} \prod_{j=1}^{N} p\left(A_{i j} | \mathbf{z}_{i}, \mathbf{z}_{j}\right), \text { with } p\left(A_{i j}=1 | \mathbf{z}_{i}, \mathbf{z}_{j}\right)=\sigma\left(\mathbf{z}_{i}^{\top} \mathbf{z}_{j}\right)
  $$
  
  其中$\sigma(\cdot)$是logistic sigmoid函数。
  
- 损失函数

  边分下界的优化的损失函数：
  
  $$
  \mathcal{L}=\mathbb{E}_{q(\mathbf{Z} | \mathbf{X}, \mathbf{A})}[\log p(\mathbf{A} | \mathbf{Z})]-\operatorname{KL}[q(\mathbf{Z} | \mathbf{X}, \mathbf{A}) \| p(\mathbf{Z})]
  $$
  
  损失函数由两部分组成：
  
  - 生成图和原始图之间的距离度量，这里用加权交叉熵损失函数表示；
  - 节点向量表示的隐变量分布和标准正态分布的散度；
  
  注意到几点：
  
  - 这里的先验假设是$$p(\mathbf{Z})=\prod_{i} p\left(\mathbf{z}_{\mathbf{i}}\right)=\prod_{i} \mathcal{N}\left(\mathbf{z}_{i} \mid 0, \mathbf{I}\right)$$，即隐变量的分布符合标准正态分布；
  - 对于极度稀疏的$$\mathbf{A}$$，因为正负样本的极度不均衡带来的影响，所以在loss中对$A_{ij}=1$的项赋予新的权重是很有必要的；
  - 对于“采样操作”，这里使用了和VAE一样的“重参数”操作，保证模型可训练；
  - 对于没有特征的网络，使用单位矩阵代替特征矩阵；



## 图自动编码器（GAE）

图自编码器模型通过两层GCN实现：

生成节点向量表示：


$$
\mathbf{Z}=\operatorname{GCN}(\mathbf{X}, \mathbf{A})
$$


重构邻接矩阵：


$$
\hat{\mathbf{A}}=\sigma\left(\mathbf{Z} \mathbf{Z}^{\top}\right)
$$


其中$$\sigma(\cdot)$$是logistic sigmoid函数。

损失函数衡量生成图和原始图之间的距离：


$$
\mathcal{L}=\mathbb{E}_{q(\mathbf{Z} | \mathbf{X}, \mathbf{A})}[\log p(\mathbf{A} | \mathbf{Z})]
$$

## 实验

将VGAE和GAE用于链路预测任务。

模型的输入是不完整的观测图（图中的部分边被移除），所有的节点特征保留。测试集（10%）和验证集（5%）由移除的边和等量随机抽样的负样本构建。

论文对比了spectral clustering (SC)和DeepWalk(DW)，这两种方法都能生成节点嵌入向量$$\mathbf{Z}$$。利用公式$$\hat{\mathbf{A}}=\sigma\left(\mathbf{Z} \mathbf{Z}^{\top}\right)$$为重构邻接矩阵的元素打分（1或0）.

实验结果如下：

![2d1b6ffd9af2a2238381c9a17cdaff9.png](http://ww1.sinaimg.cn/large/005NduT8ly1gb1tttvwjgj30y30a3tb3.jpg)

GAE\*和VGAE\*表示无节点特征的实验。

VGAE和GAE都在无节点特征的任务上获得了具有竞争力的结果。添加节点特性可以显著提高预测性能。

最后，原文中提到了一点：

![9b91f8a8080f3f243c68b0e007a4dba.png](http://ww1.sinaimg.cn/large/005NduT8ly1gb1u3abvwxj30qc03qaba.jpg)

高斯先验假设和内积解码器结合是一个比较糟糕的选择，因为内积解码器会将嵌入向量推离零中心。

<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gb1u5d12qaj30di0d90vp.jpg" alt="c9cb795d62d34c2587236d3d42ca2cc.png" style="zoom:50%;" />

怎么理解呢，我觉得是因为encoder输出标准正态分布，而decoder做内积使得输出全都大于0，不符合标准正态分布，也就将嵌入向量推离了零中心。

