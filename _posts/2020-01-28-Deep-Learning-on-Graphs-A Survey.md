---
layout: article
title: 图的深度学习综述：Deep Learning on Graphs-A Survey
date: 2020-01-28 00:10:00 +0800
tags: [GNN, Graph, Deep Learning]
categories: blog
pageview: true
key: Deep-Learning-on-Graphs-A Survey
---



## 5. Graph AutoEncoders(GAEs)

自编码器(Autoencoder, AE)及其变体广泛应用于无监督学习，适用于学习图结构数据的节点和结构表示。其隐含的假设是图具有固有的、潜在的非线性低秩结构。图自编码器模型自提出以来，以后很多的优化和改进，产生许多变体。

### AutoEncoder

在图结构数据上应用自编码器，这一思路源于稀疏自编码器（SAE）。基本的思想是将图的邻接矩阵或其变体视为节点的原始特征，自编码器可以作为一种降维方法来学习节点的低维向量表示。具体而言，稀疏自编码器采用L2重构损失函数：


$$
\begin{aligned}
\min _{\Theta} \mathcal{L}_{2} &=\sum_{i=1}^{N}\|\mathbf{P}(i,:)-\hat{\mathbf{P}}(i,:)\|_{2} \\
\hat{\mathbf{P}}(i,:) &=\mathcal{G}\left(\mathbf{h}_{i}\right), \mathbf{h}_{i}=\mathcal{F}(\mathbf{P}(i,:))
\end{aligned}
$$

其中$$\mathbf{P}$$是转移，$$\mathbf{\hat{P}}$$是重构的矩阵，$$
\mathbf{h}_{i} \in \mathbb{R}^{d}
$$是节点$v_i$的低维向量表示，$$
\mathcal{F}(\cdot)
$$是编码器，$$
\mathcal{G}(\cdot)
$$是解码器。编码器和解码器均为多层感知机。换句话说，SAE将图的原始邻接信息压缩为低维向量表示，然后重构原始节点特征。然而SAE的理论基础存在错误，其有效性机制尚未得到有效的解释。

- Structure Deep Network Embedding (SDNE)

  SDNE模型则提出了，SAE中的L2重构损失实际上代表了节点的二阶近似，也就是说，拥有相似邻域结构的节点共享相似的隐变量表示。但其实网络中节点的一阶近似也很重要，直接相连的节点也有更大概率共享相似的隐变量表示。所以最终SDNE模型结合图节点的一阶近似和二阶近似，改进了SAE原有的重构损失：

  $$
  \begin{aligned}
  \min _{\Theta} \mathcal{L}_{2}+\alpha \sum_{i, j=1}^{N} \mathbf{A}(i, j)\left\|\mathbf{h}_{i}-\mathbf{h}_{j}\right\|_{2} \\
  \mathcal{L}_{2}=\sum_{i=1}^{N}\left\|\left(\mathbf{A}(i,:)-\mathcal{G}\left(\mathbf{h}_{i}\right)\right) \odot \mathbf{b}_{i}\right\|_{2} 
  
  \end{aligned}
  $$

  其中$$\mathbf{A}$$是图的邻接矩阵，节点的低维向量表示由邻接矩阵进行编码得到，即$$\mathbf{h}_{i}=\mathcal{F}(\mathbf{A}(i,:))$$，同时对邻接矩阵中的不同元素（0或非0）赋予不同的权重，即：

  $$
  b_{i j}=\left\{\begin{array}{ll}
  {1 ;} & {\mathbf{A}(i , j)=0} \\
  {\beta>1} & {; \text { other }}
  \end{array}\right.
  $$
  
- Deep neural networks for learning graph representations(DNGR)

  DNGR模型将重构损失中的转移矩阵$$\mathbf{P}$$替换为positive pointwise mutual information (PPMI) 。这样做可以将节点的原始特征和随机游走概率相结合。但是该模型在构建输入矩阵是具有较大的时间复杂度。

- Graph convolutional matrix completion(GC-MC)

  GC-MC使用GCN作为编码器：

  
  $$
  \mathbf{H}=G C N\left(\mathbf{F}^{V}, \mathbf{A}\right)
  $$
  
  使用双线性函数作为解码器：
  
  $$
\hat{\mathbf{A}}(i, j)=\mathbf{H}(i,:) \Theta_{d e} \mathbf{H}(j,:)^{T}
  $$
  
- Deep recursive network embedding with regular equivalence(DRNE)

  DRNE利用LSTM来聚集邻居信息直接重建节点低维向量，通过最小化如下损失：
  
  $$
\mathcal{L}=\sum_{i=1}^{N}\left\|\mathbf{h}_{i}-\operatorname{LSTM}\left(\left\{\mathbf{h}_{j} | j \in \mathcal{N}(i)\right\}\right)\right\|
  $$
  
  由于LSTM要求输入是一个序列，因此作者建议根据节点的度对节点的邻域进行排序。对于度较大的节点，利用采样策略防止内存过大。
  
- Deep gaussian embedding of graphs: Unsupervised inductive learning via ranking(Graph2Gauss)

  G2G将每个节点编码为高斯分布$$
  \mathbf{h}_{i}=\mathcal{N}(\mathbf{M}(i,:), \operatorname{diag}(\mathbf{\Sigma}(i,:)))$$来捕获节点的不确定性。具体来说，作者使用了从节点属性到高斯分布均值和方差的深度映射作为编码器:
  
  $$
  \mathbf{M}(i,:)=\mathcal{F}_{\mathbf{M}}\left(\mathbf{F}^{V}(i,:)\right), \boldsymbol{\Sigma}(i,:)=\mathcal{F}_{\boldsymbol{\Sigma}}\left(\mathbf{F}^{V}(i,:)\right)
  $$
  
  使用成对约束来学习模型：
  
  $$
  \begin{aligned}
  & \mathrm{KL}\left(\mathbf{h}_{j} \| \mathbf{h}_{i}\right) <\mathrm{KL}\left(\mathbf{h}_{j^{\prime}} \| \mathbf{h}_{i}\right) \\
  & \forall i, \forall j, \forall j^{\prime} s . t . d(i, j)<d\left(i, j^{\prime}\right)
  \end{aligned}
  $$
  
  也就是说，距离越远的两个节点，它们对应的两个分布之间的KL散度越大。
  
  上面的方程难以优化，转化为基于能量的损失：
  
  $$
  \mathcal{L}=\sum_{\left(i, j, j^{\prime}\right) \in \mathcal{D}}\left(E_{i j}^{2}+\exp ^{-E_{i j^{\prime}}}\right)
  $$
  
  其中$$\mathcal{D}=\left\{\left(i, j, j^{\prime}\right) | d(i, j)<d\left(i, j^{\prime}\right)\right\}$$，$$
  E_{i j}=\mathrm{KL}\left(\mathbf{h}_{j} \| \mathbf{h}_{i}\right)$$。



### Variational Autoencoders

- VGAE：见[https://jjzhou012.github.io/blog/2020/01/19/GNN-Variational-graph-auto-encoders.html](https://jjzhou012.github.io/blog/2020/01/19/GNN-Variational-graph-auto-encoders.html)

