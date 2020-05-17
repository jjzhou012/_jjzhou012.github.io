---
layout: article
title: 图神经网络--对抗训练：对抗正则化图自编码器（ARGA）
date: 2020-02-16 00:10:00 +0800
tags: [Link Prediction, GNN, Graph]
categories: blog
pageview: true
key: Adversarially-regularized-graph-autoencoder-for-graph-embedding
---

------

- 论文链接：[https://arxiv.org/pdf/1802.04407v2.pdf](https://arxiv.org/pdf/1802.04407v2.pdf)
- github: [https://github.com/Ruiqi-Hu/ARGA](https://github.com/Ruiqi-Hu/ARGA)



## 引言

论文提出了一种基于图自编码器的对抗图嵌入框架，利用编码器将图结构和节点编码为潜在的低维向量表示，再利用解码器重构图结构，在编码解码的过程中引入了对抗训练机制，保证潜在的向量表示符合图结构数据的先验分布。

## 框架

ARGA利用对抗训练机制和图自编码器，直接对整个图进行处理，学习鲁棒性的嵌入向量。ARGA由两部分组成：

- 图卷积自编码器

  图自编码器从图结构信息$$\mathbf{A}$$和节点特征$$\mathbf{X}$$中学习节点的潜变量表示$$\mathbf{Z}$$，然后利用$$\mathbf{Z}$$重构图结构。

- 对抗正则化

  对抗网络利用对抗训练机制，限制潜变量符合先验分布，判别器用于判断生成的潜变量来自于编码器还是先验分布。

![ARGA_FLOW.jpg](http://ww1.sinaimg.cn/large/005NduT8ly1gbzfdyznauj31lf0n9acy.jpg)



### 图卷积自编码器

自编码器分为编码器和解码器两部分，

#### 编码器

其中编码器由传统的图卷积层（GCN）层构成，每层图卷积层能表示为：


$$
f\left(\mathbf{Z}^{(l)}, \mathbf{A} \mid \mathbf{W}^{(l)}\right)=\phi\left(\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{Z}^{(l)} \mathbf{W}^{(l)}\right)
$$


其中$$
\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}
$$，$$\tilde{\mathbf{D}}_{i i}=\sum_{j} \tilde{\mathbf{A}}_{i j}$$，$$\mathbf{Z}^{(l)}$$是第$l$层的输入，$\phi$为激活函数。编码器由两层GCN组成：


$$
\begin{aligned}
\mathbf{Z}^{(1)} &=f_{\text {Relu }}\left(\mathbf{X}, \mathbf{A} \mid \mathbf{W}^{(0)}\right) \\
\mathbf{Z}^{(2)} &=f_{\text {linear }}\left(\mathbf{Z}^{(1)}, \mathbf{A} \mid \mathbf{W}^{(1)}\right)
\end{aligned}
$$


主要作用是通过图卷积编码器$$
\mathcal{G}(\mathbf{Z}, \mathbf{A})=q(\mathbf{Z} \mid \mathbf{X}, \mathbf{A})
$$将图结构和节点特征编码成潜在的向量表示$$
\mathbf{Z}=q(\mathbf{Z} \mid \mathbf{X}, \mathbf{A})=\mathbf{Z}^{(2)}
$$。

变分编码器通过学习均值和方差来生成符合特定分布的潜变量，变分图自编码器的编码部分如下：


$$
\begin{aligned}
q(\mathbf{Z} \mid \mathbf{X}, \mathbf{A}) &=\prod_{i=1}^{n} q\left(\mathbf{z}_{\mathbf{i}} \mid \mathbf{X}, \mathbf{A}\right) \\
q\left(\mathbf{z}_{\mathbf{i}} \mid \mathbf{X}, \mathbf{A}\right) &=\mathcal{N}\left(\mathbf{z}_{i} \mid \boldsymbol{\mu}_{i}, \operatorname{diag}\left(\boldsymbol{\sigma}^{2}\right)\right)
\end{aligned}
$$

其中$$
\boldsymbol{\mu}=\mathbf{Z}^{(2)}
$$是潜变量的均值矩阵，$$\sigma^2$$是方差矩阵，均值和方差在第一层共享权重$$\mathbf{W^{(0)}}$$。

#### 解码器

解码器部分主要利用编码器生成的潜变量来重构图结构。重构的图结构可以用于预测两个节点之间是否存在链接（即链路预测）。

具体而言，解码器基于潜变量训练一个链路预测层：


$$
\begin{aligned}
p(\hat{\mathbf{A}} \mid \mathbf{Z}) &=\prod_{i=1}^{n} \prod_{j=1}^{n} p\left(\hat{\mathbf{A}}_{i j} \mid \mathbf{z}_{i}, \mathbf{z}_{j}\right) \\ 
p\left(\hat{\mathbf{A}}_{i j}=1 \mid \mathbf{z}_{i}, \mathbf{z}_{j}\right) &=\operatorname{sigmoid}\left(\mathbf{z}_{i}^{\top}, \mathbf{z}_{j}\right)
\end{aligned}
$$



也就是潜变量作内积，在通过逻辑回归激活函数，得到每个节点对之间的边存在概率。

#### 图自编码器模型

整个图自编码器模型可以用以下公式表达：


$$
\hat{\mathbf{A}}=\operatorname{sigmoid}\left(\mathbf{Z} \mathbf{Z}^{\top}\right), \text { here } \mathbf{Z}=q(\mathbf{Z} \mid \mathbf{X}, \mathbf{A})
$$


#### 优化

对于图自编码器模型，通过优化重构误差：


$$
\mathcal{L}_{0}=\mathbb{E}_{q(\mathbf{Z} \mid(\mathbf{X}, \mathbf{A}))}[\log p(\hat{\mathbf{A}} \mid \mathbf{Z})]
$$


对于变分图自编码器，通过优化重构误差和潜变量分布与先验分布的KL散度：


$$
\mathcal{L}_{1}=\mathbb{E}_{q(\mathbf{Z} \mid(\mathbf{X}, \mathbf{A}))}[\log p(\hat{\mathbf{A}} \mid \mathbf{Z})]-\mathbf{K} \mathbf{L}[q(\mathbf{Z} \mid \mathbf{X}, \mathbf{A}) \Vert p(\mathbf{Z})]
$$



其中 $$p(\mathbf{Z})=\prod_{i} p\left(\mathbf{z}_{i}\right)=\prod_{i} \mathcal{N}\left(\mathbf{z}_{i} \mid 0, \mathbf{I}\right)$$为先验高斯分布。



### 对抗模型

为了保证生成的潜变量分布符合先验高斯分布，ARGA通过一个对抗训练来实现。对抗模型建立在一个标准的多层感知器(MLP)上，其中输出层通过sigmoid函数输出一维。对抗模型作为一个判别器，用来区分潜变量来源于先验分布还是编码器。通过最小化训练二分类器的交叉熵损失，使嵌入在训练过程中最终得到正则化和改进：


$$
-\frac{1}{2} \mathbb{E}_{\mathbf{z} \sim p_{z}} \log \mathcal{D}(\mathbf{Z})-\frac{1}{2} \mathbb{E}_{\mathbf{X}} \log (1-\mathcal{D}(\mathcal{G}(\mathbf{X}, \mathbf{A})))
$$


整个对抗图自编码器模型的训练损失可以表达如下：


$$
\min _{\mathcal{G}} \max _{\mathcal{D}} \mathbb{E}_{\mathbf{z} \sim p_{z}}[\log \mathcal{D}(\mathbf{Z})]+\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})}[\log (1-\mathcal{D}(\mathcal{G}(\mathbf{X}, \mathbf{A})))]
$$


相当于GAN的机理，只不过这里的生成模型$$
\mathcal{G}(\mathbf{X}, \mathbf{A})
$$由图自编码器的编码器部分构成，判别模型$$
\mathcal{D}(\mathbf{Z})
$$单独构造。



### 训练过程

1. 利用图卷积编码器生成潜变量$$\mathbf{Z}$$；
2. 从原始数据和潜变量中均衡采样得到数据，用于训练判别器；
3. $K$轮训练后，图卷积编码器更新梯度信息；
4. 重复以上步骤，最终得到较好的编码器和判别器；

