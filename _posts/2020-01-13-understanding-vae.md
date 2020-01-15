---
layout: article
title: 理解变分自编码器
date: 2020-01-13 00:11:00 +0800
tags: [Summary, Deep Learning]
categories: blog
pageview: true
key: Understanding-VAE
---



------


## 1. 泛泛而谈

变分自编码器（Variational auto-encoder，VAE）和传统的自编码器有很大的不同，是一类重要的生成模型（generative model）。它与GAN的区别在于，在VAE中我们知晓了输入数据的分布，而在GAN中我们不清楚数据的分布。

一般来说，每个样本数据都受到一些因素控制，这些因素可以称之为**潜变量**，用向量$z$表示。VAE和GAN的目标基本是一致的，希望构建一个从潜变量$$\mathbf{z}$$生成目标$$\mathbf{x}$$的模型。

> 潜变量就是观测不到的变量，相对于可观测变量而言。可观测变量由潜变量产生。数据中每个数据点$$\mathbf{x}$$都有一个相应的潜变量$$\mathbf{z}$$，$$\mathbf{z}$$通过某一个变换$$f(\mathbf{z} \mid \theta)$$可以得到$$\mathbf{x}$$，$\theta$是这个变换的参数，这就是所谓的潜变量模型。

首先我们有一批样本数据$$
\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\right\}
$$，其整体用$$\mathbf{x}$$表示，我们本想根据$$
\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\right\}
$$得到$$\mathbf{x}$$的分布$$p(\mathbf{x})$$，这样我们就能直接根据分布$$p(\mathbf{x})$$采样来得到所有可能的$$\mathbf{x}$$，这是理想的生成模型。但是原始的分布很难得到，我们换个思路，从潜变量生成观测数据。一个好的潜变量模型在于，它能通过变换$$f(\mathbf{z} \mid \theta)$$将潜变量的分布转换为观测数据$$\mathbf{x}$$的分布。换句话说，它能使观测数据$$\mathbf{x}$$的似然函数$$p(\mathbf{x})$$最大。通过极大似然法优化$$p(\mathbf{x})$$，首先需要得到它关于$$\mathbf{z}$$和$\theta$的表达式：



$$
p(\mathbf{x})=\int p(\mathbf{x} | \mathbf{z} ; \theta) p(\mathbf{z}) d \mathbf{z}
$$



在VAE中，为了优化公式（1），我们需要考虑两个问题：

- **如何定义潜变量$$\mathbf{z}$$；**

  首先我们不可能手工定义潜变量，因为潜变量$$\mathbf{z}$$作为一个多维向量，过于复杂的分布无法通过人工刻画。但是我们知道，混合高斯分布可以拟合任意分布，那么我们假设输入符合一个单位高斯分布$${p}(\mathbf{z}) \sim N(0, I)$$，然后通过神经网络去对这个符合高斯分布的各个维度进行混合，最终在某一层获得真正的多维潜变量，然后再用后续的神经网络进行分布变换$$f(\mathbf{z} \mid \theta)$$，将潜变量变换为$$\mathbf{x}$$。这一过程称之为解码。

- **如何处理积分的优化过程；**

  公式（1）其实就是$$p(\mathbf{x} | \mathbf{z} ; \theta)$$关于$$\mathbf{z}$$的期望。在优化的过程中，我们一定要取到能变换到$$\mathbf{x}$$的潜变量$$\mathbf{z}$$，只有这样才能通过优化参数最终获得期望的输出。但是如果潜变量的维度过高，对于大部分的$$\mathbf{z}$$，都不能生成与$$\mathbf{x}$$相似的样本，即$$p(\mathbf{x} | \mathbf{z} ; \theta)$$通常都接近于0。因此直接优化$$p(\mathbf{x})$$很难，所以我们还是得换一个优化目标。



## 2. 理论简述

那么既然$$p(\mathbf{x})$$这个概率分布中的很多潜变量$$\mathbf{z}$$不能被变换为与输入$$\mathbf{x}$$相似的样本，那么我们能否找到一个概率分布$$q(\mathbf{z})$$，使得这个$$q(\mathbf{z})$$分布中的$$\mathbf{z}$$被变换为我们需要的$$\mathbf{x}$$的概率更大呢？

既然$$\mathbf{z}$$能变换为$$\mathbf{x}$$，那么相应的某一个$$\mathbf{x}$$也能对应一个$$\mathbf{z}$$，这里我们引入一个$$\mathbf{z}$$的后验概率分布$$p(\mathbf{z} | \mathbf{x})$$。通过后验概率分布我们可以以最大概率获得能变换为所需$$\mathbf{x}$$的$$\mathbf{z}$$。此时如果$$q(\mathbf{z})$$就是$$p(\mathbf{z} | \mathbf{x})$$的话那就再好不过了，但是现在后验概率分布无法获得。所以我们需要引入这个$$q(\mathbf{z} | \mathbf{x})$$分布来近似后验概率分布$$p(\mathbf{z} | \mathbf{x})$$，这就是变分（variational）的思想。

VAE的模型图如下所示：

![1fe4afeb3000bb44790eb50ebda3c3a.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaxafq6ly2j30k206vwgl.jpg)

$$\mathbf{x}$$是我们可以观测到的数据，也就是输入数据，输入数据受隐变量$$\mathbf{z}$$控制，即$$\mathbf{x}$$由隐变量$$\mathbf{z}$$产生：

- $$\mathbf{x} \rightarrow \mathbf{z}$$: 从$$\mathbf{x}$$推断得到$$\mathbf{z}$$，生成$$\mathbf{z}$$的后验概率分布，视为推断模型，从自编码器的角度看就是编码器；
- $$\mathbf{z} \rightarrow \mathbf{x}$$: 从隐变量中采样数据$$\mathbf{z}$$映射为与$$\mathbf{x}$$相似的样本数据，视为生成模型$$p_{\theta}(\mathbf{x} | \mathbf{z})$$，从自编码器的角度看就是解码器；

那么为了使得$$q(\mathbf{z} | \mathbf{x})$$和$$p(\mathbf{z} | \mathbf{x})$$这两个分布尽可能的相似，可以通过最小化这两个分布之间的KL散度









### 2. 理论推导





## 参考

转载：https://kexue.fm/archives/5253

- https://zhuanlan.zhihu.com/p/55557709