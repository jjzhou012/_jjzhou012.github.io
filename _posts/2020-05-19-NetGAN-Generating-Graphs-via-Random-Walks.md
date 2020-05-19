---
layout: article
title: 图神经网络--对抗训练：NetGAN
date: 2020-05-19 00:10:00 +0800
tags: [Adversarial, GNN, Link Prediction, Node Classification, Graph]
categories: blog
pageview: true
key: NetGAN-Generating-Graphs-via-Random-Walks
---



------

------

- Paper: [NetGAN: Generating Graphs via Random Walks](https://arxiv.org/pdf/1803.00816.pdf)
- Code: [https://github.com/danielzuegner/netgan](https://github.com/danielzuegner/netgan)



## 模型

NetGAN的核心思想是通过学习随机游走的分布来捕捉网络的拓扑结构的。

对于一个给定的图，首先采样获得长度为$T$的随机游走序列，作为模型的输入。游走策略使用node2vec的有偏二阶随机游走。随机游走可以维持节点排序的不变性，同时随机游走只包含$A$的非零项，因此能有效地利用真实世界网络的稀疏性。

NetGAN采用了典型的GAN架构，由生成器$G$和判别器$D$构成。生成器用于生成可靠的合成随机游走序列，判别器用来判断随机游走序列来自于生成器合成还是来自于真实分布。

### 框架

#### 生成器

生成器定义了一个隐式概率模型用于生成随机游走序列$\left(\boldsymbol{v}_{1}, \ldots, \boldsymbol{v}_{T}\right) \sim G$。

生成器被建模为一个序列化过程，基于神经网络$f_{\theta}$，每次迭代$t$，$f_{\theta}$生成两个值：

- 采样下个节点的概率分布$\boldsymbol{p}_{t}$;
- 模型当前的记忆状态$\boldsymbol{m}_t$;

从类的概率分布$\boldsymbol{v}_{t} \sim \operatorname{Cat}\left(\sigma\left(\boldsymbol{p}_{t}\right)\right)$中采样得到下一节点$\boldsymbol{v}_t$，在下一次迭代时和$\boldsymbol{m}_t$一起传入$f_{\theta}$。

从多元标准正态分布中提取潜变量$\boldsymbol{z}$，用于初始化记忆状态$\boldsymbol{m}_{0}=g_{\theta^{\prime}}(\boldsymbol{z})$。

生成器的生成过程总结如下：

![e09e082935364cfb9519a9b601569a6](C:\Users\jjzhou\AppData\Local\Temp\WeChat Files\e09e082935364cfb9519a9b601569a6.png)

论文考虑用lstm架构作为$f_{\theta}$。

那么一个自然的问题可能会出现：“当随机游走是马尔可夫过程时，为什么要使用具有内存和时间相关性的模型？”(有偏随机游走的二阶马尔可夫)。或者换句话说，使用长度大于2的随机游走有什么好处?理论上，一个具有足够大容量的模型可以简单地记住图中所有现有的边并重构它们。然而，对于大型图来说，在实践中实现这一点是不可行的。更重要的是，纯记忆并不是NetGAN的目标，相反，NetGAN想要泛化和生成具有相似属性的图形，而不是精确的复制。更长时间的随机游走加上内存有助于模型学习数据中的拓扑结构和一般模式(例如，一致性结构)。

在每一个时间步之后，网络$f_{\theta}$要在random walk中生成下一个节点，需要输出长度为$N$的，但是在这样的高维空间中运行会导致不必要的计算开销。为了解决这个问题，LSTM输出H< N的ot E RH，然后使用矩阵Wn E RHXN向上投影到IRN，这使我们能够有效地处理大型图。