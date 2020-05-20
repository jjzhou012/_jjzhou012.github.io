---
layout: article
title: 图神经网络：基于随机游走的图对抗生成网络（NetGAN）
date: 2020-05-19 00:10:00 +0800
tags: [Adversarial Train, GNN, Link Prediction, Graph Generation, Graph]
categories: blog
pageview: true
key: NetGAN-Generating-Graphs-via-Random-Walks
---

------

- Paper: [NetGAN: Generating Graphs via Random Walks](https://arxiv.org/pdf/1803.00816.pdf)
- Code: [https://github.com/danielzuegner/netgan](https://github.com/danielzuegner/netgan)



## 模型

NetGAN的核心思想是通过学习随机游走的分布来捕捉网络的拓扑结构的。

对于一个给定的图，首先采样获得长度为$T$的随机游走序列，作为模型的输入。游走策略使用node2vec的有偏二阶随机游走。随机游走可以维持节点排序的不变性，同时随机游走只包含$A$的非零项，因此能有效地利用真实世界网络的稀疏性。

NetGAN采用了典型的GAN架构，由生成器$G$和判别器$D$构成。生成器用于生成可靠的合成随机游走序列，判别器用来判断随机游走序列来自于生成器合成还是来自于真实分布。

### 框架

#### 生成器

生成器定义了一个隐式概率模型用于生成随机游走序列$(\boldsymbol{v}_{1}, \ldots, \boldsymbol{v}_{T}) \sim G$。

生成器被建模为一个序列化过程，基于神经网络$f_{\theta}$，每次迭代$t$，$f_{\theta}$生成两个值：

- 采样下个节点的概率分布$\boldsymbol{p}_{t}$;
- 模型当前的记忆状态$\boldsymbol{m}_t$;

从类的概率分布$\boldsymbol{v}_{t} \sim \text{Cat}\left(\sigma\left(\boldsymbol{p}_{t}\right)\right)$中采样得到下一节点$\boldsymbol{v}_t$，在下一次迭代时和$\boldsymbol{m}_t$一起传入$f_{\theta}$。

从多元标准正态分布中提取潜变量$\boldsymbol{z}$，用于初始化记忆状态$\boldsymbol{m}_{0}=g_{\theta^{\prime}}(\boldsymbol{z})$。

生成器的生成过程总结如下：

![](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200520102102.png)

论文考虑用LSTM架构作为$f_{\theta}$。LSTM的记忆状态$\boldsymbol{m}_t$由cell状态$\boldsymbol{C}_t$和hidden状态$\boldsymbol{h}_t$表示。

那么一个自然的问题可能会出现：“当随机游走是马尔可夫过程时，为什么要使用具有内存和时间相关性的模型？”(有偏随机游走的二阶马尔可夫)。或者换句话说，使用长度大于2的随机游走有什么好处?理论上，一个具有足够大容量的模型可以简单地记住图中所有现有的边并重构它们。然而，对于大型图来说，在实践中实现这一点是不可行的。更重要的是，纯记忆并不是NetGAN的目标，相反，NetGAN想要泛化和生成具有相似属性的图形，而不是精确的复制。更长时间的随机游走加上内存有助于模型学习数据中的拓扑结构和一般模式(例如，一致性结构)。

在每一个时间步之后，网络$f_{\theta}$要通过随机游走生成下一个节点，在采样的过程中，存在两个问题：

- 需要输出长度为$N$的logit$\boldsymbol{p}_t$，但是在这样的高维空间中运行会导致不必要的计算开销。为了解决这个问题，令LSTM输出维度远小于$N$的$\boldsymbol{o}_t \in \mathbb{R}^H$，然后使用矩阵$\boldsymbol{W}_{up}\in \mathbb{R}^{H\times N}$对其进行向上投影到$\mathbb{R}^N$，这使我们能够有效地处理大型图。

- 从类的概率分布中进行采样是一个不可导的过程，它会阻塞梯度流并防止反向传播。论文通过使用Straight-Through [Gumsolve](https://zhuanlan.zhihu.com/p/55682096)预测器来解决这一问题。

  令   $\left.\boldsymbol{v}_{t}^{*}=\sigma\left(\left(\boldsymbol{p}_{t}+\boldsymbol{g}\right) / \tau\right)\right)$，其中$\tau$是一个[temperature parameter](https://www.quora.com/What-is-Temperature-in-LSTM)，$g_i$是从Gumbel分布中采样的独立同分布样本。那么下个采样节点 $\boldsymbol{v}_{t}=\operatorname{onehot}\left(\arg \max \boldsymbol{v}_{t}^{*}\right)$。当该样本$\boldsymbol{v}_{t}$作为输入被传递到下一个时间步时，梯度将会在反向传播的时候流经可微的$\boldsymbol{v}^{*}_{t}$。控制温度参数$\tau$可以权衡

  - 当$\tau$大时：所有的激活值对应的激活概率趋近于相同（激活概率差异性较小），此时采样得到的$\boldsymbol{v}^{*}_{t}$更一致；
  - 当$\tau$小时：不同的激活值对应的激活概率差异也就越大，此时采样得到的样本更精确，即$\boldsymbol{v}_{t}^{*} \approx \boldsymbol{v}_{t}$；

当新的节点$\boldsymbol{v}_{t}$被采样时，它在输入LSTM之前需要被投影回低维向量表示，此时用一个向下投影矩阵$\boldsymbol{W}_{\text {down}} \in \mathbb{R}^{N \times H}$。



#### 判别器

判别器$D$基于标准LSTM架构。每一个时间步时，输入一个表示当前位置的节点$\boldsymbol{v}_{t}$。当完整的节点序列被传递到判别器时，判别器输出一个score来表示输入的随机游走序列是否为真的概率。



#### 示意图

![](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200520102045.png)

说明：

- 生成器：合成节点序列

  - 从多元标准正态分布$\mathcal{N}\left(\mathbf{0}, \boldsymbol{I}_{d}\right)$中提取潜变量$\boldsymbol{z}$，用于初始化记忆状态$\boldsymbol{m}_{0}=g_{\theta^{\prime}}(\boldsymbol{z})$。

    其中，LSTM的记忆状态$\boldsymbol{m}_t$由cell状态$\boldsymbol{C}_t$和hidden状态$\boldsymbol{h}_t$表示。

  - LSTM输出维度远小于$N$的$\boldsymbol{o}_t \in \mathbb{R}^H$；

  - 使用矩阵$\boldsymbol{W}_{up}\in \mathbb{R}^{H\times N}$对其进行向上投影到$\mathbb{R}^N$；

  - 从类的概率分布中进行采样获得$\boldsymbol{v}_{t}$，使用Straight-Through Gumsolve解决采样不可导问题；

  - 用向下投影矩阵$\boldsymbol{W}_{\text {down}} \in \mathbb{R}^{N \times H}$将节点$\boldsymbol{v}_{t}$投影回低维向量表示；

- 判别器：判断节点序列的真实性

  - 每一个时间步时，输入一个表示当前位置的节点$\boldsymbol{v}_{t}$。当完整的节点序列被传递到判别器时，判别器输出一个score来表示输入的随机游走序列是否为真的概率。