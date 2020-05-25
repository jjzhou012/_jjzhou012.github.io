---
layout: article
title: 对抗样本检测：adversarial example detection using autoencoder (MagNet)
date: 2020-05-22 00:10:00 +0800
tags: [DeepTest, Adversarial]
categories: blog
pageview: true
key: adversarial-example-detection-using-autoencoder-MagNet
---

------

- Paper: [MagNet: A Two-Pronged defense against adversarial examples](https://arxiv.org/abs/1705.09064)
- Github: [https://github.com/Trevillie/MagNet](https://github.com/Trevillie/MagNet)
- Github: [https://github.com/sky4689524/Pytorch_MagNet](https://github.com/sky4689524/Pytorch_MagNet)



## Introduction

提出了一个对抗样本检测框架MagNet，具有两个特点：

- 不修改分类器，不依赖分类器的属性，也就是说，将分类器视为黑盒模型，只利用分类器的输入输出，适用性广；
- 不依赖对抗样本的生成过程，训练的过程中只使用正常样本。

对抗样本不是自然出现的样本，分类器对样本的判断结果不同于一般人的判断。

分类器错误分类对抗样本的原因可能有两个：

- 对抗样本远离任务数据流形的边界。例如手写数字识别中，对抗样本的图片中不包含任何数字，但是分类器还是会输出一个预测结果；
- 对抗样本接近流行边界。但是分类器不能很好地概括对抗样本附近的数据流形，导致错误分类；

对于第一类问题，MagNet使用detectors去检测测试样本与正常样本的差距，也就是计算测试样本和正常数据流形的距离，如果距离大于阈值，那么detectors会将其判断为对抗样本。

对于第二类问题，MagNet使用reformer去改良对抗样本，具体而言，使用足够的正常样本训练一个自动编码器来学习数据流形。希望该自编码器能够将对抗样本修复到正常的数据流形。

<img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200522125636.png" alt="d48da30c74bf6bfea9dec922545d4f5" style="zoom:50%;" />

## 定义

### 符号

- $$\mathbb{S}$$：样本空间中所有样本的集合；
- $$\mathbb{C}_t$$：对分类任务$$t$$的互斥类集合；
- $$\mathbb{N}_t=\{x\mid x\in \mathbb{S} \  \text{ and } x \text{ occurs naturally with regard to the classification task } t.\}$$：从自然数据分布中采样得到的数据集合，也就是采样得到的正常数据。比如用于图像识别的CIFAR和MNIST数据集。

### 对抗样本

- 定义1：分类任务$$t$$的一个分类器表示为函数$$f_{t}: \mathbb{S} \rightarrow \mathbb{C}_{t}$$；
- 定义2：分类任务$$t$$的真实分类器表示一般人类的判断标准：$$g_t:\mathbb{S} \rightarrow \mathbb{C}_{t} \cup\{\perp\}$$；$$\perp$$表示针对对抗样本的分类判断结果；
- 定义3：对分类任务$$t$$和分类器$$f_t$$而言，对抗样本$$x$$满足以下两个条件：
  - $$f_{t}(x) \neq g_{t}(x)$$；
  - $$x \in \mathbb{S} \backslash \mathbb{N}_{t}$$；

>条件1表明，分类器对对抗样本的分类判断会出错，但是由于任何分类器都是不完美的，分类器对正常样本的预测也存在误差；
>
>条件2进一步对对抗样本限定为，仅仅由攻击者人为产生的用来欺骗分类器的样本；

- 定义4：针对分类器$f_t$的防御方法为：$$d_{f_{t}}: \mathbb{S} \rightarrow \mathbb{C}_{t} \cup\{\perp\}$$；

  防御方法$$d_{f_{t}}$$提升分类器的鲁棒性，确切地说，主要用来提升分类器对对抗样本而非正常样本的检测精度；

- 定义5：如果满足下列任意一种情况，则表明防御方法对样本$$x$$做出正确分类：

  - $$x$$是正常样本，防御方法$$d_{f_{t}}$$和真实分类器$$g_t$$的判断结果一致，即 $$x \in \mathbb{N}_{t}$$， $$d_{f_{t}}(x)=g_{t}(x)$$；

  - $$x$$是对抗样本：

    - 防御方法$$d_{f_{t}}$$直接判断其为对抗样本；
    - 防御方法$$d_{f_{t}}$$和真实分类器$$g_t$$的判断结果一致；

    即对$$x \in \mathbb{S} \backslash \mathbb{N}_{t}$$，有$$d_{f_{t}}(x)=\perp$$或者$$d_{f_{t}}(x)=g_{t}(x)$$；



### 威胁建模

假设攻击者完全了解分类器$$f_t$$，即目标模型；

根据攻击者了解防御方法$$d_{f_{t}}$$的程度，可将攻击分为:

- 黑盒攻击
- 白盒攻击
- 灰盒攻击





## 模型

MagNet是一个对抗样本防御框架，主要基于分类器对对抗样本分类错误的两点原因：

- 对抗样本远离任务数据流形的边界；
- 对抗样本接近流行边界。但是分类器泛化能力不好，不能很好地概括对抗样本附近的数据流形，导致错误分类；

提出了两个部分：

- detector：拒绝（删除）远离流形边界的样本，也就是被检测为对抗样本；
- reformer：对于一个输入样本$$x$$，若靠近流形边界，则努力找到一个接近或者就在该任务流形之上的样本$$x'$$，其中$$x'$$是$$x$$的近似，然后将$$x'$$作为分类器的输入。

MagNet中包括一个或者多个Detector模块和一个Reformer模块，

![4f359cb6a7aa0949a5adb9efde5a04f](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200522130616.png)



### Detector

Detector是一个判断样本是否是对抗样本的函数，即$$d: \mathbb{S} \rightarrow\{0,1\}$$。

> 如果检测器用小扰动的对抗样本训练，则检测器会倾向于将部分正常样本识别为对抗样本，从而具有较高的FP值；
>
> 如果检测器用大扰动的对抗样本训练，则检测器可能会忽略小扰动对抗样本；

#### 基于重构误差的检测器

为了避免在检测的过程中需要对抗样本的数据集，detector中的检测方法是通过对正常样本建模，估算测试样本与正常样本流形边界的距离，从而达到检测对抗样本的目的。

这部分主要使用自编码器作为detector，用重构误差来近似输入和流行边界之间的距离；

自编码器可以表示为：$$ae=d \circ e$$，包含两部分，编码器（$$e: \mathbb{S} \rightarrow \mathbb{H}$$），解码器（$$d: \mathbb{H} \rightarrow \mathbb{S}$$），损失函数为均方误差$$L\left(\mathrm{X}_{\text {train }}\right)=\frac{1}{\left|\mathrm{X}_{\text {train }}\right|} \sum_{x \in \mathbb{X}_{\text {train }}} \| x-a e(x) \|_{2}$$，重构误差为$$E(x)=\| x-a e(x)) \|_{p}$$

作者认为如果待测试的样本是正常样本，因为该样本与autoencoder的训练数据的生成过程相同，所以其重构误差很小，反之，如果带检测的样本是对抗样本，那么该样本的重构误差会很大。所以，在本文中作者用重构误差来衡量待检测样本与正常样本流形间的距离。但是，由于该距离是连续值，所以在本文中作者还利用验证集确定一个阈值$$t_{re}$$，用来判断是否为对抗样本。

对于计算重构误差的范数选择，经验上来说，使用两个分别具有$$L_1$$和$$L_2$$规范的基于重构误差的检测器来覆盖两端是足够的。



#### 基于概率发散的检测器

基于重构误差的Detector的适用于当样本的误差很大的时候，那么如果重构误差很小的时候呢？在本文中作者的解决方式是利用softmax目标分类器来进一步检测。

多数神经网络分类器在最后一层实现softmax函数，其输入为logit $$\boldsymbol{l}$$，其输出为一个概率分布。$$\text{rank}(\boldsymbol{l},i)$$表示输入中的第$$i$$大的元素。攻击者的目标就是攻击样本得到新的logit $$\boldsymbol{l}’$$使得$$\operatorname{rank}(l, 1) \neq \operatorname{rank}\left(l^{\prime}, 1\right)$$。

自编码器$$a e$$使用正常样本训练，接受样本输入$$x$$，输出$$ae(x)$$。$$f(x)$$是分类器的输出。当$$x$$为正常样本时，$$ae(x)$$接近于$$x$$，因此概率分布$$f(x)$$和$$f(ae(x))$$是相似的；当$$x'$$为对抗样本时，$$ae(x')$$与$$x'$$差异较大，因此概率分布$$f(x')$$和$$f(ae(x'))$$是明显不同的；

作者使用$$f(x)$$与$$f(ae(x))$$之间的差距来判断样本$$x$$是否为正常样本，差距用Jensen-Shannon散度来衡量：
$$
\operatorname{JSD}(P \| Q)=\frac{1}{2} D_{\mathrm{KL}}(P \| M)+\frac{1}{2} D_{\mathrm{KL}}(Q \| M)
$$
其中：$$D_{\mathrm{KL}}(P \| Q)=\sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$，$$M=\frac{1}{2}(P+Q)$$。

令$$\boldsymbol{l}(x)$$是输入$$x$$的logit。当$$\boldsymbol{l}(x)$$的最大元素与第二大元素相差较大时，$$\operatorname{softmax}(l(x))$$饱和；同时$$\operatorname{softmax}(l(a e(x)))$$也会发生饱和现象。这使得$$\operatorname{softmax}(l(x))$$和$$\operatorname{softmax}(l(a e(x)))$$两者的Jensen-Shannon散度很小。这里作者用了温度参数$$T>1$$来调整softmax，解决这个问题：
$$
\operatorname{softmax}(l)_{i}=\frac{\exp \left(l_{i} / T\right)}{\sum_{j=1}^{n} \exp \left(l_{j} / T\right)}
$$


### Reformer

改良器函数$$r: \mathbb{S} \rightarrow \mathbb{N}_{t}$$，用于重构测试输入。改良器的输出被送入目标分类器。注意，我们在训练目标分类器时不使用改良器，而只在部署目标分类器时使用改良器。

- 不应该改变正常样本的分类结构；
- 应该充分重构对抗样本，使得重构后的样本接近于正常样本。



#### 基于噪声的改良器

一个简单的改良器函数是一个向输入添加随机噪声的函数。如果我们使用高斯噪声，我们得到如下的转化：
$$
r(\boldsymbol{x})=\operatorname{clip}(\boldsymbol{x}+\epsilon \cdot \mathbf{y})
$$
其中$\mathbf{y} \sim \mathcal{N}(\boldsymbol{y} ; \mathbf{0}, \mathbf{I})$是标准正态分布，$$\epsilon$$表示噪声，$$\text{clip}$$表示将输入控制在范围内。

该方法的缺点是不能充分利用正常样本的分布特性。因此，它随机地、盲目地改变了正常样本和对抗样本，但我们理想的改良器不应该改变正常样本，而应该把对抗样本转向正常样本。



#### 基于自编码器的改良器

作者主要利用autoencoder来实现Reformer。通过在正常样本上训练得到一个autoencoder，然后利用该autoencoder，不论是对抗样本还是正常样本，都会输出一个与正常样本流形接近的样本，从而达到理想的改良器的要求。

最终MagNet能够在不改变对正常样本的检测精度的同时，提升对对抗样本的检测效果。



### 多样性缓和灰盒攻击

作者创建了大量的自动编码器作为候选检测器和改良器。每个会话、每个测试集，甚至每个测试样本。MagNet会为每个防御设备随机选择一个这样的自动编码器。假设攻击者无法预测我们选择了哪些自编码器来应对他的对抗样本时，攻击者必须针对所有的自编码器去训练对抗样本才能实现成功的攻击。我们可以增加这个集合的大小和多样性，使攻击更难执行。这样，我们就可以防御灰盒攻击。

为了保证自编码器的多样性，论文用随机初始化同时训练$$n$$个相同或不同架构的自动编码器。训练的时候添加正则项来控制这些自编码器的相似性：
$$
L(x)=\sum_{i=1}^{n} \operatorname{MSE}\left(x, a e_{i}(x)\right)-\alpha \sum_{i=1}^{n} \operatorname{MSE}\left(a e_{i}(x), \frac{1}{n} \sum_{j=1}^{n} a e_{j}(x)\right)
$$
其中$$a e_{i}$$是第$$i$$个自编码器，$$\alpha>0$$是超参数权衡重构误差和自编码器多样性。



