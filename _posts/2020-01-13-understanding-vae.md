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

一般来说，每个样本数据都受到一些因素控制，这些因素可以称之为**潜变量**，用向量$$\mathbf{z}$$表示。VAE和GAN的目标基本是一致的，希望构建一个从潜变量$$\mathbf{z}$$生成目标$$\mathbf{x}$$的模型。

> 相对于可观测变量而言，潜变量就是观测不到的变量，可观测变量由潜变量产生。数据中每个数据点$$\mathbf{x}$$都有一个相应的潜变量$$\mathbf{z}$$，$$\mathbf{z}$$通过某一个变换$$f(\mathbf{z} \mid \theta)$$可以得到$$\mathbf{x}$$，$\theta$是这个变换的参数，这就是所谓的潜变量模型。

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

  公式（1）其实就是$$p(\mathbf{x} \mid \mathbf{z} ; \theta)$$关于$$\mathbf{z}$$的期望。在优化的过程中，我们一定要取到能变换到$$\mathbf{x}$$的潜变量$$\mathbf{z}$$，只有这样才能通过优化参数最终获得期望的输出。但是如果潜变量的维度过高，对于大部分的$$\mathbf{z}$$，都不能生成与$$\mathbf{x}$$相似的样本，即$$p(\mathbf{x} \mid  \mathbf{z} ; \theta)$$通常都接近于0。因此直接优化$$p(\mathbf{x})$$很难，所以我们还是得换一个优化目标。



## 2. 理论简述

那么既然$$p(\mathbf{x})$$这个概率分布中的很多潜变量$$\mathbf{z}$$不能被变换为与输入$$\mathbf{x}$$相似的样本，那么我们能否找到一个概率分布$$q(\mathbf{z})$$，使得这个$$q(\mathbf{z})$$分布中的$$\mathbf{z}$$被变换为我们需要的$$\mathbf{x}$$的概率更大呢？

既然$$\mathbf{z}$$能变换为$$\mathbf{x}$$，那么相应的某一个$$\mathbf{x}$$也能对应一个$$\mathbf{z}$$，这里我们引入一个$$\mathbf{z}$$的后验概率分布$$p(\mathbf{z} \mid \mathbf{x})$$。通过后验概率分布我们可以以最大概率获得能变换为所需$$\mathbf{x}$$的$$\mathbf{z}$$。此时如果$$q(\mathbf{z})$$就是$$p(\mathbf{z} \mid \mathbf{x})$$的话那就再好不过了，但是现在后验概率分布无法获得。所以我们需要引入这个$$q(\mathbf{z} \mid \mathbf{x})$$分布来近似后验概率分布$$p(\mathbf{z} \mid \mathbf{x})$$，这就是变分（variational）的思想。

VAE的模型图如下所示：

![1fe4afeb3000bb44790eb50ebda3c3a.png](http://ww1.sinaimg.cn/large/005NduT8ly1gaxafq6ly2j30k206vwgl.jpg)

$$\mathbf{x}$$是我们可以观测到的数据，也就是输入数据，输入数据受隐变量$$\mathbf{z}$$控制，即$$\mathbf{x}$$由隐变量$$\mathbf{z}$$产生：

- $$\mathbf{x} \rightarrow \mathbf{z}$$: 从$$\mathbf{x}$$推断得到$$\mathbf{z}$$，生成$$\mathbf{z}$$的后验概率分布，视为推断模型，从自编码器的角度看就是编码器；
- $$\mathbf{z} \rightarrow \mathbf{x}$$: 从隐变量中采样数据$$\mathbf{z}$$映射为与$$\mathbf{x}$$相似的样本数据，视为生成模型$$p_{\theta}(\mathbf{x} \mid \mathbf{z})$$，从自编码器的角度看就是解码器；

回到刚才，那么为了使得$$q(\mathbf{z} \mid \mathbf{x})$$和$$p(\mathbf{z} \mid  \mathbf{x})$$这两个分布尽可能的相似，可以通过最小化这两个分布之间的KL散度：


$$
D_{KL}[q(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z} \mid \mathbf{x}) ]=E_{\mathbf{z} \sim q(\mathbf{z} \mid \mathbf{x})}[\log q(\mathbf{z} \mid \mathbf{x})-\log p(\mathbf{z} \mid \mathbf{x})]
$$
进一步简化：


$$
D_{KL}[q(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z} \mid \mathbf{x}) ]=
E_{\mathbf{z} \sim q(\mathbf{z} \mid \mathbf{x})}
[\log q(\mathbf{z} \mid \mathbf{x})-\log p(\mathbf{x} \mid \mathbf{z}) - \log p(\mathbf{z})] + \log p(\mathbf{x})
$$


最终简化到下面的形式：


$$
\log p(\mathbf{x}) - D_{KL}[q(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z} \mid \mathbf{x}) ] =
E_{\mathbf{z} \sim q(\mathbf{z} \mid \mathbf{x})}[\log p(\mathbf{x \mid z})] - D_{KL}[q(\mathbf{z \mid z} ) \|  p(\mathbf{z})]
$$


公式（4）等号左边：

- 第一项：最大化$$\mathbf{x}$$的似然概率分布；
- 第二项：最小化$$q(\mathbf{z} \mid \mathbf{x})$$和$$p(\mathbf{z} \mid  \mathbf{x})$$这两个分布的KL散度，当我们选择的$$q(\mathbf{z} \mid \mathbf{x})$$表达能力足够强时，可以近似表达$$p(\mathbf{z} \mid \mathbf{x})$$，也就是说，这一项趋近于0；

等号右边：

- 第一项：生成模型，也就是解码器，$$\mathbf{z}$$是采样得到的，$$\mathbf{x}$$是观测得到的，我们希望采样得到的$$\mathbf{z}$$变换到$$\mathbf{x}$$后能与我们的观测值尽可能接近，也就是$$\mathbf{x}$$与$$f(\mathbf{z} \mid \theta)$$两者的差尽可能的小。我们可以最小化两者的L2距离，间接最大化$$p(\mathbf{z} \mid  \mathbf{x})$$。
- 第二项：正则项
  - $$p(\mathbf{z})$$是潜变量的先验概率，前面我们近似为单位高斯分布，具有明确的解析公式；
  
  - $$ q(\mathbf{z \mid x})$$是潜变量的后验概率分布，即推理模型，也就是编码器，我们用神经网络去训练拟合这个分布。由于前面我们假设潜变量符合单位高斯分布，那么这里我们假设后验概率也符合高斯分布，由于VAE需要获得显式的潜变量分布，我们令神经网络输出两个量：均值和方差，然后据此进行采样得到潜变量$$\mathbf{z}$$，由此我们可以获得$$ q(\mathbf{z \mid x})$$的解析表达式，进行$$ q(\mathbf{z \mid x})$$与$$p(\mathbf{z})$$的KL散度计算。
  
  - 神经网络输出的两个量，均值和方差，通过构建两个神经网络$$\mu_{k}=f_{1}\left(\mathbf{x}_{k}\right), \log\sigma_{k}^{2}=f_{2}\left(\mathbf{x}_{k}\right)$$来计算。选择拟合$\log\sigma_{k}^{2}$而不是直接拟合$\sigma_{k}^{2}$，是因为$\sigma_{k}^{2}$总是非负的，需要加激活函数处理，而拟合$\log \sigma_{k}^{2}$不需要加激活函数，因为其可正可负。



### 重参数技巧

我们在encoder中输入$$\mathbf{x}$$，得到估计后验概率分布$$q$$的均值和方差后，从$$ q(\mathbf{z \mid x})$$中采样得到$$\mathbf{z}$$。尽管我们知道$$ p(\mathbf{z \mid x})$$是正态分布的，但是均值和方差都是靠模型拟合出来的，我们要靠这个过程反过来优化生成均值和方差的模型，但是“采样”的过程不可导，而采样的结果是可导的。

我们利用正态分布的概率密度函数进行转化：


$$
\begin{aligned}
& \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{(z-\mu)^{2}}{2 \sigma^{2}}\right) d z 
= \frac{1}{\sqrt{2 \pi}} \exp \left[-\frac{1}{2}\left(\frac{z-\mu}{\sigma}\right)^{2}\right] d\left(\frac{z-\mu}{\sigma}\right)
\end{aligned}
$$


<center>
    <img style="border-radius: 0.3125em"  align="left" width="160" height="170"  src="http://ww1.sinaimg.cn/mw690/005NduT8ly1gb0qi9hn5dj305205l749.jpg"/>
</center>

$$\mathbf{z}$$服从正态分布，那么$$(\mathbf{z-\mu})/\sigma = \epsilon$$服从均值为0，方差为1的标准正态分布。从$$\mathcal{N}(\mu, \sigma^2)$$中采样一个$$\mathbf{z}$$，相当于从$$\mathcal{N}(0,I)$$中采样一个$\epsilon$，令$$\mathbf{z=\mu+\epsilon \times \sigma}$$。这样一来我们可以通过从$$\mathcal{N}(0,I)$$中采样来代替从$$\mathcal{N}(\mu, \sigma^2)$$中采样，然后通过参数变换将$\epsilon$转化为$$\mathbf{z}$$。这样一来，“采样”这一过程就不用参与梯度下降了，改为采样的结果参与，使得整个模型可训练。







代码中的具体实现如下：


```
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
```

整个VAE的过程如下：

<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gb0qslmqp3j30rd0nmwiv.jpg" alt="4168876662.png"  />



### 分布标准化

在上图的训练过程中，decoder希望重构$$\mathbf{x}$$，也就是最小化$$\mathcal{D(\mathbf{\hat{x},x})^2}$$，但是这个重构过程会受到噪声影响，因为$\mathbf{z}$是通过重采样得到的，不是由encoder算出来的。噪声会增加重构的难度，不过由于噪声强度（也就是方差）是通过神经网络算出来的，所以模型为了重构得更好，肯定会让方差为0，而方差为0，结果也就没有随机性了，无论怎么采样都是确定的结果（也就是均值）。

然而其实VAE在encoder的阶段让所有的$$p(\mathbf{z \mid x})$$都向标准正态分布靠拢，也就是$$D_{KL}[q(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}) ]$$正则项希望达到的目的，这样就能防止噪声为0，保证了模型具有生成能力。因为如果所有的$$p(\mathbf{z \mid x})$$都接近标准正态分布，那么根据定义：


$$
p(\mathbf{z})=\sum_{\mathbf{x}} p(\mathbf{z} \mid \mathbf{x}) p(\mathbf{x})=\sum_{\mathbf{x}} \mathcal{N}(0, I) p(\mathbf{x})=\mathcal{N}(0, I) \sum_{\mathbf{x}} p(\mathbf{x})=\mathcal{N}(0, I)
$$


这样就能达到我们的先验假设：$$p(\mathbf{z})$$符合标准正态分布。这样我们就能放心地从$$\mathcal{N}(0, I)$$中采样。

那么如何让$$q(\mathbf{z} \mid \mathbf{x})$$向$$\mathcal{N}(0, I)$$靠拢呢？最直接的方法就是在重构误差的基础上加入额外的loss:


$$
\mathcal{L}_{\mu}=\left\|f_{1}\left(\mathbf{x} \right)\right\|^{2}  , \quad \mathcal{L}_{\sigma^{2}}=\left\|f_{2}\left(\mathbf{x} \right)\right\|^{2}
$$


它们代表了均值$\mu_k$和方差的对数$\log \sigma_k^2$，符合标准正态分布就是希望两者接近于0。但是这样又会面临怎么选取两个损失的比例问题，原文直接计算一般正态分布和标准正态分布之间的KL散度$$
K L\left(N\left(\mu, \sigma^{2}\right) \| N(0, I)\right)
$$作为这个额外的loss，计算结果为：


$$
\begin{aligned}
& K L\left(N\left(\mu, \sigma^{2}\right) \| N(0,1)\right) \\
=& \int \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-(x-\mu)^{2} / 2 \sigma^{2}}\left(\log \frac{e^{-(x-\mu)^{2} / 2 \sigma^{2}} / \sqrt{2 \pi \sigma^{2}}}{e^{-x^{2} / 2} / \sqrt{2 \pi}}\right) d x \\
=& \int \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-(x-\mu)^{2} / 2 \sigma^{2}} \log \left\{\frac{1}{\sqrt{\sigma^{2}}} \exp \left\{\frac{1}{2}\left[x^{2}-(x-\mu)^{2} / \sigma^{2}\right]\right\} d x\right.\\
=& \frac{1}{2} \int \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-(x-\mu)^{2} / 2 \sigma^{2}}\left[-\log \sigma^{2}+x^{2}-(x-\mu)^{2} / \sigma^{2}\right] d x \\
=& \frac{1}{2}\left(-\log \sigma^{2}+\mu^{2}+\sigma^{2}-1\right)
\end{aligned} 
$$


VAE中考虑各分量独立的多元正态分布，那么结果为：


$$
\mathcal{L}_{\mu, \sigma^{2}}=\frac{1}{2} \sum_{i=1}^{d}\left(\mu_{(i)}^{2}+\sigma_{(i)}^{2}-\log \sigma_{(i)}^{2}-1\right)
$$


其中$d$是隐变量$$\mathbf{z}$$的维度，直接用这个式子作为额外的loss，就避免考虑均值loss和方差loss的相对比例问题了。

代码中的具体实现如下：



```
# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
```



### VAE的本质

VAE与AE不同，VAE中的两个encoder，一个用于计算均值，一个用于计算方差。

- 对应计算均值的encoder，它在encoder的结果上加上了“高斯噪声”，使得结果的decoder对噪声有鲁棒性；额外的KL散度作为loss的一部分，目的是对encoder的正则项，希望encoder编码的结果符合标准正态分布（均值为0，方差为1）。
- 对应计算方差的encoder，它是用来动态调节噪声的强度的。
  - 当decoder还没训练好的时候，即重构误差远大于KL loss时，encoder会适当降低噪声（也就是偏离正态分布，KL loss增加），使得拟合过程更加容易（重构误差下降）；
  - 当decoder训练的较好时，即重构误差小于KL loss时，encoder会适当增加噪声（也就是靠近正态分布，KL loss减小），使得拟合过程更加困难（重构误差增加），decoder提高它的生成能力；

实际上，重构的过程就是希望噪声尽可能的小，而KL loss则希望有高斯噪声，两者是对立的。所以VAE和GAN一样，内部都包含了一个对抗的过程，只不过在VAE中，两者是混合的、共同进化的。





## 参考

转载：



- [https://kexue.fm/archives/5253](https://kexue.fm/archives/5253)
- [https://zhuanlan.zhihu.com/p/55557709](https://zhuanlan.zhihu.com/p/55557709)
- [https://github.com/bojone/vae](https://github.com/bojone/vae)

