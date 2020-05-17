---
layout: article
title: Summary：小样本学习综述
date: 2020-05-17 00:10:00 +0800
tags: [Summary, Writting]
categories: blog
pageview: true
key: Summary-of-few-shot-learning
---

------

- Paper: [Generalizing from a Few Examples: A Survey on Few-Shot Learning](https://arxiv.org/abs/1904.05046)
- Github: https://github.com/tata1661/FewShotPapers



## 一、写在前面

现在的机器学习和深度学习任务都依赖于大量的标注数据来训练，而人类的学习过程并不是这样的，人类能够利用过去所学的知识快速学习新任务。FSL就是这样一个概念，能利用一些先验知识，在新的问题上只需要少量样本就能把模型训练的很好。

FSL的关键问题是：最小化经验风险是不可靠的。

基于采用先验知识处理核心问题的方式，将不同的FSL方法分为三个视角：

- 数据：使用先验知识来增强监督经验；
- 模型：通过先验知识来约束假设空间；
- 算法：使用先验知识来改变在假设空间搜索最佳假设的过程。



## 二、定义

### 符号

| 符号 | 定义                      |
| ---- | ------------------------- |
| $E$  | experience，经验          |
| $T$  | task，任务                |
| $P$  | performance measure，指标 |

### 问题

Few-shot learning(FSL):

- 定义：特殊的机器学习问题，在有限的标注数据中获得较好的任务性能。 
- 应用场景：
  - 从小样本中学习特征
  - 为特殊样本建模
  - 减少数据收集工作和计算成本

![image-20200517111508413](C:\Users\jjzhou\AppData\Roaming\Typora\typora-user-images\image-20200517111508413.png)



## 三、数据层面

FSL使用先验知识来扩充训练集$D_{train}$。 

根据扩充的对象数据，方法可以分为：

![15e60ad444c73f4b59d6bbda1e0bc12](F:\blog\jjzhou012.github.io\images\15e60ad444c73f4b59d6bbda1e0bc12.png)

示意图：

<img src="F:\blog\jjzhou012.github.io\images\d6285c0dc532c2269fd70b56de49ebc.png" alt="d6285c0dc532c2269fd70b56de49ebc" style="zoom: 67%;" />



### 3.1. 从训练集样本转换

利用先验知识，对训练集的样本$(x_i,y_i)\in D_{train}$进行变换，生成额外的扩充样本。

相关文献：

- [1] 通过迭代地将每个样本与其他样本对齐，从相似的类中学习一组几何变换。将学习到的变换应用到每个$(x_i,y_i)$，扩充成一个大的数据集。
- [2] 利用自动编码器来学习相似类之间的转换关系。
- [3] 假设所有的类共享一些变换关系，从其他类到$(x_i,y_i)$，可以学习一个转换函数，用来对变换关系进行转换。

### 3.2. 从弱标注或无标注数据转换

从弱标注或无标注数据集中筛选样本进行扩充。这些数据的获取过程相对容易，但是主要的问题在于如何针对目标标签进行数据筛选。

- [4] 为$D_{train}$中的每个目标标签学习一个svm采样器，然后对弱标注数据集中的样本进行预测，打标签。筛选有目标标签的数据加入到训练集。
- [5] 直接使用标签传播来标记未标记的数据集。
- [6] 采用进化的策略来筛选未标注的样本，给被选择的样本赋予伪标签用于更新CNN。

### 3.3. 从相似的数据集转换

基于样本的相似性指标计算聚合权重，通过聚合和调整来自相似的但更大的数据集的输入输出对来增强$D_{train}$。

- 考虑到样本可能不属于目标类，直接扩充聚合的样本可能会给训练集带去不必要的噪声；
- 设计生成对抗网络（GAN）去生成高仿的合成样本$\tilde x$,
  - 一个生成器用于将小样本类映射到大样本类；
  - 另一个生成器用于将大样本类映射到小样本类；

### 3.4. discussion

通常，通过扩展$D_{train}$来解决FSL问题是简单易懂的，利用目标任务的先验信息来扩充数据；另一方面，通过数据扩充来解决FSL问题的缺点是，扩充策略通常以一种特殊的方式为每个数据集量身定制，不能方便地迁移到其他数据集上使用(尤其是来自其他域的数据集)。

针对这一问题，[5] 提出了自动学习深度网络训练增强策略的AutoAugment算法。



## 四、模型层面

## 五、算法层面





## Reference

1. E. G. Miller, N. E. Matsakis, and P. A. Viola. 2000. Learning from one example through shared densities on transforms. In Conference on Computer Vision and Pattern Recognition, Vol. 1. 464–471.
2. E. Schwartz, L. Karlinsky, J. Shtok, S. Harary, M. Marder, A. Kumar, R. Feris, R. Giryes, and A. Bronstein. 2018. Delta- encoder: An effective sample synthesis method for few-shot object recognition. In Advances in Neural Information Processing Systems. 2850–2860.
3. B. Hariharan and R. Girshick. 2017. Low-shot visual recognition by shrinking and hallucinating features. In Interna- tional Conference on Computer Vision.
4. T. Pfister, J. Charles, and A. Zisserman. 2014. Domain-adaptive discriminative one-shot learning of gestures. In European Conference on Computer Vision. 814–829.
5. E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le. 2019. AutoAugment: Learning augmentation policies from data. In Conference on Computer Vision and Pattern Recognition. 113–123.

