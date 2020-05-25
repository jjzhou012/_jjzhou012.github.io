---
layout: article
title: DeepTest：Feature Squeezing(Detecting Adversarial Examples in DNN)
date: 2020-03-27 00:10:00 +0800
tags: [DeepTest, Adversarial]
categories: blog
pageview: true
key: DeepTest-feature-squeezing
---

------

- Paper: [Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks](https://arxiv.org/abs/1704.01155)
- Github: [https://github.com/mzweilin/EvadeML-Zoo](https://github.com/mzweilin/EvadeML-Zoo)
- Github: [https://github.com/uvasrg/FeatureSqueezing](https://github.com/uvasrg/FeatureSqueezing)



## Contribution

样本的特征输入空间通常包含了很多的不必要的特征，特征空间的过大导致攻击者有更多的机会和自由度去实施攻击。feature squeezing主要在于“挤出”不必要的输入特性来减少攻击者的自由度。

- 提出了特征压缩（feature squeezing）策略，通过检测对抗性输入来加强DNN系统对抗有害输入的能力；

- 实验表明，特征压缩有助于DNN模型正确预测由11种不同的和最先进的攻击在不知道防御的情况下所产生的对抗样本；

- 特征压缩可以和其他对抗防御方法进行互补，因为它不会改变底层模型，相当于数据预处理的过程。

  

## Related work

- 论文讨论的几种攻击方法：

  | Attack Method                         | Perturbation                         | Target     |
  | ------------------------------------- | ------------------------------------ | ---------- |
  | FGSM (Fast Gradient Sign Method)      | $$L_{\infty}$$                       | Untargeted |
  | BIM (Basic Iterative Method)          | $$L_{\infty}$$                       | Untargeted |
  | DeepFool                              | $$L_{2}$$                            | Untargeted |
  | JSMA (Jacobian Saliency Map Approach) | $$L_{0}$$                            | Targeted   |
  | CW (Carlini/Wagner Attacks)           | $$L_{\infty}$$, $$L_{0}$$, $$L_{2}$$ | Targeted   |

  

- 论文讨论的几种防御策略：

  - **对抗训练**（Adversarial Training）：引入对抗样本，重训练模型；

    缺点：

    - 生成对抗样本的成本很高：其有效性取决于是否有一种技术，能够有效地生成类似于攻击者所使用的对抗样本，但在实践中可能很困难；
    - 由于重训练过程，导致训练成本翻倍；

  - **梯度隐藏**（Gradient Masking）：通过迫使DNN模型产生接近于零的梯度，“梯度掩蔽”防御试图降低DNN模型对输入的微小变化的敏感性。

    缺点：

    - 虽然训练后的模型对攻击表现出了更强的鲁棒性，但是这种惩罚大大降低了模型的容量，并牺牲了许多任务的准确性；
    - 由于对抗样本的迁移性，梯度隐藏的方法能力有限；

  - **输入变换**（Input Transformation）：通过转换输入来降低模型对小的输入变化的敏感性；

  

- 论文讨论的几种对抗样本检测方法，主要可以分为三大类，论文提出的feature squeezing 属于第三类：

  - 样本统计（Sample Statistics）：

    缺点：

    - 由于对抗样本本质上是不可感知的，因此使用样本统计将对抗样本与合法输入区分开似乎不太可能有效；

  - 检测器训练（Training a Detector）：

    缺点：

    - 这种策略需要大量的对抗样本，因此开销很大；
    - 容易过度拟合对抗样本；

  - 预测不一致性（Prediction Inconsistency）：测量预测未知输入时几个模型之间的差异，因为一个对抗样本可能无法愚弄所有的DNN模型；

  

## Methods

提出feature squeezing，也就是特征压缩的主要出发点在于：巨大的特征输入空间为攻击者构建对抗样本提供了很大的空间。而Feature squeezing就是通过“压缩”不必要的输入特征来减少攻击者的自由度，限制其攻击行为。

整个框架的思想在于：通过比较DNN模型对原始样本和对经过特征压缩后的样本的预测结果，如果原始的和压缩的输入产生的输出有本质的不同，那么输入很可能是对抗样本。将预测值的差异和阈值进行比较，系统就能选择对合法样本进行输出，而过滤掉对抗样本。

![671379cb90a57590d60a39b60729e4d.png](http://ww1.sinaimg.cn/large/005NduT8ly1gd8qv3syarj30gy09tabp.jpg)

### Feature Squeezing

论文主要关注两类特征压缩，**具体不展开了**：

- 减少图像的颜色深度；
  - 常见的图像表示使用颜色位深度，这会导致不相关的特征，因此论文假设降低位深度可以减少对抗机会，而不会影响分类器的准确性。
  - 压缩比特位深度：
    - 虽然人们通常喜欢更大的位深度，因为它使显示的图像更接近自然图像，但大的颜色深度通常不是解释图像所必需的(例如，人们可以毫无困难地识别大多数黑白图像)。
- 空间平滑
  - 空间平滑(亦称模糊)是广泛应用于图像处理中以降低图像噪声的技术。论文使用了两种类型的空间平滑方法:
    - 局部平滑
    - 非局部平滑



## Experiment：Detecting Adversarial Inputs

其基本思想是将模型对原始样本的预测与同一模型对压缩后样本的预测进行比较。该模型对一个合法样本的预测和它的压缩版本应该是相似的。但是对对抗样本，结果会有很大的差异。

![04106579bb445ff3f7a3658838bfd93.png](http://ww1.sinaimg.cn/large/005NduT8ly1gd8sl6vgitj30yo0bq0vz.jpg)

上图展示了原始样本和压缩样本$L_1$距离的差异。

### Detection Method 

由DNN分类器生成的预测向量通常表示为输入样本属于每个可能类别的概率分布。因此，比较模型的原始预测和压缩样本的预测涉及到比较两个概率分布向量。比较概率分布的方法有很多种，如$L_1$范数、$L_2$范数和K-L散度。论文选择$L_1$范数作为原始预测向量和压缩预测向量之间差异的自然度量：

$$
\begin{equation}\text {score}^{\left(\mathbf{x}, \mathbf{x}_{\text {squececd}}\right)}=\left\|g(\mathbf{x})-g\left(\mathbf{x}_{\text {squeezed}}\right)\right\|_{1}\end{equation}
$$

其中$g(\mathbf{x})$是DNN模型softmax层生成的概率分布向量。分数结果的区间为$[0,2]$，分数越高，意味着原始预测与压缩预测之间的差异越大。

我们期望对合法输入的打分较低，对对抗样本的打分较高。检测的有效性取决于选择一个阈值，该阈值能够准确区分合法输入和对抗性输入。

虽然可以为一种特定类型的攻击方法选择一种有效的特征压缩器，但是我们通常不知道在实践中使用的具体攻击方法。因此，论文基于以下假设：不同的特征压缩器对于不同类型的扰动是有效的，并且有效的特征压缩器可以通过输出高的$L_1$分数来识别对抗样本。因此，论文结合多个特征压缩器，输出它们中的最大距离：

$$
\begin{equation}\text {score}^{\text {joint}}=\max \left(\text {score}^{\left(\mathbf{x}, \mathbf{x}_{\text {sq1}}\right)}, \text {score}^{\left(\mathbf{x}, \mathbf{x}_{\text {sq2}}\right)}, \ldots\right)\end{equation}
$$

### Threshold Selection

最后，如何在训练阶段，简单地选择一个最优的分数联合阈值呢？

而关于阈值的选择，参考上图，合法输入的峰值在0，对抗性输入的峰值处于2。经验上，是找到一个最大限度地提高训练精度的方法。然而，由于样本的实际期望分布并不均衡，且大多是正样本（合法样本）的，因此，对于许多安全敏感的任务而言，高精度、高误置率的检测器是毫无用处的。

具体而言：与其选择使得FP低于5%的阈值，倒不如选择不超过5%合法样本的阈值。这样，训练阈值仅使用合法样本的设置，因此不依赖于对抗样本。因此，其他方法(如样本统计或训练检测器)相比，论文的方法在训练阶段的成本较低，但是由于压缩操作和多个输入，可能在推断阶段训练检测器的成本较高。

![d4176c4604ce08fc38e5cc3c8c8e128.png](http://ww1.sinaimg.cn/large/005NduT8ly1gd8tfejv71j30tf0kwgt5.jpg)