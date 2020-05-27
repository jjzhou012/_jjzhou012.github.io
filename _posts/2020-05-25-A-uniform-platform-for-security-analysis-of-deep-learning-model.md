---
layout: article
title: 对抗安全：深度模型的安全性分析（DeepSEC指标平台）
date: 2020-05-25 00:10:00 +0800
tags: [DeepTest, Adversarial, CV]
categories: blog
pageview: true
key: DEEPSEC-A-uniform-platform-for-security-analysis-of-deep-learning-model
---

- Paper: [DEEPSEC: A Uniform Platform for Security Analysis of Deep Learning Model](https://nesa.zju.edu.cn/download/DEEPSEC%20A%20Uniform%20Platform%20for%20Security%20Analysis%20of%20Deep%20Learning%20Model.pdf)
- Code: [https://github.com/kleincup/DEEPSEC](https://github.com/kleincup/DEEPSEC)



## 1. 概要

提出了一个安全性评估框架DeepSEC，集成了16种攻击和10种攻击评价指标，13种防御和5种防御评价指标。主要用于：

- 评价深度模型的脆弱性；
- 评价不同攻击、防御方法的效果；
- 以全面和翔实的方式对攻击/防御进行比较研究；

利用DeepSEC，作者系统地评估了现有的对抗攻击和防御方法，并得出了一组关键的发现，它们展示了DeepSEC丰富的功能，如：

- 从经验上证实了错误分类和不可感知的权衡；
- 大多数声称普适性强的防御措施在受限条件下只能防御有限类型的攻击;
- 并不是说扰动幅度越大的对抗样本越容易被发现；
- 多重防御的集成不能提高模型整体防御能力，但可以提高个体防御效能的下界；

进一步推进对抗实例的研究，提供一个全面、翔实的评估对抗攻击和防御的分析平台至关重要。要使这样一个平台具有实际用途，需要具有一下特点：

- 统一性：支持在相同设置下比较不同的攻击防御方法；
- 全面性：应该囊括最具代表性的攻击防御方法；
- 信息全面：包含丰富的指标来评价这些攻击防御方法；
- 可扩展的：对于新的攻防方法具有良好的可扩展性；



## 2. 攻击

### 2.1 攻击方法

根据攻击的特异性和攻击频率，将攻击分为

- 基于攻击特异性
  - 无目标攻击（UA）
  - 目标攻击（TA）
- 基于攻击频率
  - 非迭代攻击
  - 迭代攻击



### 2.2 攻击效能指标

对于想要攻击深度模型的攻击者来说，效能意味着对抗性攻击能够在多大程度上提供“成功的”对抗样本。一般来说，成功的对抗样本不仅可以被模型错误地分类，而且对人类来说是不可察觉的，对变换具有很强的抵抗能力，对基于对抗目标的防御方法具有很强的抵抗能力。

本文将误分类、不可察觉和鲁棒性作为效能需求，而将可靠性作为安全性需求。我们将首先为下面的对抗攻击定义10个效能指标。

#### 2.2.1 误分类

- Misclassification Ratio (MR)

  对抗样本被分类错误，或者分类到目标类的比例。

  - UA:	$$M R_{U A}=\frac{1}{N} \sum_{i=1}^{N} \operatorname{count}\left(\mathbf{F}\left(X_{i}^{a}\right) \neq y_{i}\right)$$
  - TA:    $$M R_{T A}=\frac{1}{N} \sum_{i=1}^{N} \operatorname{count}\left(\mathbf{F}\left(X_{i}^{a}\right)=y_{i}^{*}\right)$$

- Average Confidence of Adversarial Class (ACAC)

  ACAC定义为对不正确类的平均预测置信度。
  
  $$
  ACAC = \frac{1}{n} \sum_{i=1}^{n} P\left(X_{i}^{a}\right)_{F\left(X_{i}^{a}\right)}
  $$
  
  $$n(n \le N)$$是成功的对抗样本的数量。

- Average Confidence of True Class (ACTC)

  通过对对抗样本的true类的预测置信度进行平均，ACTC被用来进一步评估攻击在多大程度上脱离了ground truth。
  
  $$
  A C T C=\frac{1}{n} \sum_{i=1}^{n} P\left(X_{i}^{a}\right)_{y_{i}}
  $$

#### 2.2.2 能效

从本质上讲，不可感知意味着对抗样本仍然会被人类的视觉正确地分类，从而保证对抗样本和正常样本传达相同的语义信息。

- Average $$L_p$$ Distortion ($$ALD_p$$)

  平均$$L_p$$失真表示成功的对抗样本的平均标准化$$L_p$$失真。
  
  $$
  A L D_{p}=\frac{1}{n} \sum_{i=1}^{n} \frac{\left\|X_{i}^{a}-X_{i}\right\|_{p}}{\left\|X_{i}\right\|_{p}}
  $$
  
  $$ALD_p$$越小，对抗样本更加不可察觉。

- Average Structural Similarity (ASS)

  SSIM指标被用于量化图片之间的相似性，通常被认为比$$L_p$$范数更符合人类视觉感知。

  为了评估对抗样本的不可感知性，我们将ASS定义为所有成功对抗样本与其正常样本之间的平均SSIM相似性
  
  $$
  A S S=\frac{1}{n} \sum_{i=1}^{n} S S I M\left(X_{i}^{a}, X_{i}\right)
  $$
  
  ASS越大，对抗样本越不可感知。

- Perturbation Sensitivity Distance (PSD)

  基于contrast masking theory，PSD用于评估人类对扰动的感知，其中
  
  $$
  PSD = \frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{m} \delta_{i, j} \operatorname{Sen}\left(R\left(x_{i, j}\right)\right)
  $$
  
  其中，$$m$$是像素的数量，$$\delta_{i, j}$$表示第$$i$$个样本的第$$j$$个像素，$$R\left(x_{i, j}\right)$$表示$$x_{i,j}$$周围的方形区域，$$\operatorname{Sen}\left(R\left(x_{i, j}\right)\right)=1 / \operatorname{std}\left(R\left(x_{i, j}\right)\right)$$。



#### 2.2.3 鲁棒性

通常，物理世界中的图像在输入生产系统(如在线图像分类系统)之前不可避免地要进行预处理，这可能会导致对抗样本的MR下降。因此，评估对抗样本在各种实际情况下的鲁棒性是十分必要的。

- Noise Tolerance Estimation (NTE)

  对抗样本的鲁棒性通过噪声容忍度估计，反映了对抗样本在保持误分类的类标不变的情况下所能承受的噪声量。

  NTE计算误分类类的概率与所有其他类的最大概率之间的差距，
  
  $$
  NTE=\frac{1}{n} \sum_{i=1}^{n}\left[P\left(X_{i}^{a}\right)_{F\left(X_{i}^{a}\right)}-\max \left\{P\left(X_{i}^{a}\right)_{j}\right\}\right]
  $$
  
  其中$$j \in\{1, \cdots, k\}$$，$$j \neq F\left(X_{i}^{a}\right)$$。

  NTE越大，对抗样本越鲁棒；

另一方面，由于转换函数的不确定性，我们选取了高斯模糊和图像压缩两种应用最广泛的图像预处理方法来评估对抗样本的鲁棒性。

- Robustness to Gaussian Blur (RGB)

  高斯模糊是计算机视觉算法中广泛应用的预处理方式，用于降低图像中的噪声。一般情况下，鲁棒的对抗样本在高斯模糊后仍能保持其误分类效果。

  - UA:    $$R G B_{U A}=\frac{\operatorname{count}\left(\mathbf{F}\left(\mathbf{G B}\left(X_{i}^{a}\right)\right) \neq y_{i}\right)}{\operatorname{count}\left(\mathbf{F}\left(X_{i}^{a}\right) \neq y_{i}\right)}$$
  - TA:    $$RGB_{TA} =\frac{\operatorname{count}\left(\mathbf{F}\left(\mathbf{G B}\left(X_{i}^{a}\right)\right)=y_{i}^{*}\right)}{\operatorname{count}\left(\mathbf{F}\left(X_{i}^{a}\right)=y_{i}^{*}\right)}$$

  其中，$$\mathbf{G B}$$表示高斯迷糊函数。$$RGB$$越高，对抗样本越鲁棒。

- Robustness to Image Compression (RIC)

  与$$RGB$$类似，$$RIC$$能公式化为：

  - UA:    $$RIC_{UA} = \frac{\operatorname{count}\left(\mathbf{F}\left(\mathbf{I C}\left(X_{i}^{a}\right)\right) \neq y_{i}\right)}{\operatorname{count}\left(\mathbf{F}\left(X_{i}^{a}\right) \neq y_{i}\right)}$$
  - TA:    $$R I C_{T A}=\frac{\operatorname{count}\left(\mathbf{F}\left(\mathbf{I C}\left(X_{i}^{a}\right)\right)=y_{i}^{*}\right)}{\operatorname{count}\left(\mathbf{F}\left(X_{i}^{a}\right)=y_{i}^{*}\right)}$$

  其中IC为具体的图像压缩函数。RIC值越高，AEs值越高。



#### 2.2.4 计算代价

我们定义了计算成本**Computation Cost(CC)**作为攻击者平均生成AE的运行时，从而评估攻击成本。



## 3. 防御

现存的防御方法可以归纳为5类。

### 3.1 防御方法

- Adversarial Training

  对抗训练希望通过使用新生成的对抗样本对训练集进行扩充来学习鲁棒模型。

- Gradient Masking/Regularization

  梯度掩蔽/正则化方法通过降低模型对对抗样本的敏感性并隐藏梯度来防御。

- Input Transformation

  上面讨论的防御依赖于生成的对抗样本或需要对原始模型进行修改，因此设计与攻击/模型无关的防御来抵御对抗攻击就显得尤为重要。

  将测试输入输入到原始模型(我们称之为输入转换防御)之前，通过预处理消除测试输入中的对抗干扰。

- Region-based Classification

- Detection-only Defenses

  基于检测的防御方法



### 3.2 防御指标

一般来说，防御可以从两个角度来评估:效能保留和抵抗攻击的能力。特别是，效用保留反映了防御增强模型如何保留原始模型的功能，而抵抗攻击反映了防御增强模型抵御对抗攻击的有效性。防御效能对基于检测的防御方法没有意义，因为它们只检测对抗样本并拒绝它们。因此，只评估完全防御方法的效用性能。

假设$$\mathbf{F}^{\mathbf{D}}$$是原模型的防御增强模型$$\mathbf{F}$$，$$p^{D}$$表示$$\mathbf{F}^{\mathbf{D}}$$的softmax层输出。

- Classification Accuracy Variance (CAV)

  防御增强模型应尽可能地保证对正常测试样本的分类精度。为了评估防御对准确性的影响，定义
  
  $$
  C A V=A c c\left(\mathbf{F}^{\mathrm{D}}, T\right)-A c c(\mathbf{F}, T)
  $$
  
  其中，$$A c c(\mathbf{F}, T)$$表示模型$$\mathbf{F}$$在数据集$$T$$上的精度。

- Classification Rectify/Sacrifice Ratio (CRR/CSR)

  为了评估防御对测试集上的模型预测的影响，详细描述了应用防御前后预测的差异。

  - CRR（分类校准比）: 为之前被$$\mathbf{F}$$错误分类但被$$\mathbf{F}^{\mathrm{D}}$$正确分类的测试样本的百分比。
  - 
    $$
    C R R=\frac{1}{N} \sum_{i=1}^{N} \operatorname{count}\left(\mathbf{F}\left(X_{i}\right) \neq\right. \left.y_{i} \& \mathbf{F}^{\mathbf{D}}\left(X_{i}\right)=y_{i}\right)
    $$

  - CSR（分类牺牲比）: 为之前被$$\mathbf{F}$$正确分类但被$$\mathbf{F}^{\mathrm{D}}$$错误分类的测试样本的百分比。
  - 
    $$
    C S R=\frac{1}{N} \sum_{i=1}^{N} \operatorname{count}\left(\mathbf{F}\left(X_{i}\right) =\right. \left.y_{i} \& \mathbf{F}^{\mathbf{D}}\left(X_{i}\right)\neq y_{i}\right)
    $$

  实际上，$$CAV=CRR-CSR$$。

- Classification Confidence Variance (CCV)

  虽然防御增强模型可能不会影响性能的准确性，但正确分类样本的预测置信度可能会显著降低。

  为了测量由防御增强模型引起的置信变化，我们定义：
  
  $$
  C C V=\frac{1}{n} \sum_{i=1}^{n}\left|P\left(X_{i}\right)_{y_{i}}-P^{D}\left(X_{i}\right)_{y_{i}}\right|
  $$
  
  其中，$$n<N$$是被$$\mathbf{F}$$和$$\mathbf{F}^{\mathrm{D}}$$同时分类正确的样本。

- Classification Output Stability (COS)

  为了度量原始模型和防御增强模型之间的分类输出稳定性，我们使用JS散度来度量它们输出概率分布的相似性。在所有正确分类的样本上，我们平均了原始和防御增强模型输出之间的JS散度差异。
  
  $$
  C O S=\frac{1}{n} \sum_{i=1}^{n} J S D\left(P\left(X_{i}\right) \| P^{D}\left(X_{i}\right)\right)
  $$
  
  其中，$$n<N$$是被$$\mathbf{F}$$和$$\mathbf{F}^{\mathrm{D}}$$同时分类正确的样本。



## 框架简述

- 攻击方法：16种，8种UA，8种TA
  - UAs: FGSM, R+FGSM, BIM, PGD, U-MI-FGSM, DF, UAP, OM; 
  - TAs: LLC, R+LLC, ILLC, T-MI-FGSM, BLB, JSMA, CW2, EAD;
- 防御方法：13种
  - adversarial training defenses: NAT, EAT, PAT;
  - gradient masking defenses: DD, IGR;
  - input transformation based defenses: EIT, RT, PD, TE;
  - region-based classification defense: RC;
  - detection-only defenses: LID, FS, MagNet;

