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

