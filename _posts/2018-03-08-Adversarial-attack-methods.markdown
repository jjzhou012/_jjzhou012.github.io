---
layout: article
title: The Adversarial Attacks methods
date: 2018-03-08 18:38:04 +0800
tags: [Adversarial, Summary]
categories: blog
pageview: true
---




翻译文献：[Threat of Adversarial Attacks on Deep Learning
in Computer Vision: A Survey](https://arxiv.org/abs/1801.00553)

## 常用术语介绍

- Adversarial example/image(对抗样本)：由干净样本添加了对抗扰动得到的新样本，用于欺骗机器学习模型。
- Adversarial perturbation(对抗扰动)：使干净样本转化为对抗样本的噪声。
- Adversarial training(对抗训练)：使用混合的对抗样本和干净样本去训练模型。
- Adversary (攻击者)：制造对抗样本的人，有时候也指代对抗样本本身。
- Black-box attacks(黑盒攻击)：攻击者在不清楚（或了解很少）模型的参数和结构的情况下，生成对抗样本攻击模型。
- Detector(检测器)：用于检测一幅图像是否是对抗样本的装置。
- Fooling ratio/rate：(欺骗率)：样本被扰动后，一个训练过的模型改变预测标签的比例。
- One-shot/one-step methods(单步法)：使用单步计算去生成对抗扰动，相对于迭代法，后者计算量大。
- Quasi-imperceptible(难以察觉)：轻微扰动图像，在人类视觉下无法察觉。
- Rectifier(整流器，校正器)：整流器修改对抗样本，以恢复目标模型的预测，对应于干净版本的预测。
- Targeted attacks(目标攻击)：欺骗机器学习模型，使之将对抗图片识别为指定标签。相对于non-targeted attacks(无目标攻击)。
- non-targeted attacks(无目标攻击)：欺骗机器学习模型，使之将对抗图片识别为错误的任意标签。
- Threat model(威胁模型)：指由方法考虑的潜在攻击的类型，例如， 黑盒攻击。
- Transferability(转移性)：对抗样本对于生成模型之外的其他模型依然有效。
- Universal perturbation(通用扰动)：添加到任何图像中，都能够以高概率愚弄的给定模型。 请注意，普遍性是指扰动的"图像的不可知"性，与'具有良好的可转移性'相反。
- White-box attacks(白盒攻击):攻击者完全掌握了目标模型的参数，架构，训练方法，有时候包括训练数据。

##  攻击方法分类

&emsp; 从不同角度分类现有的攻击策略：

- Black/White box

- Targeted/Non-targeted

- Gradient/Optimization/others
- Specific/Universal
- One-shot/Iterative

&emsp;<span id="Summary-">各方法属性总结：</span>

Method|<font size=2>Black/ White box</font>|<font size=2>Targeted/ Non-targeted</font>|<font size=2>Gradient/ Optimiza tion/others</font>|<font size=2>Specific/ Universal</font>|<font size=2>perturbation norm</font>|learn|strength
---|---|---|---|---|---|---|---
[L-BFGS](#L-BFGS)|White-box|Targeted|-|Specific|$L_∞$|One-shot|3*
[FGSM](#FGSM)|White-box|Targeted|Gradient|Specific|$L_∞$|One-shot|3*
[BIM&ILCM](#BIM&ILCM)|White-box|Non-|Gradient|Specific|$L_∞$|Iterative|4*
[JSMA](#JSMA)|White-box|Targeted|Gradient|Specific|$L_0$|Iterative|3*
[One-Pixel](#One-Pixel)|Black-box|Non-|-|Specific|$L_0$|Iterative|2*
[C&W attacks](#C.W-attacks)|White-box|Targeted|-|Specific|$L_0,L_2,L_∞$|Iterative|5*
[DeepFool](#DeepFool)|White-box|Non-|-|Specific|$L_2,L_∞$|Iterative|4*
[Uni.perturbation](#Uni-pert)|White-box|Non-|-|Universal|$L_2,L_∞$|Iterative|5*
[UPSET](#UPSET)|black-box|Targeted|-|Universal|$L_∞$|Iterative|4*
[ANGRI](#ANGRI)|Black-box|Targeted|-|Specific|$L_∞$|Iterative|4*
[Houdini](#Houdini)|Black-box|Targeted|-|Specific|$L_2,L_∞$|Iterative|4*
[ATNs](#ATNs)|White-box|Targeted|-|Specific|$L_∞$|Iterative|4*

### L-BFGS

通过对图像添加人眼不可察的微小扰动来误导神经网络做出错误分类。他们试图求解让神经网络做出错误分类的最小扰动方程，限于问题的高复杂度，他们简化了过程转而寻找最小的代价函数添加项，来误导神经网络，从而将问题转化为一个凸优化过程。  



### FGSM

#### 快速梯度符号法（FGSM）：

<picture>

符号     | 说明
-------- | ---
$θ$      | 模型参数
$x$      | 模型输入
$y$      | 关于x的目标
$J(θ,x,y)$ | 损失函数  
&emsp;将损失函数线性化为  $θ$ 的当前值，得到最优的**最大范数**约束扰动
$$
η=εsign(∇_xJ(θ,x,y))
$$
&emsp;称之为产生对抗样本的“快速梯度符号法”。 请注意，可以使用反向传播有效地计算所需的梯度。  

- FGSM在各种模型上的效果 :  

$ε$      | 0.25  |0.25   |  0.1
-------- | --    |--     |-----
测试集    |MNIST  |MNIST  |CIFAR-10预
分类器    |softmax|maxout |maxout
错误率    |99.9％ |89.4％ |87.15％
平均置信度|79.3％ |97.6％ |96.6％

> [预处理代码](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout)   
其产生大约0.5的标准偏差。

&emsp;其他生成对抗样本的简单方法也是可能的。 例如，我们还发现[**在梯度方向上以小角度旋转**](https://jjzhou012.github.io/2018/03/11/adversaria-%E6%AD%A3%E7%A1%AE%E5%88%86%E7%B1%BB-%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC-%E9%B2%81%E6%A3%92%E6%80%A7/) $x$ 也能可靠地产生对抗样本。

- 在ImageNet上应用GoogLeNet的快速对抗样本生成演示：  

![](http://p5bxip6n0.bkt.clouddn.com/18-3-10/67792085.jpg-Watermark)

&emsp;通过添加一个不可察的小矢量，其元素等于损失函数梯度的元素的符号，可以改变GoogLeNet的图像分类。这里，$ε$ 对应于GoogLeNet转换为实数后8位图像编码的最小位数的大小， 因等于0.007。  

&emsp;对抗样本生成过程：
$$
x ̃=x+εsign(∇_xJ(θ,x,y))
$$

![](http://p5bxip6n0.bkt.clouddn.com/18-3-10/43011594.jpg-Watermark)

#### FGSM的one-step target class变体

符号     | 说明
-------- | ---
$X$|干净样本
$X^{adv}$ |对抗图像
Misclassified adversarial image|误分类的对抗图像
$ε$|对抗扰动大小
$J(X,y_{true})$|用于训练模型的损失函数


- 最大化某个特定目标类 $y_{target}$ 的概率 $p(y_{target}X)$ ，该目标类不可能是给定图像的真实类。 对于具有交叉熵损失的神经网络，一步目标类方法的公式：

$$
X^{adv}=X - εsign(∇_XJ(X,y_{target}))
$$
- 对于目标类，我们可以选择被网络
$$
y_{LL} = arg \min_y\{p(y \mid X)\}
$$
&emsp;&emsp;&emsp;预测的最小概率的类

- 对于这里给定图像 $X$ 和 $y$ 的神经网络的交叉熵代价函数 $J(X,y)$ 。 故意忽略代价函数中的网络权重（和其他参数）$θ$,
假设它们是固定的。对于softmax输出层的神经网络，则有 整数类标签的交叉熵代价函数等于给定图像的真实类的负对数概率：
$$
J(X,y)= - \log\;p（y \mid X）
$$




### BIM&ILCM

#### Basic iterative method(BIM):

- $Clip_{X,ε}\{X'\}$ 基于像素剪切$X'$, 结果在原图像$X$的$ε$的最大范数邻域内，剪切方程：

$$
Clip_{X,ε}\{X'\}(x,y,z) = \min\{255,X(x,y,z)+ε,\max\{0,X(x,y,z)-ε,X'(x,y,z)\}\}
$$

&emsp;&emsp;&emsp;$X(x,y,z)$ 的值为图像$X$在坐标 $(x,y)$ 处 $z$ 通道的值。

- 对于 单步类的方法，将它扩展为很多小步，在每个步骤之后剪切中间结果的像素值以确保它们处于
原始图像的一个 $ε$ 邻域内：
$$
\sideset{}{^{adv}_0}X = X,  \sideset{}{^{adv}_{N+1}}X=Clip_{X,ε}\{\sideset{}{^{adv}_{N}}X+αsign(ᐁ_XJ(\sideset{}{^{adv}_{N}}X,y_{true}))\}
$$
- 在作者的实验中，使用了 $α=1$，即在每一步中只将每个像素的值改为1。 选择迭代次数为 $\min(ε+ 4,1.25ε)$。 这种迭代量是通过启发式选择的; 对抗样本足以达到$ε$的最大范数球但有足够的限制，以保持实验的计算成本可控。

#### ITERATIVE LEAST-LIKELY CLASS METHOD(ILCM)
&emsp;前面描述的两种方法都只是试图增加正确类的成本，而没有指定模型应该选择哪些不正确的类。 这样的方法足以应用于诸如MNIST和CIFAR-10数据集，其中类别数量很少，并且所有类别彼此差异很大。 在ImageNet上，类别数量多得多，类别之间的差异程度也不同，这些方法可能导致无趣的错误分类，例如将一种雪橇犬误认为另一种雪橇犬。 为了创造更多有趣的错误，引入了迭代最不可能的类方法。

- 这种迭代方法试图制作一个对抗图像，将其分类为特定的期望目标类别。 对于期望的类别，根据图像$X$上训练网络的预测选择最不可能的类：
$$
y_{LL} = arg \min_y\{p(y\mid X)\}
$$

&emsp;对于训练很好的分类器，最不可能的分类通常与真实分类非常不相似，因此这种攻击方法会导致更多有趣的错误，例如将狗误认为是飞机。

- 生成一张被分类为 $y_{LL}$ 的对抗图像，在 $sign\{∇_X {\log}\;p(y_{LL}\mid X)\}$ 方向上进行迭代来最小化 ${\log}\;p(y_{LL}\mid X)$，交叉熵损失的神经网络最后的表达式为 $sign\{-∇_X J(X,y_{LL})\}$ ，过程为下：
$$
\sideset{}{^{adv}_0}X = X,  \sideset{}{^{adv}_{N+1}}X=Clip_{X,ε}\{\sideset{}{^{adv}_{N}}X-αsign(ᐁ_XJ(\sideset{}{^{adv}_{N}}X,y_{LL}))\}
$$

### JSMA

&emsp;Papernot 等人通过限制扰动的 $L_0$ 范数，创造了一种新的对抗攻击“JSMA”。从物理上来说，这意味着只需修改图像中的几个像素，而不是扰动整个图像就能欺骗分类器。他们的算法一次修改干净图像的一个像素，并监视改动所引起的分类变化。通过使用网络层输出的梯度来计算显着图执行监视。显著图数值越大，表示欺骗网络的可能性越高，越容易获得目标标签。一旦图被计算出来，算法选择最有效的像素进行修改来欺骗网络。重复该过程，直到对抗图像中允许的像素的最大数量被改变或者愚弄成功。




### One-Pixel

&emsp;一种极端的攻击方法，仅仅改变图像的一个像素实现对抗攻击。该攻击基于差分进化算法，迭代的修改每个像素值，生成子图像，和母图像进行对比，保留攻击效果最佳的子图像最终实现攻击。它对多种类型的DNN模型有效，且需要极少的对抗信息。


### C&W attacks

通过限制扰动的l_0，l_2和l_∞范数使它们难以被察觉，并且成功的突破了防御净化法。


### DeepFool

&emsp;[github](https://github.com/lts4/deepfool)  

通过迭代生成最小规范扰动，将位于分类边界的图像逐步推到边界外，直到产生错误分类。

### Uni.perturbation 

### UPSET 

### ANGRI 

### Houdini 

通过限制l_2和l_∞范数生成能适应任务损失的对抗样本来欺骗基于梯度的ML模型。该算法利用神经网络的可微损耗函数的梯度来计算扰动。


### ATNs 
