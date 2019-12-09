---
layout: article
title: 知识图谱：知识表示之HyTE模型
date: 2019-01-20 00:08:00 +0800
tag: knowledge representation
categories: blog
pageview: true
---



pageview: true
---

# HyTE

Hyperplane-based Temporally aware KG Embedding

论文地址：[http://talukdar.net/papers/emnlp2018_HyTE.pdf](http://talukdar.net/papers/emnlp2018_HyTE.pdf)



## 模型改进

现有的 KG embedding方法很少考虑到**时间**维度，因为它们假设所有的三元组总是永远正确的，可是现实中很多情况下不是这样。

>  For example, (Bill Clinton, presidentOf, USA) was true only from 1993 to 2001;

考虑到三元组**时效性**问题，提出了 HyTE 模型，定义了三元组有效成立时间段为 **temporal scopes** ，这些temporal scopes随着时间的推移对许多数据集会产生影响（比如YAGO，Wikidata），可以用于：

- 利用**时间导向**进行知识图谱图推理；
- 为缺失时间注释的事实预测 temporal scopes； 



考虑一个四元组 $(h,r,t,[τ_s, τ_e])$，这里的 $τ_s$ 和 $τ_e$ 分别定义了三元组成立时间段的**起始**与**截止**。TransE模型将实体和关系考虑到相同的语义空间，但是在不同的时间段，实体与关系组成的 $(h，r)$ 可能会对应到不同的尾实体  $t$ ，所以在本文的模型中，希望**实体能够随不同的时间点有着不同的表示**。为了达到这一目的，文中将时间表示成超平面,模型示意图如下：

![](http://ww1.sinaimg.cn/large/005NduT8ly1g35i5vr9vwj30ng0hb0xa.jpg)

> $e_h，e_t，e_r$，分别表示三元组中头实体，尾实体以及关系所对应的向量表示;
>
> $τ_1$ 和 $τ_2$ 分别表示此三元组有效成立时间段的**起始时间**与**截止时间**;
>
> $e_h(τ_1)$, $e_r(τ_1)$ 以及表示各向量在时间超平面 $τ_1$上的投影;
>
> 最终，模型通过最小化 translational distance 来完成结合时间的实体与关系的embedding学习过程。



给定时间戳，可以将图分解为几个静态图，其中包含在各个时间步骤中有效的三元组，如：

知识图 $G$ 能被表示为 ${\Bbb {G=G_{τ_1} \cup G_{τ_2} \cup ... \cup  G_{τ_T} } }$ , 其中 $τ_i, i \in 1,2,...,T$ 是离散时间点。

构建时间组成图 ${\Bbb G_τ}$ 时，对于一个四元组 $(h,r,t,[τ_s,τ_e])$ ，考虑每个在 $τ_s , τ_e$ 之间的时间点，该四元组为正样本。$τ$ 时刻正样本集合定义为 ${\scr D_τ^+}$ 。

对于 $T$ 个时间点的 KG，将会有 $T$ 个用不同法向量（$w_{t_1}, w_{t_2}, ..., w_{t_T}$）表示的超平面，在超平面的帮助下将空间隔离成不同时间域。在时间 $τ$ 下有效的三元组被投影到特殊超平面 $w_τ$，在超平面上平移距离被最小化。

计算在 $w_τ$ 上的投影表示，其中 $\|w_τ\|_2=1$ ：

$$
P_τ(e_h)=e_h - (w_τ^Te_h)w_τ  \\
P_τ(e_t)=e_t - (w_τ^Te_t)w_τ  \\
P_τ(e_r)=e_r - (w_τ^Te_r)w_τ  \\
$$

> 向量投影



## 优化

### 得分函数

对于在时间 $τ$ 有效的正样本，希望映射满足这样的关系 ：$P_τ(e_h)=P_τ(e_r) \approx P_τ(e_t)$, 因而使用以下的得分函数：

$$
f_τ(h,r,t)=\|P_τ(e_h)+P_τ(e_r)-P_τ(e_t)\|_{l1/l2}
$$

在实体和关系嵌入过程中，对每个时间戳 $τ$ , 学习对应法向量 $\{w_τ\}_{τ=1}^T$  。

通过将三元组投影到时间超平面，我们可以将时间信息融入关系实体嵌入，利用相同的分布式表示在不同的时间点呈现不同的表达。

### loss函数

$$
{\cal L} = \sum_{\tau \in [T]} \sum_{\tau \in \scr D_τ^+} \sum_{τ \in \scr D_τ^-} \max(0,f_τ(x)-f_τ(y)+\gamma)
$$

- $\scr D_τ^+$: 有效三元组集合，即正样本集合；
- $\scr D_τ^-$: 负样本集合；
- 实体约束： $\|e_p\|_2 \leq 1, \forall p \in {\varepsilon}$    实体向量的$L_2$正则化
- 法向量约束：$\|w_{\tau}\|_2=1,\forall \tau \in [T]$      法向量归一化                



### 负样本构造

考虑了两种负样本：

- **Time agnostic negative sampling(TANS)** 

  **与时间无关的负采样**：忽略时间戳，考虑不属于KG的负样本，在时间 $τ$ 采样该负样本:
  $$
  \scr D_τ^- = \{(h',r,t,\tau)|h' \in \scr {\varepsilon}, (h',r,t) \notin \scr D^+ \} \cup \{(h,r,t',\tau)|t' \in \scr {\varepsilon} , (h,r,t') \notin \scr D^+ \}
  $$

- **Time dependent negative sampling(TDNS)**

  **与时间相关的负采样**： 考虑时间戳，考虑属于KG，但不属于特定时间戳 $\tau$ 的负样本：
  $$
  \scr D_τ^- = \{(h',r,t,\tau)|h' \in \scr {\varepsilon}, (h',r,t) \in \scr D^+，(h',r,t,\tau) \notin \scr D_{\tau}^+ \} \cup \\
  \{(h,r,t',\tau)|t' \in \scr {\varepsilon} , (h,r,t') \in \scr D^+, (h,r,t',\tau) \notin \scr D_{\tau}^+ \}
  $$

