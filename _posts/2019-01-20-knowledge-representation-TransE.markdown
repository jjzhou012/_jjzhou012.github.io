---
layout: article
title: 知识图谱：知识表示之TransE模型
date: 2019-01-20 00:01:00 +0800
tag: knowledge representation
categories: blog
pageview: true
---


# TransE

论文地址：[https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546)  

## 模型概述

- 三元组：   $(h,{\scr l},t)$  	
- embedding之后， 头部实体嵌入向量加上关系嵌入向量，更接近与尾部实体嵌入向量
- 依赖于简化的参数集，只学习每个实体和每个关系的一个**低维向量**表示

![kXcCWD.png](https://s2.ax1x.com/2019/03/05/kXcCWD.png)

细节：
- 训练集 $S$: 包含三元组 $(h,{\scr l},t)$， 实体 $h,t\in E(实体集)$, 关系 ${\scr l} \in L(关系集)$；
- embeddings 在 $\Bbb R^k$ 中取值
- ${\bf h}+{\scr l}\approx {\bf t}$ , ${\bf t}$ 应该为 ${\bf h}+{\scr l}$ 的最邻近， 然后 ${\bf h}+{\scr l}$ 与其他的 $t$ 尽可能远；  这里的“接近”程度可以用 $L_1$或$L_2$范数衡量；  
  理想状态下一个正确的三元组的embedding 之间存在 ${\bf h}+{\scr l}={\bf t}$ 的关系，错误的三元组没有；
- 利用基于能量的框架，三元组的势能表示为 $d(h, {\scr l}, t)=\|h+{\scr l}-t\|_2$ ， 正确的三元组势能越低越好，错误的三元组势能越高越好；



## 损失函数： 

### 带negative sample的max margin损失函数

训练方法：margin-based ranking criterion


$$
{\cal L}=\sum_{(h,{\scr l},t)\in S} \sum_{(h',{\scr l},t')\in S'_{(h,{\scr l}, t)} } [\gamma + d(h+{\scr l}, t)-d(h'+{\scr l}, t') ]_+
$$

$$
d(h, {\scr l}, t)=\|h+{\scr l}-t\|_2
$$

- $S$: 正确三元组集合
- $S'$: 错误三元组集合
- $\gamma$: margin 距离超参数，表示正负样本之间的距离，常数;
- $[x]_+$: $max(0,x)$

> 最小化loss可以使正样本势能越低，负样本势能越高，但两者的能量差距达到一定程度 $\gamma$ 就足够了， 再大loss也只是0；

### 负样本生成

$$
S'_{(h,{\scr l},t)} = \{(h',{\scr l},t)|h'\in E\}\bigcup \{(h,{\scr l},t')|t' \in E\}
$$

- 对于三元组 $(h,{\scr l},t)$， 随机使用知识库中的某个实体 $h'$ 替换 $h$，或用某个实体 $t'$ 替换 $t$, 得到两个负样本 $(h',{\scr l},t)$ 和 $(h,{\scr l},t')$;  
- 对生成的负样本进行筛选过滤，若该负样本原本存在于知识库，则重新生成；  
- 然后，有人认为，生成负样本时不应该完全随机，而是应该选择与被替换实体类型相似的实体来进行替换；

## 训练过程

![](http://p5bxip6n0.bkt.clouddn.com/18-11-1/93529922.jpg)



## TransE局限性

考虑在没有错误embedding的情况下，$h+{\scr l}=t$  当 $(h,{\scr l},t) \in \Delta$ 时，我们可以从TransE模型中看出：

- 若 $(h,{\scr l},t) \in \Delta$ 且 $(t,{\scr l},h) \in \Delta$, $r$是一个自反映射， 因此 $r=0$ 且 $h=t$;
- 若 $\forall i \in \{0,...,m\},(h_i,r,t)\in \Delta$ , $r$是一个N-1映射，且 $h_0=...=h_m$;  
类似的，$\forall i \in \{0,...,m\},(h,r,t_i)\in \Delta$ , $r$是一个1-N映射，且 $t_0=...=t_m$;  

当涉及相同关系时，忽略了实体的分布式表示，导致实体呈现相同的嵌入表示。

> 例如，假如知识库中有两个三元组，分别是(美国, 总统, 奥巴马)和(美国, 总统, 布什)。这里的关系“总统”是典型的 1-N 的复杂关系。如果用 TransE 从这两个三元组学习知识表示，将会使奥巴马和布什的向量变得相同。
> 因而，TransE 模型在处理 1-N、N-1、N-N 复杂关系时存在局限性。




