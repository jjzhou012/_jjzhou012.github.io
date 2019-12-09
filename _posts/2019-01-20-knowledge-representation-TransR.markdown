---
layout: article
title: 知识图谱：知识表示之TransR模型
date: 2019-01-20 10:01:00 +0800
tag: knowledge representation
categories: blog
pageview: true
---

# TransR

论文地址: [http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/)

## 模型改进
TransE 和 TransH 都假设实体和关系嵌入在相同的空间中。然而，一个实体是多种属性的综合体，不同关系对应实体的不同属性，即头尾节点和关系可能不在一个向量空间中，在公共语义空间中不能表示它们。

<img src="https://s2.ax1x.com/2019/03/05/kXcAOA.png" alt="kXcAOA.png" style="zoom:67%;" />

- 对每个三元组，实体空间中的实体首先通过运算 $M_r$ 映射到与它相关的 $r$ 关系空间, 保证 ${\bf h}_r + r \approx {\bf t}_r$ , 关系特定投影可以使实际持有该关系(表示为彩色圆圈)的头/尾实体彼此靠近，也可以使不持有该关系(表示为彩色三角形)的头/尾实体远离。
- TransR对每个关系$r$都分配了一个空间${\bf M}_r \in R^{k \times d}$。
- 特定关系下，头尾实体会表现不同的模式；



## 方法

提出了一种新的方法，**在不同的空间中建模实体和关系**，即，**实体空间**和**关系空间**，并在关系空间中进行转换，因此命名为TransR。

### TransR

TransR 模型中，对每一个三元组，实体嵌入 ${\bf h,t} \in {\Bbb R}^k$ , 关系嵌入 ${ {\bf r} \in {\Bbb R}^d}$, 两个空间的维度不一定相同（属于不同空间）。  
- 对每一个关系$r$，定义一个投影矩阵 ${\bf M}_r \in {\Bbb R}^{k \times d}$, 将实体从实体空间投影到关系空间：


$$
{ {\bf h_r} = {\bf hM}_r}
$$

$$
{ {\bf t_r} = {\bf tM}_r}
$$

- 得分函数：

$$
f_r(h,t)=\|{ {\bf h}_r} + {\bf r} - {\bf t}_r\|_2^2
$$

- 实体和关系嵌入的范数约束  

$$
\|{\bf h}\|_2 \leq 1, \|{\bf t}\|_2 \leq 1,\|{\bf r}\|_2 \leq 1, \|{\bf hM}_r\|_2 \leq 1, \|{\bf tM}_r\|_2 \leq 1
$$



### Cluster-based TransR (CTransR)

上述模型包括TransE, TransH和TransR，仅仅通过单个的关系向量还不足以建立从头实体到尾实体的所有转移，**即对于同一条关系$r$  来讲，$r$ 具有多种语义上的表示**。为了更好地建模这些关系，引入了**分段线性回归**的思想来扩展TransR。

- 对于一个特定的关系$r$，把训练数据中的所有实体对 $(h, t)$ 聚类到多个组中，每个组中的实体对都期望表现出相似的$r$关系。其中所有实体对$(h,t)$通过的向量偏移$(h−t)$来聚类；

  > 为什么根据$(h-t)$(也就是$r$)就能聚类:
  >
  >  CTransR考虑的问题是对一个关系只用一个表示无法体现这一种关系的多义性，比如关系（location location contains）其实包含country-city、country-university、continent-country等多种含义。
  >
  > ![kXc1yj.png](https://s2.ax1x.com/2019/03/05/kXc1yj.png)
  >
  > 原文提到，这里的$h,t$是经过TRansE模型预训练得到的: 
  >
  > ![kXcJwq.png](https://s2.ax1x.com/2019/03/05/kXcJwq.png)
  >
  > ----
  >
  > ![kXcNkV.png](https://s2.ax1x.com/2019/03/05/kXcNkV.png)
  >
  >  而TransE模型的映射是唯一（one to one）的，大关系下的不同子关系通过映射后其实是不同的向量表示，那么$(h-t)$的结果也不相同，根据不同的结果可以用来进行聚类。

- 对每一个簇, 学习一个**分离关系向量** ${\bf r}_c$ ，对每种关系，学习投影矩阵 ${\bf M}_r$

- 定义实体和关系的投影向量为：

$$
{ {\bf h}_{r,c} = {\bf hM}_r}
$$

$$
{ {\bf t}_{r,c} = {\bf tM}_r}
$$

- 得分函数：

$$
f_r(h,t)=\|{ {\bf h}_{r,c}} + {\bf r}_c - {\bf t}_{r,c}\|_2^2 + \alpha \|{\bf r}_c - {\bf r}\|_2^2
$$

$\|{\bf r}_c - {\bf r}\|_2^2$ 这一项确保群特异性关系向量 ${\bf r}_c$ 距离原始关系向量 ${\bf r}$ 不太远， $\alpha$ 控制约束效果
- 实体和关系嵌入的范数约束  

$$
\|{\bf h}\|_2 \leq 1, \|{\bf t}\|_2 \leq 1,\|{\bf r}\|_2 \leq 1, \|{\bf hM}_r\|_2 \leq 1, \|{\bf tM}_r\|_2 \leq 1
$$



## 训练过程

损失函数：

$$
{\cal L}=\sum_{(h,r,t)\in \Delta} \sum_{(h',r,t')\in \Delta'} max (0, \gamma + f_r(h, t)-f_{r}(h', t') )
$$

### 负样本生成

为首尾实体替换分配不同的概率。对于那些1- n、n -1和n -n关系，给“one”设置更高的概率，产生假阴性样本的机会将会减少 

