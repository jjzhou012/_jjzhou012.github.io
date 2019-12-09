---
layout: article
title: 知识图谱：知识表示之TransD模型
date: 2019-01-20 00:01:00 +0800
tag: knowledge representation
categories: blog
pageview: true
---


# TransD

论文地址: [http://www.aclweb.org/anthology/P15-1067](http://www.aclweb.org/anthology/P15-1067)

TransR /CTransR模型存在的问题：

- 对于一种关系，所有实体共享一样的映射矩阵${\bf M}_r$。然而被同一个关系链接的实体拥有不同的类型和属性，这些差异需要用不同的映射来体现；
- 实体与关系间的映射是可逆的，没有理由只通过关系来构建映射矩阵；
- 矩阵向量乘法运算，参数多，模型运算量大；

创新点：

- TransD可以说是TransR/CTransR的简化版本，它同时考虑了实体和关系之间的多样性，用两个向量来动态重构mapping矩阵；
- 相比TransR/CTransR有更小的计算量，且没有矩阵运算，可以在大规模KG上应用；

**实体关系的多语义表示**：

![](https://ws1.sinaimg.cn/mw690/005NduT8ly1g9pt720skmj30jp0i2wj1.jpg)



## 模型改进

TransD 模型同 CTransR 模型一样，都是为了解决关系的多种语义表示。相比较 CTransR 采用聚类的方式，TransD 提出一种**动态变化矩阵**的方法。

![kXc0l4.png](https://s2.ax1x.com/2019/03/05/kXc0l4.png)



每一个命名对象（实体关系）用两个向量表示：

- $(h,r,t)$ :  自身的关系（语义）表示；
- $(h_p, r_p, t_p)$ :  实体空间中的投影，用于构建映射矩阵的表示；

具体公式如下图所示：

$$
{\bf M}_{rh} = {\bf r}_p{\bf h}_p^T + {\bf I}^{m \times n}
$$

$$
{\bf M}_{rt} = {\bf r}_p{\bf t}_p^T + {\bf I}^{m \times n}
$$

- 下标$p$表示投影向量， ${\bf h} , {\bf h}_p, {\bf t} , {\bf t}_p \in {\Bbb R}^n$ ,  ${\bf r}, {\bf r}_p \in {\Bbb R}^m$
- 映射矩阵的每一个元素都包含了实体和关系信息；
- 通过向量相乘生成的矩阵对单位矩阵（代表不做变换）进行调整；

定义向量在关系空间的投影：

$$
{\bf h_{\perp} }={\bf M}_{rh} {\bf h}
$$

$$
{\bf t_{\perp} }={\bf M}_{rt} {\bf t}
$$

得分函数：

$$
f_r({\bf h,t})= -\| {\bf h_\perp} + {\bf {r-t}_\perp}\|_2^2
$$

- 限制条件：  

$$
\|{\bf h}\|_2 \leq 1, \|{\bf t}\|_2 \leq 1, 	\|{\bf r}\|_2 \leq 1, \|{\bf h_\perp}\|_2 \leq 1, \|{\bf t_\perp}\|_2 \leq 1
$$



## 训练过程

假设训练集中有$n_t$个三元组，用$(h_i,r_i,t_i)(i=1,2,...,n_t)$表示第$i$个三元组。

每个三元组有标签$y_i$表示三元组的正负性质：

$$
\Delta = \{(h_j,r_j,t_j)|y_j=1\}
$$

$$
\Delta' = \{(h_j,r_j,t_j)|y_j=0\}
$$

### 负样本生成

$$
\Delta' = \{(h_l,r_k,t_k)|h_l\neq h_k \wedge y_k=1\} \cup \{(h_k,r_k,t_l)|t_l\neq t_k \wedge y_k=1\}
$$

### 损失函数

$$
{\cal L}=\sum_{\xi \in \Delta} \sum_{\xi' \in \Delta' } [\gamma + f_r(\xi')-f_{r}(\xi) ]_+
$$

- $\xi$ : 正样本， $\xi'$ : 负样本

为加快收敛和避免过拟合：

- 利用TransE模型初始化实体和关系嵌入向量；

- 用单位矩阵初始化转移矩阵；   


## 和其他模型的联系

### TransE

当向量维度满足$m=n$，且所有用于构建映射矩阵的投影向量都为0时，TransE是TransD的一种特殊情况。

### TransH

当向量维度满足$m=n$时，实体的投影向量能被表示为：

$$
{\bf h_{\perp} }={\bf M}_{rh} {\bf h} = {\bf h}+{\bf h}_p^T {\bf h} {\bf r}_p
$$

$$
{\bf t_{\perp} }={\bf M}_{rt} {\bf t} = {\bf t}+{\bf t}_p^T {\bf t} {\bf r}_p
$$

此时，投影向量仅由关系表示。

### TransR/CTransR

相比TransR对每个关系直接定义一个映射矩阵，TransD通过为每个实体关系对设置一个投影向量，来为三元组动态的构造映射矩阵。

另外，TransD没有矩阵向量乘法操作，可以用向量运算代替：

假设$m \geq n$，投影向量能按以下方式计算：

$$
{\bf h_{\perp} }={\bf M}_{rh} {\bf h} = {\bf h}_p^T {\bf h} {\bf r}_p + [{\bf h}^T, {\bf 0}^T]^T
$$

$$
{\bf t_{\perp} }={\bf M}_{rt} {\bf t} = {\bf t}_p^T {\bf t} {\bf r}_p + [{\bf t}^T, {\bf 0}^T]^T
$$
