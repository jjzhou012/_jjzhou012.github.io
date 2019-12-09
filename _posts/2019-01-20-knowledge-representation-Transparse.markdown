---
layout: article
title: 知识图谱：知识表示之TransParse模型
date: 2019-01-20 14:01:00 +0800
tag: knowledge representation
categories: blog
pageview: true
---



# Transparse

论文地址：[http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11982/11693](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11982/11693)



## 模型改进

KG中面临的两个主要问题，分别是 **Heterogeneity（异构性）**和 **Imbalance（不平衡性）**：

- Heterogeneity： 知识图谱中不同关系连接的实体（节点）数量不同；
- Imbalance：同一关系连接的头尾实体（节点）数量不同；

![kXgFhT.png](https://s2.ax1x.com/2019/03/05/kXgFhT.png)

> 上图展示了子图FB15k的数据结构，可以看出：
>
> - 不同关系的复杂性差异很大；
> - 不平衡关系在知识图谱中占了很大比例；
>
> 早期模型未关注这两个问题，用同样的方法构建关系。
>
> **异构性可能导致简单或复杂关系的过拟合；关系的不平衡性表明平等的对待头尾实体是不合理的；**



因此提出了两种模型来解决这两个问题：

为克服异质性，我们提出了一种**TranSparse(share)**模型，其中转移矩阵的稀疏程度由关系所链接的实体对的数量决定，关系的两边共享相同的转移矩阵。其中，复杂关系的转移矩阵要比简单关系的转移矩阵稀疏。

为克服不平衡性，对TranSparse(share)模型修改，提出了**TranSparse(separate)**模型，该模型中每个关系有两个分离稀疏转移矩阵，一个对头部实体一个对尾部实体，稀疏程度由关系连接的实体数量决定。

> **Sparse Matrix**：稀疏矩阵中大多数元素为0，0元素占总元素的比例称为稀疏程度（$\theta$）,用$M(\theta)$表示稀疏程度为 $\theta$ 的矩阵，稀疏矩阵更容易压缩，需要更少的存储空间，且只有非零元参与计算，减少计算复杂度。下图展示了结构化和非结构化数据：
>
> ![kXgVc4.png](https://s2.ax1x.com/2019/03/05/kXgVc4.png)
>
> 结构化模式有助于向量矩阵运算，而非结构化模式则不是。因此，结构化模式可以使我们的模型更容易地扩展到大型知识图，而非结构化模式更多的用在其他文献中，有更好的实验结果。

> **稀疏矩阵 VS 低秩矩阵**
>
> 我们需要分别使用自由度高和自由度低的矩阵来学习复杂关系和简单关系。权重矩阵的自由度是指独立变量的个数。对于权矩阵M，低秩和稀疏都可以降低自由度，因为它们都是对M的约束。具体来说，低秩强制一些变量满足特定的约束，使得M中的变量不能被自由分配。这样，自由度就降低了。对于稀疏矩阵，我们让一些元素为零，在训练过程中不改变它们的值，另一些元素为自由变量，可以通过训练来学习。因此，自由度就是通过训练学会的变量的数量。但是稀疏矩阵更适合我们的任务有两个原因：
>
> - 稀疏矩阵比低秩矩阵更灵活，假设 $M \in {\Bbb R^{m*n} }$,  $rank(M) \leq \min (m,n)$ 。 因为秩能控制自由度（$m*n$矩阵秩为$k$，有自由度 $k(m+n)-k^2$ ），如果用低秩去控制自由度，只能获得 $\min (m,n)$个低秩矩阵；若用稀疏矩阵去控制自由度，能获得 $m*n$ 个稀疏矩阵；
> - 稀疏矩阵比低秩矩阵更高效。稀疏矩阵中仅非零元参与计算，减少了计算复杂度。而且稀疏矩阵更容易迁移到大型知识图谱上； 



### TranSparse

#### TranSparse(share)

- $M_r(\theta_r)$：转移矩阵；

- $\bf r$：每个关系$r$的转移向量；

- $N_r$：关系$r$连接的实体对数量；

- $N_{r^*}$：$N_r$中最大的数量；

- $\theta_{min}$：$M_{r^*}$的最小稀疏度，为一个超平面，$0 \leq \theta_{min} \leq 1$;

  转移矩阵的稀疏度被定义为：
  
  $$
  \theta_r = 1-(1-\theta_{min})N_r/N_{r^*}
  $$
  
  投影向量：
  
  $$
  {\bf h}_p = {\bf M}_r(\theta_r){\bf h} \\
  {\bf t}_p = {\bf M}_r(\theta_r){\bf t}
  $$
  
  其中：$M_r(\theta_r) \in {\Bbb R^{m*n} }$, ${\bf h,t} \in {\Bbb R^n}$, ${\Bbb h}_p, {\Bbb t}_p \in {\Bbb R^m}$.

#### TranSparse(separate)

- $M_r^h(\theta_r^h), M_r^t(\theta_r^t)$：头尾实体转移矩阵；

- $N_r^l (l=h,t)$：关系$r$下$l$对应实体的数量；

- $N_{r^*}^{l^*}$ : $N_r^l$ 中最大数量；

- $\theta_{min}$ : $M_{r^*}^{l^*}$ 的最小稀疏度，$0 \leq \theta_{min} \leq 1$;

  转移矩阵的稀疏度被定义为：

$$
\theta_r^l = 1-(1-\theta_{min})N_r^l/N_{r^*}^{l^*}   ; (l=h,t)
$$

​	投影向量：

$$
{\bf h}_p = {\bf M}_r^h(\theta_r^h){\bf h} \\
{\bf t}_p = {\bf M}_r^t(\theta_r^t){\bf t}
$$
​	其中：$M_r^h(\theta_r^h), M_r^t(\theta_r^t)\in \Bbb R^{m*n}$  .



以上两种模型的得分函数是：

$$
f_r({\bf h,t})= \|{\bf h_p+ r -t_p}\|_{l_{1/2} }^2
$$
​	其中 ${\bf r} \in {\Bbb R^m}$。



## 训练对象

用 $(h_i,r_i,t_i) (i=1,2,...)$ 表示第 $i$ 个三元组，标签 $y_i$ 代表了三元组的正负情况， 正样本和负样本分别表示为 $\Delta =\{(h_i, r_i,t_i) | y_i=1\}$ 和  $\Delta' =\{(h_i, r_i,t_i) | y_i=0\}$, 知识图谱仅编码正样本。因此将负样本集合构建为 
$$
\Delta' = \{(h_i',r_i,t_i)|h_i' \neq h_i \wedge y_i = 1\} \cup \{(h_i,r_i,t_i')|t_i' \neq t_i \wedge y_i = 1 \}
$$

### 损失函数

$$
L =\sum_{(h,r,t)\in\Delta} \sum_{(h',r,t')\in\Delta'} [\gamma+f_r({\bf h,t})-f_r(\bf h',t')]_+
$$

​	约束限制： $\|{\bf h}\|_2 \leq 1, \|{\bf r}\|_2 \leq 1, \|{\bf t}\|_2 \leq 1, \|{\bf h}_p\|_2 \leq 1, \|{\bf t}_p\|_2 \leq 1$

 

## 算法实现

对于转移矩阵 ${\bf M(\theta)} \in {\Bbb R}^{n \times n}$ , 有 $nz=[\theta \times n \times n]$ 个非零元素 


