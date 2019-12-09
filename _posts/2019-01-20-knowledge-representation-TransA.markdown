---
layout: article
title: 知识图谱：知识表示之TransA模型
date: 2019-01-20 00:05:00 +0800
tag: knowledge representation
categories: blog
pageview: true
---


# TransA

论文地址: [https://arxiv.org/pdf/1509.05490.pdf](https://arxiv.org/pdf/1509.05490.pdf)

## 模型改进

### 缺陷

![kXcy01.png](https://s2.ax1x.com/2019/03/05/kXcy01.png)

TransE模型本质上是一种欧式距离的计算，对应一个等势超球面。

>  上图中蓝色部分为正例，红色部分为负例，TransE模型错误划分7个点；利用本文提出的基于马氏距离的TransA模型，其PCA降维图对应一个椭圆，该模型只错误划分三个点。

- 目前基于转移的方法构造了等式超球面，不容易划分匹配与不匹配的尾部实体；而且等式超球面形状固定、不够灵活；

  > 疑问：“等式超球面在等式面上权重处处相等，且等势面广，容易将不匹配的实体包含进来？？？

- 权重的问题：损失函数过于简单，向量的每一维度等价考虑，无法突出维度的重要性差异；

![kXc6Tx.png](https://s2.ax1x.com/2019/03/05/kXc6Tx.png)

> 上图所示，对于关系 has-part 而言，TransE模型根据欧式距离计算生成了像 ”Room-has-Goniff“这样的三元组。而正确的结果是”Room-has-Wall“。
>
> 对x,y轴进行分解，发现Room在x轴上距离Wall更近，因此可以认为该图在x轴维度上更重要。TransA模型通过引入加权矩阵，对每一个维度赋予不同权重。
>
> 轴分量损失： $loss_x = (h_x+r_x-t_x)$ ,  $loss_y=(h_y+r_y-t_y)$



## Adaptive Metric Approach

- TransA利用椭圆表面，而不是球面，这样可以更好的表示由复杂关系引起的复杂的嵌入拓扑；
- 根据自适应度量方法，TransA可以自动从数据中学习权重，加权变换特征维度；

得分函数：


$$
f_r(h,t)=(|{\bf h+r-t}|)^T {\bf W}_r(|{\bf h+r-t}|)
$$

- $\mid {\bf h+r-t} \mid  \doteq (\mid h_1+r_1-t_1 \mid,\mid h_1+r_1-t_1 \mid,...,\mid h_n+r_n-t_n \mid)$
- ${\bf w}_r$ 是与自适应度相关的对称非负权重矩阵；
- 采用绝对值运算能很好的定义得分函数，保证${\bf w}_r$是非负的；

将得分函数扩展为一个诱导范数:

$$
N_r({\bf e})=\sqrt{f_r(h,t)}
$$

- ${\bf e} \doteq {\bf h+r-t}$  

- $N_r$ 是非负的，单位的，绝对齐次的。

  $$
  N_r({\bf e}_1+{\bf e}_2)=\sqrt{|{\bf e_1}+{\bf e_2}|^T{\bf W_r} |{\bf e_1}+{\bf e_2} |} \leq  \sqrt{|{\bf e_1}|^T {\bf W_r} |{\bf e_1}|} + \sqrt{|{\bf e_2}|^T {\bf W_r} |{\bf e_2}|} = N_r({\bf e}_1) + N_r({\bf e}_2)
  $$




$$
{\bf W_r}={\bf L_r}^T{\bf D_r L_r}
$$

$$
f_r=({\bf L_r}|{\bf h+r-t}|)^T {\bf D_r} ({\bf L_r}|{\bf h+r-t}|)
$$

- ${\bf D_r}$ : 对角矩阵$diag (w_1, w_2,..., w_n)$，对角元素代表向量每一维度$i$以权重$w_i$嵌入 ;


#### 等势面

其他基于平移的方法：

- 欧式距离定义等势面

  $$
  \|({\bf t-h-r})\|_2^2={\cal C}
  $$






TransA方法：

- [马氏距离](https://www.cnblogs.com/likai198981/p/3167928.html)定义等势面

  $$
  |{\bf t-h-r}|^T {\bf W_r} |{\bf t-h-r}|= {\cal C}
  $$






> 马氏距离利用协方差，有效计算样本各特性之间的联系，与尺度无关。
>
> 从而可以看出，TransA利用马氏距离，可以更好的应对1-N关系，由于矩阵对称，反过来对于N-1关系也有效；N-N关系可以看成多个1-N关系；因此TransA对于复杂关系的处理很有效。



## 训练过程

### 损失函数

$$
\left( \sum_{e \in E}\|{\bf e}\|_2^2 + \sum_{r \in R}\|{\bf r}\|_2^2 \right)
$$

$$
{\cal L}=\sum_{(h,r,t) \in \Delta} \sum_{(h',r',t') \in \Delta'} [\gamma + f_r(h, t)-f_{r'}(h', t')]_+ + C \left (\sum_{e \in E}\|{\bf e}\|_2^2 + \sum_{r \in R}\|{\bf r}\|_2^2 \right) + \lambda \left (\sum_{r \in R}\|{\bf W_r}\|_F^2 \right)
$$

- $\lambda$ : 正则化自适应权重矩阵
- $C$ : 控制缩放比例

- ${\bf [W_r]}_{ij} \geq 0$ :  在每轮的训练中，${\bf W_r}$ 可以通过将推导值设为零直接计算出来。为保证${\bf W_r}$的非负，将${\bf W_r}$的所有负项都设为零。

  $$
  \bf W_r = -\sum_{(h,r,t)\in \Delta} \left(|{\bf h+r-t}| |{\bf h+r-t}|^T \right) + \sum_{(h',r',t')\in \Delta'} \left(|{\bf h'+r'-t'}| |{\bf h'+r'-t'}|^T \right)
  $$

 

