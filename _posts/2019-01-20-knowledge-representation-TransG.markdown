---
layout: article
title: 知识图谱：知识表示之TransG模型
date: 2019-01-20 00:07:00 +0800
tag: knowledge representation
categories: blog
pageview: true
---



# TransG

论文地址： [https://arxiv.org/pdf/1509.05488.pdf](https://arxiv.org/pdf/1509.05488.pdf)



## 模型改进

针对一种关系存在的多语义问题：

![kXcqtf.png](https://s2.ax1x.com/2019/03/05/kXcqtf.png)

由上图，嵌入向量的可视化结果所示表明，一个特定的关系存在不同的簇，不同的簇表示不同的潜语义。

> 例如，关系HasPart至少有两个潜在的语义:与合成相关的as(Table、HasPart、Leg)和与位置相关的as (Atlantics、HasPart、NewYorkBay)。
>
> 再例如，Freebase中，(Jon Snow, birth place, Winter Fall)和(George R. R. Martin, birth place, U.S.)分别映射到模式 ：/fictional_universe/fictional_character/place of birth  和 /people/person/place of birth，表明出生地有不同的含义。



多关系语义的可视化

![kXcj1g.png](https://s2.ax1x.com/2019/03/05/kXcj1g.png)

> 点是正确的三元组，属于HasPart关系，而圆是不正确的。
>
> 点坐标是头部和尾部实体之间的差向量，应该靠近中心。
>
> (a)正确的三元组很难与错误的三元组区分开来；
>
> (b)通过应用多个语义分量，TransG 可以区分正确的三元组和错误的三元组

### 创新点

- 提出了一种关系会因为实体对的差异存在多重语义的问题，相当于对关系进行了细化；

- 提出了一种新的**贝叶斯非参数无限混合嵌入模型TransG**。该模型可以自动发现关系的语义集群，并利用多个关系分量的混合来转换实体对;

### 方法

模型生成过程：

- 对一个实体 $e \in E$ :

  (a)  从标准正态分布中提取每个嵌入的平均向量作为先验： ${\bf u_e} \sim  {\cal N}(0,1)$.

- 对于一个三元组 $(h,r,t) {\in \Delta}$ ： 

  (a)	通过$CRP$过程对一种关系构造语义成分， $\pi _{r,m} \sim CRP(\beta)$；

  (b)	用正态分布构造头实体嵌入向量，${\bf h} \sim  {\cal N}({\bf u_h}, \sigma_h^2 {\bf E})$.

  (c)	用正态分布构造尾实体嵌入向量，${\bf t} \sim  {\cal N}({\bf u_t}, \sigma_t^2 {\bf E})$.

  (d)	对该语义构造一个关系嵌入向量，
${\bf u_{r,m} }= {\bf {t-h} } \sim \cal N ({\bf {u_t-u_h, (\sigma_h^2+\sigma_t^2)E} })$ .

其中：

- ${\bf u_h}$ 和 ${\bf u_t}$ 代表头尾实体的平均嵌入向量，
-  $\sigma _h$ 和 $\sigma _t$  代表对应实体分布的方差；
- ${\bf u_{r,m} }$ 代表关系 $r$ 的第 $m$ 个 成分转移向量

- [$CRP$](https://segmentfault.com/a/1190000010694630) 过程是一个 Dirichlet 过程，它能自动检测语义成分。

得分函数：

$$
{\Bbb P \{(h,r,t)\} }  \propto  \sum_{m=1}^{M_r} \pi_{r,m} \Bbb P({\bf u_{r,m} } | h,t)=\sum_{m=1}^{M_r} \pi_{r,m} e^{-\frac{\|{\bf u_h+u_r}, \bf m-u_t\|_2^2}{\sigma_h^2+\sigma_t^2} }
$$

- $\pi_{r,m}$ : 混合系数，代表第$i$个语义成分的权重；
- $M_r$ : 关系$r$的语义成分数量， 通过$CRP$自动的从数据中学习得到；

TransG利用了特定关系的关系分量向量的组合。每个成分代表一个特定的潜在含义。通过这种方法，TransG可以区分多个关系语义。值得注意的是，CRP可以生成多个语义分量，并从数据中自适应地学习关系语义分量数$M_r$。



### 几何解释

TransG 推广：

$$
m_{h,r,t}^* = {\arg\min}_{m=1,...,M_r} \left(  \pi_{r,m} e^{-\frac{\|{\bf u_h+u_r,m-u_t}\|_2^2}{\sigma_h^2+\sigma_t^2} }     \right)
$$

$$
{\bf h} + \bf u_{ {r,m}_{(\bf h,r,t)}^*} \approx \bf t
$$

- $m_{h,r,t}^*$ : 主分量，虽然所有分量都对模型有贡献，但是由于指数效应，主分量贡献最大；

- 给定一个三元组，TransG计算出主分量，然后用主转换向量将头实体转化为尾实体；

- 对于大多数三元组而言，应该只有一个分量有明显的非零值  $ \left(  \pi_{r,m} e^{-\frac{\|{\bf u_h+u_r, m-u_t}\|_2^2}{\sigma_h^2+\sigma_t^2} }     \right)$ ， 而其他分量由于指数衰减应该足够小，所有TransG中所有分量都有贡献，但主分量贡献最少；

- 该性质有效的减少了来自其他语义分量的噪声，更好的描述了多种语义关系；

- 在TransG中， ${\bf t-h}$ 近乎只有一个转换向量  $\bf u_{ {r,m}_{(\bf h,r,t)}^*}$ ，

  当 $m \neq m_{h,r,t}^*$ 时，$\frac{\|{\bf{u_h+u_r, m-u_t} }\|_2^2}{ \sigma_h^2+ \sigma_t^2}$ 很大，经过指数操作后值很小，故其他分量可以忽略；



## 训练算法

训练中运用了最大数据似然原理。对于无参数部分，$\pi _{r,m}$ 通过 Gibbs 采样 从 CRP 生成。从三元组采样新的分量可利用以下概率：

$$
\Bbb P (m_{r.new}) = \frac{\beta e^{-\frac{\|{\bf h-t}\|_2^2}{\sigma_h^2 + \sigma_t^2 +2} } }{\beta e^{-\frac{\|{\bf h-t}\|_2^2}{\sigma_h^2 + \sigma_t^2 +2} }+ \Bbb P\{(h,r,t)\} }
$$

- $\Bbb P\{(h,r,t)\}$ : 当前后验概率

### 目标函数

其他部分，为了更好地区分正样本和负样本，将正样本和负样本的可能性比最大化。值得注意的是，嵌入向量是由(Glorot and Bengio, 2010)初始化的。将所有其他约束组合在一起，得到最终目标函数，如下所示:

$$
\min_{\bf u_h,u_r,m,u_t}  {\cal L}    \\
{\cal L} = -\sum_{(h,r,t)\in \Delta} ln \left(\sum_{m=1}^{M_r} \pi_{r,m} e^{-\frac{\|{\bf u_h+u_r, m-u_t}\|_2^2}{\sigma_h^2+\sigma_t^2} } \right)  \\ +  \sum_{(h',r',t')\in \Delta'} ln \left(\sum_{m=1}^{M_r} \pi_{r',m} e^{-\frac{\|{\bf u_{h'}+u_{r'} ,m-u_{t'}}\|_2^2}{\sigma_{h'}^2+\sigma_{t'}^2} } \right) + C\left(\sum_{r\in R} \sum_{m=1}^{M_r} \|{\bf u_{r,m} }\|_2^2 + \sum_{e\in E}\|{\bf u_e}\|_2^2 \right)
$$

> 此外还应用了一个技巧来控制训练过程中的参数更新过程。对于那些非常不可能的三元组，将跳过更新过程。因此，引入了与TransE 相似的条件: 训练算法只在满足以下条件时更新嵌入向量:
> 
> $$
> \frac{\Bbb P\{(h,r,t)\} }{\Bbb P\{(h',r',t')\} }= \frac{\sum_{m=1}^{M_r} \pi_{r,m} e^{-\frac{\|{\bf u_h+u_r, m-u_t}\|_2^2}{\sigma_h^2+\sigma_t^2} } }{\sum_{m=1}^{M_{r'} } \pi_{r',m} e^{-\frac{\|{\bf u_{h'}+u_{r'} , m-u_{t'}}\|_2^2}{\sigma_{h'}^2+\sigma_{t'}^2} } }  \leq M_r e^{\gamma}
> $$
>
> - $\gamma$ : 控制更新条件

