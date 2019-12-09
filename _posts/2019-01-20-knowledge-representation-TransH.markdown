---
layout: article
title: 知识图谱：知识表示之TransH模型
date: 2019-01-20 00:01:10 +0800
tag: knowledge representation
categories: blog
pageview: true
---

# TransH

论文地址：[https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546)

## 模型改进

![kX6zo6.png](https://s2.ax1x.com/2019/03/05/kX6zo6.png)

将特定关系的转移向量 $d_r$ 放置于特定关系的超平面 $w_r$ ,而不是映射到相同的实体嵌入空间；  

对于三元组 $(h,r,t)$，嵌入向量 ${\bf h}$ 和 ${\bf t}$ 投影到超平面 ${\bf {w_r}}$ , 投影被表示为 ${\bf h_{\bot}}$ 和 ${\bf t_{\bot}}$ 。投影 ${h_{\bot}}$ 和 ${t_{\bot}}$ 能被转移向量 ${\bf d_r}$ 连接， 当三元组为正样本时有更低的错误率，为负样本时错误率上升；   

> TransH将一个关系$r$表示成两个向量：超平面法向量${\bf W}_r$和超平面内的关系向量${\bf d_r}$表示；
>
> 该关系连接的不同实体对在超平面上的投影只对应一个向量表示： ${\bf h_{\bot}}$ 和 ${\bf t_{\bot}}$，
>
> 也就是，唯一的${\bf W}_r$和${\bf d_r}$ 确定了该关系在空间中的唯一超平面；


- 定义一个得分函数,衡量三元组的合理性：

  
$$
\|{\bf h_{\bot} } + {\bf d_r} - {\bf t_{\bot} }\|_2^2
$$

- 超平面法向量限制为 $\|{\bf w}_r \|_2 = 1$, 超平面投影的表示为

  

$$
{\bf h_{\bot} } = {\bf h} - {\bf w}_r^T {\bf h}{\bf w}_r
$$

$$
{\bf t_{\bot} } = {\bf t} - {\bf w}_r^T {\bf h}{\bf w}_r
$$

> 推导： 嵌入向量在超平面法向量上的投影长度，再乘上法向量方向向量：  
$d=\frac{ | {\bf w}_r^T h| }{| |{\bf w}| |} \cdot \frac{ {\bf w}_r}{| |{\bf w}| |}=\frac{ {\bf w}_r^T {\bf h}{\bf w}_r}{| |{\bf w}^2| |}={\bf w}_r^T {\bf h}{\bf w}_r$

最终得分函数：

$$
f_r({\bf h},{\bf t}) = | |({\bf h} - {\bf w}_r^T {\bf h} {\bf w}_r) + {\bf d}_r -({\bf t} - {\bf w}_r^T {\bf h} {\bf w}_r)| |_2^2
$$

模型参数：
- 所有实体的嵌入向量：  ![image-20191209013227380](images/image-20191209013227380.png)  ; 
- 所有关系超平面和转移向量  ![image-20191209013240687](images/image-20191209013240687.png)  ;

在TransH中，通过引入投射到特定关系超平面上的机制，使实体在不同的关系/三元组中扮演不同的角色。

## 训练过程

### 损失函数
$$
{\cal L}=\sum_{(h,r,t)\in \Delta} \sum_{(h',r',t')\in \Delta'_{(h,r, t)} } [\gamma + f_r({\bf h}, {\bf t})-f_{r'}({\bf h'}, {\bf t'}) ]_+
$$

- $\Delta$: 正确三元组集合
- $\Delta'$: 错误三元组集合
- $\gamma$: margin 距离超参数，表示正负样本之间的距离
- $[x]_+$: $max(0,x)$



训练的限制条件：

- $\forall e\in E$ ， $\|{\bf e}\|_2 \leq 1$ ,   # 控制实体嵌入大小
- $\forall r \in R$ ，$ \frac{\mid \bf{ w_t^T } \bf{ d_r } \mid}{ {\| \bf{ d_r }  \|}_2 } \leq \epsilon$ , # 正交，控制转移向量落在超平面
- $\forall r \in R$ ， $\|{\bf w}_r\|_2=1$ , # 单位法向量



软约束：

$$
{\cal L}=\sum_{(h,r,t)\in \Delta} \sum_{(h',r',t')\in \Delta'_{(h,r, t)} } [\gamma + f_r({\bf h}, {\bf t})-f_{r'}({\bf h'}, {\bf t'}) ]_+ + C\left\{\sum_{e\in E}[\|{\bf e}\|_2^2 -1]_+ + \sum_{r \in R}\left[\frac{({\bf w}_t^T {\bf d}_r)^2}{\|{\bf d}_r\|_2^2} - \epsilon ^2\right]_+ \right\}
$$

- C是软约束的超参数权重
- 注意到约束3不在loss公式里，在访问每一个mini-batch时将每一个${\bf w}_r$投影到单位  [$L_2- \bf {ball}$](https://blog.csdn.net/zouxy09/article/details/24971995) 
- 随机梯度下降，正样本的集合被随机遍历多次，当访问一个正样本时，一个负样本被随机构造。



### 负样本构造

相对于TransE模型的随机采样生成负样本，TransH模型的创新点：
**赋予头尾实体采样概率**   

- 处理1-N关系时，赋予头部实体更高的采样概率；
- 处理N-1关系时，赋予尾部实体更高的采样概率；  

在所有关系r的三元组中，首先获取两个数据：
- $tph$: 每一个头部实体对应的平均尾部实体;
- $hpt$: 每一个尾部实体对应的平均头部实体;

针对这两个数据进行映射的分类：

$$
\left \{ \begin{array}{rcl}
1-1    & \mbox{for} & tph_r<1.5,hpt_r <1.5 \\ 
N-N    & \mbox{for} & tph_r \geq 1.5,hpt_r \geq 1.5 \\
1-N    & \mbox{for} & tph_r \geq 1.5,hpt_r < 1.5 \\
N-1    & \mbox{for} & tph_r < 1.5,hpt_r \geq 1.5 \\
\end{array}\right.
$$

定义一个伯努利分布进行采样：
- $\frac{tph}{tph+hpt}$: 采样头部实体的概率;
- $\frac{hpt}{tph+hpt}$: 采样尾部实体的概率;



## 实验部分(未完待补充)

在三个任务上研究评估模型：从不同视角和应用层面，评估对陌生三元组的预测精度
- link prediction
- triplets classification
- relational fact extraction

实验数据使用：
![](http://p5bxip6n0.bkt.clouddn.com/18-11-1/13704280.jpg)

#### link prediction
**任务**：三元组的头部或者尾部实体缺失，给定 $(h,r)$ 预测 $t$ 或者给定 $(r,t)$ 预测 $h$ , 返回预测的候选排名；   
**评估协议**： 对于给定的三元组，用每个实体 $e$ 取代尾部实体 $t$ ，并计算损坏样本的差异性分数，升序排列获取原始三元组的排名；  
**过滤**： 生成的损坏三元组可能原本就存在于知识图谱中，需要删去；  
**报告指标**：  
- Mean:平均排名:
- $Hits@10$: 排名不超过10的比例

#### triplets classification
**任务**: 对给定三元组进行二分类判断正负样本；
#### relational fact extraction

(待补充)