---
layout: article
title: 图数据增强：NodeAug：Semi-Supervised Node Classification with Data Augmentation
date: 2021-03-31 00:11:00 +0800
tags: [GNN, Graph, Data Augmentation, Node Classification]
categories: blog
pageview: true
key: NodeAug-Semi-Supervised Node Classification with Data Augmentation
---



- (KDD 2020) **NodeAug: Semi-Supervised Node Classification with Data Augmentation**
  - [[Paper]](https://www.kdd.org/kdd2020/accepted-papers/view/nodeaug-semi-supervised-node-classification-with-data-augmentation)
  - Task: Node Classification



## 一、Motivation

- Question

  <u>半监督节点分类中，未标注的节点对训练没有帮助。如何将未标注的节点合并到GCN中？</u>

- Intuition

  graph中，增删节点之间的边会影响对附近节点的标签的预测。

  所以希望：DA扰动数据，使得对input的特征有离散的变化，对输出的节点分类预测有较小的影响。

  也就是说，保证预测结果不受DA对于输入数据修改带来的影响。

## 二、Contribution

- 数据增强方法：

  一种新的基于数据增强的GCN一致性训练方法，NodeAug.

  NodeAug对每个节点独立进行数据扩充，即每次DA只针对一个节点，也只预测该节点。修改方式设计了三种，针对节点属性和图结构。

- Subgraph Mini-Batch Training：

  利用了接收域(receptive field)提取目标节点所在的子图，将full-batch training方式转化为 subgraph mini-batch training方式，降低资源消耗。（原来做节点分类是将整个图，也就是整个邻接矩阵作为输入，现在是用子图做输入，相当于将节点分类转化为子图分类）

- 结合NodeAug，可以提升现有GNN模型的预测性能

## 三、Method

NodeAug module 有两部分

- ‘parallel universes’ DA scheme

  conduct DA for different nodes individually and separately；

- consistency training scheme

  最小化原始节点和增强节点的预测之间的分类分歧。

### 3.1 Consistency Training with DA on Graph

![image-20210409193252457](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409193252.png)

- 首先提取标注节点的子图，将节点分类问题转化为图分类；

![image-20210409212154821](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409212154.png)

- 一致性损失$\mathcal{L}_C$

  保证增强后的节点预测和原始节点预测尽可能一致，（图中所有节点，标注&未标注）；

- 监督分类损失$\mathcal{L}_S$

  保证标注节点的分类正确率；

### 3.2 Data Augmentation

- replacing attributes
- removing edges
- adding edges

第一种策略用与目标节点潜在相关的其他属性替换目标节点的属性；后面两种更改了GCN模型将其他节点的属性聚合到目标节点的路径。

#### 3.2.1 Replace Attributes Node

<img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409221242.png" alt="image-20210409221242195" style="zoom: 80%;" />

- 移除目标节点的属性：

  利用邻域中和目标节点相关的属性，替换目标节点中信息量较少的属性。

  - 属性的移除权重
  
    $$
    p_{a-r e m}=\min \left(p \frac{w_{\max }-w}{w_{\max }-w_{\text {ave }}}, 1\right)
    $$
    
    其中$$w_\text{max}$$,$$w_\text{ave}$$,$$w_\text{min}$$是目标节点的属性的最大、平均和最小权重。$$p$$是超参，控制概率大小。

    权重越大的属性被删除的概率越小。

  - 添加属性的采样概率
  
    $$
    p_{a-s a m}=\frac{s-s_{\min }}{\mid X\mid\left(s_{\text {ave }}-s_{\min }\right)}
    $$
    
    其中$$s_{\min }$$,$$s_\text{ave}$$,$$s_{\max}$$是用权重计算的分数，$$\mid X\mid$$是所有属性的集合。

    分数越大的属性被采样的概率越大。

#### 3.2.2 Remove Edges

移除外围的度小的节点

$$
p_{e-r e m}=\min \left(p l \frac{s_{e-\max }^{(l)}-s_{e}}{s_{e-\max }^{(l)}-s_{e-\text { ave }}^{(l)}}，1\right)
$$

其中$$s_{e-\max }^{(l)}$$,$$s_{e-\text{ave}}^{(l)}$$是外层$l$的边分数，$$s_{e}=\log \left(d_{\text {low}}\right)$$，$l$表示距离带来的影响。

度越大且越在外层的节点被删除的概率越小。

<img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409224253.png" alt="image-20210409224253390" style="zoom: 67%;" />

#### 3.2.3 Add Edges

$$
p_{e-a d d}=\min \left(\frac{p}{l} \frac{s_{n}-s_{n-\min }^{(l)}}{s_{n-\text { ave }}^{(l)}-s_{n-\min }^{(l)}}, 1\right)
$$

其中$$s_{n-\max }^{(l)}$$,$$s_{n-\text{ave}}^{(l)}$$是外层$l$的边分数, $$s_{n}=\log (d)$$。

度越大且越在外层的节点更容易被采样。

<img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409225409.png" alt="image-20210409225409236" style="zoom:80%;" />



## 四、results

- 对比实验

<img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409231723.png" alt="image-20210409231723157" style="zoom:67%;" />

- 不同数据划分

  <img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409231749.png" alt="image-20210409231749751" style="zoom:67%;" />

- subgraph mini-batch 

  ![image-20210409232155751](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409232155.png)

- 不同DA性能比较

  ![image-20210409232906512](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409232906.png)

- DA参数敏感性

  ![image-20210409232949666](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409232949.png)

- 训练曲线

  ![image-20210409233141272](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409233141.png)

- 节点表示可视化

  ![image-20210409233245981](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210409233246.png)