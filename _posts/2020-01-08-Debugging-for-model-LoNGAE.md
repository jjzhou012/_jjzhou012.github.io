---
layout: article
title: Debugging for model-LoNGAE
date: 2020-01-08 00:21:00 +0800
tags: [Summary, Model, Debugging]
categories: blog
pageview: true
key: Debugging-for-model-LoNGAE
---



------



## 基本信息：

目标模型：LoNGAE

出处：[Learning to Make Predictions on Graphs with Autoencoders]( https://arxiv.org/pdf/1811.02798.pdf)

需求：

- 环境迁移：
  - python 3.6
  - CUDA 9.0/9.1
- 复现论文实验
- 实现邮件数据集的链路预测和节点分类；
- 实现链路预测和节点分类的攻击；



## 环境依赖

- CUDA 9.0

- python 3.6
- tensorflow-gpu 1.12.0-gpu_py36he68c306_0
- keras 2.0.6 using TensorFlow GPU 1.12.0 backend

- networkx 2.4                   # 加速图邻接矩阵读取
- scipy 1.1.0
- numpy 1.16.5
- scikit-learn 0.22.1 



## 链路预测

### 1. 论文实验

#### 1.1. 数据集

```
protein, metabolic, conflict, powergrid, cora, citeseer, pubmed
```

![](http://ww1.sinaimg.cn/large/005NduT8ly1gaobk9fsltj30f309o3zp.jpg)



### 2. 实际数据集

| Dataset | Nodes | Edges | Ave. Degree | Pos/Neg | Node Feature | Node Classes | Label rate | AUC  |
| ------- | ----- | ----- | ----------- | ------- | ------------ | ------------ | ---------- | ---- |
| test 2  | 10381 |       | 3.14        |         |              |              |            |      |
|         |       |       |             |         |              |              |            |      |
|         |       |       |             |         |              |              |            |      |
|         |       |       |             |         |              |              |            |      |
|         |       |       |             |         |              |              |            |      |
|         |       |       |             |         |              |              |            |      |

AUC与划分比例

| Dataset | test_frac | AUC   | AP    |
| ------- | --------- | ----- | ----- |
| test 2  | 10%       | 0.960 | 0.349 |
| test 2  | 50%       | 0.918 | 0.270 |
| test 2  | 80%       | 0.780 | 0.095 |

> 疑问：
>
> - test set 来源