---
layout: article
title: 聚类：确定K-means最优簇数量K & 评价指标
date: 2021-02-07 00:12:00 +0800
tags: [Algorithm, Cluster, Tutorials]
categories: blog
pageview: true
key: determined-the-optimal-num-of-clusters-for-kmeans
---



> 参数：
>
> - [Elbow Method for optimal value of k in KMeans](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)
>
> - [Tutorial: How to determine the optimal number of clusters for k-means clustering](https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f)
>
> - [Selecting the number of clusters with silhouette analysis on KMeans clustering](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)



## 聚类结果评价指标

### Silhouette（轮廓系数）

轮廓系数（Silhouette Coefficient）结合了聚类的凝聚度（Cohesion）和分离度（Separation），用于评估聚类的效果。取值范围为[-1,1]，值越大，表示聚类效果越好。计算过程如下：

- $a_i$：某个样本$x_i$与其所在簇内其他样本的平均距离（**样本$x_i$的簇内不相似度**）。$a_i$越小，说明样本$a_i$越应该被聚类到该簇。簇$C$中所有样本的$a_i$均值为簇$C$的簇不相似度。
- $b_i$：某个样本$x_i$与其他簇样本的最小平均距离（**样本$x_i$的簇间不相似度**）。假设有$k$个簇，样本$x_i$对簇$C_j$样本的平均距离为$d_{iC_j}=\frac{1}{\mid C_j \mid} \sum_{x_t\in C_j} d_{it}$，则$b_i=\min\{\{d_{iC_j}\mid i\neq i\} \}$。

针对某个样本的轮廓系数$s_i$为:
$$
s_i=\frac{b_i-a_i}{\max (a_i, b_i)}
$$
聚类的总体轮廓系数$SC$为：
$$
\mathrm{SC}=\frac{1}{N} \sum_{i=1}^{N} s_i
$$
判断
$$
s_i=\left\{\begin{array}{cc}
1-\frac{a_i}{b_i}, & a_i<b_i \\
0, & a_i=b_i \\
\frac{b_i}{a_i}-1, & a_i>b_i
\end{array}\right.
$$

- $s_i$接近1，则说明样本$x_i$聚类合理；
- $s_i$接近-1，则说明样本$x_i$更应该分类到另外的簇；
- 若$s_i$近似为0，则说明样本$x_i$在两个簇的边界上。

 



### NMI（标准化互信息）



### ARI



### Modularity

[-0.5,1]