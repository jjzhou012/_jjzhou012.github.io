---
layout: article
title: 图的对抗攻击： Adversarial Attack on Graph Structured Data
date: 2019-12-28 00:18:00 +0800
tags: [Adversarial attack, Graph]
categories: blog
pageview: true
key: Adversarial-Attack-on-Graph-2
---



------
论文链接：https://arxiv.org/pdf/1806.02371.pdf

github链接：https://github.com/Hanjun-Dai/graph_adversarial_attack



## Introduction

本文研究了一些列GNN模型的的对抗攻击问题，主要任务为图分类（graph classification）和节点分类（node classification）问题。





在论文中考虑了几种不同的对抗攻击设置。当目标分类器中有更多的信息可以利用时，提出了一种基于梯度的方法和一种基于遗传算法的方法。这里主要关注以下三个设置:

- 白盒攻击 (WBA)：在这种情况下，攻击者可以访问目标分类器的任何信息，包括预测结果、梯度信息等。

- 实际黑盒攻击(PBA)：这种情况下，只有目标分类器的预测结果是可以利用的。当预测置信度可以利用时，将攻击表示为PBA-C；当只有离散的预测标签可以利用时，将攻击表示为PBA-D。

- 受限黑盒攻击（RBA）：这个设置比PBA更严格。在这种情况下，我们只能对一些样本进行黑盒查询，对其他样本进行对抗性修改。

考虑攻击者能够从目标分类器中获得的信息量，可以对上述攻击方式进行排序 WBA > PBA-C > PBA-D > RBA 。同时本文主要关注无目标攻击。



## Background

考虑到图分类以图为单位，一个包含$$
|\mathcal{G}|=N
$$个图的集合表示为$$
\mathcal{G}=\left\{G_{i}\right\}_{i=1}^{N}
$$，每个图$G_i=(V_i,E_i)$表示为节点$$
V_{i}=\left\{v_{j}^{(i)}\right\}_{j=1}^{\left|V_{i}\right|}
$$和边$$
E_{i}=\left\{\mathbf{e}_{j}^{(i)}\right\}_{j=1}^{\left|E_{i}\right|}
$$的集合。边表示为$$
\mathbf{e}_{j}^{(i)}=\left(\mathbf{e}_{j, 1}^{(i)}, \mathbf{e}_{j, 2}^{(i)}\right) \in V_{i} \times V_{i}
$$。

论文只考虑无向边。节点的特征表示为$$
x\left(v_{j}^{(i)}\right) \in \mathbb{R}^{D_{n o d e}}
$$，边的特征表示为$$
w\left(\mathbf{e}_{j}^{(i)}\right)=w\left(\mathbf{e}_{j, 1}^{(i)}, \mathbf{e}_{j, 2}^{(i)}\right) \in \mathbb{R}^{D_{e d g e}}
$$。

这篇论文考虑两个不同的监督学习任务：

- 归纳图分类