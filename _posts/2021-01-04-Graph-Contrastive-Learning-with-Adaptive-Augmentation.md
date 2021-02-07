---
layout: article
title: 图对比学习：Graph Contrastive Learning with Adaptive Augmentation
date: 2021-01-04 00:10:00 +0800
tags: [Graph, Data Augmentation, Graph Contrastive Learning]
categories: blog
pageview: true
key: Graph-Contrastive-Learning-with-adaptive-Augmentations
---

------

- Paper: **Graph Contrastive Learning with Adaptive Augmentations**
- Code: 







本质上来说，对比学习通过最大化不同视角间的一致性来学习表征，学习到的表征对数据增强方案引入的扰动具有鲁棒性（不变性）。也就是说，不同数据增强方案引入不同的扰动，生成不同视角的样本，对比学习能够通过最大化不同视角间的一致性，学习到不变性的特征表示。