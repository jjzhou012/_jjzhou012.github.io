---
layout: article
title: 图对比学习：Graph Contrastive Learning with Augmentations
date: 2020-12-22 00:10:00 +0800
tags: [Graph, Data Augmentation, Graph Contrastive Learning]
categories: blog
pageview: true
key: Graph-Contrastive-Learning-with-Augmentations
---

------

- Paper: **Graph Contrastive Learning with Augmentations** (NeurIPS 2020) 
- Code: [https://github.com/Shen-Lab/GraphCL](https://github.com/Shen-Lab/GraphCL)



## Data Augmentation

![image-20201222205415864](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201222205416.png)

`Data augmentation aims at creating novel and realistically rational data through applying certain transformation without affecting the semantics label. `

在图上应用DA存在的问题：

- 图结构数据是从不同领域中抽象出来的，不像图像数据一样有普适的DA方法；

提出了四种DA方法：

- 节点删除（Node dropping）: 随机从图中删除一部分节点来扰动图的完整性，每个节点的删除概率遵循默认的独立同均匀分布(或任何其他分布)，约束的先验为删除的节点不影响整个图的语义信息。
- 边扰动（Edge perturbation）: 随机从图中增删一部分边来扰动图的邻接信息，每条边的增删概率遵循默认的独立同均匀分布(或任何其他分布)，用来反映图的语义对边连接模式的改变具有一定的鲁棒性。
- 属性掩盖（Attribute masking）：随机删除部分节点的属性信息，迫使模型根据节点的上下文信息（邻接信息）来重构被掩盖掉的节点属性。基于的假设是缺少部分节点属性不会对模型的预测造成太大影响。
- 子图（Subgraph）：使用随机游走从图中采样子图。基本的假设是图的语义信息能够在图的局部结构中保留。



## Graph Contrastive Learning
![image-20201222215513835](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201222215513.png)

提出了一个图对比框架GraphCL。在图对比学习中，通过潜在空间中的对比损失来最大化同一图的两个增强视图之间的一致性来执行预训练。整个框架由4部分组成：

- Graph data augmentation. 

  对给定图$\mathcal{G}$使用数据增强生成两个视图作为正样本对，其中$$
  \hat{\mathcal{G}}_{i} \sim q_{i}(\cdot \mid \mathcal{G}), \hat{\mathcal{G}}_{j} \sim q_{j}(\cdot \mid \mathcal{G})$$。
  
- GNN-based encoder.

  基于GNN的自编码器$$f(\cdot)$$用于提取图级别的向量表示$$\boldsymbol{h}_{i}, \boldsymbol{h}_{j}$$。
  
- Projection head.

  名为投影头的非线性变换$$g(\cdot)$$将增广表示映射到另一个计算对比损失的潜在空间。

- Contrastive loss function.

  对比损失函数$$\mathcal{L}(\cdot)$$用于强制最大化正样本对$$z_i,z_j$$之间的一致性。



## 讨论：DA在GCL中的作用

![image-20201223144505608](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223144505.png)

在生化分子和社交网络数据集上进行了实验。

### 结论一：从DA的作用出发

![image-20201223144922056](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223144922.png)

- **数据扩充对图对比学习至关重要（引入图数据扩充很容易带来性能提升）**

  从右上角可以看出，在不使用DA的情况下，对比学习是没有帮助的，相比从零开始学习，精度有所降低。

  从最上面一行或者最右边一列可以看出，加入合适的DA可以提升性能。

  不加入DA的情况下，graphCL将两个原始样本作为负样本对（正样本对损失为0），使得特征空间中所有图表示相互远离。当应用适当的DA时，数据分布的相应先验就会被加入，强制模型通过最大化图与其增广之间的一致性来学习对期望扰动具有鲁棒性的表示。

- **组合不同的扩充方式会带来更多的性能收益（对角线性能不佳）**

  相比对角线上的结果（组合相同DA），对角线之外的结果一般表现更好，说明组合不同的DA可以带来更多的性能收益。

  组合不同的DA避免了学习到的特征过度拟合，可以使特征更加一般化。

### 结论二：从DA的类型、范围、模式出发

- **对于不同类型的数据集来说，最优的DA组合也是不同的（dataset-specific）**

- **边扰动 有益于社交网络，但是不利于生化分子网络。**

  图中可以看出，边扰动在COLLAB、RDT-B和PROTEINS上效果不错，但是在NCI1上不利。具体问言，NCI1上的边扰动对应于化学键的移除或添加，这会极大地改变化合物的身份甚至有效性，更不用说影响其下游语义的性质了。相比之下，社交网络的语义对个体边扰动的容忍度更高。因此，对于化合物来说，边扰动违背了一个先验，这在概念上与领域知识不相容，在经验上对下游性能也没有帮助。

  ![image-20201223152433492](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223152433.png)

  图4AB中，展示不同边扰动强度对不同类型数据集的影响，在NCI1上，都是下降的；在COLLAB上，边扰动强度增大，性能提升。

- **属性掩盖 在更密集的图上有更好的表现。**

  对社交网络数据集而言，（identical+attMask）在COLLAB上有5.12%，在RDT-B上仅有0.17%；在生化分子网络中有同样的现象，PROTEINS上有1.07%，NCI1上只有-0.17%。

  图4CD中，掩盖较多或较少的属性对比较稀疏的RDT-B而言没什么帮助，掩盖较多的属性对较密集的COLLAB而言依然有效。

  进一步假设，掩蔽模式也很重要，再更密集的图中高程度的掩蔽更多的hub节点也是有效的，因为根据消息传递机制，GNNs无法重建孤立节点缺失的信息。为了验证这一假设，我们进行了一个实验，在更密集的图PROTEINS和COLLAB下，以更高的概率掩盖度更大的节点。使用一个掩盖分布$$\text{deg}_n^\alpha$$，其中$$\text{deg}_n$$是节点的度值，$\alpha$是控制因子。

  **从图5C和D可以看出，对于非常密集的COLLAB，当掩蔽度越大的节点时，性能有更明显的提升。**

  **在密集的图中可以通过更丰富的邻接信息来重构被掩盖的属性信息。**

  ![image-20201223153958471](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223153958.png)

- **节点删除和子图，尤其是后者，在研究的数据集中似乎是普遍有益的。**

  对于节点的删除，强调缺失某些顶点(例如化合物中的氢原子或社交网络中的边缘用户)不会改变语义信息的先验性，直观地符合认知。也就是说：**节点删除在丢失某些节点不会改变图的语义信息的先验条件下，不会妨碍我们的认知。**

  对于子图，先前的研究表明，**加强局部（提取的子图)和全局信息一致性有助于表征学习**，这解释了观察到的结果。即使对于NCI1中的化合物，子图也可以表示对下游语义很重要的结构和功能“motif”。

  在图5B中，在密集的COLLAB中，删除度越大的节点，性能有更明显的提升；

  在图5A中，在不是特别密集PROTEINS中，改变节点的删除分布对结果的影响不是特别大。

### 结论三：从任务的难易出发

- **过于简单的对比任务对性能的表现无济于事。**

  任务的难易程度：

  - 删除/掩盖： 度大的节点 > 度小的节点;  掩盖率大 > 掩盖率小

  - 边扰动：扰动比率大 > 扰动比率小

    ![image-20201223193603721](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223193603.png)

  - 子图： subgraph-DFS >  subgraph > subgraph-WFS

    ![image-20201223193728369](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223193728.png)

    *任务越难，性能越好。*



## 对比实验

- 半监督学习

  ![image-20201223200047554](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223200047.png)

- 无监督表示学习

  ![image-20201223204637864](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223204637.png)

- 迁移学习

  ![image-20201223205148531](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223205148.png)

- 对抗攻击

  ![image-20201223205456832](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20201223205456.png)

