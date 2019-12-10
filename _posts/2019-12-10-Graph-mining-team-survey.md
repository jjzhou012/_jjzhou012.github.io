---
layout: article
title:  国内外 Graph mining 团队工作调研
date: 2019-12-10 00:08:00 +0800
tag: [Graph, Summary]
categories: blog
pageview: true
---



## 1. TUM Department of Informatics (慕尼黑工业大学)

相关链接：

- [Team Home](https://www.kdd.in.tum.de/en/home/)



### Head of Research Group

<center>
    <center><img align="left" width=160 height=220 src="https://ws1.sinaimg.cn/large/005NduT8ly1g9rlc6hei2j304q06c0u5.jpg"></center>
    <p style="font-size:22px"><b>Stephan Günnemann</b></p>
    <ul style="margin:0 0 0 200px">
        <p style="font-size:18px;margin:0px;" align="left"><b>Research Focus:</b></p>
    <li><p align="left">Machine Learning for Graphs/Networks, Graph Mining, Network Analysis</p></li>
    <li><p align="left">Robust and Adversarial Machine Learning</p></li>
    <li><p align="left">Probabilistic Models, (Deep) Generative Models</p></li>
    <li><p align="left">Analysis of Temporal and Dynamic Data</p></li>
    </ul>
</center>



### Team profile

Our group’s research centers around the development of *robust machine learning methods*, with major focus on mining and learning principles for *graphs and networks*.

主要研究包括：

- 图/网络数据挖掘 结合 机器学习
- 鲁棒性、对抗性机器学习
- 概率模型、生成模型
- 时序、动态数据分析

团队重要贡献:

- 针对 GNN 模型的对抗攻击 （节点分类）、针对图嵌入算法的对抗攻击
- 提出新的 GNN 模型用于图挖掘，特别是节点分类任务
- 

### Recent publication

#### 算法鲁棒性研究

对抗性攻击 & 鲁棒性：图学习算法鲁棒吗？如何去有效的攻击图算法？如何提升它们的鲁棒性？

- [**Adversarial Attacks**] [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://openreview.net/pdf?id=Bylnx209YX) [ICLR 2019]

  - **Overview:** 针对**节点分类**，提出了一种**基于属性图的训练时间对抗攻击算法**。使用**元梯度**来解决双层优化问题的基础上的**类中毒对抗攻击**。实验表明，元梯度方法创建的攻击（即使很微小的扰动）始终导致gcn模型的分类性能大幅下降，甚至转移到**无监督模型**。

    <center><img src="https://ws1.sinaimg.cn/large/005NduT8ly1g9rmj7537yj30pc07tgnx.jpg" width=700 height=180/></center>
  
- <div><img align="right" src="https://ws1.sinaimg.cn/large/005NduT8ly1g9rn2djo6tj30lx096zml.jpg" width=300 height=160/></div>
  [Adversarial Attacks] [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984.pdf) (Best Research Paper Award) [kdd 2018]

  - **Overview:** 提出了第一个关于（属性）图的对抗性攻击的研究，特别关注利用gcn进行节点分类的任务。针对节点特征和图结构，提出了测试阶段的直接攻击和影响者攻击，和训练阶段的中毒/致因攻击。通过保留重要的数据特征来确保扰动的隐蔽性；提出了一种增量计算的高效算法；攻击具有迁移性。

    

- <img align="right" src="https://ws1.sinaimg.cn/large/005NduT8ly1g9rnpn95cwj308j060jrr.jpg" width=300 height=190/>[Clustering] [Bayesian Robust Attributed Graph Clustering: Joint Learning of Partial Anomalies and Group Structure](https://pdfs.semanticscholar.org/2230/c5946b37ab891389a034396a4e2f1830cdfa.pdf) [AAAI 2018]  
	
  - **Overview:** 研究了鲁棒属性图聚类问题。实际数据中聚类结构常常由于异常或破环而变得模糊，考虑到：图包含两个视图（网络结构和属性），异常可能只影响一部分，即实例可能会在其中一个视图中损坏，但在另一个视图中完好。在这种情况下，仍然可以派生有意义的聚类集群。通过对度修正随机块模型和伯努利混合模型的推广，提出了一种属性图聚类的概率模型PAICAN，实验表明了PAICAN在部分和全部异常下的鲁棒性。
  
    

- <img align="right" src="https://ws1.sinaimg.cn/large/005NduT8ly1g9rod09tahj309404d3z5.jpg" width=300 height=160/>[Clustering] [Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings](http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p737.pdf) [kdd 2017]
  
  - **Overview:** 谱聚类是最主要的聚类方法之一，但是它对有噪声的输入数据非常敏感。提出了一种用于谱聚类的相似图的稀疏和潜在分解方法。该模型联合学习了谱嵌入和被破坏的数据，从而整体上提高了聚类性能。



- [Adversarial Attacks on Node Embeddings via Graph Poisoning](https://arxiv.org/pdf/1809.01093.pdf) [Arxiv]



#### 图神经网络架构

如何设计新的神经网络结构来捕捉图的基本特性？如何捕获关系依赖来提高学习的性能？

- [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/pdf/1810.05997.pdf) [ICLR 2019]
  - **Overview:** 提出GCN + personalized PageRank 的神经消息传播框架(PPNP)，用于节点的半监督分类。
- [Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/pdf/1811.05868.pdf) [NIPS 2018]
  - **Overview:** 证明了现有的GNN模型评价策略存在严重缺陷。评估表明，使用相同的训练/验证/测试分割相同的数据集，以及对训练过程进行重大更改(例如，early stopping)，会妨碍对不同架构的公平比较；考虑不同的数据分割会导致模型的排名显著不同；如果对所有模型的超参数和训练过程进行适当的调整，那么更简单的GNN体系结构能够胜过更复杂的结构。
- [Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings](https://pdfs.semanticscholar.org/2b76/b6e766547b3c6dbc2785a084ec3b72cb760d.pdf) [kdd 2017]
  - **Overview:** 提出的Graph2Gauss方法可以有效地学习大规模(带属性)图的通用节点嵌入，不同于大多数将节点表示为低维连续空间中的点向量的方法，Graph2Gauss将每个节点嵌入为高斯分布，从而允许捕获关于表示的不确定性。此外，提出了一个无监督的方法，处理归纳学习场景，适用于不同类型的图:普通/属性，有向/无向。

#### 图推理的生成模型

如何自动生成具有真实属性的图?我们能否从其他信息来源推断出图/网络？

- [GhostLink: Latent Network Inference for Influence-aware Recommendation](https://arxiv.org/abs/1905.05955) [WWW 2019]

- [NetGAN: Generating Graphs via Random Walks](https://arxiv.org/pdf/1803.00816.pdf) [ICML 2018]













## 5. **AMLab** in University of Amsterdam

相关链接：

- [Thomas Kipf’s blog](http://tkipf.github.io/)
- [AMLab](https://staff.fnwi.uva.nl/)

### Head of Research Group

<img align="left" src="https://scholar.googleusercontent.com/citations?view_op=view_photo&user=8200InoAAAAJ&citpid=2" width=160 height=110/><font size=5>**[Max Welling](https://staff.fnwi.uva.nl/m.welling/)**</font>

Professor Machine Learning

**Research Focus:** `Machine Learning`, `Artificial Intelligence`, `Statistics`

### Team profile

**AMLab:** The Amsterdam Machine Learning Lab (AMLab) conducts research in the area of **large scale modeling of complex data sources.** This includes the development of new methods for **deep learning, probabilistic graphical models, Bayesian modeling, approximate inference, causal inference, reinforcement learning and the application of all of the above to large scale data domains in science and industry.**

主要研究包括：

- 大规模复杂数据的建模
  - 深度学习、强化学习
  - 图模型、贝叶斯模型
  - 近似推理、因果推理

团队重要贡献:

- 提出 **gcn**: [Semi-supervised classification with graph convolutional networks](https://arxiv.org/abs/1609.02907) [ICLR 2017]

### Recent publication

- <img align="right" src="https://ws1.sinaimg.cn/large/005NduT8ly1g9rjo6iamrj30ie076ta2.jpg" width=350 height=110/>[Graph Convolutional Matrix Completion](https://arxiv.org/pdf/1706.02263.pdf) [kdd 2019]
  
  - **Overview:** 从图上链接预测的角度考虑了推荐系统的矩阵补全问题，提出了一种基于可微消息在二部交互图上传递的图自动编码器框架。编码器包含一个图卷积层，它通过在二部user-item交互图上传递消息来构造用户和项嵌入，结合双线性译码器，以标记边缘的形式预测新的评级。
  
    
  
- <img align="right" src="https://ws1.sinaimg.cn/large/005NduT8ly1g9rhflyy9yj30bc05zaag.jpg" width=350 height=110/>[Neural relational inference for interacting systems](http://proceedings.mlr.press/v80/kipf18a.html). [ICML 2018]
  
    - **Overview:** 提出了基于变分自编码器的神经关系推理(NRI)模型，用于非监督的学习交互系统的dynamics同时推断关系结构。该模型可以在无监督的情况下准确的恢复真实的交互图。
    
      
    
- <img align="right" src="https://ws1.sinaimg.cn/large/005NduT8ly1g9rkekla1oj30k80ao0ur.jpg" width=350 height=110/>[MODELING RELATIONAL DATA WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1703.06103.pdf) [ESWC 2018]
  
    - **Overview:** 提出了关系图卷积网络(R-GCNs)，用于知识图谱的**链路预测**和**实体分类**。


