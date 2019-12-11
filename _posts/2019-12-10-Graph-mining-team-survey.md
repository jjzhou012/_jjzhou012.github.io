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

- [Team Home](http://www.kdd.in.tum.de/en/home/)



### Head of Research Group

<center>
    <img style="border-radius: 0.3125em"  align="left" width="160" height="220"  src="http://ww1.sinaimg.cn/large/005NduT8ly1g9ru7rdz1mj304q06c0u5.jpg"/>
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

### Recent publication

#### 算法鲁棒性研究

对抗性攻击 & 鲁棒性：图学习算法鲁棒吗？如何去有效的攻击图算法？如何提升它们的鲁棒性？

- [**Adversarial Attacks**] [Adversarial Attacks on Graph Neural Networks via Meta Learning](http://openreview.net/pdf?id=Bylnx209YX) [ICLR 2019]

  - **Overview:** 针对**节点分类**，提出了一种**基于属性图的训练时间对抗攻击算法**。使用**元梯度**来解决双层优化问题的基础上的**类中毒对抗攻击**。实验表明，元梯度方法创建的攻击（即使很微小的扰动）始终导致gcn模型的分类性能大幅下降，甚至转移到**无监督模型**。

    <center>
        <img src="http://ws1.sinaimg.cn/large/005NduT8ly1g9rmj7537yj30pc07tgnx.jpg" width="700" height="180"/></center>
  
- <div><img align="right" src="http://ws1.sinaimg.cn/large/005NduT8ly1g9rn2djo6tj30lx096zml.jpg" width="300" height="190"/></div> [Adversarial Attacks] [Adversarial Attacks on Neural Networks for Graph Data](http://arxiv.org/pdf/1805.07984.pdf) (Best Research Paper Award) [kdd 2018]
	- **Overview:** 提出了第一个关于（属性）图的对抗性攻击的研究，特别关注利用gcn进行节点分类的任务。针对节点特征和图结构，提出了测试阶段的直接攻击和影响者攻击，和训练阶段的中毒/致因攻击。通过保留重要的数据特征来确保扰动的隐蔽性；提出了一种增量计算的高效算法；攻击具有迁移性。
  
  
  
- <img align="right" src="http://ws1.sinaimg.cn/large/005NduT8ly1g9rnpn95cwj308j060jrr.jpg" width="300" height="220"/>[Clustering] [Bayesian Robust Attributed Graph Clustering: Joint Learning of Partial Anomalies and Group Structure](http://pdfs.semanticscholar.org/2230/c5946b37ab891389a034396a4e2f1830cdfa.pdf) [AAAI 2018]  
	
  - **Overview:** 研究了鲁棒属性图聚类问题。实际数据中聚类结构常常由于异常或破环而变得模糊，考虑到：图包含两个视图（网络结构和属性），异常可能只影响一部分，即实例可能会在其中一个视图中损坏，但在另一个视图中完好。在这种情况下，仍然可以派生有意义的聚类集群。通过对度修正随机块模型和伯努利混合模型的推广，提出了一种属性图聚类的概率模型PAICAN，实验表明了PAICAN在部分和全部异常下的鲁棒性。
  
    

- <img align="right" src="http://ws1.sinaimg.cn/large/005NduT8ly1g9rod09tahj309404d3z5.jpg" width="300" height="160"/>[Clustering] [Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings](http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p737.pdf) [kdd 2017]
  
  - **Overview:** 谱聚类是最主要的聚类方法之一，但是它对有噪声的输入数据非常敏感。提出了一种用于谱聚类的相似图的稀疏和潜在分解方法。该模型联合学习了谱嵌入和被破坏的数据，从而整体上提高了聚类性能。



- [Adversarial Attacks on Node Embeddings via Graph Poisoning](http://arxiv.org/pdf/1809.01093.pdf) [Arxiv]



#### 图神经网络架构

如何设计新的神经网络结构来捕捉图的基本特性？如何捕获关系依赖来提高学习的性能？

- [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](http://arxiv.org/pdf/1810.05997.pdf) [ICLR 2019]
  - **Overview:** 提出GCN + personalized PageRank 的神经消息传播框架(PPNP)，用于节点的半监督分类。
- [Pitfalls of Graph Neural Network Evaluation](http://arxiv.org/pdf/1811.05868.pdf) [NIPS 2018]
  - **Overview:** 证明了现有的GNN模型评价策略存在严重缺陷。评估表明，使用相同的训练/验证/测试分割相同的数据集，以及对训练过程进行重大更改(例如，early stopping)，会妨碍对不同架构的公平比较；考虑不同的数据分割会导致模型的排名显著不同；如果对所有模型的超参数和训练过程进行适当的调整，那么更简单的GNN体系结构能够胜过更复杂的结构。
- [Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings](http://pdfs.semanticscholar.org/2b76/b6e766547b3c6dbc2785a084ec3b72cb760d.pdf) [kdd 2017]
  - **Overview:** 提出的Graph2Gauss方法可以有效地学习大规模(带属性)图的通用节点嵌入，不同于大多数将节点表示为低维连续空间中的点向量的方法，Graph2Gauss将每个节点嵌入为高斯分布，从而允许捕获关于表示的不确定性。此外，提出了一个无监督的方法，处理归纳学习场景，适用于不同类型的图:普通/属性，有向/无向。

#### 图推理的生成模型

如何自动生成具有真实属性的图?我们能否从其他信息来源推断出图/网络？

- [GhostLink: Latent Network Inference for Influence-aware Recommendation](http://arxiv.org/abs/1905.05955) [WWW 2019]

- [NetGAN: Generating Graphs via Random Walks](http://arxiv.org/pdf/1803.00816.pdf) [ICML 2018]



## 2. SNAP: Stanford Network Analysis Project

相关链接：

- [Jure’s personal homepage](https://cs.stanford.edu/~jure/)

### Head of Research Group

<center>
    <img style="border-radius: 0.3125em"  align="left" width="160" height="220"  src="http://ww1.sinaimg.cn/large/005NduT8ly1g9ruxqdrg5j303d046q37.jpg"/>
    <p style="font-size:22px"><b>Jure Leskovec</b></p>
    <ul style="margin:0 0 0 200px">
        <p style="font-size:18px;margin:0px;" align="left"><b>Research Focus:</b></p>
    <li><p align="left">Mining and modeling large social and information networks.</p></li>
    <li><p align="left">Networks evolution and diffusion of information.</p></li>
         <p style="font-size:18px;margin:0px;" align="left"><b>Main contributions:</b></p>
    <li><p align="left">Node2vec, GraphSAGE, GIN</p></li>
    </ul>
</center>



团队重要贡献:

- GNN以及GNN的可解释性研究，应用于节点分类、动态网络分析、推荐系统
- 

### Recent publication

#### GNN以及GNN的可解释性研究

- [可解释性] [GNNExplainer: Generating Explanations for Graph Neural Networks](https://cs.stanford.edu/~jure/pubs/gnnexplainer-neurips19.pdf). [NeurIPS 2019]

  - **Blog:** [[Project website, code and data](http://snap.stanford.edu/gnnexplainer/)]

  - **Overview:** 提出了第一个用于GNN解释预测的通用的、模型无关的工具 GNNExplainer，用于在任何基于图的机器学习任务上为任何基于GNN的模型的预测提供说明性的解释。给定一个实例，GNNExplainer能识别出一个紧凑的子图结构和一小部分在GNNs预测中起关键作用的节点特征。此外，GNNExplainer可以为整个实例类生成一致和简洁的解释。实验表明，GNNExplainer在解释精度上比其他baseline方法高出43.0%。GNNExplainer提供了各种各样的好处，从可视化语义相关结构的能力到可解释性。。。

    <center>
        <img src="http://ww1.sinaimg.cn/large/005NduT8ly1g9rwukb5f5j30qu05h0vy.jpg" width="900" height="180"/></center>

- [How Powerful Are Graph Neural Networks?](https://cs.stanford.edu/~jure/pubs/gin-iclr19.pdf). [ICLR 2019] [[code]](https://github.com/weihua916/powerful-gnns)

  - **Overview：**提出了一个理论框架来分析GNNs捕捉不同图结构的表达能力。结果描述了流行的GNN变体，如GCNs和GraphSAGE的区别能力，并表明它们不能学会区分某些简单的图结构。此外开发了一个简单的架构，它被证明是所有GNNs类中最富表现力的，并且与WeisfeilerLehman图同构测试一样强大。

- [Hyperbolic Graph Convolutional Neural Networks](https://cs.stanford.edu/~jure/pubs/hgcn-neurips19.pdf). [NeurIPS 2019] [[code]](http://snap.stanford.edu/hgcn/#code)

  - **Overview：**提出了双曲图卷积神经网络(HGCN)，这是第一个利用GCNs的表达性和双曲几何性质来学习层次图和无标度图的归纳节点表示的归纳双曲神经网络。实验证明HGCN能学习嵌入，保护层次结构，降低预测错误率。

- <img align="right" src="http://ww1.sinaimg.cn/large/005NduT8ly1g9skn1a66qj30t40cctbl.jpg" width="280" height="140"/>[Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://cs.stanford.edu/~jure/pubs/jodie-kdd19.pdf). [kdd 2019] [[code]](https://github.com/srijankr/jodie/)

  - **Overview：**JODIE是动态网络中所有节点的表示学习框架。给定一个节点动作序列，JODIE为每个节点学习一个动态嵌入轨迹(与静态嵌入相反)。这些轨迹对于后续的机器学习任务很有用，比如链接预测、节点分类和聚类。JODIE速度很快，可以准确地预测未来的交互和异常检测。

    

- [Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems](https://cs.stanford.edu/~jure/pubs/kgnnls-kdd19.pdf). [kdd 2019]

  - **Overview：**提出了带有标签平滑正则化的知识感知图神经网络(KGNN-LS)来为推荐系统提供更好的建议。从概念上讲，我们的方法首先应用一个可训练的函数来识别给定用户的重要知识图关系，从而实现特定于用户的项嵌入。将知识图转化为用户特定的加权图，利用图神经网络计算个性化的项嵌入。为了提供更好的归纳偏差，我们依赖于标签平滑度假设，该假设假定知识图谱中相邻的项目可能具有相似的用户相关性标签/分数。
  
- [Hierarchical Temporal Convolutional Networks for Dynamic Recommender Systems](https://cs.stanford.edu/~jure/pubs/hiertcn-www19.pdf). [WWW 2019]
  
  - **Overview：**考虑到现有的基于序列模型的推荐系统受到速度和内存消耗的限制，提出了一种适用于实时大规模推荐系统的层次化时域卷积网络(HierTCN)。这是一种层次化的深度学习体系结构，它基于用户与项目的连续多会话交互进行动态推荐。HierTCN是为具有数十亿项和数亿用户的网络级系统设计的。它包括两个层次的模型:高级模型使用递归神经网络(RNN)聚合用户发展的长期利益在不同会话，而低级模型实现时间卷积网络(TCN)，利用长期利益和短期内交互会话来预测下一个交互。
  
- [Hierarchical Graph Representation Learning with Differentiable Pooling](https://cs.stanford.edu/~jure/pubs/diffpool-neurips18.pdf). [NeurIPS 2018]

  - **Overview：**提出了DIFFPOOL，一个可微分的图池模块，它可以生成图的层次表示，并可以端到端的方式与各种图神经网络架构相结合。DIFFPOOL为深度GNN的每一层的节点学习可微分的软集群分配，将节点映射到一组集群，然后形成下一个GNN层的粗糙输入。



#### 网络分析

- [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://cs.stanford.edu/~jure/pubs/jodie-kdd19.pdf). [kdd 2019][The Local Closure Coefficient: A New Perspective On Network Clustering](https://cs.stanford.edu/~jure/pubs/motifs-wsdm19.pdf). [WSDM 2019]
  
  - **Overview：**边缘聚类的许多常见解释都将三元闭合归结为一个头部节点，而不是长度为2路径的中心节点，例如，我朋友的一个朋友也是我的朋友。虽然这种解释在网络分析中很常见，但是对于边缘聚类，没有一种测量方法可以归因于头节点。在此，我们将局部闭包系数作为量化基于头节点的边缘聚类的度量。我们将局部闭包系数定义为由头节点发散出的长度为2的路径中产生三角形的部分。这种定义上的细微差异导致了与传统聚类系数显著不同的属性。我们分析了与节点度的相关性，将闭合系数与社区检测联系起来，表明闭合系数作为一种特征可以改善链路预测。
  
    <center>
        <img src="http://ww1.sinaimg.cn/large/005NduT8ly1g9sues94tzj30ev04edhh.jpg" width="500" height="180"/></center>

- [Network enhancement as a general method to denoise weighted biological networks.](https://cs.stanford.edu/~jure/pubs/ne-natcom18.pdf) [Nature Communications 2018]

  - **Overview：**开发了网络增强(NE)，一种去噪加权生物网络的通用方法。该方法可用于任何加权网络分析管道，提高了无向加权网络的信噪比，并可改进下游分析，提高下游网络分析的性能。



## 3. GEMS Lab ：Graph Exploration and Mining at Scale

相关链接：

- https://web.eecs.umich.edu/~dkoutra/
- https://gemslab.github.io/

### Head of Research Group

<center>
    <img style="border-radius: 0.3125em"  align="left" width="210" height="220"  src="http://ww1.sinaimg.cn/large/005NduT8ly1g9t0o5iz0qj32bc1yox72.jpg"/>
    <p style="font-size:22px"><b>Danai Koutra</b></p>
    <ul style="margin:0 0 0 200px">
        <p style="font-size:18px;margin:0px;" align="left"><b>Research Focus:</b></p>
    <li><p align="left">large-scale graph mining, graph similarity, matching, summarization and visualization.</p></li>
    <li><p align="left">graph anomaly and event detection.</p></li>
    </ul>
</center>





### Team profile

 Some of ongoing [projects](https://gemslab.github.io/research/) include:

- [多网络分析](https://gemslab.github.io/research/#multi-network-analysis), 如跨网络节点匹配
- [网络表示学习](https://gemslab.github.io/research/#representation-learning)
- Abstracting or [“summarizing” a graph](https://gemslab.github.io/research/#brain-network-analysis) with a smaller network
- Analyzing [network models of the brain](https://gemslab.github.io/research/#brain-network-analysis) derived from fMRI scans
- [Distributed graph methods](https://gemslab.github.io/research/#distributed-graph-methods) for iteratively solving linear systems
- Network-theoretical [user modeling](https://gemslab.github.io/research/#user-modeling) for various data science applications

### Recent publication

#### 多网络分析

<center>
    <img src="http://ww1.sinaimg.cn/large/005NduT8ly1g9t1pzg6rij30po0ci45a.jpg" width="700" height="200"/>
</center>

- [Graph Classification] [Distribution of Node Embeddings as Multiresolution Features for Graphs](https://gemslab.github.io/papers/heimann-2019-RGM.pdf) (Best Student Paper award) [ICDM 2019]

  提出了随机网格映射(RGM)用于图分类，它是一个快速计算的特征映射，能通过节点嵌入在特征空间的分布来表示图。

- [REGAL: Representation Learning-based Graph Alignment](https://gemslab.github.io/papers/heimann-2018-regal.pdf) [CIKM 2018]
  - **Overview：**提出了REGAL(基于表示学习的图对齐)，这是一个利用自动学习节点表示的能力来跨不同图匹配节点的框架。

#### 表示学习

- [Distribution of Node Embeddings as Multiresolution Features for Graphs](https://gemslab.github.io/papers/heimann-2019-RGM.pdf) (Best Student Paper award) [ICDM 2019]
- [Smart Roles: Inferring Professional Roles in Email Networks](https://gemslab.github.io/papers/jin-2019-roles.pdf) [kdd 2019]
  - **Overview：**研究了从电子邮件数据中进行专业角色推断的任务，这对于电子邮件优先级划分和联系人推荐系统至关重要。我们要解决的核心问题是:鉴于有关员工的有限数据(这在第三方电子邮件应用程序中很常见)，我们能否根据这些员工的电子邮件行为推断出他们在组织层次结构中的位置？为了实现我们的目标，在本文中，我们研究了一个独特的新电子邮件数据集上的专业角色推断，该数据集包含了数千个组织中数十亿的电子邮件交换。采用节点是雇员、边缘代表电子邮件通信的网络方法，我们提出了EMBER或嵌入基于电子邮件的角色，它发现以电子邮件为中心的网络节点嵌入可用于专业角色推断任务。EMBER自动捕获电子邮件网络中员工之间的行为相似性，从而自然地将不同层次角色的员工区分开来。
- [Latent Network Summarization: Bridging Network Embedding and Summarization](https://gemslab.github.io/papers/jin-2019-latent.pdf) [kdd 2019]
- [GroupINN: Grouping-based Interpretable Neural Network for Classification of Limited, Noisy Brain Data](https://gemslab.github.io/papers/yan-2019-groupinn.pdf) [kdd 2019]





## 4. KEG: Center for Knowledge & Intelligence , Knowledge Engineering Group in Tsinghua University

相关链接：

- http://keg.cs.tsinghua.edu.cn/jietang/

### Head of Research Group

<center>
    <img style="border-radius: 0.3125em"  align="left" width="180" height="220"  src="http://ww1.sinaimg.cn/large/005NduT8ly1g9t2bi5bwnj30he0heha6.jpg"/>
    <p style="font-size:22px"><b>Jie Tang</b></p>
    <ul style="margin:0 0 0 200px">
        <p style="font-size:18px;margin:0px;" align="left"><b>Research Focus:</b></p>
    <li><p align="left"> artificial intelligence、 data mining and machine learning</p></li>
    <li><p align="left">social networks、 knowledge graph</p></li>
    </ul>
</center>







### Recent publication

#### 表示学习

- [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](http://keg.cs.tsinghua.edu.cn/jietang/publications/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf) [WSDM 2018]
  - **Overview：**对四种有效的网络嵌入方法DeepWalk、LINE、PTE和node2vec进行了理论分析，证明了上述所有带有负抽样的模型都可以统一到具有封闭形式的矩阵分解框架中，它们的矩阵的封闭形式不仅提供了这些方法之间的关系，而且提供了它们与图拉普拉斯变换的内在联系。给出了计算网络嵌入的NetMF方法及其近似算法。
- [DeepInf: Social Influence Prediction with Deep Learning](http://keg.cs.tsinghua.edu.cn/jietang/publications/kdd18_jiezhong-DeepInf.pdf) [kdd 2018]
  - **Overview：**设计了一个端到端框架DeepInf，学习用户的潜在特征表征来预测社会影响。一般情况下，Deeplnf将用户的局部网络作为输入到一个图神经网络中，学习其潜在的社会表征。我们设计的策略是将网络结构和用户特定的特性结合到卷积神经网络和注意力网络中。明显优于传统的基于特征工程的方法，这表明了表示学习对于社交应用的有效性。
- [NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorization](http://keg.cs.tsinghua.edu.cn/jietang/publications/www19-Qiu-et-al-NetSMF-Large-Scale-Network-Embedding.pdf) [www 2019]
  - **Overview：**研究了大规模网络嵌入问题。提出了大规模网络嵌入的稀疏矩阵分解(NetSMF)算法。NetSMF杠杆理论从光谱稀疏化，有效稀疏上述密集矩阵，使嵌入学习的效率显着提高。稀疏化后的矩阵在谱上与原始的稠密矩阵非常接近，理论上近似误差有界，这有助于保持学习嵌入的表示能力。

#### 知识图谱

- [ArnetMiner: Extraction and Mining of Academic Social Networks](http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD08-Tang-et-al-ArnetMiner.pdf) [kdd 2008]
  - **Overview：**该系统主要针对学术社交网络的提取和挖掘，整合了来自在线网络数据库的数据，构建 [ArnetMiner](https://www.aminer.cn/)
- [Fast and Flexible Top-k Similarity Search on Large Networks](http://keg.cs.tsinghua.edu.cn/jietang/publications/TOIS17-Zhang-et-al-Fast-Top-K-Similarity-Search.pdf) [TOIS 2017]
  - **Overview：**相似度搜索是网络分析中的一个基本问题，可以应用于许多领域，如合著者网络中的协作者推荐、社交网络中的朋友推荐、医疗信息网络中的关系预测等。在本文中，我们提出了一种基于随机路径的方法来估计相似点，这种方法基于共同邻居和结构上下文，在非常大的同构或异构信息网络中有效。





## 5. Marcin Waniek: a Post-Doctoral Associate at New York University Abu Dhabi.

相关链接：

- https://www.mimuw.edu.pl/~vua/

### Introduction

<center>
    <img style="border-radius: 0.3125em"  align="left" width="160" height="180"  src="http://ww1.sinaimg.cn/large/005NduT8ly1g9t44cj9hlj302h03kaaa.jpg"/>
    <p style="font-size:22px"><b>Marcin Waniek</b></p>
    <ul style="margin:0 0 0 200px">
        <p style="font-size:18px;margin:0px;" align="left"><b>Research Focus:</b></p>
    <li><p align="left">Link prediction attack</p></li>
    </ul>
</center>





### Recent publication

- [How to Hide One's Relationships from Link Prediction Algorithms.](https://www.nature.com/articles/s41598-019-48583-6.pdf) [Scientific Reports]
  - **Overview：**提出启发式方法，弱化社交网络中的链路预测，隐私保护。
- [Attacking Similarity-Based Link Prediction in Social Networks](https://arxiv.org/pdf/1809.08368.pdf) [AAMAS 2019]
  - **Overview：**调研了两类基于相似度的链路预测攻击：局部和全局。对于局部度量，我们提出了一种最小化局部度量上界的算法，它对应于在基数约束下最大化子模函数。此外，我们确定了两种特殊情况，攻击单个链接和攻击一组节点，其中第一种情况确保对所有本地指标的最佳攻击，而后者确保对CND指标的最佳攻击。对于全局度量，我们证明了即使在攻击单个链接时，最小化Katz和最大化ACT的问题都是NP-Hard。然后针对这两个问题分别提出了一种有效的贪婪算法和一种有原则的启发式算法。



## 6. **AMLab** in University of Amsterdam

相关链接：

- [Thomas Kipf’s blog](http://tkipf.github.io/)
- [AMLab](http://staff.fnwi.uva.nl/)

### Head of Research Group

<center>
    <img style="border-radius: 0.3125em"  align="left" width="160" height="110"  src="https://scholar.googleusercontent.com/citations?view_op=view_photo&user=8200InoAAAAJ&citpid=2"/>
    <p style="font-size:22px"><b>Max Welling</b></p>
    <ul style="margin:0 0 0 200px">
        <p style="font-size:18px;margin:0px;" align="left"><b>Research Focus:</b></p>
    <li><p align="left">Machine Learning, Artificial Intelligence, Statistics</p></li>
    </ul>
</center>

### Team profile

**AMLab:** The Amsterdam Machine Learning Lab (AMLab) conducts research in the area of **large scale modeling of complex data sources.** This includes the development of new methods for **deep learning, probabilistic graphical models, Bayesian modeling, approximate inference, causal inference, reinforcement learning and the application of all of the above to large scale data domains in science and industry.**

主要研究包括：

- 大规模复杂数据的建模
  - 深度学习、强化学习
  - 图模型、贝叶斯模型
  - 近似推理、因果推理

团队重要贡献:

- 提出 **gcn**: [Semi-supervised classification with graph convolutional networks](http://arxiv.org/abs/1609.02907) [ICLR 2017]

### Recent publication

- <img align="right" src="http://ws1.sinaimg.cn/large/005NduT8ly1g9rjo6iamrj30ie076ta2.jpg" width="350" height="130"/>[Graph Convolutional Matrix Completion](http://arxiv.org/pdf/1706.02263.pdf) [kdd 2019]
  
  - **Overview:** 从图上链接预测的角度考虑了推荐系统的矩阵补全问题，提出了一种基于可微消息在二部交互图上传递的图自动编码器框架。编码器包含一个图卷积层，它通过在二部user-item交互图上传递消息来构造用户和项嵌入，结合双线性译码器，以标记边缘的形式预测新的评级。
  
    
  
- <img align="right" src="http://ws1.sinaimg.cn/large/005NduT8ly1g9rhflyy9yj30bc05zaag.jpg" width="350" height="110"/>[Neural relational inference for interacting systems](http://proceedings.mlr.press/v80/kipf18a.html). [ICML 2018]
  
    - **Overview:** 提出了基于变分自编码器的神经关系推理(NRI)模型，用于非监督的学习交互系统的dynamics同时推断关系结构。该模型可以在无监督的情况下准确的恢复真实的交互图。
    
      
    
- <img align="right" src="http://ws1.sinaimg.cn/large/005NduT8ly1g9rkekla1oj30k80ao0ur.jpg" width="350" height="110"/>[MODELING RELATIONAL DATA WITH GRAPH CONVOLUTIONAL NETWORKS](http://arxiv.org/pdf/1703.06103.pdf) [ESWC 2018]
  
    - **Overview:** 提出了关系图卷积网络(R-GCNs)，用于知识图谱的**链路预测**和**实体分类**。


