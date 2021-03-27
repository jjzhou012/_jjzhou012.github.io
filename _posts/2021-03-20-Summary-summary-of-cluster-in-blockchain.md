---
layout: article
title: Summary：区块链数据中的聚类应用
date: 2021-03-20 00:11:00 +0800
tags: [Summary, blockchain, Cluster]
categories: blog
pageview: true
key: summary-of-cluster-in-blockchain
---



------
参考：
- [https://jimwongm.github.io/2020/03/18/paper_however/](https://jimwongm.github.io/2020/03/18/paper_however/)
- 中山大学综述：Analysis of Cryptocurrency Transactions from a Network Perspective: An Overview






## 1.区块链中的“聚类”影子

聚类分析在区块链中有着广泛的应用，但是可能不叫聚类，而是有着其他的命名，比如

- Entity Recognition

  > Ron and Shamir proposed to use the neutral word “entity” to describe the common owner of multiple addresses (accounts).
  >
  > (Paper: Quantitative analysis of the full Bitcoin transaction graph.)

  在这里，实体一般表示拥有多重账户地址的用户。实体识别主要用于去匿名化拥有多重账户的用户。

- User Re-identification

  用户重识别，同上面的实体识别一样，也是用来去匿名化。

- Address Clustering

  地址聚类，得到的clusters代表同一用户的地址集合。
  
  > Address clustering tries to construct the one-to-many mapping from entities to addresses in the Bitcoin system.





## 2. Cluster in Bitcoin

- (2013) **An Analysis of Anonymity in the Bitcoin System**

  指出TX网络和User网络之间的互补性，可以用于比特币网络的去匿名化。

  - **目的**：去匿名化

  - **数据处理**：构建了TX网络和User网络。

  - **方法**：基于UTXO模型的多输入启发式：假设一个特定事务的输入地址被同一个实体拥有。

    也就是说，**同一交易的输入地址大概率属于同一用户。**

  - **其他**：一段时间内，在相似时间内使用的public key可能属于同一用户;

    可以构建并聚类地址共现网络，来推断公钥与用户之间的映射。

    > Over an extended time period, several public-keys, if used at similar times, may belong to the same user. It may be possible to construct and cluster a co-occurrence network to help deduce mappings between public-keys and users.

    

- (2013) **A Fistful of Bitcoins: Characterizing Payments Among Men with No Names**

  使用启发式聚类对比特币钱包地址进行分组，再使用重识别对得到的用户进行分类。

  - **目的**：去匿名化

  - **数据处理**：

    - 数据标注：地址受真实世界的用户控制的，标记地址所属的服务（7种标签）。

  - **方法**：

    - 启发式1：**同一交易的输入地址大概率属于同一用户。**

    - 启发式2：**交易输出中存在零钱地址（change address），属于交易输入所属的用户。**（基于使用习惯，存在漏洞）

      一次性零钱地址（one-time change address）的标注：

      - 一般作为输出地址首次出现在交易中；
      - 所属的交易不是创币交易；
      - 一般不出现在交易的输入地址中；（因为self-change address可以指定为输入地址）

      - 一般只有输入没有输出；

    - 启发式3：考虑到启发式2的不安全性，重新设计了基于零钱地址的启发式方法。

  - **其他**：



- (2013) **Evaluating User Privacy in Bitcoin**
  - **目的**：去匿名化，比特币交易的隐私问题
  - **方法**：
    - 构建了Adversarial Model，将去匿名化重定义为攻击。
    - 量化了“Privacy in Bitcoin” 。
  - **实验**：
    - 评估了上述的两种启发式算法。
    - 对比特币交易进行仿真，测试了提出的方案。



- (2014) **BitIodine: Extracting Intelligence from the Bitcoin Network**

  提出了一个模块化框架BitIodine，它可以解析区块链，将可能属于同一用户或用户组的地址聚类，并对这些用户进行分类和标记，最终将从比特币网络中提取的复杂信息可视化。

  - Code: 
    - [https://github.com/mikispag/bitiodine](https://github.com/mikispag/bitiodine)
    - [https://github.com/tzarskyz/bitiodine](https://github.com/tzarskyz/bitiodine)



- (2015) **Data-Driven De-Anonymization in Bitcoin**
  - 除了常用的聚类技术外，还引入了两种新的聚类策略。
  - 获取数据+真实标签。
  - 对bitcoin网络上的一些聚类策略进行了大量的评估。



- (2016) **The Unreasonable Effectiveness of Address Clustering.**

  探讨了使用**多输入启发式**进行地址聚类的有效性背后的原因。该启发式假设在多输入事务中兑换的事务输出中的地址由同一实体控制。虽然在一般情况下是不准确的，但在实践中这是一个有用的启发式方法。

  - the high-levels of address reuse and avoidable merging.
  - the existence of super-clusters with high centrality.
  - the incremental growth of address clusters.

  这些因素可能导致多输入启发式方法产生假阳性。



- (2018) **Tracking bitcoin users activity using community detection on a network of weak signals.**

  结合多输入启发式方法和社团检测方法，提出了新的实体识别方法，能够牺牲一定的precision来提升recall,并根据实际应用调整召回量。

  - **方法**：**在地址网络中进行聚类（节点之间的边用于提示相连的address属于同一user）**
    - 构建identity hint network
    - 应用community detection方法



- (2017) **Behavior pattern clustering in blockchain networks**

  交易行为分析

  - 代码：[https://github.com/cakcora/Chainoba/tree/master/clustering](https://github.com/cakcora/Chainoba/tree/master/clustering)
  - 方法：
    - 将节点的交易量提取为时间序列；
    - 提出Behavior Pattern Clustering算法对时间序列进行聚类；
      - 利用时间序列相似性指标`DTW distance`进行聚类中心初始化；
      - 迭代更新聚类中心；



## 3. Cluster in Ethereum

- (2018) **De-anonymisation in Ethereum Using Existing Methods for Bitcoin**

  讨论将bitcoin上的两种去匿名化（攻击）策略用到以太坊上面，发现不可行。（纯讨论）

  - link IP addresses to Bitcoin addresses；
    - IP发现攻击依赖于存在于比特币P2P网络中的静态入口节点
  - cluster different Bitcoin addresses belonging to the same user (BitIodine)；
    - 聚类攻击使用基于UTXO的交易模型来聚类地址。而以太坊中不存在UTXO模型，也不会在交易的过程中创建零钱地址。



- (2020) **Address clustering heuristics for Ethereum**

  提出了三种启发式方法来聚类以太坊中的账户。

  - 方法：

    - Deposit Address Reuse 

      ==向同一存款地址发送资金的多个地址可能属于同一实体。==

    - Airdrop Multi-participation

      ==用户通常会将空投获得的所有token汇总到一个账户，我们可以利用这个模式来识别多次接收token的单个实体。==

    - Self-authorization

      ==用户授权他们拥有的其他地址，自授权。==

  - Code: [https://github.com/etherclust/etherclust](https://github.com/etherclust/etherclust)

  

- 

