---
layout: article
title: Summary：聚类思想在区块链中的发展
date: 2021-03-20 00:11:00 +0800
tags: [Summary, blockchain, Cluster, Bitcoin]
categories: blog
pageview: true
key: summary-of-cluster-in-bitcoin
---



## 一、区块链中的“聚类”影子

聚类分析在比特币中有着广泛的应用，任务的命名也多种多样：

- Entity Recognition

  > Ron and Shamir proposed to use the neutral word “entity” to describe the common owner of multiple addresses (accounts).
  >
  > *Paper: Quantitative analysis of the full Bitcoin transaction graph*

  在这里，实体一般表示拥有多重账户地址的用户。实体识别主要用于去匿名化拥有多重账户的用户。

- User Re-identification

  > Community detection can be efficiently used to re-identify multiple addresses belonging to a same user.
  >
  > *Paper: Tracking bitcoin users activity using community detection on a network of weak signals*

  用户重识别，同上面的实体识别一样，也是用来去匿名化。

- Address Clustering

  地址聚类，得到的clusters代表同一用户的地址集合。

  > Address clustering tries to construct the one-to-many mapping from entities to addresses in the Bitcoin system.
  >
  > *Paper: The Unreasonable Effectiveness of Address Clustering*



## 二、任务总结

- Entity Recognition (User Re-identification, Address clustering)

  确定用户的多重（相同功能）地址。

  - **对象：**比特币
  - fraud detection

- Dynamic community detection for tracking network evolving

  动态社区检测，追踪网络演化

  - **对象：**比特币、以太坊

- Advertising recommendation

  用户、Dapp聚类, 广告推荐
  
  - **对象：**以太坊



## 三、Entity Recognition

### 3.1 比特币

在这里，实体一般表示拥有多重账户地址的用户。实体识别作为一种去匿名化手段，主要通过地址聚类（address clustering），将同一用户的多重地址关联起来。



#### 3.1.1 Heuristic 1 - *multi-input heuristic*

==Multi-input heuristic: 交易输入的多个地址属于同一用户控制。==

- 出处：(2013) [An analysis of anonymity in the bitcoin system](https://arxiv.org/abs/1107.4524)

- Several description in related papers:

  > 1. All addresses used as input of the same transaction belong to the same controlling entity, called a User.
  > 2. if two (or more) addresses are used as inputs to the same transaction, then they are controlled by the same user.
  > 3. assumes that the input addresses of a particular transaction are possessed by the same entity.
  > 4. the addresses of the input end of a transaction are under the control of the same entity.

- 分类：Heuristic based on transaction inputs only

- 实现：

  > 1. 建网：<u>节点</u>表示<u>账户地址</u>；
  >
  > 2. 对一个交易的$n$个输入地址，在它们之间增加$n-1$条边，构成地址路径；
  >
  > 3. 确定网络的连通子图，每个<u>连通子图</u>视为一个<u>用户(User)</u>；      
  >
  >    connected component $\rightarrow$ set of addresses $\rightarrow$ user



#### 3.1.2 Heuristic 2 - *shadow heuristic*

==Shadow heuristic: 交易输出中存在的零钱地址(change/shadow address)，属于交易输入所属的用户。==

- 出处：

  - (2013) [Evaluating user privacy in bitcoin.](https://eprint.iacr.org/2012/596.pdf)
  - (2013) [A Fistful of Bitcoins: Characterizing Payments Among Men with No Names](https://cseweb.ucsd.edu/~smeiklejohn/files/imc13.pdf)
  - (2013) [Bitiodine: Extracting intelligence from the bitcoin network.](https://www.ifca.ai/fc14/papers/fc14_submission_11.pdf)
  - (2015) [Data-Driven De-Anonymization in Bitcoin.](https://nickler.ninja/papers/thesis.pdf)

- Several description in related papers:

  > 1. One of the defining features of the Bitcoin protocol is the way that bitcoins must be spent. When the bitcoins re- deemed as the output of a transaction are spent, they must be spent all at once: the only way to divide them is through the use of a change address, in which the excess from the input address is sent back to the sender. 
  > 2. In the current implementation of Bitcoin, a new address—the “shadow” address [1]—is automatically created and used to collect back the “change” that results from any transaction issued by the user. Besides the reliance on pseudonyms, shadow addresses constitute the only mechanism adopted by Bitcoin to strengthen the privacy of its users.

- 分类：Heuristics based on inputs and outputs of transactions

- 实现：

  > 启发式的判断零钱地址：
  >
  > - 双输出（$a_1$和$a_2$）的交易，如果$a_1$第一次出现，$a_2$已经被使用过，那么$a_1$可能是零钱地址；
  >
  > 论文（A Fistful of Bitcoins: Characterizing Payments Among Men with No Names） 提出：
  >
  > - 零钱地址不存在于创币交易；
  > - 不出现在交易的输入地址中（no address reuse）；
  > - 作为输出地址，首次出现在交易中；
  >
  > 论文 (Data-Driven De-Anonymization in Bitcoin) 提出 最优零钱地址：
  >
  > - 值唯一且不超过任意一个输入值；
  > - 双输入双输出：零钱地址一定小于另一个输出；



#### 3.1.3 Heuristic 3 - *H1 + cluster methods*

- 出处：[Tracking bitcoin users activity using community detection on a network of weak signals](https://arxiv.org/abs/1710.08158)

- 实现：

  创建一个身份提示网络，其中节点之间的边表示对应的地址集可能属于同一用户的提示。

  > - 一次建网：
  >
  >   利用H1获得第一层地址聚类，每一个连通子图视为一个user；
  >
  > - 二次建网：
  >
  >   对于每一条交易，考虑H1获得的users，如果该交易的发送者（输入user）唯一，且满足以下条件：
  >
  >   - 交易的接收者（输出user）少于10个；
  >   - 交易的所有接收者，与交易发送者不同；（也就是说输出中没有已知的零钱地址）
  >
  >   则在交易的发送者和每一个接收者之间加边（<u>我觉得类似于社区间加边</u>）；
  >
  > - 在最终获得的网络上，使用社团检测算法挖掘社团，最终一个社团对应于一个user;

- 结论：

  - 该启发式算法发现的users和H1发现的一个或多个user关联；（有一个分层的关系）



## 3.2 以太坊

由于比特币和以太坊在交易模式上的区别，现有的用于比特币的地址聚类方法不适用于以太坊的交易模型。

- 论文：[Address Clustering Heuristics for Ethereum](https://fc20.ifca.ai/preproceedings/31.pdf)
  - 代码：[https://github.com/etherclust/etherclust](https://github.com/etherclust/etherclust)
  - 脉络：
    - 







# 四、Dynamic、Evolving

==比特币交易有时间属性，可以利用动态网络分析方法，研究交易网络演化。==

## 4.1 Dynamic network evolving

- 论文：(2018) <u>Community Detection and Observation in Large-scale Transaction-based Networks</u>
    - 代码：
      - [https://github.com/dalwar23/neochain](https://github.com/dalwar23/neochain)
      - [https://github.com/pinebud/neochain](https://github.com/pinebud/neochain)
      - [https://github.com/dalwar23/ncprep](https://github.com/dalwar23/ncprep)
      - [https://github.com/plrectco/NeoChain](https://github.com/plrectco/NeoChain)
    - 脉络：
      - 介绍并分析了一些社团检测算法
      - 提出了 <u>N</u>etwork <u>E</u>volution <u>O</u>bservation for Block<u>chain</u> (NEOchain)

    - 实现：

      利用网络快照，对相邻时间片的社团进行对比，分析社团演化。

      > - 在$t$时刻，提取网络快照$G_t$，利用社团检测算法发现$\text{top}-n$个目标社团$C_t^{top}$；
      >
      >   $G_t^{top} \leftarrow \mathsf{communityDetection}(G_t)$
      >
      > - 针对获得的社团$C_t^{top}$，利用其中的节点$V_t\in C_{top}$，构建网络子快照$G_{sub_t}$，其中$V_t\in G_{sub_t}$；
      >
      >   $G_{sub_t}=(V_t\in C_{top}, E_t\in G_t) \leftarrow \mathsf{networkExtract}(C_t^{top},G_t)$
      >
      > - 将下一时刻（$t+1$）的网络快照$G_{t+1}$和当前时刻的网络子快照$G_{sub_t}$进行融合，获得新的网络$G_{t,t+1}$；
      >
      >   $G_{t,t+1} \leftarrow \mathsf{merge}(G_{t+1} , G_{sub_t})$
      >
      > - 对网络$G_{t,t+1}$，利用社团检测算法发现$\text{top}-n$个社团$C_{t+1}^{top}$；
      >
      >   $G_{t+1}^{top} \leftarrow \mathsf{communityDetection}(G_{t,t+1})$
      >
      > - 从$C_t^{top}$和$C_{t+1}^{top}$中发现最大重叠社团$G_{t,t+1}^{max}$，分析社团在其生命周期中的状态；
      >
      >   $G_{t,t+1}^{max} \leftarrow \mathsf{findMaxOverlappingCommunities}(C_t^{top}, C_{t+1}^{top})$
      >
      >   社团重合度：$C(t)=\frac{C(t)\cap C(t+1)}{C(t)\cup C(t+1)}$
      >
      > - 社团生命周期中的不同状态：
      >
      >   ![image-20210326145321693](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210326145321.png)

  

# 五、Recommendation

对聚类的结果做推荐（一般用于以太坊）

## 5.1 Advertisement

- 论文：(2021) <u>Community Detection in Blockchain Social Networks</u>

  - 脉络：

    - 构建区块链社交网络

      - 针对比特币，利用“common spend” 和“change address”构建比特币社交网络；

        > 本质上来说就是利用上述地址聚类启发式，将地址聚合成“super address”，也就是用户；

      - 针对以太坊，构建EoA-CA的二分网络

    - 利用谱聚类算法进行社团检测

    - 检测结果用于广告推荐
