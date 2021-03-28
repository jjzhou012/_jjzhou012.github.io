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

- **Entity Recognition (User Re-identification, Address clustering)**

  确定用户的多重（相同功能）地址。

  - **对象：**比特币
  - fraud detection

- **Dynamic community detection for tracking network evolving**

  动态社区检测，追踪网络演化

  - **对象：**比特币、以太坊

- **Advertising recommendation**

  用户、Dapp聚类, 广告推荐
  
  - **对象：**以太坊
  
- **Behavior pattern analysis**

  交易行为模式分析



## 三、Entity Recognition

### 3.1 比特币

在这里，实体一般表示拥有多重账户地址的用户。实体识别作为一种去匿名化手段，主要通过地址聚类（address clustering），将同一用户的多重地址关联起来。



#### 3.1.1 Heuristic 1 - *multi-input heuristic*

**==Multi-input heuristic: 交易输入的多个地址属于同一用户控制。==**

- 出处：(2013) [An analysis of anonymity in the bitcoin system](https://arxiv.org/abs/1107.4524)

- Several description in related papers:

  > 1. All addresses used as input of the same transaction belong to the same controlling entity, called a User.
  > 2. if two (or more) addresses are used as inputs to the same transaction, then they are controlled by the same user.
  > 3. assumes that the input addresses of a particular transaction are possessed by the same entity.
  > 4. the addresses of the input end of a transaction are under the control of the same entity.
  > 5. The multiple input heuristic is based on the idea that multiple UTXOs which are used as input for a transaction are most likely controlled by the same entity.

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

**==Shadow heuristic: 交易输出中存在的零钱地址(change/shadow address)，属于交易输入所属的用户。==**

- 出处：

  - (2013) [Evaluating user privacy in bitcoin.](https://eprint.iacr.org/2012/596.pdf)
  - (2013) [A Fistful of Bitcoins: Characterizing Payments Among Men with No Names](https://cseweb.ucsd.edu/~smeiklejohn/files/imc13.pdf)
  - (2013) [Bitiodine: Extracting intelligence from the bitcoin network.](https://www.ifca.ai/fc14/papers/fc14_submission_11.pdf)
  - (2015) [Data-Driven De-Anonymization in Bitcoin.](https://nickler.ninja/papers/thesis.pdf)

- Several description in related papers:

  > 1. One of the defining features of the Bitcoin protocol is the way that bitcoins must be spent. When the bitcoins re- deemed as the output of a transaction are spent, they must be spent all at once: the only way to divide them is through the use of a change address, in which the excess from the input address is sent back to the sender. 
  > 2. In the current implementation of Bitcoin, a new address—the “shadow” address [1]—is automatically created and used to collect back the “change” that results from any transaction issued by the user. Besides the reliance on pseudonyms, shadow addresses constitute the only mechanism adopted by Bitcoin to strengthen the privacy of its users.
  > 3. the change heuristic assumes that a previously unused one-time change address created by a transaction is likely controlled by the same entity that created the transaction.

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



### 3.2 以太坊

由于比特币和以太坊在交易模式上的区别，现有的用于比特币的地址聚类方法不适用于以太坊的交易模型。

#### 3.2.1 Deposit Address Reuse 

**==向同一存款地址发送资金的多个地址可能属于同一实体。==**

- 出处：[Address Clustering Heuristics for Ethereum](https://fc20.ifca.ai/preproceedings/31.pdf)
  - 代码：[https://github.com/etherclust/etherclust](https://github.com/etherclust/etherclust)

  - 简述：

    为了将资产贷到正确的账户，交易所通常会创建所谓的存款地址，然后将收到的资金转到一个主地址。由于每个客户都创建了这些存款地址，因此<u>向同一存款地址发送资金的多个地址很可能由同一实体控制</u>。

    如何去确定存款地址：它们的特点是它们将收到的款项转到一个主要的外汇账户。<u>由于交易所必须支付交易费用，转寄的金额往往比收到的金额略少。</u>在大多数情况下，存款地址是EOAs，但它们也可以是CAs。

    <img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210326234206.png" alt="image-20210326234206354" style="zoom:67%;" />

    两个关键参数用于确定存款地址：

    - 接收和转发的最大金额差异：$a_{max}$：对应于在转发过程中支付的交易费用。

    - 接受和转发的最大时间差异：$t_{max}$：时间限制保证了转发特性，避免冲突匹配。

      > - 如果存款地址是CAs，交易费用为0，因为EoA在创建交易的时候支付了费用；
      > - 如果足够小的Ether被转移到一个存款地址，交易所可能会等待更多的存款，以使其足够支付交易费用；
      > - 交易费用无法通过token支付，所以这种情况下$a_{max}=0$；

      > - 有时交易所之间会相互发送资金。由于这些地址可能在转发跟踪中意外地作为存款地址出现，所以我们排除已知的交换地址；
      > - 要求存款地址只转发到一个单一的交易所地址

  - 实现：

    <img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210327003902.png" alt="image-20210327003902158" style="zoom: 67%;" />



#### 3.2.2 Airdrop Multi-participation

**==用户通常会将空投获得的所有token汇总到一个账户，我们可以利用这个模式来识别多次接收token的单个实体。==**

- 出处：[Address Clustering Heuristics for Ethereum](https://fc20.ifca.ai/preproceedings/31.pdf)

  - 代码：[https://github.com/etherclust/etherclust](https://github.com/etherclust/etherclust)

  - 简述：

    空投是一种流行的分配令牌的机制。在以太坊区块链上，它们是通过智能合约执行的。智能合约的所有者可以根据过去的活动随机选择收件人，也可以要求用户通过在线表单进行注册。这些注册过程中的某些过程要求用户在社交媒体上执行某些操作，例如发布文章或成为关注者。分配给每个用户的令牌数量是固定的，或基于现有帐户余额。如果金额是固定的，则有诱骗系统的动机。

    单个用户可以使用多个电子邮件地址注册并使用多个社交媒体帐户执行操作。空投完成后，用户将在其所有注册地址上收到令牌。<u>由于在所有账户上管理令牌是困难的，因此用户通常会收集令牌并将其汇总到一个地址。</u> 

    <img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210327143513.png" alt="image-20210327143513738" style="zoom: 67%;" />



#### 3.2.3 Self-authorization

**==用户授权他们拥有的其他地址，自授权。==**

- 出处：[Address Clustering Heuristics for Ethereum](https://fc20.ifca.ai/preproceedings/31.pdf)
  - 代码：[https://github.com/etherclust/etherclust](https://github.com/etherclust/etherclust)

  - 简述：

    ERC20令牌标准需要一个approve函数来允许另一个地址代表实际所有者使用令牌。通过执行，发送者地址获得对有限数量令牌的访问权。该功能主要用于连接智能合约，特别是去中心化交易所。虽然使用智能合约是主要目的，但这种类型的授权也可以用于常规EOA地址。这里将在<u>假设有用户批准他们拥有的另一个地址的情况下使用该功能。我们将此过程称为自授权。</u>



## 四、Dynamic、Evolving

**==比特币交易有时间属性，可以利用动态网络分析方法，研究交易网络演化。==**

### 4.1 Snapshot + clustering

- 论文：(2018) <u>Community Detection and Observation in Large-scale Transaction-based Networks</u>

    时间切片网络+聚类，分析网络社区演化。

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

  



## 五、Recommendation

对聚类的结果做推荐（一般用于以太坊）

### 5.1 Networking + clustering

- 论文：(2021) [<u>Community Detection in Blockchain Social Networks</u>](https://arxiv.org/abs/2101.06406)

  - 脉络：

    - 构建区块链社交网络

      - 针对比特币，利用“common spend” 和“change address”构建比特币社交网络；

        > 本质上来说就是利用上述地址聚类启发式，将地址聚合成“super address”，也就是用户；

      - 针对以太坊，构建EoA-CA的二分网络

    - 利用谱聚类算法进行社团检测

    - 检测结果用于广告推荐



## 六、Behavior Pattern Analysis

### 6.1 Time Sequence + clustering

- 论文：(2017) [<u>Behavior pattern clustering in blockchain networks</u>](https://www.semanticscholar.org/paper/Behavior-pattern-clustering-in-blockchain-networks-Huang-Liu/433c8b1160247d96aa4c5b84993c4b160523f2d3)
  - 代码：[https://github.com/cakcora/Chainoba/tree/master/clustering](https://github.com/cakcora/Chainoba/tree/master/clustering)
  - 方法：
    - 将节点的交易量提取为时间序列；
    - 提出Behavior Pattern Clustering算法对时间序列进行聚类；
      - 利用时间序列相似性指标`DTW distance`进行聚类中心初始化；
      - 迭代更新聚类中心；



### 6.2 Feature + clustering

- 论文：(2015) [Identifying Bitcoin users by transaction behavior](https://vmonaco.com/papers/Identifying%20Bitcoin%20Users%20by%20Transaction%20Behavior.pdf)

  - 方法：

    - 提出了一些特征

      - `Random time-interval (RTI)`：连续事务时间戳之间的时间间隔，以秒为单位。值总是非负的，表示事务行为的速度。
        $$
        R T I_{i}=t_{i}-t_{i-1}
        $$

      - `Hour of day (HOD)`：一天中交易发生的时间（小时）。
        $$
        H O D_{i}=t_{i} / / 3600 \bmod 24
        $$

      - `Time of hour (TOH)`：一个小时开始后经过的秒数。
        $$
        T O H_{i}=t_{i} \bmod 3600
        $$

      - `Time of day (TOD)` ：一天开始后经过的秒数。
        $$
        T O D_{i}=t_{i} \bmod 86400
        $$

      - `Coin flow (CF)`：用户的比特币价值吞吐量。
        $$
        C F_{i}=b_{i}
        $$

      - `Input/output balance (IOB)`：从其他<u>用户</u>获得的输入减去发送给其他<u>用户</u>的输出。
        $$
        I O B_{i}=u_{i}^{i n}-u_{i}^{o u t}
        $$
        

- 论文：(2019) [BITSCOPE: Scaling Bitcoin Address Deanonymization using Multi-Resolution Clustering](https://izgzhen.github.io/bitscope-public/paper.pdf)

  多分辨率地址聚类。

  - Resource: [https://izgzhen.github.io/bitscope-public/](https://izgzhen.github.io/bitscope-public/)
  - 方法：
    - 提出了一些aggregated features，进行账户分类；
    - 利用迭代搜索提取子图，进行社团检测；





