---
layout: article
title: Summary：Paper Writting
date: 2020-01-08 00:10:00 +0800
tags: [Summary, Writting]
categories: blog
pageview: true
key: Summary-of-Paper-Writting
---

------



## 写在前面

阅读了这么多论文（没那么多），有些精彩的表达方式确实值得借鉴。话不多说，笔记笔记☞。

- REF-N-WRITE：[https://www.ref-n-write.com/trial/academic-phrasebank/](https://www.ref-n-write.com/trial/academic-phrasebank/)

关于词汇

- Linggle: [https://linggle.com/](https://linggle.com/)

  词汇搭配

- Netspeak: [https://netspeak.org/](https://netspeak.org/)

  生词，模糊查找

关于语句

- Academic Phrasebank: [http://www.phrasebank.manchester.ac.uk/](http://www.phrasebank.manchester.ac.uk/)



## 一、Abstract

- 背景
  - in the context of

- 方法
  - We **present a novel XXX architecture capable of** learning a joint representation of both local graph structure and available node features for the multi-task learning of link prediction and node classification.
  - We discuss practical and theoretical motivations, considerations and strategies for …
- 结尾
  - Its experimental results show unprecedented performance, working consistently well on a wide range of problems.



## 二、Introduction

- **问题难**
  - is intractable in：棘手的

- **缩小差距、解决问题**
  - Bridge/narrow this gap between ... and ...：缩小差距
  - narrow the gap：缩小差距
  - To solve the above problem
  - To tackle these challenges
  - To bypass this difficulty, motivated by [17]
- **引起关注，流行**
  - attract considerable attention
  - increasing attention is drawn to
  - The past few years have **seen the growing prevalence of** XXX on application domains such as image
  - have seen **widespread adoption** in fields such as
  - the emergence of：涌现，出现
  - … have emerged as a rising approach for …
- **转折**
  - Despite the advances of；即使进步
- **启发**
  - Armed with this insight，
- **提出idea**
  - propose = take an effective approach = introduce
  - aim = the idea is 
  - effectively utilize
  - establish\introduce a framework
  - the application of … **to** …
  - … is employed here：被使用
  - Driven by the academic goal for ... to ...：目标驱使

### 1.1 contribution

- 现状

  - One major **obstacle** is that, **in contrast t**o other data, **where** structure is encoded by position, the structure of graphs is encoded by node connectivity, **which** is irregular. 
  - **But the question remains, which** edges to change.

- 创新点
  - our work ……, **whereas previous related methods** ……
  - We provide **a comprehensive empirical evaluation of our models on** nine benchmark graph-structured datasets **and demonstrate significant improvement over related methods** for graph representation learning.
  - The core contribution of this paper is two-fold, which is summarized as follows.
- to the best of our knowledge, has not been studied before
  



- The remainder of this paper is organized as follows. 



## 三、Related work

- Advances in this direction are often categorized as … (这方面的进展可以归纳为…)
- There has been a surge of algorithms that seek to …
- There are a limited number of existing reviews on the topic of …
- To tackle these challenges, tremendous effort has been made towards this area, resulting in a rich literature of related papers and methods. 
- one of the dominating paradigms is  
- XXX generalizes poorly off … ： 不能很好的概括
- In an approach more akin to A, or B, than C：一种更类似于AB而不是C的方法；
- A is helpful but far from sufficient, whereas 
- share the same drawback as



## 四、Method

> 方法部分涉及到we的时候，可用one can代替，即：
>
> we => one can
>
> we can’t => one cannot

- **[简化符号]** To simplify notations, we drop the subscript/superscript for the xxx, i.e. xxx.
- In order to overcome this limitation, we develop a XXX method, which is a simple yet effective approach to ...
- the tie is broken by comparing …
- and α is a hyperparameter **mediating** the influence of



## 五、Experiment setup

- In this section, we **expound our protocol for the empirical evaluation of our models’ capability for** learning and generalization on the tasks of link prediction and semi-supervised node classification.
- data split
  - we split the data into disjoint test, and validation, sets of 1,000, and 500, examples, respectively.
  - Each dataset is separated into a training, testing set and validation set. The validation set contains 5% citation edges for hyperparameter optimization, the test set holds 10% citation edges to verify the performance, and the rest are used for training.
  - We follow the semi-supervised setting in most GNN literature [24, 41] for train/validation/test splitting on CORA and CITESEER, and a 10/20/70% split on other datasets due to varying choices in prior work.
- In comparison to the baselines, we evaluate our model on the same data splits over 10 runs with XXX and report mean AUC/AP scores for link prediction and accuracy scores for node classification.
- **[parameter]** varying parameter XX from … to …
  - fixing a = 1 then varying b from 1 to 100 …
  - vary a in (1,2,3,4,5)
  - select a from (1,2,3,4,5)



## 六、Evalution、Discussion

### 6.1 图表

- [折线图中的大起大落](https://zhuanlan.zhihu.com/p/24372300)
- [折线图](http://blog.sina.com.cn/s/blog_5ac7f5100100soyp.html)
- [折线图趋势](http://www.joozone.com/ielts/12268.html)
- **[描述框架]** We introduce XXX, schematically depicted in Figure 1, capable of … for XXX task.
  - **We introduce** the Multi-Task Graph Autoencoder (MTGAE) architecture, **schematically depicted in Figure 1, capable of** learning a shared representation of latent node embeddings from local graph topology and available explicit node features **for** LPNC. 
- [图标题]
  - Schematic depiction of XXX architecture.
  - Illustration of the proposed XXX
  - Illustration of applying XXX
  - depict
- 图表内容说明
  - Table 2 is organized per architecture (row), per dataset (column), and original-graph and modified-graph settings (within-row). We bold best-performance per architecture and dataset,



### 6.2 分析

- 结合图表分析
  - These results **provide positive answers** to our motivating questions …
  - **from the evolution results,** we can observe that
  - table 1 shows\reports ……, **indicating** that …
  - As shown\visualized in …
  - For sake of visualization brevity：
- **指标提升数值**
  - GAUG-M improves 4.6% (GCN), 4.8% (GSAGE), 10.9% (GAT) and 5.7% (JK-NET). GAUG-O improves 4.1%, 2.1%, 6.3% and 4.9%, respectively.
- 符合预期
  - These results meet our expectation since …
- 猜想，推测，例外
  - For one of the exceptions (XXX), we **speculated** that the reason is …
  - 结果有好、有持平，如何分析
    - ![e8a7e2a160c6d212acb744cf5716f9c](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200711220058.png)
- 参数敏感性
  - is roughly not sensitive to
  - is not strictly sensitive to different hyperparameter settings.

- 性能
  - 泛化能力弱
    - XXX generalizes poorly off … ： 不能很好的概括

### 6.3 模型、方法评价/比较

- XXX always **produces a significant boost in** predictive performance.
- XXX model **performs favorably well on the task of** …
- XXX achieving **promising** results on tasks such as
- model1 **achieves competitive performance against** model2, and **significantly outperforms** model3.
- We **empirically compare** our models **against** four strong baselines summarized in Table, which were designed specifically for XXX task.
- This complexity is on par with the baseline methods such as …
- We show that **this method obtains competitive results compared to state-of-the-art algorithms**.
- **Due to the large literature, we could not compare to every** graph kernel, **but to some** classical ones and those closely related to our approach.
- and ran DGCNN **using exactly the same folds as used in** graph kernels **in all the 100 runs of each dataset.**





## 七、Acknowledgement

- 感谢审稿人
  - and gracious reviewers for constructive feedback on the paper.



## 八、Other

- 脚注
  - 符号特殊说明
    - [符号简化]：For notational convenience, we assume that
- 开源
  - Our implementation is made publicly available.



## 九、高级词汇

### 缩略词

- e.g.: 例如 (for example)
- i.e.: 即，也就是 (id est)
- aka: 又名，亦称 (also known as)
- w.r.t: with regard to/with reference to 关于
- N.B.：注意

### 动词

- 利用：leverage
- 生成：yield
- 优点：merit
- 缺点：drawback, shortcoming
- 包含：contain, comprise, consist of
- 应对，解决，操作：tackle, handle
- 缓解：mitigate
- 谋取，获取：enlist
- 实现：achieve => accomplish

### 形容词、副词

- 在前面、提前：upfront
  - beforehand：adj.提前的；预先准备好的 adv.事先；预先
  - 上述的：aforementioned
- 普遍的：ubiquitous
- 具体地：concretely， specifically
- 令人满意地：satisfactorily
- underneath：背后，在。。后面（learn important patterns underneath the input graph）
- illuminating：启发式的

### 连接词、搭配

- Nevertheless：然而、不过、虽然如此
- whereas：然而、鉴于、反之
- albeit：尽管specifically

  - Albeit its prosperity：尽管取得了很大的成功
- moreover
- furthermore
- overall
- Notably
- Intuitively：直观的
- In this regard：在这点上，从这方面
- to our knowledge: 
- in terms of：依据；按照；在…方面；以…措词
- **To name a few：例如，举例**
  - for instance
- Armed with this insight：按照这样的直觉

### 固定搭配

- task agnostic：任务无关的
- have no analogs for：没有类似的
- with the formal format as follows：正式格式如
- data-intensive： 数据密集型
- the pros and cons of：优缺点
- categorize ... from three perspectives：分类
  - With this taxonomy：基于这种分类
- The defense makes a correct decision on an example x **if either** of the following applies:
- draw from: 来自
- meagre amount of：少量的，贫瘠的
- is compatible with：兼容
- attend to A over B：关注A而不是B
- in an approach reminiscent of：让人想起，回忆起，联想到
- XXX for brevity:可以用于缩写（we refer to as Graph Contrastive learning with Adaptive augmentation, GCA for brevity.）
- shed light on：阐明
- be skewed for: 倾斜，偏向于
- coincide with：等同于，一致





## 专著

### 摘要

- In this chapter, we

- We then show how we can leverage these

- One of the main contributions of this book chapter is to propose using

- can be treated as a trace







## 搭配

- 流行，发展
  - has gained significant popularity over the recent years since …
  - with the massive popularity of …

 