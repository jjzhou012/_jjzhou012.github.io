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

- 推荐工具：[http://www.phrasebank.manchester.ac.uk/](http://www.phrasebank.manchester.ac.uk/)
- REF-N-WRITE：[https://www.ref-n-write.com/trial/academic-phrasebank/](https://www.ref-n-write.com/trial/academic-phrasebank/)

## 摘要

- We **present a novel XXX architecture capable of** learning a joint representation of both local graph structure and available node features for the multi-task learning of link prediction and node classification.



## 引言

- 缩小差距、解决问题
  - To bridge this gap
  - To solve the above problem
  - To tackle these challenges
  - Bridging this gap between ... and ...：缩小差距
- 引起关注
  - attract considerable attention
  - increasing attention is drawn to
- 转折
  - Despite the advances of；即使进步
- 提出idea
  - propose = take an effective approach = introduce
  - aim = the idea is 
  - effectively utilize
  - establish\introduce a framework
  - the application of … **to** …
  - … is employed here：被使用

## 综述

- Advances in this direction are often categorized as … (这方面的进展可以归纳为…)
- There has been a surge of algorithms that seek to …
- There are a limited number of existing reviews on the topic of …
- To tackle these challenges, tremendous effort has been made towards this area, resulting in a rich literature of related papers and methods. 
- one of the dominating paradigms is  



## 方法

> 方法部分涉及到we的时候，可用one can代替，即：
>
> we => one can
>
> we can’t => one cannot



- **[简化符号]** To simplify notations, we drop the subscript/superscript for the xxx, i.e. xxx.
- In order to overcome this limitation, we develop a XXX method, which is a simple yet effective approach to ...
- 使用



## 模型评价/比较

- XXX always **produces a significant boost in** predictive performance.
- XXX model **performs favorably well on the task of** …
- model1 **achieves competitive performance against** model2, and **significantly outperforms** model3.
- We **empirically compare** our models **against** four strong baselines summarized in Table, which were designed specifically for XXX task.
- This complexity is on par with the baseline methods such as …
- We show that **this method obtains competitive results compared to state-of-the-art algorithms**.



## 实验说明

- In this section, we **expound our protocol for the empirical evaluation of our models’ capability for** learning and generalization on the tasks of link prediction and semi-supervised node classification.
- **[data split]** we split the data into disjoint test, and validation, sets of 1,000, and 500, examples, respectively.
- **[data split]** Each dataset is separated into a training, testing set and validation set. The validation set contains 5% citation edges for hyperparameter optimization, the test set holds 10% citation edges to verify the performance, and the rest are used for training.
- In comparison to the baselines, we evaluate our model on the same data splits over 10 runs with XXX and report mean AUC/AP scores for link prediction and accuracy scores for node classification.
- **[parameter]** varying parameter XX from … to …
  - fixing a = 1 then varying b from 1 to 100 …
  - vary a in (1,2,3,4,5)
  - select a from (1,2,3,4,5)

### 图表

- [折线图中的大起大落](https://zhuanlan.zhihu.com/p/24372300)
- [折线图](http://blog.sina.com.cn/s/blog_5ac7f5100100soyp.html)
- [折线图趋势](http://www.joozone.com/ielts/12268.html)

- **[描述框架]** We introduce XXX, schematically depicted in Figure 1, capable of … for XXX task.
  - **We introduce** the Multi-Task Graph Autoencoder (MTGAE) architecture, **schematically depicted in Figure 1, capable of** learning a shared representation of latent node embeddings from local graph topology and available explicit node features **for** LPNC. 
- [图标题]
  - Schematic depiction of XXX architecture.
  - Illustration of the proposed XXX
  - Illustration of applying XXX

### 分析

- 结合图表分析
  - These results **provide positive answers** to our motivating questions …
  - **from the evolution results,** we can observe that
  - table 1 shows\reports ……, **indicating** that …
  - As shown\visualized in …
- 参数敏感性
  - is roughly not sensitive to
  - is not strictly sensitive to different hyperparameter settings.
- 猜想，推测，例外
  - For one of the exceptions (XXX), we **speculated** that the reason is …



## 工作的创新点

- our work ……, **whereas previous related methods** ……
- We provide **a comprehensive empirical evaluation of our models on** nine benchmark graph-structured datasets **and demonstrate significant improvement over related methods** for graph representation learning.



## 高级词汇

### 缩略词

- e.g.: 例如 (for example)
- i.e.: 即，也就是 (id est)
- aka: 又名，亦称 (also known as)
- w.r.t: with regard to/with reference to 关于

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

### 连接词

- Nevertheless：然而、不过、虽然如此
- whereas：然而、鉴于、反之
- albeit：尽管specifically

  - Albeit its prosperity：尽管取得了很大的成功
- moreover
- furthermore
- overall
- Notably
- Intuitively：直观的

### 固定搭配

- task agnostic：任务无关的
- with the formal format as follows：正式格式如下
- in terms of：依据；按照；在…方面；以…措词
- to our knowledge: 
- To bridge this gap：To solve the above question；
- data-intensive： 数据密集型
- the pros and cons of：优缺点
- categorize ... from three perspectives：分类
  - With this taxonomy：基于这种分类
- the emergence of：涌现，出现
- **To name a few：例如，举例**
  - for instance
- Driven by the academic goal for ... to ...：目标驱使
- XXX generalizes poorly off … ： 不能很好的概括
- with regard to： 关于
- The defense makes a correct decision on an example x **if either** of the following applies:
- draw from: 来自
- meagre amount of：少量的，贫瘠的
- is compatible with：兼容

