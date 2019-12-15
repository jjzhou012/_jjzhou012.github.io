---
layout: article
title: GNN 教程：GNN 模型有多强？
key: GNN_tutorial_theory_power
tags: GNN
category: blog
pageview: true
date: 2019-06-22 12:00:00 +08:00
---
## 引言

**此为原创文章，未经许可，禁止转载**

前面的文章中，我们介绍了GNN的三个基本模型GCN、GraphSAGE、GAT，分析了经典的GCN逐层传播公式是如何由谱图卷积推导而来的。GNN模型现在正成为学术研究的热点话题，那么我们不经想问，GNN模型到底有多强呢？之前的[文章](https://archwalker.github.io/blog/2019/06/22/GNN-Theory-WL.html)我们介绍了用来衡量GNN表达能力的算法—Weisfeiler-Leman，这篇文章我们将以该算法为基础，向大家介绍ICLR 2019的oral论文 [How powerful are graph neural networks](http://arxiv.org/abs/1810.00826) 。

## 图神经网络 Graph Neural Network

GNNs 利用图结构和节点初始特征$X_0$学习图节点的表示(embeddings)$h_v$，或者整个图的表示$h_G$。前面我们介绍了GCN、GraphSAGE和GAT这三种模型，图内节点都是通过各种聚合邻居的策略(neighborhood aggregation strategy)迭代式更新的，在$K$步迭代之后，每个节点的Embedding都融合了它$k$ hop所有邻居的信息。转化成数学形式：

$$
a_{v}^{(k)}=\text { AGGREGATE }^{(k)}\left(\left\{h_{u}^{(k-1)} : u \in \mathcal{N}(v)\right\}\right)
$$

$$
h_{v}^{(k)}=\operatorname{COMBINE}^{(k)}\left(h_{v}^{(k-1)}, a_{v}^{(k)}\right)
$$

$h_v^{(k)}$ 即为$k$步迭代之后节点$v$的Embedding。$$\text{AGGREGATE}$$ 以及 $$\text{COMBINE}$$ 的不同区分了不同的图神经网络模型。比如在使用max-pooling聚合的GraphSAGE模型中，$$\text{AGGREGATE}$$ 具有如下形式：

$$
a_{v}^{(k)}=\operatorname{MAX}\left(\left\{\operatorname{ReL} \mathrm{U}\left(W \cdot h_{u}^{(k-1)}\right), \forall u \in \mathcal{N}(v)\right\}\right)
$$

$$\text{COMBINE}$$ 的形式为

$$
h_{v}^{(k)}=\operatorname{CONCAT}\left(h_{v}^{(k-1)}, a_{v}^{(k)}\right)
$$


而在GCN中，$$\text{AGGREGATE}$$ 和 $$\text{COMBINE}$$ 被组合到了一起：

$$
h_{v}^{(k)}=\operatorname{ReLU}\left(W \cdot \operatorname{MEAN}\left\{h_{u}^{(k-1)}, \forall u \in \mathcal{N}(v) \cup\{v\}\right\}\right)
$$

具体细节可以参考我们之前的博文[GCN](https://archwalker.github.io/blog/2019/06/01/GNN-Triplets-GCN.html) [GraphSAGE](https://archwalker.github.io/blog/2019/06/01/GNN-Triplets-GraphSAGE.html)，在此不在赘述。

对于节点分类的问题，学习到$h_v^{(k)}$就可以接到下游的分类器了。而另一种图机器学习任务，即对于整个图的分类，我们需要融合节点Embedding以表示出整个图的Embedding，再接到下游分类器。具体来说，通过设计一个读出器函数$$\text{READOUT}$$, 我们聚合节点Embedding而求出整个图的Embedding：

$$
h_{G}=\operatorname{READOUT}\left(\left\{h_{v}^{(N)} \vert v \in G\right\}\right)
$$

## 理论分析

对于GNN中的每个节点来说，他们都是通过递归的融合邻居信息来捕获图的结构信息和邻居特征信息，因此每个节点的更新路径是一个树结构，节点位于树根，从叶子节点逐层向上更新Embedding直到根节点。比如在下图中，对于一个两层的图神经网络，节点$B$ 的更新路径是一个高为2的数，$B$ 位于树根。更新方向如下箭头方向所示：

![Screen Shot 2019-06-24 at 8.06.32 PM](http://ww2.sinaimg.cn/large/006tNc79ly1g4cjiq909tj31ao0q0wkg.jpg)

为了分析GNN的表示能力，我们转而研究什么样的GNN结构能够保证将两个不同的节点投影到不同的Embedding空间中(即为它们生成不同的Embedding)。直觉上可知，最强大的GNN结构仅会将拥有完全相同子树结构的两个节点投影到相同的Embedding空间(即这两个节点的Embedding相同，他们的邻居的Embedding相同，数量也相同)。因为子树结构可以通过节点递归得定义得到(如上图中$B$的二度子树结构可以定义为$B$的一度子树结构以及一度子树$(A, C, E)$的1度子树结构)，因此我们的分析可以简化为"最强大的GNN结构仅会将拥有完全相同1度邻域的两个节点投影到相同的Embedding空间"。这里1度领域不仅包括节点的1度邻居，也包括节点自身。即为以节点为根，高为1的子树结构。

自此，我们的分析框架就建立了，那么下一步是将节点的一度邻居表示出来，由于图上节点邻居没有相对的次序性，因此MultiSet这个数据结构最适合表示这样的领域。Multiset和set不同，Multiset允许集合中的元素出现多次。**Multiset是一个2元组 $X=(S, m)$, 其中$S$是集合中的元素，$m$表示该元素出现的次数**。

在此做一个小结，GNN的表示能力可以这样分析：最强大的GNN能够将两个节点投影到不同的Embedding空间(为这两个节点生成不同的Embedding向量)，除非这两个节点Embedding和1度邻居完全相同。节点的1度邻居可以用Multiset表示，因此最强大的GNN一定能够将两个不同的Multiset映射到不同的Embedding空间。所以最终问题就转化成了设计这样的GNN函数，使得
$$
GNN(v_1, Multiset_1) = GNN(v_2, Multiset_2)
$$
成立当且仅当  $h_1 = h_2$ and $Multiset_1 = Multiset_2$，$h$ 为节点$v$的Embedding表示，multiset中包含节点的1度邻居信息，包括邻居的数量和Embedding。

## GIN 和Weisfeiler-Leman算法一样强大

有了这些理论分析后，要想设计一个强大的GNN模型，我们要做到是设计$$\text{AGGREGATE}$$、$$\text{COMBINE}$$ 以及 $$\text{READOUT}$$ 函数使得经过这三个函数映射后不同的multiset能够保持不同，即这些函数都是单射函数(injective function)。为此，作者证明了几个定理：

首先是为设计单射性质的 $$\text{AGGREGATE}$$ 和 $$\text{COMBINE}$$ 而证明的：

**定理5. ** 存在函数 $f: \mathcal{X}\rightarrow\mathbb{R}^n$ 使得 $h(X)=\sum_{x \in X} f(x)$ 对于每个multiset $X\in \mathcal{X}$ 是不同的。再者，任何作用于multiset上的函数 $g$ 能够被如下的形式分解: $g(X)=\phi\left(\sum_{x \in X} f(x)\right)$

具体证明详见论文，在这里通俗的说下这个定理的目的是什么。这个定理说，对于任意multiset $X$, 都有一个对应的单射函数$g(X)$，这就意味着如果节点的邻居有差别，那么我们可以通过 $g(X)$ 这个函数将它们区分出来，回忆在[Weisfeiler-Leman算法](https://archwalker.github.io/blog/2019/06/22/GNN-Theory-WL.html)的例子中，函数$g$ 将邻居Embedding排序之后拼起来，虽然简单，但是满足单射的要求。

上文中，我们将节点自身归到节点的1度领域中，GCN和GAT中就是这样表示的。在GraphSAGE中，节点自身和其1度邻居是分开对待的，为此，作者证明了如下引理：

**引理6.** 存在函数 $$f:\mathcal{X}\rightarrow\mathbb{R}^n$$ 使得存在数 $$\epsilon$$，函数$$h(c, X) = (1 + \epsilon)\cdot f(c) + \sum_{x\in X} f(x)$$ 对于每一对 $(c, X)$ 是单射的，其中 $c\in \mathcal{X}$ 和 $X\subset \mathcal{X}$ 是一个有限的multiset，再者，任何作用于$(c, X)$ 这样的multiset对的函数 $g$ 能够被分解成  $g(c, X)=\phi\left((1+\epsilon) \cdot f(c)+\sum_{x \in X} f(x)\right)$ 这样的形式，其中 $$\phi$$ 是一个函数。

通俗来说，对于任意multiset对$(c, X)$，都有一个对应的单射函数 $g(c, X)$，这里如果把 $c$ 看成节点自身，$X$ 看成是节点的1度领域，那么意味着如果节点自身或者其邻居有差别，我们可以通过 $g(c, X)$ 把它们区分出来，回忆在[Weisfeiler-Leman算法](https://archwalker.github.io/blog/2019/06/22/GNN-Theory-WL.html)的例子中，函数 $g$ 的定义是，将节点自身和其1度邻居的拼接向量再拼接起来并用","分开，同样简单有效。

**引理6** 从数学的角度上证明了如果GNN能够学习到$g(c, X)$，GNN能够达到Weisfeiler-Leman的分类能力。那么我们面临的问题是如何设计GNN的结构以学习到这样的$g(c, X)$，观察$g(c, X)$ 的形式可知，它主要是将复合函数$f\circ g$ 作用在节点Embedding上，而根据神经网络的**通用近似理论**，我们可以将复合函数 $f\circ \phi$ 建模为一个多层感知机MLP，综合以上推导，作者得出了一个和Weisfeiler-Leman具有相同能力的GNN模型，叫做 GIN (Graph Isomorphism Network)：
$$
h_{v}^{(k)}=\operatorname{MLP}^{(k)}\left(\left(1+\epsilon^{(k)}\right) \cdot h_{v}^{(k-1)}+\sum_{u \in \mathcal{N}(v)} h_{u}^{(k-1)}\right)
$$

小结一下：这个式子设计了合理的 $$\text{AGGREGATE}$$ 和 $$\text{COMBINE}$$ ，使他们成为单射函数，对于图中节点的分类任务，这个式子已经和 Weisfeiler-Leman 算法能力一致了，对于图的分类任务，我们还要设计 $$\text{READOUT}$$ 单射函数，使得所有的节点Embedding综合而成的图Embedding仍然是唯一的。$$\text{READOUT}$$ 函数的设计博文中不再赘述，详情见论文。

## GIN模型和其他图模型的比较

上面我们介绍了这么多，得出了GIN的逐层embedding更新公式，我们说他和Weisfeiler-Leman的能力相同，然而之前提到的图模型比如GraphSAGE和GCN却没有这样强大的能力，下面我们从他们的逐层更新公式上举例子来看看原因：它们的逐层更新公式区别主要有三点

- GIN 采用了MLP来对节点聚合后的Embedding做非线性隐射，而GCN和GraphSAGE采用的单层感知机的结构(一个权重矩阵$W$加上非线性激活函数$$\sigma$$)。然而只有MLP才具有通用近似的能力(MLP 能够拟合 $f\circ g$)，根据上面的理论分析，使用单层感知机的结构会导致逐层更新公式不再是单射函数，导致对于不同的节点和邻居，模型有分辨不出来的风险；
- GIN 使用 $(1+\epsilon)$  对自身的Embedding进行了加权处理，通过引入对自身Embedding的扰动避免了自身节点Embedding $a$和邻居聚合后Embedding $b$加和相同的情况下， 如果$a, b$ 互换，那么无法分辨的问题。
- GIN 使用加和(sum)作为邻居Embedding的聚合方式，sum 比 mean 或者 pooling 的方式更强，论文中举了几个例子帮助理解

![Screen Shot 2019-06-24 at 6.39.40 PM](http://ww1.sinaimg.cn/large/006tNc79ly1g4cjizlte0j30yw0b4go2.jpg)

假设我们关注的都是位于图中间蓝色节点一次更新后的Embedding，因为该节点的邻居个数在左右图不同 ，因此节点Embedding更新后的结果应当要有差异。对(a)，通过mean或者max后的邻居信息均是1个蓝色节点代表的Embedding，不能分辨邻居有差异；对(b)，max聚合后的邻居信息是绿色和红色节点Embedding的最大值，不能分辨邻居有差异；对(c)，mean聚合后是绿色和红色节点代表Embedding的均值，不能分辨邻居有差异；max聚合后是绿色和红色节点代表Embedding的最大值，同样不能分辨邻居差异。

而如果采用sum的方式，以上3中情况邻居聚合后的差异都能分辨出来，因此sum 比 mean 或者 max-pooling 的表达能力更强，至少在图同构测试或者节点分类的任务上。



## 关于表达能力的讨论

谈到这里，有些对图神经网络比较熟悉的读者可能有疑惑了，在GraphSAGE的论文中，作者比较了各种聚合方式(mean, max-pooling等)，实验结论是mean聚合方式在在实验结果上比较好啊，这不是与该篇论文的分析相悖吗？确实，笔者在多种数据集的实验中也发现mean聚合的效果比较好。作者在接下来的几个小节中解释了可能的原因。

### mean 学习邻居Embedding分布

考察这样两个multiset $X_1=(S, m)$， $X_2=(S, k\cdot m)$，即， $X_2$ 包含 multiset $X_1$ 中的所有元素，且每个元素的数量是 $X_1$ 的 $k$ 倍，那么mean aggregator是无法区分他们的，因此我们可以说，mean aggregator可以用来分辨multiset中元素的分布(比率)，而不是multiset。

因此 mean aggreator 在特定的任务可能表现得更好，比如在图上统计或者分布信息比特征信息更重要的时候。或者，考虑另一种情况，当节点Embedding极少重复的话，mean aggregator和sum aggregator的表达能力基本是一致的，在很多machine learning的任务中，由于节点的Embedding基本都存在差异，并且mean aggregator能够保持聚合后的每一维的scale和聚合前一致，因此在任务中可能会更好。

### max-pooling 学习具有表示性的Embedding

考察这样两个multiset $X_1=(S_1, m_1)$， $X_2 = (S_2, m_2)$ 其中$max(S_1) = max(S_2)$，即$X_1$中最大的元素和$X_2$ 相同，那么max-pooling aggregator 是无法区分他们的，因此我们可以说，max-pooling aggregator 可以用来分辨multiset具有代表性的元素或者multiset的“骨架”(skeleton)。在[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)中，作者阐述了max-pooling aggreator 可以用来分辨3D 点云的“骨架”，对噪音和离群点具有鲁棒性。max-pooling aggreator 可能在这类的任务中表现得更好。



## 后记

这篇博文重点介绍了GNN的表示能力，简单来说，可以通过设计特殊GNN的架构使得其达到Weisfeiler-Leman算法在图同构分类中的效果。到目前为止，我们介绍了图神经网络的三个基本模型，GCN、GraphSAGE、GAT；介绍了图卷积神经网络和谱图卷积的关系；介绍了图神经网络模型的表达能力；至此，图神经网络的理论部分暂时告一段落，接下来的博文将会向大家介绍目前图神经网络存在的框架，使大家对图神经网络编程有所了解，毕竟纸上谈兵终非长远之策。




















