---
layout: article
title: GNN 教程：DGL框架-消息和GCN的实现
key: GNN_tutorial_framework_dgl_gcn
tags: GNN
category: blog
pageview: true
date: 2019-07-07 20:00:00 +08:00
---
**此为原创文章，转载务必保留[出处](https://archwalker.github.io)**

## 引言

图神经网络的计算模式大致相似，节点的Embedding需要汇聚其邻接节点Embedding以更新，从线性代数的角度来看，这就是邻接矩阵和特征矩阵相乘。然而邻接矩阵通常都会很大，因此另一种计算方法是将邻居的Embedding传递到当前节点上，再进行更新。很多图并行框架都采用详细传递的机制进行运算(比如Google的Pregel)。而图神经网络框架DGL也采用了这样的思路。从本篇博文开始，我们DGL](https://docs.dgl.ai/index.html)做一个系统的介绍，我们主要关注他的设计，尤其是应对大规模图计算的设计。这篇文章将会介绍DGL的核心概念 — 消息传递机制，并且使用DGL框架实现GCN算法。


## DGL 核心 — 消息传递 

DGL 的核心为消息传递机制（message passing），主要分为消息函数 （message function）和汇聚函数（reduce function）。如下图所示：

![](http://ww3.sinaimg.cn/large/006tNc79ly1g4r9g38x1lj316n0ewgom.jpg)

- 消息函数（message function）：传递消息的目的是将节点计算时需要的信息传递给它，因此对每条边来说，每个源节点将会将自身的Embedding（e.src.data）和边的Embedding(edge.data)传递到目的节点；对于每个目的节点来说，它可能会受到多个源节点传过来的消息，它会将这些消息存储在"邮箱"中。

- 汇聚函数（reduce function）：汇聚函数的目的是根据邻居传过来的消息更新跟新自身节点Embedding，对每个节点来说，它先从邮箱（v.mailbox['m']）中汇聚消息函数所传递过来的消息（message），并清空邮箱（v.mailbox['m']）内消息；然后该节点结合汇聚后的结果和该节点原Embedding，更新节点Embedding。

下面我们以GCN的算法为例，详细说明消息传递的机制是如何work的。

## 用消息传递的方式实现GCN

### GCN 的线性代数表达

GCN 的逐层传播公式如下所示：

$$
H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
$$

从线性代数的角度，节点Embedding$H^{(l+1)}$的的更新方式为首先左乘邻接矩阵$\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$以汇聚邻居Embedding，再为新Embedding做一次线性变换(右乘$W^{(l)}$)，简而言之：每个节点拿到邻居节点信息汇聚到自身 embedding 上在进行一次变换。具体 GCN 内容介绍可参考之前的[博文](https://archwalker.github.io/blog/2019/06/01/GNN-Triplets-GCN.html)。

### 从消息传递的角度分析

上面的数学描述可以利用消息传递的机制实现为：

1. 在 GCN 中每个节点都有属于自己的表示 $h_i$;
2. 根据消息传递（message passing）的范式，每个节点将会收到来自邻居节点发送的Embedding；
3. 每个节点将会对来自邻居节点的 Embedding进行汇聚以得到中间表示 $\hat{h}_i$ ；
4. 对中间节点表示 $\hat{h}_i$ 进行线性变换，然后在利用非线性函数$f$进行计算：$h^{new}_u=f\left(W_u \hat{h}_u\right)$;
5. 利用新的节点表示 $h^{new}_u$ 对该节点的表示 $h_u$进行更新。

### 具体实现

step 1，引入相关包

```python
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

```

step 2，我们需要定义 GCN 的 message 函数和 reduce 函数， message 函数用于发送节点的Embedding，reduce 函数用来对收到的 Embedding 进行聚合。在这里，每个节点发送Embedding的时候不需要任何处理，所以可以通过内置的`copy_scr`实现，`out='m'`表示发送到目的节点后目的节点的mailbox用`m`来标识这个消息是源节点的Embedding。目的节点的reduce函数很简单，因为按照GCN的数学定义，邻接矩阵和特征矩阵相乘，以为这更新后的特征矩阵的每一行是原特征矩阵某几行相加的形式，"某几行"是由邻接矩阵选定的，即对应节点的邻居所在的行。因此目的节点reduce只需要通过`sum`将接受到的信息相加就可以了。

```python
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
```

step 3，我们定义一个应用于节点的 node UDF(user defined function)，即定义一个全连接层（fully-connected layer）来对中间节点表示 $\hat{h}_i$ 进行线性变换，然后在利用非线性函数$f$进行计算：$h^{new}_u=f\left(W_u \hat{h}_u\right)$。

```python
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}
```

step 4，我们定义 GCN 的Embedding更新层，以实现在所有节点上进行消息传递，并利用 NodeApplyModule 对节点信息进行计算更新。

```python
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
```

step 5，最后，我们定义了一个包含两个 GCN 层的图神经网络分类器。我们通过向该分类器输入特征大小为 1433 的训练样本，以获得该样本所属的类别编号，类别总共包含 7 类。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
net = Net()
print(net)
```

step 6，加载 cora 数据集，并进行数据预处理。
```python
from dgl.data import citation_graph as citegrh
def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    mask = th.ByteTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(g.selfloop_edges())
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask
```

step 7，训练 GCN 神经网络。
```python
import time
import numpy as np
g, features, labels, mask = load_cora_data()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(30):
    if epoch >=3:
        t0 = time.time()

    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))
```

## 后话

本篇博文介绍了如何利用图神经网络框架DGL编写GCN模型，接下来我们会介绍如何利用DGL实现GraphSAGE中的采样机制，以减少运算规模。

## Reference

1. [DGL Basics](https://docs.dgl.ai/tutorials/basics/2_basics.html)
2. [Graph Convolutional Network](https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html)
3. [PageRank with DGL Message Passing](https://docs.dgl.ai/tutorials/basics/3_pagerank.html)
4. [DGL 作者答疑！关于 DGL 你想知道的都在这里](https://mp.weixin.qq.com/s?__biz=MzI2MDE5MTQxNg==&mid=2649695390&idx=1&sn=ad628f54c97968d6fff55907c47cb77e)