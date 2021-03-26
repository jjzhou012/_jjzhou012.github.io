---
layout: article
title: Pytorch：交叉熵损失函数CrossEntropyLoss实现
date: 2021-03-12 00:11:00 +0800
tags: [Pytorch]
categories: blog
pageview: true
key: Pytorch-crossentropyloss
---











## nn.CrossEntropy

![image-20210311164203219](https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20210311164203.png)



按官方文档定义所述，pytorch中实现的交叉熵损失函数，应该是`nn.LogSoftmax`和`nn.NLLLoss`的结合。

首先测试`nn.CrossEntropy`:

```python
import torch
import torch.nn as nn
 
input = torch.Tensor([[1,2,3]])    #(1,3)
target = torch.Tensor([2]).long()

ce = nn.CrossEntropyLoss()

# 测试CrossEntropyLoss
ce_loss = ce(input, target)
print(ce_loss)
# return tensor(0.4076)
```



测试`nn.LogSoftmax`加`nn.NLLLoss`：

```python
logsoftmax = nn.LogSoftmax()
nll = nn.NLLLoss()

# 测试LogSoftmax + NLLLoss
lsm_loss = logsoftmax(input)
nll_lsm_a = nll(lsm_loss, target)
# return tensor(0.4076)
```

直接用 `nn.CrossEntropy` 和`nn.LogSoftmax`+`nn.NLLLoss`是一样的结果。



> `nn.NLLLoss`的作用是**取出`input`中对应`target`位置的值并取负号**。



回顾交叉熵的表达式

