---
layout: article
title: Keras 搭建图神经网络常见问题总结
date: 2020-01-27 00:10:00 +0800
tags: [Summary, Keras, GNN]
categories: blog
pageview: true
key: Summary-of-keras-on-GNN
---

------

## 模型输入

### **Problem 1：keras使用稀疏矩阵输入进行训练**

稀疏矩阵一般用于表示那些数值为0的元素数目远大于非零元素数目的矩阵。图数据集一般都以稀疏矩阵表示其邻接矩阵。一般用普通的ndarray存储稀疏矩阵会造成很大的内存浪费，在python中可以使用scipy的sparse模块构建稀疏矩阵。

在用keras搭建神经网络的时候，使用稀疏矩阵作为神经网络的输入时，需要做一些处理才能使用sparse格式的数据。

#### 方法一：使用keras函数式API中的sparse参数实现

keras的Sequential顺序模型是不支持稀疏输入的，如果非要用Sequential模型，可以参考方法二。在使用函数式API模型时，Input层初始化时有一个sparse参数，用来指明要创建的占位符是否是稀疏的，如图：

<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gbawzwpj7ej30os0mywgi.jpg" alt="9adca6237dc9db8d1125ee984073e82.png" style="zoom:50%;" />

使用的过程中，设置sparse参数为True即可：

```python
G = Input(batch_shape=(None, None), name='A', sparse=True)
```



**注意：**这么使用有一个问题，就是**指定的batch_size无效**，不管设置多大的batch_size，训练的时候都是按照batch_size为1来进行。



#### 方法二：使用生成器方法实现

参考链接：

- [https://www.jianshu.com/p/a7dadd842f78](https://www.jianshu.com/p/a7dadd842f78)
- [https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue](https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue)

