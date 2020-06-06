---
layout: article
title: 对抗安全：深度模型覆盖率和鲁棒性的有限关联
date: 2020-05-27 00:10:00 +0800
tags: [DeepTest, Adversarial, CV]
categories: blog
pageview: true
key: There-is-Limited-Correlation-between-Coverage-and-Robustness-for-Deep-Neural-Networks
---

- Paper: [There is Limited Correlation between Coverage and Robustness for Deep Neural Networks](https://arxiv.org/pdf/1911.05904)
- Code: [https://github.com/ase2019/DRTest](https://github.com/ase2019/DRTest)



## 概要

论文的灵感来自于软件程序代码覆盖标准的成功。

我们期望，如果一个DNN模型根据这样的覆盖率指标经过良好的测试（重训练），它就可能是鲁棒的。

在这项工作中，作者进行了实证研究来评估覆盖率、鲁棒性和攻击/防御指标之间的关系，基于100个DNN模型和25个指标。

研究发现，覆盖率和鲁棒性之间的关联是有限的，提高覆盖率并不能提升鲁棒性。

研究如下，针对每个数据集，首先训练25个不同的种子模型(采用最先进的架构设计)，用不同的攻击方法对每个种子模型进行攻击，生成具有不同攻击参数的对抗样本，并用生成的对抗样本对训练数据集进行扩充，然后重训练模型。接着应用DRTest来计算每个模型的指标范围。然后使用一个标准的相关分析算法——肯德尔等级相关系数，来分析指标之间的相关性。



## 相关研究

### DNN测试

- Neuron Coverage

  神经元覆盖是第一个提出用于测试DNN的覆盖率标准，它通过测试套件中的至少一个测试样本来量化激活神经元的百分比。

- DeepGauge

- Surprise Adequacy

### DNN鲁棒性 





## 框架

<img src="https://raw.githubusercontent.com/jjzhou012/image/master/blogImg20200527103737.png" alt="56d30e20f5df2de76b2c3c48d6437a4" style="zoom:50%;" />



### 对抗攻击







