---
layout: article
title: Pytorch安装
date: 2020-03-23 00:10:00 +0800
tags: [Tutorials]
categories: blog
pageview: true
key: install-pytorch
---

## 镜像配置

- 清华镜像源：[https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/)

  ```
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  conda config --set show_channel_urls yes
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  ```

## 环境依赖

我自己主要用的几个环境依赖

| pytorch | python      | cuda     | tensorflow |
| ------- | ----------- | -------- | ---------- |
| 1.1.0   | 3.6/3.7     | 9.0      | 1.14       |
| 1.4.0   | 3.6/3.7/3.8 | 9.2/10.1 | 1.14       |
| 1.0.0   | 3.6/3.7     | 8.0      | 1.14       |

```
# cuda9.0 / py37 / tf1.14 / pytorch 1.1
conda create -n tf14 python=3.7 tensorflow-gpu=1.14 cudatoolkit=9.0 pytorch=1.1.0 torchvision=0.3.0

```



## 在线安装

- windows

  ```
  # cuda 9.0
  conda install pytorch=1.1.0 torchvision cudatoolkit=9.0
  ```

  > 不用加-c pytorch，因为已经配置了镜像，默认用国内镜像 

- linux

  ```
  # cuda 8.0
  conda install pytorch==1.0.0 torchvision==0.2.1 cuda80
  # cuda 9.0
  conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0
  ```

  > 早期版本参考： [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)



## 测试

```
import torch
print(torch.cuda.is_available())   	 # 返回结果是True，则PyTorch的GPU安装成功
print(torch.__version__) 			# 版本
print(torch.cuda.device_count()) 	 # 返回gpu数量
```





## 相关问题

- [Anconda安装的pytorch依赖的cuda版本和系统cuda版本不一致问题](https://blog.csdn.net/qq_38156052/article/details/103663759)

- [Pytorch 使用不同版本的 cuda](https://www.cnblogs.com/yhjoker/p/10972795.html)

