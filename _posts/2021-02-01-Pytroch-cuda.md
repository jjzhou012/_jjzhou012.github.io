---
layout: article
title: Pytroch：cuda数据处理
date: 2021-02-01 00:01:00 +0800
tag: [Tutorials, Pytorch] 
categories: Tutorials
pageview: true
---



## 1.设置GPU

- 设置os端哪些GPU可见

  ```python
  import os
  GPU = '0,1,2'
  os.environ['CUDA_VISIBLE_DEVICES'] = GPU
  ```

- 基本命令

  ```python
  # 查看可用GPU
  torch.cuda.device_count()
  # 查看gpu算力，返回gpu最大和最小计算能力，是一个tuple
  torch.cuda.get_device_capability()
  # 设置默认哪一个gpu运行，int类型
  torch.cuda.set_device()
  ```

## 2.从CPU转移到GPU

- `tensor.to()`

    ```python
    #
    gpu_id = '0'
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    # or device = 'cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu'

    a = torch.tensor([1,2,3])
    b = a.to(device)
    a1.is_cuda
    b1.is_cuda
    # out: a: tensor([1, 2, 3]), 
    #      b: tensor([1, 2, 3], device='cuda:0')
    #      a: False
    #      b: True
    ```

> 注意：`.to()`不仅可以转移device，还可以修改数据类型，比如：`a.to(torch.double)`

- `tensor.cuda()`

  ```python
  c = torch.tensor([1,2,3]).cuda()
  # out: tensor([1, 2, 3], device='cuda:0')
  ```

- `tensor.type()`

  ```python
  dtype = torch.cuda.FloatTensor
  d = a.type(dtype)
  # out: tensor([1., 2., 3.], device='cuda:0')
  ```

  

## 3.使用GPU创建

- 传参`device=`

  ```python
  a2 = torch.ones(3,4, device=device)
  ```

- GPU tensor类型创建

  ```python
  a3 = torch.cuda.FloatTensor()
  ```

  



