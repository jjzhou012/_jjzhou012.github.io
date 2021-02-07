---
layout: article
title: Tutorials：安装pytorch-geometric
date: 2020-01-18 00:10:00 +0800
tags: [Tutorials, Pytorch, GNN]
categories: blog
pageview: true
key: Summary-of-install-pytorch-geo
---

------

## 一、Linux平台

> 参考官方文档：[https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### 1.1 版本关联

```
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

`${TORCH}`和`${CUDA}`用相关版本号替换。

- torch-1.4.0  |  cu101
- torch-1.6.0  |  cu102
- torch-1.7.0  |  cu110

### 1.2 安装

举例：

```shell
# 安装pytorch
conda install pytorch==1.4.0 torchvision==0.5.0

# 验证cuda和pytorch版本
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"

# 安装依赖 torch-1.4.0 + cuda 10.1
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-geometric

# 安装依赖 torch-1.6.0 + cuda 10.2
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-geometric

# 安装依赖 torch-1.7.0 + cuda 11.0
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric
```





### 二、Windows平台

举例：环境： py3.7 | cuda 10.2 | torch-1.6

### 2.1 安装cuda

### 2.2 安装pytorch

```shell
# 安装pytorch
conda install pytorch==1.6.0 torchvision==0.7.0
```

### 2.3 安装pytorch-geometric

下载相关whl文件：

`https://pytorch-geometric.com/whl/torch-1.6.0.html`

根据版本关联，修改网址末尾的版本号，下载对应不同版本的依赖：

- `torch_cluster-latest+cu102-cp37-cp37m-win_amd64.whl`
- `torch_scatter-latest+cu102-cp37-cp37m-win_amd64.whl`
- `torch_sparse-latest+cu102-cp37-cp37m-win_amd64.whl`
- `torch_spline_conv-latest+cu102-cp37-cp37m-win_amd64.whl`

安装whl文件

```shell
# 
pip install torch_cluster-latest+cu102-cp37-cp37m-win_amd64.whl
pip install torch_scatter-latest+cu102-cp37-cp37m-win_amd64.whl
pip install torch_sparse-latest+cu102-cp37-cp37m-win_amd64.whl
pip install torch_spline_conv-latest+cu102-cp37-cp37m-win_amd64.whl
```

安装pytorch-geometric

```
pip install pytorch-geometric
```

> 有时候最后一步安装不成功，可能是因为国内镜像源和代理的冲突，关闭代理试试。