---
layout: article
title: scipy 矩阵类型
date: 2020-01-09 00:13:00 +0800
tags: [Summary, Scipy]
categories: blog
pageview: true
key: Scipy-matrix-type
---



------
总结scipy中的多种类型的稀疏矩阵格式。

- `coo_matrix`: 
- `lil_matrix`: Row-based list of lists sparse matrix



### lil_matrix

```
class scipy.sparse.lil_matrix(arg1, shape=None, dtype=None, copy=False)
# 实例化方式
# 1. with a dense matrix or rank-2 ndarray D
lil_matrix(D)
# 2. with another sparse matrix S (equivalent to S.tolil())
lil_matrix(S)
# 3. to construct an empty matrix with shape (M, N) dtype is optional, defaulting to dtype=’d’.
lil_matrix((M,N), [dtype])
```

- LIL格式的优点：
  - 支持灵活的切片；
  - 增量式创建稀疏矩阵的结构，适合逐个添加数据；
- 缺点：
  - LIL格式之间加法运算效率低；（考虑CSR或CSC格式）
  - 列切片慢；（考虑CSC格式）
  - 矩阵向量乘法效率低；（考虑CSR和CSC格式）

- 数据结构：

  LIL使用两个列表保存非零元素：

  - `self.rows`：列表形式保存行向量索引，`[list(row1(ind)), list(row2(ind)), …]`，每一行保存非零元素的列索引；
  - `self.data`：列表形式保存行向量数据，`[list(row1(data)), list(row2(data)), …]`，每一行保存非零元素的数据；

  示例：

  ```
  lil = sp.lil_matrix((4, 5))
  lil[0, 0] = 0.0
  lil[1, 2] = 2.0
  lil[1, 3] = 3.0
  lil[3, 4] = 4.0
  
  print(lil.rows)    # [list([]) list([2, 3])     list([]) list([4])]
  print(lil.data)    # [list([]) list([2.0, 3.0]) list([]) list([4.0])]
  ```






— 待补充