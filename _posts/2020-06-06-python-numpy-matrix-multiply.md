---
layout: article
title: Numpy使用技巧
date: 2020-06-06 00:10:00 +0800
tags: [Python, Numpy]
categories: blog
pageview: true
key: python-numpy-using
---



------

## 一、乘法符号的区别

| 数据类型      | $$*$$        | @        | np.dot   | np.multiply  |
| ------------- | ------------ | -------- | -------- | ------------ |
| numpy.ndarray | 对应元素乘法 | 矩阵乘法 | 矩阵乘法 | 对应元素乘法 |
| numpy.matrix  | 矩阵乘法     | 矩阵乘法 | 矩阵乘法 | 对应元素乘法 |



## 二、除法与数据类型

- `ndarray`/`ndarray`

  行操作

  - 数组/向量

    数组每一行/向量

  - 数组/数组

    对应行相除

  - 向量/向量

    对应元素相除

  ```python
  c = np.array([(1,1,1), (2,2,2),(3,3,3)])
  d = c.sum(axis=0)
  # c:  array([[1, 1, 1],
  #            [2, 2, 2],
  #            [3, 3, 3]]),
  #d:    array([6, 6, 6])
  
  c / d
  # array([[0.16666667, 0.16666667, 0.16666667],
  #        [0.33333333, 0.33333333, 0.33333333],
  #        [0.5       , 0.5       , 0.5       ]])
  c/c.T
  # array([[1.        , 0.5       , 0.33333333],
  #        [2.        , 1.        , 0.66666667],
  #        [3.        , 1.5       , 1.        ]])
  c[0]/d.T
  # array([0.16666667, 0.16666667, 0.16666667])
  ```

- 涉及`np.matrix`
  - `np.matrix`/`np.ndarray`
    - 矩阵每一行/向量，行操作
    - 矩阵每一行/数组，行相除
    - 行矩阵/数组每一行，行操作
    - 列矩阵/数组每一列，列操作
  - `np.matrix`/`np.matrix`
    - 矩阵每一行/行矩阵，行相除
    - 矩阵每一列/列矩阵，列相除
    - 行矩阵/矩阵每一行，行相除
    - 列矩阵/矩阵每一列，列相除
    - 矩阵每一行/矩阵每一行，行相除
  - `np.matrix`/`np.matrix`
    - 行矩阵/列矩阵 (1,N) / (N,1) ==>  (N,N)
      - 





## 三、采样

- `random.sample()` or `numpy.random.sample()`

  ```python
  random.sample(population, k)
  ```

  - 优点：
    - 可**指定抽样的个数**；
    - 采样**列表的维数没有限制**
    - 默认**不重复抽样**（不放回的抽样）
  - 缺点：
    - 当N的值比较大的时候，该方法执行速度很慢，替代方案为`np.random.choices`
    - 无法设置采样权重

- `np.random.choice()`

  ```python
  numpy.choice(a, size=None, replace=True, p=None)
  '''
  replace 代表的意思是抽样之后还放不放回去，如果是False的话，那么通一次挑选出来的数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。
  '''
  ```

  - 优点：
    - 效率高
    - 可**指定抽样的个数**
    - 可不重复抽样
    - 可设置采样权重
  - 缺点：
    - 抽样对象有要求，必须是整数或者一维数组（列表），**不能对超过一维的数据进行抽样**；
    - 默认是可以重复抽样，要想不重复地抽样，需要**设置replace参数为False**

- `random.choices()`

  ```python
  random.choices(population, weights=None, *, cum_weights=None, k=1)
  # Return a k sized list of population elements chosen with replacement
  ```

  - 重复采样

- `random.choice()`

  ```python
  random.choice(seq):
  # Choose a random element from a non-empty sequence.
  ```

  - 采样一个样本