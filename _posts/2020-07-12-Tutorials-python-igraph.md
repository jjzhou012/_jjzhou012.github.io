---
layout: article
title: Tutorials：安装python-igraph包
date: 2020-07-12 00:21:00 +0800
tags: [Python, Tutorials]
categories: blog
pageview: true
key: install-python-igraph-package
---


## 安装python igraph包



## 一、Windows平台

> Christoph提供了一个非官方的python模块的库[点击进入该网址](http://www.lfd.uci.edu/~gohlke/pythonlibs)。在他的网站中提供了很多pypi里面没有提供的解决方案。（很多pypi里面的模块可能在安装的时候可能会报错，对于这些模块不如上他的网站上找找）

### 1.1 安装python-igraph

根据自己的虚拟环境，`python 3.7` 64位安装包：

```shell
pip install python_igraph‑0.7.1.post6‑cp37‑cp37m‑win_amd64.whl
```

### 1.2 安装Pycairo

python-igraph在绘图的时候依赖pycairo模块（如果你不绘图，不用安装），这个非官方库也提供了对于的whl文件.

根据自己的虚拟环境，`python 3.7` 64位安装包：

```shell
pip install pycairo‑1.19.1‑cp37‑cp37m‑win_amd64.whl
```





## 二、Linux平台

### 2.1 Ubuntu

- 安装编译igraph所需要的库，下载libigraph0-dev

  ```shell
  sudo apt-get install -y libigraph0-dev
  ```

- 安装pkg-config

  在官网可以了解到，从版本0.5开始，igraph库的C core不包含在Python发行版中 ，必须单独编译和安装C core。在ubuntu中，需要下载`build-essential` 和 `python-dev 编译C core。`

  ```shell
  sudo apt-get install build-essential
  sudo apt-get install python-dev
  ```

- 如果igraph的C core已经安装到可以使用pkg-config检测到的位置，则pip将编译并将扩展链接到已安装的C core，实际上这时候我没有安装pkg-config，所以无法编译c core，所以其实只需要安装pkg-config即可

  ```shell
  sudo apt install pkg-config
  ```

- 安装python-igraph

  ```shell
  pip install python-igraph==0.7.1.post6
  ```

  

