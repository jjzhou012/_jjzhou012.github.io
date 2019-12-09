---
layout: article
title: jupyterLab安装与配置
date: 2018-12-01 00:01:00 +0800
tag: [Config] 
categories: Tutorials
pageview: true
---


## 安装与更新 jupyterLab

- 安装jupyterLab

```
conda install -c conda-forge jupyterlab
```

- 更新jupyterlab

```
pip install -U jupyterlab
```



## 安装nodejs

- 安装nodejs

```
conda install -c conda-forge nodejs
```

- 安装扩展

```
jupyter labextension install my-extension
```

- 列出扩展

```
jupyter labextension list
```

- 卸载扩展

```
jupyter labextension uninstall my-extension
```

- 启用扩展

```
jupyter labextension enable my-extension
```

- 禁用扩展

```
jupyter labextension disable my-extension
```



## 插件

- 目录

```
jupyter labextension install @jupyterlab/toc
```

- tensorboard

```
jupyter labextension install jupyterlab_tensorboard
```

- html

```
jupyter labextension install @mflevine/jupyterlab_html
```

- latex

```
jupyter labextension install @jupyterlab/latex
```

- go-to-definition

```
jupyter labextension install @krassowski/jupyterlab_go_to_definition
```

- voyager

```
jupyter labextension install jupyterlab_voyager
```

- statusbar

```
jupyter labextension install @jupyterlab/statusbar
```

- celltags

```
jupyter labextension install @jupyterlab/celltags
```

- drawio

```
jupyter labextension install jupyterlab-drawio
```

