---
layout: article
title: jupyterLab安装与配置
date: 2018-12-01 00:01:00 +0800
tag: [Tutorials] 
categories: Tutorials
pageview: true
---



## Linux安装jupyterLab

- ### 安装jupyterlab

  ```
  pip install jupyterlab
  ```

- ### 配置

  > 注意：以下命令是将 JupyterLab 安装在服务器上的，本机电脑安装的话并不需要。

  生成登陆密码：

  ```
  (env) wx@ubuntu:~$ ipython3
  
  In [1]: from notebook.auth import passwd                       
  
  In [2]: passwd()                                               
  Enter password: 
  Verify password: 
  Out[2]: 'sha1:a68b5838f88b:f6f9fb4340dd081a4d06a5f9f36f841ffa8ae0c6'
  ```

  生成配置文件

  ```
  jupyter lab --generate-config
  ```

  注意终端上会出现对应的配置文件的路径。找到下面这几行文件，删掉注释并修改成如下格式：

  ```
  c.NotebookApp.allow_root = True
  c.NotebookApp.ip = '0.0.0.0'
  c.NotebookApp.open_browser = False
  c.NotebookApp.password = u'sha1:a68b5838f88b:f6f9fb4340dd081a4d06a5f9f36f841ffa8ae0c6'
  ```

- ### 运行

  想要运行 jupyterlab 的话首先得需要进入到相对应的 python 虚拟运行环境中，之后在运行环境中输入：

  ```
  # nohup表示ssh终端断开后仍然运行
  # &表示允许后台运行
  nohup jupyter lab &
  ```

  运行成功之后在该网络上的随意一台电脑上输入 **ip:8888** 便可以访问到 jupyterlab 上。



## Windows安装与更新 jupyterLab

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



