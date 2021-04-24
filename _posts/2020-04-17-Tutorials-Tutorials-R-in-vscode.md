---
layout: article
title: Tutorials：在VSCode中配置R语言运行环境
date: 2020-04-17 00:10:00 +0800
tags: [Tutorials, R]
categories: blog
pageview: true
key: Tutorials-R-in-vscode
---



> ​	参考文献：
>
> - https://sspai.com/post/47386

## 



## 插件安装

- ### [R support for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=Ikuyadeu.r)

- ### [R LSP Client](https://marketplace.visualstudio.com/items?itemName=REditorSupport.r-lsp)

  LSP 是Language Server Protocol 的缩写。简单来说，LSP 为不同语言在不同编辑器或IDE中的自动补全、查找定义、悬停查看函数文档等功能搭建了桥梁，使得开发者可以减少针对不同语言和不同编辑器的重复开发。

  插件只是在编辑器一侧提供了实现LSP 的条件，而在R 语言一侧还需要另外的包——`languageserver`——来完成搭接。在R 环境中运行如下安装指令：

  ```r
  install.packages("languageserver")
  ```

  然后重启一下VSCode，整个LSP 的功能就可以实现了。

- ### [rtichoke](https://github.com/randy3k/rtichoke)（现更名为radian）

  ```
  pip install -U rtichoke
  ```

  等待安装进程结束后，就可以在终端中直接使用命令：

  ```shell
  rtichoke
  ```

  开始一个多彩的R 进程。

