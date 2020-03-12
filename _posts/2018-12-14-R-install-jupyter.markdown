---
layout: article
title: R语言安装与关联jupyter
date: 2018-12-14 00:01:00 +0800
tag: [Tutorials] 
categories: Tutorials
pageview: true
---



- 安装R

  版本：[Microsoft R Open 3.5.1](https://mran.revolutionanalytics.com/download)

- cmd进入R会话



- 输入如下配置命令;

  ```R
  install.packages("devtools")
  devtools::install_github("IRkernel/IRkernel")
  IRkernel::installspec()
  ```

  


