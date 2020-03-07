---
layout: article
title: 批量删除Github中的项目
date: 2020-03-07 00:10:00 +0800
tags: [Github]
categories: blog
pageview: true
key: delete-github-repos
---



## 批量删除github中的项目

- 将需要删除的项目按照`username\repos-name`的格式以一行一个存放于文本文件`repos.txt`中：

  ```latex
  username\NRLPapers
  username\KB2E
  username\Viterbi
  username\kdd2018
  ```

- 在[GitHub](https://github.com/settings/tokens/new)上申请具有删除`repos`权限的`token`：

  ![27374b0e9d2cddc15f9903361fcbad8.png](http://ww1.sinaimg.cn/large/005NduT8ly1gcln7wx8hqj30fa02jmx0.jpg)

- 在命令行中运行下面的命令：

  - Linux

    ```
    while read r;do curl -XDELETE -H 'Authorization: token xxx' "https://api.github.com/repos/$r ";done < repos.txt
    ```

  - Windows(PowerShell)

    ```
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    
    get-content repos.txt | ForEach-Object { Invoke-WebRequest -Uri https://api.github.com/repos/$_ -Method “DELETE” -Headers @{"Authorization"="token xxx"} }
    ```

    > ​	XXX = token

