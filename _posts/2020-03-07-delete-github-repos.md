---
layout: article
title: Tutorials：Delete repositories in batches from github
date: 2020-03-07 00:10:00 +0800
tags: [Tutorials, Github]
categories: blog
pageview: true
key: delete-github-repos
---



## Delete repositories in batches from github

- Place these repositories that need to be deleted in the text file `repos.txt` as lines in the format of `username\repos-name` :

  ```txt
  username\NRLPapers
  username\KB2E
  username\Viterbi
  username\kdd2018
  ```

- Apply for `token` with `delete repos` permission in [github](https://github.com/settings/tokens/new)

  ![27374b0e9d2cddc15f9903361fcbad8.png](http://ww1.sinaimg.cn/large/005NduT8ly1gcln7wx8hqj30fa02jmx0.jpg)

- Run the following command in cmd：

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
  
- Finish, all selected repositories has been deleted!

