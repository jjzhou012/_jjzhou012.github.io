---
layout: article
title: Summary：Paper Writting using Latex
date: 2020-01-08 00:11:00 +0800
tags: [Summary, Writting, Latex]
categories: blog
pageview: true
key: Summary-of-Paper-Writting-Using-Latex
---

------

## 写在前面

整理了一些Latex写作的技巧！



## 排版

- **文本对齐**

  - **两端对齐**

    一般用于摘要；

    ```latex
    % 开头加载如下包
    \usepackage{ragged2e}
    
    % 需要对齐的段落之上使用如下命令 `\justifying`
    % 例如：
    标题
    \justifying 
    段落一
    \justifying 
    ```

- **biography 作者间距**

  - 重定义the bibliography环境

    修改`IEEEtran.cls`: 打开`IEEEtran.cls`，找到
    `\def\@IEEEBIOskipN{4\baselineskip}% nominal value of the vskip above the biography` 把数字4改小就可以，这个数字可以是零，也可以是负数，这个根据自身情况去进行更改。

  - 在`\end{IEEEbiography}`和`\begin{IEEEbiography}`中间添加`\vspace{-10 mm}`。

    ```latex
    \end{IEEEbiography}
    \vspace{-100 mm}  % \vspace{-10ex}, \vspace{-10pt} 
    \begin{IEEEbiography}
    ```

    