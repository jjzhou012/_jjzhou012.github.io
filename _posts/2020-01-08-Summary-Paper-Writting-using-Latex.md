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



## 一、排版

### 1.1 **文本对齐**

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



### 1.2 图片/表格与上下文

```latex
\vspace{-0.8cm}   
\begin{figure}

\vspace{-0.8cm}   % ACM放前面
\end{figure}
\vspace{-0.8cm}   % 其他放后面
```

上面的减小间距对双栏的论文可以。如果在双栏的情况下，图片是一栏的情况，则要把上面`\vspace{***cm}`放在`\end{figure*}`这句之前。

```latex
% 图片与标题
\begin{figure}
\centering
  \includegraphics[width=\textwidth]{}
  \setlength{\abovecaptionskip}{-6pt}      % 调整图片标题与图距离
  \caption{}
  \setlength{\belowcaptionskip}{-1cm} %调整图片标题与下文距离
\end{figure}
```



### 1.3 公式与上下文

```latex
% example 1
\begin{equation}
\setlength{\abovedisplayskip}{3pt}
\setlength{\belowdisplayskip}{3pt}
y(t)=a(t)-b(t).
\end{equation}

% example 2
$$
\setlength{\abovedisplayskip}{3pt}
\setlength{\belowdisplayskip}{3pt}
y(t)=a(t)-b(t).
$$

% example 3
\begin{eqnarray}
\setlength{\abovedisplayskip}{3pt}
\setlength{\belowdisplayskip}{3pt}
y(t)=a(t)-b(t).
\end{eqnarray}

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

    

## 二、 公式设置

### 2.1 上下标设置

在内联公式中，上下标一般出现在目标的右上方/右下方；而在行间公式中，上下标一般出现在目标的正上方/正下方；

我们可以强制用`\limits`和`\nolimits`来控制上下标的出现位置。

比如内联公式中一般呈现为$$\sum_{i=0}^n {x_i}$$ (`\sum_{i=0}^n {x_i}`);

加入强制约束后变为$$\sum\limits_{i=0}^n {x_i}$$ (`\sum\limits_{i=0}^n {x_i}`);

```latex
% \limits 强制约束上下标位于 正上方/正下方
$\sum\limits_{i=0}^n {x_i}$
```

又比如行间公式中一般呈现为：

$$
\sum_{i=0}^n {x_i}
$$

加入强制约束`\nolimits`后变为：

$$
\sum\nolimits_{i=0}^n {x_i}
$$

```latex
% \nolimits 强制约束上下标位于 右上方/右下方
$\sum\nolimits_{i=0}^n {x_i}$
```

此外，有时候我们需要上/下标出现在一段**非数学符号**的正上/下方，而直接用强制约束会报错，怎么办？

解决方法是用`\mathop{notation}`命令将notation转化成数学符号，写成

```latex
\mathop{expr1}\limits_{expr2}^{expr3}
```

这样就可以使用`\limits`命令了，例如命令

```latex
$f_3(d) = \mathop{max}\limits_{x_3}(2x_3 + f_4(d-x_3))$
```

$$
f_3(d) = \mathop{max}\limits_{x_3}(2x_3 + f_4(d-x_3))
$$
