---
layout: article
title: Tutorials：优雅的配置Latex写作环境
date: 2020-04-04 00:10:00 +0800
tags: [Tutorials, Latex, VSCode]
categories: blog
pageview: true
key: Tutorials-Elegant-configuration-for-latex-writing
---

------



## 写在前面

最近有很多小伙伴来问我写论文用什么工具好，他们之中有师弟师妹，也有科研大佬，令我瑟瑟发抖。其实谷歌能够解决大多数问题，只是大家比较懒（我可不是真相帝）。

言归正传，他们吐槽在线的overleaf老是中断连接，本地的winedt又不好用。。。其实，我们首先需要考虑清楚我们的需求是什么，很简单，一个稳定的latex写作环境肯定是放在本地好；但是作为研究生，你写的论文肯定需要导师浏览、协同编辑修改，放在本地老板看不到；当然，最重要的是能备份。

明确了这三点，那我就推荐自己用起来比较舒服一套配置：`VSCode + MikTex + SumatraPDF + Github + Overleaf`，其中：

- `VSCode + MikTex + SumatraPDF`: 本地Latex写作环境
- `Github`: 存放论文项目（私有库）
- `Overleaf`: 在线latex编辑
- `VSCode -> Github -> Overleaf`: 实现论文本地到云端的同步

下面就介绍各个部分的配置，在此之前，我默认你已经安装并熟悉了VSCode，MikTex，拥有Github和Overleaf的账户。



## 本地配置

### VSCode配置

#### 在VSCode中安装扩展`Latex Workshop`

![ee8eb86ad5822f8e765c867fa16c3a8.png](http://ww1.sinaimg.cn/large/005NduT8ly1gdhvwzfw09j30ed05zt8v.jpg)

#### 配置编译方案

LaTeX Workshop 默认的编译工具是 latexmk，大家根据需要配置所需的编译工具（tool）和方案(recipe)。

打开VSCode设置文件，根据自身需求选择性的加入设置代码。

我主要用到以下几个编译工具：

```json
"latex-workshop.latex.tools": [
        {
          "name": "texify",
          "command": "texify",
          "args": [
            "--synctex",
            "--pdf",
            "--tex-option=\"-interaction=nonstopmode\"",
            "--tex-option=\"-file-line-error\"",
            "%DOC%.tex"
          ]
        },
        {
            // 编译工具和命令
            "name": "xelatex",
            "command": "xelatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOC%"
            ]
        },
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOC%"
            ]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
                "%DOCFILE%"
            ]
        }
      ],
```



接下来配置编译方案：

```json
"latex-workshop.latex.recipes": [
        {
          "name": "PDFLaTeX",
          "tools": [
            "pdflatex"
          ]
      	},
        {
          "name": "XeLaTeX",
          "tools": [
            "xelatex"
          ]
        },
        {
          "name": "latexmk",
          "tools": [
            "latexmk"
          ]
        },
        {
          "name": "BibTeX",
          "tools": [
            "bibtex"
          ]
        },
        {
          "name": "pdflatex -> bibtex -> pdflatex*2",
          "tools": [
            "pdflatex",
            "bibtex",
            "pdflatex",
            "pdflatex"
          ]
        },
        {
          "name": "xelatex -> bibtex -> xelatex*2",
          "tools": [
            "xelatex",
            "bibtex",
            "xelatex",
            "xelatex"
          ]
        }
    ],
```

配置好之后，latex插件的`COMMANDS`区就会出现配置好的构建方案`Build LaTeX project`:

![8bb1b5ec853d1e215dbf80383e2f707.png](http://ww1.sinaimg.cn/large/005NduT8ly1gdhwdb9kyoj30a107lt8o.jpg)

> 注意：
>
> - 将 tools 中的 `%DOC%`替换成`%DOCFILE%`就可以支持编译**中文路径下的文件了**；
>
> - 要使用 pdflatex，只需在 tex 文档首加入以下代码：
>
>   ```tex
>   %!TEX program = pdflatex
>   ```
>
> - 在文档的开头添加以下代码就可以自动处理 bib 了：
>
>   ```tex
>   %!BIB program = bibtex
>   ```

#### 配置外部pdf浏览器

Latex Workshop其实自带了pdf浏览选项：

![3588d8829ae4d6d1f53e1f04ae513fe.png](http://ww1.sinaimg.cn/large/005NduT8ly1gdhwkzlnafj30cl04l3yd.jpg)

- View in VSCode tab: VSCode内部查看
- View in web browser: 默认浏览器查看
- View in external viewer: 外部pdf浏览器查看

要使用 SumatraPDF 预览编译好的PDF文件，添加以下代码进入设置区：

```json
"latex-workshop.view.pdf.viewer": "external",
"latex-workshop.view.pdf.external.viewer.command": "D:/Sumatrapdf/SumatraPDF.exe",
"latex-workshop.view.pdf.external.viewer.args": [
    "-forward-search",
    "%TEX%",
    "%LINE%",
    "-reuse-instance",
    "-inverse-search",
    "\"D:\\Microsoft VS Code\\Code.exe\" \"D:\\Microsoft VS Code\\resources\\app\\out\\cli.js\" -gr \"%f\":\"%l\"",
    "%PDF%"
],
```

> 注意：`viewer.command`和 `viewer.args`需要根据自己电脑上 SumatraPDF 和 VSCode 的安装位置进行修改;

#### 配置正向反向搜索

```json
"latex-workshop.view.pdf.external.synctex.command": "D:/Sumatrapdf/SumatraPDF.exe",
"latex-workshop.view.pdf.external.synctex.args": [
    "-forward-search",
    "%TEX%",
    "%LINE%",
    "-reuse-instance",
    "-inverse-search",
    "code \"D:\\Microsoft VS Code\\resources\\app\\out\\cli.js\" -r -g \"%f:%l\"",
    "%PDF%"
],
```

> 注意：
>
> - `synctex.command`和 `synctex.args`需要根据自己电脑上 SumatraPDF 和 VSCode 的安装位置进行修改;
>
> - 如果不加双引号，在文件路径有空格的情况下会导致无法反向搜索

- 正向搜索

  配置好之后，在`COMMANDS`区单击`SyncTex from cursor`就可以进行正向搜索，在pdf中定位到tex文件光标指示的位置；

![134f971f2aaabf9493db3348904a5b9.png](http://ww1.sinaimg.cn/large/005NduT8ly1gdhysypfmxj30ce03qglh.jpg)

- 反向搜索

  在PDF中双击即可反向搜索；



#### 其他设置

- 如果编译出错，插件会弹出气泡提示，不喜欢的话可以在设置中添加以下代码：

  ```json
  "latex-workshop.message.error.show": false,
  "latex-workshop.message.warning.show": false,
  ```

- LaTeX Workshop 默认保存的时候自动编译，如果不喜欢这个设置，可以添加以下代码进入设置区：

  ```json
  "latex-workshop.latex.autoBuild.run": "never",
  ```

  

### MikTex配置

- 安装完成后，以**管理员身份运行**"Miktex console"进行配置；

- 点击"Packages"标签，然后依次安装需要的包等(或者**待编译时再按需安装也可**)；

  <img src="http://ww1.sinaimg.cn/large/005NduT8ly1gdhz9z3n0mj30ns0gwdgf.jpg" alt="6c4a7ec1f1fdbcdb0fcafd3bc495fbe.png" style="zoom: 50%;" />

  <img src="http://ww1.sinaimg.cn/large/005NduT8ly1gdhzbidggyj30nq0h60tq.jpg" alt="47ec412dc1e407c465ee1733fbc99d7.png" style="zoom:50%;" />



## Github配置

关于创建github仓库，以及VSCode关联github，这些你已经懂了！:kissing_heart:

不过提醒一句，**如果你的论文未发表，希望你在github创建私有库，出现问题不负责哦**；

<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gdhzgej6a8j30r30gcgmc.jpg" alt="01ba35136010a93ab74639644a18a17.png" style="zoom:50%;" />



## Overleaf配置

Overleaf是一个易于使用的在线Latex编辑器。

- 中文版：[https://cn.overleaf.com/](https://link.zhihu.com/?target=https%3A//cn.overleaf.com/)

  英文版：[https://www.overleaf.com](https://link.zhihu.com/?target=https%3A//www.overleaf.com)

这里罗列一些优点：

- **免安装，免配置**

  因为它是**在线编辑器**，不用在本地安装各种latex相关的软件，省去了不少麻烦。

- **package全集**

  **什么package都有**，只需要使用\usepackage命令随意使用就行。

- **多人协作**

  现在假设你的论文写好了，想让老师帮你做些修改、批注啥的，通常我们会把word、或者.tex文件发给老师，老师打开之后编译很有可能会报错（缺包）。**导师的心情可能会有一点点不快**，看起论文来自然不会很舒畅。

  现在，只需要添加你的导师为合作者，参与编辑，就能实时掌握你的进度了。

  ![cc1e76d569432c99a190b51946ca634.png](http://ww1.sinaimg.cn/large/005NduT8ly1gdhzqm428ij30hk02at8n.jpg)

- **版本控制**

  保存各个时期保存的版本，备份！

那既然优点这么多，为什么不直接在Overleaf上写论文呢。因为服务器在国外，不稳定，总是会连接超时，有时候需要科学上网才能使用。

### 创建项目

我们前面已经配置好github，论文也上传了github私有库，我们可以直接点击`Import from Github`，从github中选择论文项目导入到overleaf；

<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gdhztiutr5j307v087jre.jpg" alt="f43ce874e238da931c5245830b82946.png" style="zoom:50%;" />

当然你也可以从本地上传，不过问题也显而易见。



### 同步更新

我们可以在github和overleaf之间进行双向同步，及时更新论文。

<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gdhzxi9w4kj308t05k0sl.jpg" alt="3b72447723f8f74a553126908bc4ca4.png" style="zoom:50%;" />

<img src="http://ww1.sinaimg.cn/large/005NduT8ly1gdhzytjnwzj30jv0ftq3r.jpg" alt="f009c5544cbaf4e0498abdbc7704edd.png" style="zoom:50%;" />

自此，我们已经配置好了`VSCode <-> Github <-> Overleaf`三者之间的关联，就可以愉快的写论文了。



## 参考文献

- https://zhuanlan.zhihu.com/p/67182742
- https://zhuanlan.zhihu.com/p/38178015
- https://blog.csdn.net/yinqingwang/article/details/79684419



## 后记

作为研究生，从小白到老油条，这一路上我们不断摸索，不断踩坑，到最后总结出一套适合自己的学习、工作方法。你认为怎么做高效，舒适，那你就怎么做，可以听从别人的意见，但是要学会自己判断决策。

有时候小伙伴会调侃我总是搞一些花里胡哨的东西，我承认自己是个工具控，不用winedt就是因为它丑！VSCode不漂亮吗，用户体验不好吗，虽然“颜值即是正义”这句话放在人身上不合适，但是放在产品上，放在用户体验上，那是再合适不过了。

我个人追求美感，产品的美感、工具的美感、做事的美感，这样的习惯让我在学习工作的过程中充满乐趣和坚持下去的动力。我希望每个人都能找到自己的学习方式，不断优化，持续进步。

最后，希望大家的科研道路一切顺利，收获满满，共勉！

