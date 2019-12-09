---
title: 'TensorFlow环境 人脸识别 FaceNet 应用（一）:FaceNet安装与验证测试集'
mathjax: true
date: 2018-3-13 00.00.00
tags: [facenet, Summary, face recognition]
categories: 人脸识别
---

>作者：Andy_z  
文献：[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)  
[数据集及模型下载通道](https://pan.baidu.com/s/1R70SWpSmF7SoZB5vkHdfpw)：(密码：3wty)

##一、前提条件

###1.&emsp;已安装Tensorflow
###2.&emsp;已在安装下列包(二选一):   

&emsp;&emsp;a.&emsp;python下安装scipy, scikit-learn, opencv-python, h5py, matplotlib, Pillow, requests, psutil

&emsp;&emsp;b.&emsp;安装Anaconda集成环境

###3.&emsp;已更新Sklearn至最新版本(二选一):

&emsp;&emsp;a.&emsp;可在propmt下"conda update conda "

&emsp;&emsp;b.&emsp;直接在cmd命令行下"pip install -U scikit-learn"

###4.&emsp;已安装git

>备注:如果没有完成以上的第3点,之后执行align时,可能会出现"no module named facenet","no module named align","no module named scikit-learn"等情况



##二、安装和配置FaceNet

&emsp;&emsp;1.&emsp;在cmd命令行，定位到自己想下载的文件夹,用git下载FaceNet源代码工程:

```
git clone --recursive https://github.com/davidsandberg/facenet.git
```
>建议：最好定位在&emsp;&emsp;Anaconda3\Lib\site-packages&emsp;&emsp;下安装。因为FaceNet也相当于一个python库。

![](http://p5bxip6n0.bkt.clouddn.com/18-3-15/23373226.jpg)


&emsp;&emsp;2.&emsp;下载数据集LFW。LFW数据集是由美国马萨诸塞大学阿姆斯特分校计算机视觉实验室整理的。下载地址：http://vis-www.cs.umass.edu/lfw/lfw.tgz, 下载完成后，把数据解压到目录 ..facenet\data\lfw\raw  下面,新建一个空文件夹命名为"lfw_160"。可以看到数据集中每张图像的分辩率是250*250。


![](http://p5bxip6n0.bkt.clouddn.com/18-3-15/8023931.jpg)

&emsp;&emsp;3.&emsp;设置环境变量,以下方法二选一:

&emsp;&emsp;a.&emsp;在cmd命令行键入：set PYTHONPATH=...\facenet\src, 例如笔者的是:set PYTHONPATH=D:\Anaconda2\envs\py3.6\Lib\site-packages\facenet\src

&emsp;&emsp;b.&emsp;在 计算机-->属性-->高级系统设置-->环境变量中,新建PYTHONPATH,键入 D:\Anaconda2\envs\py3.6\Lib\site-packages\facenet\src

检验:在cmd命令行下面，键入set，查看设置情况

![](http://p5bxip6n0.bkt.clouddn.com/18-3-15/66907272.jpg)


![](http://p5bxip6n0.bkt.clouddn.com/18-3-15/56240084.jpg)


##三、图像数据预处理

>也可直接使用下载的已处理数据集

&emsp;&emsp;我们需要将待检测所使用的数据集校准为和预训练模型所使用的数据集大小一致。

&emsp;&emsp;1.&emsp;使用&emsp;facenet\src\align\align_dataset_mtcnn.py&emsp;进行校准,校准后的图片存在&emsp;..facenet\data\lfw\lfw_160&emsp;下面。在cmd命令行 或者 对应语言版本的propmt下，定位到facenet所在位置，键入

```
python src\align\align_dataset_mtcnn.py data/lfw/raw data/lfw/lfw_160
```
&emsp;&emsp;官方Wiki说明

```
python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/ ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44
```
>上述命令生成的脸部缩略图是182x182像素。


&emsp;&emsp;2.&emsp;校准后发现图像大小变了


##四、评估谷歌预训练模型在数据集的准确率

&emsp;&emsp;1.&emsp;下载预训练的模型。把下载的文件解压到src\models\目录下面。

![](http://p5bxip6n0.bkt.clouddn.com/18-3-15/34564036.jpg)


&emsp;&emsp;2.&emsp;程序下载好了,测试数据集LFW也有了,模型也有了,接下来就可以评估模型在数据集的准确率了。在cmd命令行或者propmt下定位到facenet文件夹下，输入

```
python src\validate_on_lfw.py data\lfw\lfw_160 src\models\20170512-110547
```
紧接着,预测中,结果如图：

![](http://p5bxip6n0.bkt.clouddn.com/18-3-15/47530520.jpg)


##五、其他

###5.1 对比
&emsp;&emsp;facenet可以直接比对两个人脸经过它的网络映射之后的欧氏距离，运行程序为facenet-master\src\compare.py。
-1、在compare.py所在目录下放入要比对的文件1.jpg和2.jpg，打开cmd命令行窗口
-2、cd到compare.py所在路径
-3、输入 python compare.py models/20170512-110547 1.png 2.png

![](http://p5bxip6n0.bkt.clouddn.com/18-3-16/42045957.jpg)
