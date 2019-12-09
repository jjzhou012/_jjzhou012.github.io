---
title: 'TensorFlow环境 人脸识别 FaceNet 应用（二）:FaceNet官方WiKi解读'
mathjax: true
date: 2018-3-14 00.00.00
tags: [facenet, Summary, face recognition]
categories: 人脸识别
---

>作者：Andy_z
文献：[官方WiKi](https://github.com/davidsandberg/facenet/wiki)

##一、分类器训练

###1.1 运行 train_softmax.py 文件训练
```
python src/train_softmax.py
--logs_base_dir ~/logs/facenet/
--models_base_dir ~/models/facenet/
--data_dir ~/datasets/casia/casia_maxpy_mtcnnalign_182
--image_size 160
--model_def models.inception_resnet_v1
--lfw_dir /home/david/datasets/lfw/lfw_mtcnnalign_160
--optimizer RMSPROP
--learning_rate -1
--max_nrof_epochs 80
--keep_probability 0.8
--random_crop
--random_flip
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt
--weight_decay 5e-5
--center_loss_factor 1e-2
--center_loss_alfa 0.9
```
- **log_base_dir**:  
**models_base_dir**:
训练开始时，以数据/时间训练开始的训练会话的子目录以yyyymmdd-hhmm的格式在以上两个目录中创建。  
- **data_dir**：  用于指出训练数据集的位置，可以通过用冒号分隔路径来使用几个数据集的联合。  
model_def： 给出推理网络的描述符，上述样例中  models.inception_resnet_v1 指向models包中的
inception_resnet_v1模块。 该模块定义一个函数 inference(images, ...)，images是输入图像的占位符(Inception-ResNet-v1的尺寸<?, 160,160,3>),并返回一个embeddings变量的引用。  
- **lfw_dir**：如果将参数lfw_dir设置为指向LFW数据集的基本目录，那么每1000个批次将在LFW上对该模型进行评估。有关如何在LFW上评估现有模型的信息，请参阅 Validate-on-LFW 页面。 如果在训练期间不需要对LFW进行评估，则可以将lfw_dir参数留空。 但请注意，此处使用的LFW数据集应与训练数据集一致。  
- **max_nrof_epochs**：最大训练周期。  
- **learning_rate_schedule_file**：为了改善最终模型的性能，当训练开始收敛时，学习速率降低10倍。 这是通过在参数learning_rate_schedule_file指向的文本文件中定义的学习速率时间表来完成的，同时还将参数learning_rate设置为负值。 为了简单起见，本例中data / learning_rate_schedule_classifier_casia.txt中使用的学习率也包括在库中。

>注：train_tripletloss.py和train_softmax.py的区别：这是作者对论文做出的一个延伸，除了使用facenet里提到的train_tripletloss三元组损失函数来训练，还实现了用softmax的训练方法来训练。当然，在样本量很小的情况下，用softmax训练会更容易收敛。但是，当训练集中包含大量的不同个体(超过10万)时，最后一层的softmax输出数量就会变得非常大，但是使用train_tripletloss的训练仍然可以正常工作。

###1.2 运行 train_softmax.py 文件训练

```
python src/train_tripletloss.py
--logs_base_dir ~/logs/facenet/
--models_base_dir ~/models/facenet/
--data_dir ~/datasets/casia/casia_maxpy_mtcnnalign_182_160
--image_size 160
--model_def models.inception_resnet_v1
--lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160
--optimizer RMSPROP
--learning_rate 0.01
--weight_decay 1e-4
--max_nrof_epochs 500
```


##二、可视化TensorBoard
&emsp;&emsp;监视训练过程，使用TensorBoard:

```
tensorboard --logdir=~/logs/facenet --port 6006
```
&emsp;&emsp;打开浏览器：http://localhost:6006/



##三、用自己的图像训练分类器
###3.1 在LFW上训练分类器  

&emsp;&emsp;对于这个实验，我们使用LFW图像的子集来训练分类器。 LFW数据集分为训练和测试集。 然后加载预训练模型，然后使用此模型为选定图像生成特征。 预训练模型通常在更大的数据集上进行训练以提供良好的性能（本例中为MS-Celeb-1M数据集的一个子集）。

- 将数据集分解为训练和测试集
- 加载预训练模型进行特征提取
- 计算数据集中图像的嵌入
- 模式= TRAIN：
    - 使用来自数据集的训练部分的嵌入来训练分类器  
    - 将训练好的分类模型保存为python pickle

- 模式= CLASSIFY：
    - 加载分类模型
    - 使用来自数据集测试部分的嵌入来测试分类器  


>classifier.py定义参数：
```
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' +
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. ' +
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)

    return parser.parse_args(argv)
```

- **mode**：  指示训练新分类器还是进行分类测试集。'TRAIN',    'CLASSIFY'
- **data_dir**：  包含对齐的LFW面部补丁的数据目录路径。
- **model**：  可能是包含meta_file和ckpt_file或模型protobuf(.pb)文件的目录
- **classifier_filename**：  分类器模型文件名称作pickle（.pkl）文件，对于训练过程，这是输出；对于分类过程，这是输入。
- **use_split_dataset**：  指示由data_dir指定的数据集应该分为训练集和测试集。 否则可以使用test_data_dir选项指定单独的测试集。
- **test_data_dir**：  包含用于测试的对齐图像的测试数据目录的路径。
- **batch_size**：  一个批次的图像运行数量。
- **image_size** ：  图像的像素尺寸。
- **seed**:   随机seed。
- **min_nrof_images_per_class**：  仅包含数据集中至少包含这些数量的图像的类。
- **nrof_train_images_per_class**：  从每个类中使用这个数量的图像进行训练，其余的进行测试。

&emsp;&emsp;在数据集的训练集部分训练分类器的步骤如下：

```
python src/classifier.py
TRAIN
data/lfw/lfw_align_mtcnnpy_160/
src/models/20170512-110547/20170512-110547.pb
src/models/lfw_classifier.pkl
--batch_size 1000
--min_nrof_images_per_class 40
--nrof_train_images_per_class 35
--use_split_dataset
```

```
python src/classifier.py TRAIN data/lfw/lfw_align_mtcnnpy_160/ src/models/20170512-110547/20170512-110547.pb src/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
```

&emsp;&emsp;训练输出：
```
Number of classes: 19
Number of images: 665
Loading feature extraction model
Model filename: src/models/20170512-110547/20170512-110547.pb
Calculating features for images
Training classifier
Saved classifier model to file "src/models/lfw_classifier.pkl"
```

&emsp;&emsp;训练好的分类器可以稍后用于使用测试集进行分类：

```
python src/classifier.py
CLASSIFY
data/lfw/lfw_align_mtcnnpy_160/
src/models/20170512-110547/20170512-110547.pb
src/models/lfw_classifier.pkl
--batch_size 1000
--min_nrof_images_per_class 40
--nrof_train_images_per_class 35
--use_split_dataset
```

```
python src/classifier.py CLASSIFY data/lfw/lfw_align_mtcnnpy_160/ src/models/20170512-110547/20170512-110547.pb src/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
```
&emsp;&emsp;单独指定测试集

```
python src/classifier.py CLASSIFY data/lfw/test_lfw src/models/20170512-110547/20170512-110547.pb src/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --test_data_dir data/lfw/test_lfw
```


&emsp;&emsp;这里使用数据集的测试集部分进行分类，并显示分类结果和分类概率。 该子集的分类准确度为〜0.98。
```
Number of classes: 19
Number of images: 1202
Loading feature extraction model
Model filename: src/models/20170512-110547/20170512-110547.pb
Calculating features for images
Testing classifier
Loaded classifier model from file "src/models/lfw_classifier.pkl"
   0  Ariel Sharon: 0.712
   1  Ariel Sharon: 0.771
   2  Ariel Sharon: 0.807
   3  Ariel Sharon: 0.785
   4  Ariel Sharon: 0.750

...
...
...
1197  Vladimir Putin: 0.536
1198  Vladimir Putin: 0.723
1199  Vladimir Putin: 0.715
1200  Vladimir Putin: 0.663
1201  Vladimir Putin: 0.732
Accuracy: 0.999
```



![](http://p5bxip6n0.bkt.clouddn.com/18-3-21/53475149.jpg)

![](http://p5bxip6n0.bkt.clouddn.com/18-3-21/76473615.jpg)



##四、基于mtcnn与facenet的人脸识别（单张图像识别分类）

&emsp;&emsp;代码：facenet/contributed/predict.py  

&emsp;&emsp;主要功能：

- ① 使用mtcnn进行人脸检测并对齐与裁剪

- ② 对裁剪的人脸使用facenet进行embedding

- ③ 执行predict.py进行人脸识别（需要训练好的svm模型）

&emsp;&emsp;参数：
```
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_files', type=str, nargs='+', help='Path(s) of the image(s)')
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. ' +
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)
```
- **image_files**： 被识别图像路径
- **model**：包含meta_file和ckpt_file或模型protobuf（.pb）文件的目录
- **classifier_filename**：分类器模型文件名称作为pickle（.pkl）文件


&emsp;&emsp;测试：用三中生成的lfw_classifier.pkl作为分类器模型进行

```
python predict.py d:/Anaconda3/Lib/site-packages/facenet/data/images/3.png D:/Anaconda3/Lib/site-packages/facenet/src/models/20170512-110547 D:/Anaconda3/Lib/site-packages/facenet/src/models/lfw_classifier.pkl
```

![](http://p5bxip6n0.bkt.clouddn.com/18-3-22/3119407.jpg)


python contributed/predict.py data/images/2.png src/models/20170512-110547 src/models/lfw_classifier_whole.pkl
