## 队伍简介

   队伍名 

> 机器还在学习
队长：来自电子科技大学的“无能的万金油”
四位队员分别是:
来自西南财经的"忆苦思甜""和"大道”
来自天津大学的“有什么关系”
来自电子科技大学的“这个宇宙不太短”。

## 赛题回顾
[初赛](https://www.dcjingsai.com/common/cmpt/2019%E5%B9%B4%E2%80%9C%E5%88%9B%E9%9D%92%E6%98%A5%C2%B7%E4%BA%A4%E5%AD%90%E6%9D%AF%E2%80%9D%E6%96%B0%E7%BD%91%E9%93%B6%E8%A1%8C%E9%AB%98%E6%A0%A1%E9%87%91%E8%9E%8D%E7%A7%91%E6%8A%80%E6%8C%91%E6%88%98%E8%B5%9B-AI%E7%AE%97%E6%B3%95%E8%B5%9B%E9%81%93_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html)唇语数据集是在室内环境下利用手机录制采集而成，样本录制人员包含多名男性和女性，录制时要求人员正面面向镜头用普通话朗读中文常用词语。数据集中每个样本由唇语序列图片和对应的说话内容文本组成，数据被分为训练集和测试集，总共的预测语句总共有313类

初赛数据：[下载链接](https://pan.baidu.com/s/1WDr-T7nRdpDDJIORDbtZKw) 密码：s6qt

备用下载连接：链接：https://pan.baidu.com/s/1Fl8xmlSRR6oHJwR1PUzhsQ 提取码：3Bt0 

##环境
tensorboard   = 1.10.0                
tensorboardX  =  1.6                   
tensorflow-estimator =  1.13.0                
tensorflow-gpu = 1.10.0                
tensorlayer  = 1.10.1 
scikit-learn=0.21.3
Keras=2.2.4                 
Keras-Applications=1.0.6                 
keras-efficientnets =0.1.7                 
Keras-Preprocessing =1.0.5
numpy = 1.14.5
pandas = 0.23.4 
Cuda compilation tools, release 9.0, V9.0.176
## 文件说明

├─code
│  │  lip_train.txt  赛方给的标签
│  │  network.py  网络模型代码
│  │  network1.py
│  │  network2.py
│  │  one_hot_label.txt 处理后的ont-hot标签
│  │  predict.py   预测代码
│  │  processing.py  预处理代码
│  │  train.py  训练模型代码（主函数）
│  │  utils.py  辅助代码
│  │  综合.csv  最总提交结果文件
│  │  融合.ipynb 将结果文件融合的代码
│  │
│  ├─model 保存训练模型
│  │
│  ├─yolo3
│  │  │  kmeans.py 将标签K均值聚类代码
│  │  │  manul_label.txt 提取标签xml文件生成的txt文件
│  │  │  my_yolo.py yolo3核心代码
│  │  │  yolo_anchors.txt 生成的anchors文本
│  │  │  __init__.py
│  │  │
│  │  ├─labels 存放xml手工标签文件
│  │  │
│  │  ├─weight  存放yolo训练模型
│  │  │      best_weight.h5  保存的最好的yolo模型权重
│  │  │      trained_weights_stage_1.h5  最后一次训练权重
│  │  │      yolo_architeture.json    —— yolo模型结构
│  │  │
│  │  ├─yolo3 源码文件
│  │  │  │  model.py
│  │  │  │  utils.py
│
└─DATA
    ├─test
    │  ├─lip_test 原始测试集
    │  └─processing_lip_test  处理后的测试集
    └─train
        ├─lip_train 原始数据集
        ├─lip_train_np  处理后的tensor数据集
        └─processing_lip_train 处理后的训练集

## 解决方案概述
本赛题提提供了视频均匀采样的图片，样本帧数是不一样的，最大帧为24帧，首先对数据集进行裁剪（使用yoloV3），随后resize到112*112*3大小，并将不满24帧的图片，根据填充个数：`[24/n]` 取整来对数据进行填充（n为样本实际帧数），最后得到一个样本的张量形状为（24，112，112，3）。最后训练了DENNET,RESNET，EFFICIENT等3D改进网络，以及配合LSTM,GRN自然语言处理方面的网络，通过预测结果加权进行模型融合。

## 数据集划分
通过分成采样对训练集进行7：3的训练集验证集划分，保证划分的数据集的各类样本保持原有比例

## 前期处理
赛题提供了train的语料标签，应先转化为one-hot编码形式。
由于我们只有单卡TitanXp,12g显存，所以模型参数和输入数据的大小受到限制，必须充分考虑来做均衡处理。

以下简要地说明训练步骤和细节部分：
- **文本处理**
	- 1.将比赛给的标签文件进行onehot编码（onehot文件我已给出）
	- 2.对比赛数据集重新命名便于训练和排序，新命名格式为:ID_number.png
	- 如(./DATA/train/cda1355113s/1.png------------>./DATA/train/cda1355113s/cda1355113s_01.png)
- **训练yolo模型**
	- 1.用labelImage对数据进行嘴部区域标记，大概标记了2K张（标签文件保存在./code/yolo3/labels）
	- 2.运行./code/yolo3/my_yolo.py中的creat_label2txt()函数，提取xml标签文件信息转化为txt文本形式，生成manul_label.txt文件（./yolo3/目录下）
	- 3.运行./code/yolo3/kmeans.py进行聚类（9类anchor），生成yolo_anchors.txt文件（./yolo3/目录下）
	- 4.运行./code/yolo3/my_yolo.py中的train_yolo_model()函数训练yolo模型（./code/yolo3/weight文件夹下存放着训练权重，可根据需要选择是否预加载权重）

- **提取嘴部区域**
	- 1.运行./code/utils.py中的 clip_raw_data()函数，将训练集嘴巴裁剪为正方形并保存到./code/DATA/train/processing_lip_train/ (有些瑕疵图片或者部分图片yolo识别不到，后面需要手动裁剪或者重新训练yolo模型)
	- 2.运行./code/utils.py中的creat_generate_tensor()函数将裁剪的图片resize并整合成（24,112,112,3）的张量（这么做是为了加快训练），保存在./DATA/train/lip_train_np/  ，去除噪声样本，以及只有一张图片的样本，最后得到9993个npy样本文件
	- 3.对测试集的图片做上述同样的处理，但不做第二步的处理（预测的时候不需要预处理加快，样本量少）
- **训练模型**
	- 1.tran.py为主要代码文件，main中描述了数据处理的主要步骤，模型的训练是经过多次训练和调参了的，所以复现的时候只需要使用迁移训练即可，由于调参和更改模型的过程中，部分模型代码已经删除或者改变，所以部分模型只能使用load_model的方法来从jason文件中重构模型，本代码只以最后一次训练的模型为列进行演示，该模型的最高线上准确率能达到0.61左右。

	
- **其他处理**
	-归一化
	-裁剪框微调 
	-yolo模型多次调参训练
	-剔除噪声图片



## 模型设计与模型融合

基于以上预处理，进行模型设计与融合。

- 单模型

	最基本的直接跑了lipnet网络，线下val_acc：0.34左右，线上能达到0.42的准确率

	其后主要搭建了三种网络：
	Resnet+BiLSTM
	Densenet+BiLSTM
	Efficient+BiLSTM
	三种主题框架网络都改成了3D卷积并进行了多次调整（卷积核大小，特征数，层数，学习率，优化器，模型结构微调等）
	
	经过测试并考虑设备问题，两层双向LSTM+一层双向GRU可以达到较能接受的效果，其中Efficient的网络框架效果最好，初期模型线下val_acc:0.45，线上:0.59. 最后最高单模型能达到线下:0.58线上0.65

- Trick

	我们尝试了多种模型的trick，比如：
	- 非同步学习率 (卷积层和LSTM)
	- 增加TTL层（多尺度时空卷积）
	- Inception
	- effcient_wise （特征图选择）
	- swish
	- self-attention（最终模型中并未使用）
	- 结合时空信息ConvLSTM
	- DropConnect3D
	- BN
	- 流型 （由于时间关系，提取的流型特征并未使用）
	
将有较好提升的模型都保存下来，根据奥卡姆剃刀原则，把不必要的参数尽可能地缩减。由于设备的限制，我们无法通过增加图像分辨力或者网络的深度和广度来探索模型的极限从而达到模型最优的准确率，只能尝试优化模型结构，在有限的算力和时间下达到所能达到的比较好的结果。

- 加权融合
	
	由于设备限制，不能进行端到端的模型融合，再得到了单模型的预测结果后，直接将前几个最好的概率预测值进行加权融合，我们简单地用` (predict1+predict2+...)/n`就将成绩上升了0.08个百分点，最终取得了线上A榜0.7，18名的成绩。
