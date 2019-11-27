import sys
from keras.callbacks import *
from processing import *
sys.path.append('./')
sys.path.append('./yolo3/')
from keras.optimizers import *
from sklearn.model_selection import StratifiedShuffleSplit
from utils import *
from network1 import *  #调用的不同版本的网络结构，由于训练期间结构多次修改删减，
                        # 所以前期部分模型代码（效果不是特别好的）可能删除了 主要以network1这份为主
from yolo3.my_yolo import *
# from network2 import *
# from network import *
params={
        "conv_init": "glorot_uniform",
        "conv_subsample_lengths": [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1,2],
        # "conv_subsample_lengths":[1,1],
        "conv_filter_length": (3,3), ##卷积核的尺寸
        "conv_activation": "relu",
        "conv_dropout": 0,
        # 多少增加一次通道数
        "conv_increase_channels_at": 4,
        "conv_num_skip": 2,#一个block里卷积的数量
        "conv_num_filters_start": 16, #初始卷积核数量
        "learning_rate": 0.001,
        ###

        'save_weight_path': './model/',
        'type': 'lips1.9', ## 1.2 是扩大了卷积尺度的efficient，1.3是Densenet 1.1是DENsenet 1.4 efficient+3lstm 1.5 efficient_复现
        ##1.6加大图片尺寸 1.7 +1Dcnn  1.8 +gai  1.9 +convlstm   1.2 1.4 1.5 1.7 1.9 1.9_
        'batch_size':12,
        'step':1000,
        'epoch': 100,
    }
from processing import *
def train():
    # models=conv2VGG(**params)
    # models=network(**params)
    # models=Lip_net(**params)
    # models=CONV3Dlayer(**params)
    models=Efficient(**params)
    # models=lip_model2(**params)
    # models=EfficientCONvlstm(**params)
    data = pd.read_csv('one_hot_label.txt', sep='\t').values  # 读取onehot数据
    sgd = SGD(lr=0.001,
              decay=1e-6,
              momentum=0.9,
              nesterov=True)
    adm=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    models.compile(optimizer=sgd, loss=mycrossentropy, metrics=['accuracy'])#'categorical_crossentropy'score_loss,mycrossentropy
    # 保存模型结构
    jos_model=models.to_json()
    open('./model/'+params['type']+'_architecture.json','w').write(jos_model)
    # models.save('./model/'+params['type']+'_architecture.h5')

    # 划分训练集
    splips=StratifiedShuffleSplit(1,test_size=0.2,random_state=888)
    X=data[:,0]
    Y=[list(i).index(1) for i in data[:,2:]]
    ss = splips.split(X, Y)

    for train_in, test_in in ss:
        train=data[train_in,:]
        test=data[test_in,:]
    print('train:',len(train))
    print('test:',len(test))

    # 设置回调函数
    earlyStopping = EarlyStopping(monitor='val_acc', patience=12, verbose=1, mode='auto')
    saveBestModel1 = ModelCheckpoint(params['save_weight_path'] + params['type'] + '.hdf5',
                                     monitor='val_acc',verbose=1,
                                     save_best_only=True, mode='auto', save_weights_only=True)

    reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001,
                                 cooldown=6, min_lr=0)

    # 开始训练
    models.load_weights(params['save_weight_path'] + 'lips1.9_'+ '.hdf5', by_name=True,skip_mismatch=True,reshape=True)
    history = models.fit_generator(generateA(train,random=False,**params), steps_per_epoch=params['step']
                                 , epochs=params['epoch'], callbacks=[earlyStopping, saveBestModel1, reducelr],
                                 validation_data=generateA(test,random=False,**params),
                                 validation_steps=100)
#
def migration_train():
    # 选择模型训练
    # models=model_from_json(open('./model/'+'lips1.0'+'_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.1' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.2' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.3' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.4' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.5' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.6' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.7' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.7_dialation' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.8' + '_architecture.json').read())
    # models = model_from_json(open('./model/' + 'lips1.9' + '_architecture.json').read())
    models = model_from_json(open('./model/' + 'lips1.9_' + '_architecture.json').read())

    data = pd.read_csv('one_hot_label.txt', sep='\t').values  # 读取onehot数据
    sgd = SGD(lr=0.001,
              decay=1e-6,
              momentum=0.9,
              nesterov=True)
    adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    models.compile(optimizer=sgd, loss=mycrossentropy,
                   metrics=['accuracy'])  # 'categorical_crossentropy'score_loss,mycrossentropy

    # 划分训练集
    splips = StratifiedShuffleSplit(1, test_size=0.2, random_state=888)
    X = data[:, 0]
    Y = [list(i).index(1) for i in data[:, 2:]]
    ss = splips.split(X, Y)

    for train_in, test_in in ss:
        train = data[train_in, :]
        test = data[test_in, :]
    print('train:', len(train))
    print('test:', len(test))

    # 设置回调函数
    earlyStopping = EarlyStopping(monitor='val_acc', patience=12, verbose=1, mode='auto')
    saveBestModel1 = ModelCheckpoint(params['save_weight_path'] + params['type'] + '.hdf5',
                                     monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='auto', save_weights_only=True)

    reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001,
                                 cooldown=6, min_lr=0)

    # 开始训练
    models.load_weights(params['save_weight_path'] + 'lips1.9_' + '.hdf5', by_name=True, skip_mismatch=True,
                        reshape=True)
    history = models.fit_generator(generateA(train, random=False, **params), steps_per_epoch=params['step']
                                   , epochs=params['epoch'], callbacks=[earlyStopping, saveBestModel1, reducelr],
                                   validation_data=generateA(test, random=False, **params),
                                   validation_steps=100)


if __name__ == '__main__':
    # step 1:更改文件名
    rename_file(p_files='../DATA/train/lip_train/')
    rename_file(p_files='../DATA/test/lip_test/')

    # step 2: 提取xml标签文件信息转化为txt文本形式
    creat_label2txt(path='./yolo3/labels/')

    # step 3:  运行 kmeans.py

    # step 4:  训练 yolo3
    train_yolo_model(weight_path='./yolo3/weight/', manul_label_path='./yolo3/manul_label.txt', anchors_path='./yolo3/yolo_anchors.txt')

    # step 5：生成裁剪图片 训练集
    clip_raw_data(path='../DATA/train/lip_train/', dspath='../DATA/train/processing_lip_train/')
    # step 6:生成generate的tensor
    creat_generate_tensor(path='../DATA/train/processing_lip_train/', dspath='../DATA/train/lip_train_np/')
    # step 7: 生成裁剪图片 测试集
    clip_raw_data(path='../DATA/test/lip_test/', dspath='../DATA/test/processing_lip_test/')

    # step8: 训练模型 复现时不需要运行
    # 训练时有多个模型进行选择，这里以最后一次训练的模型为例
    # train()

    # step9: 迁移训练  将之前训练过的模型和权重 重新加载并全部重新训练
    migration_train()

    # step10: 运行predict.py进行预测 生成目标文件

    # step7: 将高分文件进行融合 (再notebook上运行的 融合.ipynb)


