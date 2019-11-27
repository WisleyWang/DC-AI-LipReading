from keras.models import *
import sys
from keras.callbacks import *
sys.path.append(',/')
sys.path.append(',/yolo3/')
from processing import *
from yolo_model.my_yolo import *
from PIL import Image
import pandas as pd
from keras_efficientnets.custom_objects import *
from network1 import *

def warp_picture(img,input_shape):
    cut_part = img.shape[0] // 4
    img = img[cut_part -50:, :, :]
    img=cv2.resize(img,input_shape,interpolation=cv2.INTER_AREA)
    # img=cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img#.astype(np.uint8)
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
        'name': 'resnet_incept_',
        'save_weight_path': './model/',
        'type': 'lips1.9_',   #1.2 1.4 1.5 1.7 1.9 1.9_  #模型的名字
        'batch_size': 12,
        'step': 80,
        'epoch': 100,

    }
data=pd.read_csv('one_hot_label.txt',sep='\t')
tag=data.columns[2:]

path='./DATA/test/processing_lip_test/' # 处理好的测试集目录
models=model_from_json(open('./model/'+params['type']+'_architecture.json').read())
# models=load_model('./model/'+params['type']+'_architecture.h5')
# models=Efficient(**params)
models.load_weights(params['save_weight_path'] + params['type'] + '.hdf5', by_name=True )

ID=os.listdir(path)
finish=pd.DataFrame(np.zeros((len(ID),2)),columns=['ID','Name'])
# finish=pd.DataFrame(np.zeros((len(ID),1+len(tag))),columns=['ID']+tag.to_list())
for i in range(len(ID)):
    p_files=path+ID[i]
    # p_img=os.listdir(p_files)
    batch_imgs=np.zeros((1,24,112,112,3))
    n_x = os.listdir(p_files)
    n_x.sort()
    repeat = 24 // len(n_x)
    a = range(len(n_x))
    b = [i for i in a for k in range(repeat)]
    b = b + [a[-1]] * (24 - len(b))
    for j in range(len(b)):
        img_p=p_files+'/'+n_x[b[j]]
        # img = Image.open(img_p)
        # img = np.array(img)[..., :3]
        img=np.array(np.load(img_p))
        # img=yolo_img(yolo,img_p ,out_shape=(256,256))

        img=cv2.resize(img,(112,112))
        img = img.astype('uint8') / 255
        batch_imgs[0,j]=img

    print('ID:'+ID[i]+' n_img:'+str(len(n_x)))
    result=models.predict(batch_imgs)
    # finish.loc[i, 'ID'] = ID[i]
    # finish.loc[i, tag] = result
    result=np.argmax(result) # 找到概率最大的索引
    name=tag[result]  # 索引对应种类字符
    print(name)
    finish.loc[i,'ID']=ID[i]
    finish.loc[i,'Name']=name
finish.to_csv('lips1.9_.csv',header=False,index=False)