import os
import shutil
from PIL import Image
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import sys
sys.path.append('./yolo3/')
sys.path.append('./')
import pandas as pd
from yolo3.my_yolo import *


def clip_raw_data(path='../DATA/train/lip_train/',dspath='../DATA/train/processing_lip_train/'):
    '''
     使用Yolo得到裁剪框，将裁剪的图片保存为npy

    :param path: 存放图片的路径（图片存放结构：path/ID/*.png）
    :param dspath: 生成npy图片的目录
    :return: Void
    '''
    path=path
    # data = pd.read_csv('one_hot_label.txt', sep='\t').values
    # all_img = np.zeros(( 24, 112, 112, 3))
    # ID=data[:,0]
    ID=os.listdir(path)
    yolo=YOLO()
    for i in range(len(ID)):
        p_path = path + ID[i] + '/'
        # if os.path.exists('../DATA/lip_test/'+ID[i] ):
        #     continue
        # shutil.copytree(p_path,'../DATA/'+ID[i])
        n_x = os.listdir(p_path)
        n_x.sort()
        if os.path.exists(dspath + ID[i]):
            continue
        # shutil.copytree(p_path,'../DATA/'+ID[i])

        os.makedirs(dspath+ ID[i], exist_ok=True)
        print('creat files:', dspath + ID[i])
        for k in range(len(n_x)):
            img_path = p_path+ '/' + n_x[k]
            img = Image.open(img_path)
            left, top, right, bottom = yolo.detect_image(img)
            # print(left)
            if left | top | right | bottom == 0:
                # fask_list.append(ID[i])
                print('fiall:', dspath+ ID[i])
                shutil.rmtree(dspath + ID[i])
                # shutil.copyfile(img_path,'./new_train/'+ID[i]+'/'+n_x[k])
                break
            else:
                img = np.array(img, dtype='uint8')

                lenght=bottom-top
                wdith=right-left

                d = abs((wdith - lenght) // 2)
                ###裁剪成正方形的 因为要resize到（112，112），以保证图片不变形
                if wdith>lenght:
                    top=top-d if(top-d)>0 else 0
                    bottom = bottom + d if (bottom + d) < img.shape[0] else img.shape[0]
                else:
                    left = left - d if (left - d) > 0 else 0
                    right = right + d if (right + d) < img.shape[1] else img.shape[1]

                img = img[top:bottom, left:right]

                np.save(dspath + ID[i] + '/' + n_x[k][:-4]+'.npy',img)
def stastic():
    path='train/lip_train/'
    data = pd.read_csv('one_hot_label.txt', sep='\t').values
    init={}
    init=init.fromkeys(range(1,25),0)

    stastic={'1':init.copy(),'2':init.copy(),'3':init.copy(),'4':init.copy()}

    for i in range(len(data)):
        ID=data[i][0]
        name=data[i][1]
        img_f=path+ID
        n_x=os.listdir(img_f)
        stastic[str(len(name))][len(n_x)]+=1
    #
    print(stastic)

def creat_generate_tensor(path = '../DATA/train/processing_lip_train/',dspath='../DATA/train/lip_train_np/'):
    '''
    将裁剪后的npy图片resize到合适大小并填充tensor到（24,112,112,3）的张量

    :param path: npy图片路径
    :param dspath: tensor的保存路径
    :return:
    '''

    all_img = np.zeros((24, 112, 112, 3))
    ID = os.listdir(path)
    for i in range(len(ID)):
        p_path = path + ID[i] + '/'
        if os.path.exists(dspath + ID[i] + '.npy'):
            os.remove(dspath + ID[i] + '.npy')
        n_x = os.listdir(p_path)
        n_x.sort()
        repeat = 24 // len(n_x)
        a = range(len(n_x))
        b = [i for i in a for k in range(repeat)]
        b = b + [a[-1]] * (24 - len(b))
        try:
            for j in range(len(b)):
                img = np.array(np.load(p_path + n_x[b[j]]))

                img = cv2.resize(img, (112, 112))
                all_img[j] = img.astype('uint8')
            np.save(dspath + ID[i] + '.npy', all_img)
            print("save " + ID[i] + '.npy')

        except:
            print('fail')

def read_labels():
    '''
    读取标签文件输出查看裁剪图片
    :return:
    '''
    path = './no_regenize/labels/'
    labels = os.listdir(path)
    for i in range(len(labels)):
        ID = labels[i][:-7]
        img_name = labels[i][:-4]
        left, top, right, bottom = convert_annotation(path + labels[i])
        img_path = './no_regenize/' + ID + '/' + img_name + '.png'
        img = Image.open(img_path)
        img = np.array(img)
        img = img[top:bottom, left:right]
        img = cv2.resize(img, (112, 112), img)
        plt.imsave(img_path, img)
        print('save:' + ID)

def rename_file(p_files='./DATA/train/lip_train/'):
    '''
    将lip_train内的图片重新以ID_number的方式命名
    :param p_files: 图片路径
    :return:
    '''
    p_img=os.listdir(p_files)
    for i in range(len(p_img)):
        files=p_files+p_img[i]+'/'
        n_x=os.listdir(files)
        for k in range(len(n_x)):
            # if n_x[k][-3:]=='png':
            #     print(n_x[k])
            #     continue
            old_name=files+n_x[k]
            # if len(old_name)>7:
            #     continue
            number=int(n_x[k][:-4])
            new_name=files+p_img[i]+'_'+'%02d'%int(number)+'.png'
            os.renames(old_name,new_name)
            print(old_name + '--------->>>' + new_name)
def fix_train_name():
    '''
    修正错误命名的文件
    :return:
    '''
    img_path='./new_train/'
    ID=os.listdir(img_path)
    for i in range(len(ID)):
        p_file_path=img_path+ID[i]
        p_img=os.listdir(p_file_path)
        if len(p_img)<2:
            print(ID[i])
            continue
        for k in range(len(p_img)):
            if len(p_img[k])<10:
                print(p_img[k])
                continue
            a = re.findall('_\d+', p_img[k])
            a=int(a[-1][1:])
            print(p_file_path+'/'+p_img[k]+'----------->'+p_file_path+'/'+ID[i]+'_%02d'%a+'.png')
            os.renames(p_file_path+'/'+p_img[k],p_file_path+'/'+ID[i]+'_%02d'%a+'.png')



