
import pandas as pd
import os
import cv2
from PIL import Image
import numpy as np
import random

import h5py
def rgb_to_hsv(arr):
    """
    convert float rgb values (in the range [0, 1]), in a numpy array to hsv
    values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    hsv : (..., 3) ndarray
       Colors converted to hsv values in range [0, 1]
    """
    # make sure it is an ndarray
    arr = np.array(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {} was found.".format(arr.shape))

    in_ndim = arr.ndim
    if arr.ndim == 1:
        arr = np.array(arr, ndmin=2)

    # make sure we don't have an int image
    arr = arr.astype(np.promote_types(arr.dtype, np.float32))

    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = arr.ptp(-1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    if in_ndim == 1:
        out.shape = (3,)

    return out


def hsv_to_rgb(hsv):
    """
    convert hsv values in a numpy array to rgb values
    all values assumed to be in range [0, 1]

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.array(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=hsv.shape))

    # if we got passed a 1D array, try to treat as
    # a single color and reshape as needed
    in_ndim = hsv.ndim
    if in_ndim == 1:
        hsv = np.array(hsv, ndmin=2)

    # make sure we don't have an int image
    hsv = hsv.astype(np.promote_types(hsv.dtype, np.float32))

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.stack([r, g, b], axis=-1)

    if in_ndim == 1:
        rgb.shape = (3,)

    return rgb
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
def disort_picture(image,hue=.1, sat=1.5, val=1.5):
    if len(image.shape)!=3:
        print('img expect 3 dim but it is :%d'%len(image.shape))
        # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    return image_data

def warp_picture(imgs):
    for i in range(len(imgs)):
        imgs[i]=cv2.normalize(imgs[i],imgs[i], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return imgs
def write_hd5py(data, file):
    with h5py.File(file, 'w') as f:
        f.create_dataset(data=data, dtype=data.dtype, name='image')

def load_hdf5(file):
    with h5py.File(file, 'r') as f:
        return f['image'][:]
def yolo_img(yolo,img_path,out_shape=(128,128)):
    yolo=yolo
    img=Image.open(img_path)

    left, top, right, bottom=yolo.detect_image(img)
    if (left, top, right, bottom)==(0,0,0,0):

        return np.zeros((1))
        # shutil.copyfile(img_path,name)
    # img=yolo.detect_image(img)
    img = np.array(img)
    img=img[top:bottom,left:right]
    img = cv2.resize(img, out_shape)
    # img=cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img/255
def generateA(files,random,**params):
    '''
    fit_generate的生成器
    :param files:  数据，[ID,...]矩阵
    :param random: 是否随机crop
    :param params:
    :return: all_img shape:(batch,24,112,112,3),all_label shape:(313)
    '''
    path = './DATA/train/lip_train_np/'  # 处理好的tensor目录

    count=0
    all_img = np.zeros((params['batch_size'], 24, 112, 112, 3))
    all_label = np.zeros((params['batch_size'], 313))
    # all_number=np.zeros((params['batch_size'], 1))
    # statue=True
    while True:
        index = np.random.randint(0,len(files),1)
        per_filse = files[index[0]]
        ID = per_filse[0]
        p_path = path + ID + '.npy'
        # img_path='./train/lip_train/'+ID
        # n_x=os.listdir(img_path)
        # print(p_path)
        if not os.path.exists(p_path):
            continue
        try:
            arrs_img=np.array(np.load(p_path))
            all_img[count] = arrs_img.astype('uint8')/255
            #
            all_label[count]=per_filse[2:]
            # print(np.argwhere(per_filse[2:]))
            # all_number[count]=len(n_x)
            count+=1
        except Exception as e :
            print("load numpy erro",e)
            continue
        if count==params['batch_size']:
            # print(np.argmax(all_label, axis=1))
            count=0
            yield all_img,all_label

def generateB(files,**params):
    path = './train/lip_train/'
    count=0
    all_img = np.zeros((params['batch_size'], 24, 112, 112, 3))
    # all_img=[]
    all_label=np.zeros((params['batch_size'], 313))
    while True:
        per_filse=random.choice(files)
        ID = per_filse[0]
        p_path = path + ID + '/'
        if os.path.exists(p_path):
            n_x = os.listdir(p_path)
            n_x.sort()   ###!!!!!!
            # print(n_x)
            repeat = 24 // len(n_x)
            a = range(len(n_x))
            b = [i for i in a  for k in range(repeat)]
            b = b + [a[-1]] * (24 - len(b))
            for j in b:
                img=Image.open(p_path+n_x[j])
                # print(n_x[b[j]])
                img=np.array(img)[:,:,:3]
                img=cv2.resize(img,(128,128))
                all_img[count, j, :, :] = img/255
            count+=1
            all_label[count, :] = per_filse[2:]


        if count == params['batch_size']:
            count = 0
            yield all_img, all_label



def creat_yolo_img(yolo,filess):
    path = './train/lip_train/'
    yolo = yolo
    problem=[]
    # all_label=np.zeros((params['batch_size'],313))
    # all_label = bathch_files[:, 2:]  # 标签是从第3列开始的
    for i in range(len(filess)):
        print('--第%d个文件--'%i)
        ID = filess[i][0]
        p_path = path + ID + '/'
        n_x = os.listdir(p_path)
        for k in range(len(n_x)):
            imgs = yolo_img(yolo, p_path + n_x[k])
            if len(imgs) ==1:
                print('ID:', ID, n_x[k])
                with open('./fask_yolo.txt', 'a+') as f:
                    f.write(n_x[k])
                    f.write('\n')
                problem.append(n_x[k])
    return problem


# a=Image.open('./new_train/fc21d4f37a269ea4fa8dae48d1a7bb4f/fc21d4f37a269ea4fa8dae48d1a7bb4f_01.png')
# a=np.array(a)[...,:3]
#
# b=disort_picture(a)
# print(b.max())
# plt.subplot(1,2,1)
# plt.imshow(a)
# plt.subplot(1,2,2)
# plt.imshow(b)
# plt.show()
