"""YOLO_v3 Model Defined in Keras."""

from functools import wraps,reduce
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Lambda
from keras.models import Model
import xml.etree.ElementTree as ET
from PIL import Image
from os import getcwd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys
sys.path.append('./yolo3')
from xml import  etree
# from  yolo3.utils import compose
import re

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)

    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])
def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    """输入的分别是yolo_body的输出张量Y，9个锚中的三个，分类数量，最大张量的大小"""
    num_anchors = len(anchors) #这里是3
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])  #有三个锚，每个锚分别是（x,y）坐标

    grid_shape = K.shape(feats)[1:3] # height, width  #取得该输出张量Y的大小
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),#tile（A,reps）A：array_like 输入的array
                                                                              # reps：array_like A沿各个维度重复的次数，
                                                                                # 其实就是得到了一个（grid_shape[0],grid_shape[1],1,1,1）的张量
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y]) #变成（grid_shape[0],grid_shape[1],1,1,2）  其实就是个网格表
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5]) ##理所当然的形状（batch_size,h,w,3,n_class+5）
    #feats_shape=(batch_size,h,w,3,5+n_class)
    # Adjust preditions to each spatial grid point and anchor size.  (x, y, w, h, confidence)
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats)) #::-1列表反转的操作 得到x,y的系数(归一化了)
    #box_xy_shape=(batch_size,h,w,2) 最后一维存放着x,y坐标
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats)) #对输出的长宽根据给的锚进行了尺度缩放
    #box_wh_shape=(batch_size,h,w,2) 最有一维存放着宽和长
    box_confidence = K.sigmoid(feats[..., 4:5])
    # box_confidence=(batch_size,h,w,1) 最有一维存放着confidence
    box_class_probs = K.sigmoid(feats[..., 5:])
    # box_class_probs=(batch_size,h,w,n_class) 最有一维存放着分类结果
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    ###总结，这个函数是将yolobody的输出转换为box的网格输出，从这里可以知道yolo的输出其实是输出x,y相对于网格的偏移量，box的长框系数,以及confidence
    return box_xy, box_wh, box_confidence, box_class_probs

## 修正box 并且得到在原图中左上右下坐标
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape)) #将原图缩放为输入图片的尺寸
    offset = (input_shape-new_shape)/2./input_shape ##误差
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale  #修正误差
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)  #左上角
    box_maxes = box_yx + (box_hw / 2.)  #右下角
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)  ###得到和y_true一样的格式
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    # boxes_shape=(-1,4),box_scores=(-1,n_class)
    return boxes, box_scores




##将标签变为yolo的输出形式
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape. m个样本t个框，5个指标
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  ##计算中心区域的位置 （m,T,2） 其实这里的T目前还是20，因为设置的最多盒子数量是20
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]          ##计算长宽 （m,T,2）
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]          #归一化
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]           #归一化

    m = true_boxes.shape[0]    #表示样本个数
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]  ## 三个Y的网格大小
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]  #所以一个y_true的OP 是（m,gw,gh,3,5+n_class）

    # Expand dim to apply broadcasting.  anchors_shape=(-1,2) 其实就是(9,2)
    anchors = np.expand_dims(anchors, 0) #变成了(1,9,2)
    anchor_maxes = anchors / 2.      #因为xy都变成了中心点位置 不在是左上和右下角了,所以anchor也要除半
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0  #(（m,T,bool）)  有box的地方为true，因为盒子数量可能少于20

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  #这时候是(-1,2),-1表示一张图片盒子的数量
        if len(wh)==0: continue              #如果图片不存在目标 就下一张图片
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)   ##(-1,1,2)
        box_maxes = wh / 2.          ##这里和anchors一样的操作
        box_mins = -box_maxes

        ##这部分是求各个盒子和anchors的IOU
        intersect_mins = np.maximum(box_mins, anchor_mins) #(-1,9,2) 所有盒子和所有的anchors做比较
        intersect_maxes = np.minimum(box_maxes, anchor_maxes) #(-1,9,2) 所有盒子和所有的anchors做比较
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area) #(-1,9) 所有盒子对应9个anchor的iou

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)  #（-1）  每个盒子对应的anchors最大的index

        for t, n in enumerate(best_anchor):  #t表示盒子的索引，n表示对应anchors的索引
            for l in range(num_layers): #I:0,1,2
                if n in anchor_mask[l]: #判断该anchor是否是该层的anchor
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32') #该样本的该盒子的x*网格的大小尺度 就是对应的该张量的位置
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')  #其实就是将在原图上box的中心位置

                    # 转换到输出张量上对应的尺度的位置
                    k = anchor_mask[l].index(n)  #该anchor在对应张量Y上对应anchors的索引
                    c = true_boxes[b,t, 4].astype('int32')   #box的类别
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1    # 对应类别的confidence为1  #比如如果是相同的BOX的尺度，那么在这个张量上的对应的网格的位置是相同的
                    y_true[l][b, j, i, k, 5+c] = 1   # 另对应类别为1

    return y_true

##计算iou  注意输入的b1,b2形状是不一样的
def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    args=[*yolo_outputs,*y_true]
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]  ## 这里的arg 前num_layers个是yolobody的输出张量，后面是多个y_true的输入张量 见train.py
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))  #yolo_output 应该是（batch_size,h,w,3*(5 + 80) = 255） 所以这里取的是模型的输出尺寸
                                                                                    #乘32是因为第一个是以32的比例切割的，则元素大小要乘32
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)] #三个输出张量y1,y2,y3的h,w尺寸
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor  得到tensor的batch_size
    mf = K.cast(m, K.dtype(yolo_outputs[0])) ##保证所有的op的类型都和y_true是一样的

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]  #...表示:,:,:  可以少些几个冒号而已 这里取的 (x, y, w, h, confidence)中的confidence
        true_class_probs = y_true[l][..., 5:] #这里取的是类别

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],  ##输入的分别是yolo_body的输出张量Y，9个锚中的三个，分类数量，最大张量的大小
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid  #真实的x,y偏移量 （之前归一化了）
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]) #长宽做了个Log的变换
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf #如果confidence是1者有值，0的话为0值
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]  # h*w

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)  ##可以 随时变换size的tensor矩阵
        object_mask_bool = K.cast(object_mask, 'bool')  ##这个是真实的mask 就是网格的confidence
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0]) #得到的是（j,4）的张量，j表示所有cell的数量，4表示x,y,h,w
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='./weight/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape   #输入的图片大小
    num_anchors = len(anchors)   #钩子的数量

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]   #三种比例的label [input((h,w,n_anchors,n_class))*3]
                                                            ## num_classes+5 其中5是因为每个框还需要(x, y, w, h, confidence)五个基本参数

    model_body = yolo_body(image_input, num_anchors//3, num_classes)   ### 输入到yolo body中 得到主体模型
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:    #是否加载预训练的模型
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:    #是否冻结所有层 还是保留最后3层输出
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,'print_loss':False})(
        [*model_body.output, *y_true])   #keras 的lambda层 一般是用来做op的运算的简单layer，这里是用来构建loss层
                                          #   *可以让多参以list的形式输入   yolo_loss这个函数见model.py
    model = Model([model_body.input, *y_true], model_loss)
    model.summary()
    return model

###解析xml标注文件
def convert_annotation(f_name):
    list_file = open('manul_label.txt', 'a+')  #生成标签的.txt文件
    in_file = open('./labels/'+f_name,encoding='utf8')  ## xml文件的路径
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        # cls = obj.find('name').text
        # if cls not in classes or int(difficult)==1:
        #     continue
        # cls_id = classes.index(cls)
        cls_id=0  #自己设置为0  因为只有一个类
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(f_name[:-4]+"," + ",".join([str(a) for a in b]) + ',' + str(cls_id)) #一行的保存格式为ID,xmin,ymin,xmax,ymax,cls_id
        list_file.write('\n')
#
def creat_label2txt(path='./labels/'):
    f_labels=os.listdir(path)  #path为存放xml文件的路径
    for f_name in f_labels:
        if f_name[-3:]=='xml':  #查找xml的文件
            convert_annotation(f_name)  #解析并写入txt中

##从path中读取anchors.txt return shape=(9,2)
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
##generate
def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)  #每遍历完一次数据集就在打乱一次
            image, box = get_random_data(annotation_lines[i], input_shape, random=True) #input_shape是输入模型的大小，不是真实图片的大小
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)  #(batch_size,T,5),T表示box的个数
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

##解析标签文件，得到数据 这里的输入是一行的数据annotation_line[i]=[id,xmin,ymin,xmax,ymax,clas_id]
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split(',')
    img_path='../train/lip_train/' ##注意这里填写训练图片的路径
    fs=re.split('_',line[0])[0]
    image = Image.open(img_path+fs+'/'+line[0]+'.png')
    # print(image)
    iw, ih = image.size
    h, w = input_shape
    # box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])  #这里box的形状应该是(-1,5)  左上和右下+类别标签
    #由于我们只有一个box 所以不需要上面这条
    box = np.array([np.array(list(map(int,line[1:]))) ])
    if not random:
        # resize image
        scale = min(w/iw, h/ih)  #以比例更小的那个为放大缩小标准
        nw = int(iw*scale)      #真实图片的h,W都以相同的最小的比例放大或者缩小
        nh = int(ih*scale)
        dx = (w-nw)//2           #改变了大小之后 计算原图和模型所需要图大小之间的误差取一半 这样保证上下左右都有一定的空隙
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)   #将原图resize
            new_image = Image.new('RGB', (w,h), (128,128,128))      #创建模型图大小的画布
            new_image.paste(image, (dx, dy))       #将原图放到所创建的模型图大小的画布上
            image_data = np.array(new_image)/255.   #归一化我们所需要的图片像素

        # correct boxes
        box_data = np.zeros((max_boxes,5))  #创建一个（20,5)的0矩阵
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx  #box的尺度也要对应变换，然后加上偏移量
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box   #(20,5)  只有前len(box)有值，后面的都为0
        #返回输入所需要的图片尺度，返回对应变化后的里面的box
        return image_data, box_data
    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

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

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
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
    arr = np.asarray(arr)

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
    hsv = np.asarray(hsv)

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

from timeit import default_timer as timer


from keras import backend as K
from keras.models import *
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import os
from keras.utils import multi_gpu_model
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.5,
              iou_threshold=.4):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32   ##输入大小
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)  ##三个尺度的所有box concate  (-1,4) 4为左上右下坐标
    box_scores = K.concatenate(box_scores, axis=0) ##三个尺度的所有scores concate  (-1,4)

    mask = box_scores >= score_threshold   ##保留大于阈值的
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')  ##一个常数tensor
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.

        class_boxes = tf.boolean_mask(boxes, mask[:, c]) #第c类的mask，取得第c类的所有boxs  class_boxes_shape=(-1,4)
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])#第c类的mask，取得第c类的所有scores  class_box_scores_shape=(-1,1)
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)  ##非极大化抑制，先选择秩信区间大的框，其他框如果和他的iou过大着删除
        class_boxes = K.gather(class_boxes, nms_index)  #得到非极大化抑制之后的框
        class_box_scores = K.gather(class_box_scores, nms_index)  ##同样得到那些框的分数
        classes = K.ones_like(class_box_scores, 'int32') * c  #1乘类的标签  告诉你这是哪一类
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)  ##把所有类的盒子合并起来 （-1,4）
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)
    #返回 这张图片中所有检测到的box和类，以及分数
    return boxes_, scores_, classes_

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class YOLO(object):
    _params={
        # 'score':0.3,
        # 'iou':0.7,
        'model_image_size':(416,416),
        'anchors_path':'./yolo3/yolo_anchors.txt',
        'model_path':'./yolo3/weight/trained_weights_stage_1.h5'
    }
    @classmethod
    def get_params(cls,n):
        if n in cls._params:
            return cls._params[n]

    def __init__(self,**kwargs):
        self.__dict__.update(self._params)
        self.__dict__.update(kwargs)
        # self.class_name=self._get_class()
        self.anchors=self._get_anchors()
        self.score=0.1
        self.iou=0.5
        self.sess=K.get_session()
        self.boxes,self.scores,self.classes=self.generate()


    def _get_anchors(self):
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    def generate(self):
        model_path=self.model_path
        # num_anchors=len(self.anchors)
        # num_classes=1
        # try:
        #     self.yolo_model = load_model(model_path, compile=False)
        # except:
            # print('load model failes')
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), 9 // 3, 1)
        self.yolo_model.load_weights('./yolo3/weight/best_weight.h5',by_name=True,skip_mismatch=True)  # make sure model, anchors and classes match

        # self.yolo_model=load_model(model_path,compile=False)
        # self.yolo_model.load_weights('./weight/ep033-loss3.816-val_loss3.894.h5')
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes,scores,classes=yolo_eval(self.yolo_model.output,self.anchors,1,self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes,scores,classes

    def detect_image(self,image):
        start=timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
               # K.learning_phase(): 0
            })
        try:
            # print(out_scores)
            if len(out_boxes)==0:
                left, top , right , bottom =0,0,0,0
                return left, top , right , bottom
            else:
                box = out_boxes[0]
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                # print('lip', (left, top), (right, bottom))
                return left-10, top-5, right+10, bottom+5
        except Exception as e:
            print(e)
    def close_session(self):
        self.sess.close()

def train_yolo_model(weight_path='./weight/',manul_label_path='manul_label.txt',anchors_path='yolo_anchors.txt'):

    log_dir = weight_path
    annotation_path = manul_label_path
    num_classes = 1
    input_shape = (416, 416)
    anchors = get_anchors(anchors_path) # 读取anchors
    val_split = 0.3
    batch_size = 21
    with open(annotation_path) as f:  # 读取标签
        lines = f.readlines()  # 依行读取
    # 划分数据
    np.random.seed(888899)
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    print('train:', num_train)
    print('val:', num_val)


    model = create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path=log_dir+'best_weight.h5')  # make sure you know what you freeze
    # model = create_tiny_model(input_shape, anchors, num_classes,
    #                           freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    jsonss = model.to_json()
    open(log_dir+'yolo_architeture.json', 'w').write(jsonss)
    model.load_weights(log_dir + 'best_weight.h5', by_name=True)
    model.compile(optimizer=Adam(lr=1e-4), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    # logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'best_weight.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1)
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=800,#max(1, num_train // batch_size)
                        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                               num_classes),
                        validation_steps=100,#max(1, num_val // batch_size),
                        epochs=200,
                        initial_epoch=0,
                        callbacks=[ checkpoint,early_stopping,reduce_lr])
    # if True:
    #     for i in range(len(model.layers)):
    #         model.layers[i].trainable = True
    #     model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    #     print('Unfreeze all of the layers.')
    #
    batch_size = 32 # note that more GPU memory is required after unfreezing the body
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=100,
        initial_epoch=50,
        callbacks=[ checkpoint, reduce_lr, early_stopping])
    # model.save(log_dir + 'trained_weights_stage_1.h5')

# if __name__ == '__main__' :
    # step 1
    # creat_label2txt(path='./labels/')  # 提取xml标签文件信息转化为txt文本形式
    # step 2
    # 运行 kmeans.py
    # step 3  训练yolo3
    # train_yolo_model()

    # yolo=YOLO()  #yolo对象，用于后面调用 裁剪图片
