"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


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


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
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
