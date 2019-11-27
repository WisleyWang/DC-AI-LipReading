from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split
from keras.optimizers import *
import tensorflow as tf
import sys
from keras.callbacks import *
sys.path.append('./')
sys.path.append('./yolo3/')

from keras_efficientnets.custom_objects import *
from functools import reduce

class DropConnect3D(layers.Layer):

    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect3D, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += K.random_uniform([batch_size, 1, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {
            'drop_connect_rate': self.drop_connect_rate,
        }
        base_config = super(DropConnect3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({
    'EfficientNetConvInitializer': EfficientNetConvInitializer,
    'EfficientNetDenseInitializer': EfficientNetDenseInitializer,
    'DropConnect3D': DropConnect3D,
    'Swish': Swish,
})



class SelfAttention(Layer):
    def __init__(self,**kwargs):
        super(SelfAttention,self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel=self.add_weight(name='self_weight',shape=tuple(input_shape[2:])
                                    ,initializer='uniform',trainable=True)

        super(SelfAttention, self).build(input_shape)
    def call(self, inputs, **kwargs):
        y=inputs*self.kernel
        # print(inputs.shape)
        return y
    def compute_output_shape(self, input_shape):
        return input_shape


class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight)/lamb ) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)
class Swish(Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)
###定义时间卷积层
def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,name='res_conv_',
        **params):
    from keras.layers import Conv1D

    layer = TimeDistributed(Conv2D(
            filters=num_filters,
            kernel_size=filter_length,
            strides=subsample_length,
            padding='same',
            kernel_initializer=params["conv_init"],name=name))(layer)

    return layer
##定义BN层，主要是加了激活函数和drop 但是drop一般不用了
def _bn_relu(layer, dropout=0, **params):
    # from keras.layers import BatchNormalization


    layer = BatchNormalization(axis=-1)(layer)
    layer =Activation(params["conv_activation"])(layer) ##激活的函数conv_activation，一般是relu

    if dropout > 0:
        layer = SpatialDropout3D(params["conv_dropout"])(layer)

    return layer


## 定义resnet的block
def resnet_block( layer,
        num_filters,
        subsample_length,
         zero_pad, ##如果比初始层通道数多一倍则填充
        name='',
        **params):
    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=4)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 5
        shape[4] *= 2
        return tuple(shape)

    shortcut = TimeDistributed(AveragePooling2D(pool_size=subsample_length))(layer) ##池化层 ##如果subsample_length=1，就相当于没有池化

    # 看是否增加通道的数量 这里 "conv_increase_channels_at": 4 设置的是每4层就增加一次特征图的数量 是前一次的两倍
    # zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
    #     and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    # 对输入的layer做卷积，卷积的数量由conv_num_skip 决定，这里一般设置是2层的卷积之后在add下


    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        num_filters,  # 这里输入的特征图数量由 get_num_filters_at_index给出
        subsample_length ,  # 如果subsample_length是2  那么第一次先做步长为2的卷积来降维，在之后都是步长为1以保证和池化操作的维度一样
        name=name+'_'+str(1),
        **params)
    layer = _bn_relu(
        layer,
        dropout=params["conv_dropout"],
        **params)
    layer = TimeDistributed(SeparableConv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=params["conv_init"],name=name+'_'+str(2)))(layer)
    layer = _bn_relu(
        layer,
        dropout=params["conv_dropout"],
        **params)
    layer = Add()([shortcut, layer])
    return layer
###用来计算当前层特征图数量的
def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

## 万事大吉 现在来搭建resnet提取特征：
def add_resnet_layers(layer, **params):
    ##先进行一次卷积
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    ###开始迭代resnet ，conv_subsample_lengths是每个resnet_block的步长
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params) ##确定当前Block的特征图数量
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
def identity_block(input_tensor, filters,filter_size, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'



    x = Conv3D(filters1, (9,1,1),
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters2,filter_size,
                      kernel_initializer='he_normal',padding='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x
import tensorflow as tf
def zero_pading(x):
    # shape=list(K.int_shape(x))
    #
    #
    # shape[-1]=32
    # shape=shape[1:]
    y=K.zeros_like(x)
    # y=y[...,:32]
    s=K.concatenate([x,y],axis=-1)
    return s

def zero_pading_shape(input_shape):
    shape=list(input_shape)
    shape[-1]=shape[-1]*2
    return tuple(shape)
def conv_block(input_tensor,
               filters,filter_size,
               stage,stride,
               block,
               ):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x =  Conv3D(filters2, (9, 1,1), strides=(1,1,1),
                      kernel_initializer='he_normal',padding='same',
                      name=conv_name_base + '2a')(input_tensor)
    x =  BatchNormalization(axis=-1 , name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x =  Conv3D(filters3, filter_size, padding='same',strides=stride,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)
    shortcut=input_tensor
    if stride==(1,2,2):
        shortcut = AveragePooling3D( (1, 2,2),padding='same',strides=(1,2,2),
                             name=conv_name_base + '1')(shortcut)
    if filters3!=filters2:
        shortcut=Lambda(function=zero_pading,output_shape=zero_pading_shape)(shortcut)

    x =  add([x, shortcut])
    x = Activation('relu')(x)

    return x


def Resnet34(inputs):
    x = conv_block(inputs, [16, 16,16], filter_size=(9,5,5), stage=1, stride=(1, 1, 1), block='a')
    x = identity_block(x, [16, 16], filter_size=(9,5,5), stage=1, block='b')
    x = identity_block(x, [16, 16], filter_size=(9,5,5), stage=1, block='c')

    x = conv_block(x,  [16, 16, 32],filter_size=(9,5,5), stage=2,stride=(1,2,2), block='a')
    x = identity_block(x, [32, 32], filter_size=(9,5,5),stage=2, block='b')
    x = identity_block(x, [32, 32], filter_size=(9,5,5), stage=2, block='c')

    x = conv_block(x, [32, 32, 32], filter_size=(9,3,3), stage=3, stride=(1, 2, 2), block='a')
    x = identity_block(x, [32, 32], filter_size=(9,3,3), stage=3, block='b')
    x = identity_block(x, [32, 32], filter_size=(9,3,3), stage=3, block='c')
    # x = identity_block(x, [64, 64], filter_size=(9,3,3), stage=3, block='d')

    x = conv_block(x, [32, 32, 64], filter_size=(9,3,3),stage=4,stride=(1,2,2), block='a')
    x = identity_block(x, [64, 64],filter_size=(9,3,3), stage=4, block='b')
    x = identity_block(x, [64, 64], filter_size=(9,3,3),stage=4, block='c')
    # x = identity_block(x, [64, 64], filter_size=(9,3,3), stage=4, block='d')

    x = conv_block(x, [65, 64, 64],filter_size=(9,3,3), stage=5,stride=(1,2,2), block='a')
    x = identity_block(x,[64, 64], filter_size=(9,3,3),stage=5, block='b')
    x = identity_block(x, [64, 64], filter_size=(9,3,3), stage=5, block='c')
    x = identity_block(x, [64, 64], filter_size=(9,3,3), stage=5, block='d')
    # x = identity_block(x,  [ 64, 64], filter_size=(5,3,3),stage=4, block='c')
    # x = conv_block(x, [128, 128, 256], filter_size=(9,3,3), stage=6, stride=(1, 2, 2), block='a')
    # x = identity_block(x, [256, 256], filter_size=(9,3,3), stage=6, block='b')
    x = conv_block(x, [65, 64, 128], filter_size=(9,3,3), stage=6, stride=(1, 2, 2), block='a')
    x = identity_block(x, [128, 128], filter_size=(9,3,3), stage=6, block='b')
    x = identity_block(x, [128, 128], filter_size=(9,3,3), stage=6, block='d')

    return x

def Lip_net(**params):
    input_data = Input(name='the_input', shape=(24,112,112,3), dtype='float32')
    conv1 = Conv3D(16, (9,5,5), strides=(1, 1, 1), activation='relu',padding='same', kernel_initializer='he_normal',
                        name='conv_1')(input_data)
    res=Resnet34(conv1)
    resh1 = TimeDistributed(GlobalAveragePooling2D())(res)
    gru_1 = Bidirectional(LSTM(256,return_sequences=True,kernel_initializer='Orthogonal', name='lstm1'),
                           merge_mode='concat',name='bid1')(resh1)
    gru_2 =  Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2'),
                           merge_mode='concat',name='bid2')( gru_1)
    gru_2=Flatten()(gru_2)
    dense1 = Dense(313, kernel_initializer='he_normal', name='dense1')( gru_2)
    y_pred = Activation('softmax', name='softmax')( dense1)
    models=Model(input_data,y_pred)
    models.summary()
    return models

##---------------------------------------------------------------------
def Dens_conv(inputs,filters):

    x = Conv3D(filters, (1, 1, 1), strides=(1, 1, 1),
               kernel_initializer='he_normal', padding='same'
            )(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Swish()(x)
    x = Conv3D(filters, (9, 3, 3), strides=(1, 1, 1),
               kernel_initializer='he_normal', padding='same' )(x)
    x = BatchNormalization(axis=-1)(x)
    x = Swish()(x)
    return x
def Densblock(inputs,filters,block,n=3,**params):
    name = 'Dense' + str(block)
    # x1=Dens_conv(inputs, filters)
    # c1=concatenate([inputs,x1],axis=-1)
    # x2=Dens_conv(c1, filters)
    # c2 = concatenate([c1, x2], axis=-1)
    # x3 = Dens_conv(c2, filters)
    # c=concatenate([x1,x2,x3],axis=-1)

    x=inputs
    for i in range(n):
        fornt=x
        x=Dens_conv(x, filters)
        x = concatenate([fornt, x], axis=-1)

    x=Conv3D(filters,(1,1,1),strides=(1,1,1),kernel_initializer='he_normal',name=name+'_finall')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Swish()(x)
    return x
def TTL(inputs,filters,time_size,ttl,**params):
    name='TTL' + str(ttl)
    size1,size2,size3=time_size
    f_size1=(size1,1,1)
    f_size2=(size2,3,3)
    f_size3=(size3,3,3)
    x1 = Conv3D(filters, f_size1, strides=(1, 1, 1),
               kernel_initializer='he_normal', padding='same',
          name= name+'_1' )(inputs)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Swish()(x1)
    x2 = Conv3D(filters, f_size2, strides=(1, 1, 1),
               kernel_initializer='he_normal', padding='same',
                name=name + '_2' )(inputs)
    x2 = BatchNormalization(axis=-1)(x2)
    x2 = Swish()(x2)

    x3 = Conv3D(filters, f_size3, strides=(1, 1, 1),
                kernel_initializer='he_normal', padding='same',
                name=name + '_3' )(inputs)
    x3 = BatchNormalization(axis=-1)(x3)
    x3 = Swish()(x3)

    x=concatenate([x1,x2,x3],axis=-1)
    x=AveragePooling3D((1,2,2),strides=(1,2,2),padding='same')(x)
    return x
def CONV3Dlayer(**params):

    inputs=Input(name='the_input', shape=(24,112,112,3), dtype='float32')
    x=Conv3D(16, (9, 7,7), strides=(1,2,2),
                      kernel_initializer='he_normal',padding='same',
                      name='conv1')(inputs)
    x = BatchNormalization(axis=-1, name='bn_1')(x)
    x = Activation('relu')(x)
    # x=MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),padding='same', name='max1')(x)
    x=Densblock(x,24,block=1)
    x=TTL(x,24,[1,3,9],ttl=1,**params)
    x = Densblock(x, 24,n=3, block=2)
    x = TTL(x, 48, [1, 3, 6],ttl=2, **params)
    x = Densblock(x, 48, block=3,n=3)
    x = TTL(x, 96, [1, 3, 6], ttl=3, **params)
    x = Densblock(x, 96, block=4,n=3)
    x=TimeDistributed(GlobalAveragePooling2D())(x)
    # x=TimeDistributed(Dense(48))(x)
    x=Bidirectional(LSTM(256,return_sequences=True,kernel_initializer='Orthogonal', name='lstm1'),
                           merge_mode='concat',name='bid1')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2'),
                          merge_mode='concat', name='bid2')(x)
    x=Flatten()(x)
    dense1 = Dense(313, kernel_initializer=EfficientNetDenseInitializer(), name='dense1')(x)
    y_pred = Activation('softmax', name='softmax')(dense1)
    models=Model(inputs,y_pred)
    models.summary()
    return models
##---------------------------------------------------------------------
def effcient_wise(inputs,filters):
    x = inputs
    x = Lambda(lambda a: K.mean(a, axis=[2,3], keepdims=True))(x)
    x = Conv3D(
        filters//8,
        kernel_size=(1,1,1),
        strides=(1,1,1),
        kernel_initializer='he_normal',
        padding='same',
        use_bias=True)(x)
    x = Swish()(x)
    # Excite
    x = Conv3D(
        filters,
        kernel_size=(1, 1, 1),
        strides=(1, 1, 1),
        kernel_initializer='he_normal',
        padding='same',
        use_bias=True)(x)
    x = Activation('sigmoid')(x)
    out = Multiply()([x, inputs])
    return out

def efficient_block(inputs,filters,stride=(1,1,1)):
    x=Conv3D(filters, (5, 3, 3), strides=(1, 1, 1),
               kernel_initializer='he_normal', padding='same' )(inputs)
    x=BatchNormalization( axis=-1,momentum=0.99,epsilon=1e-3)(x)
    x=Swish()(x)
    x=Conv3D(filters, (5, 3, 3), strides=stride,
           kernel_initializer='he_normal', padding='same')(x)
    x=BatchNormalization(axis=-1, momentum=0.99,epsilon=1e-3)(x)
    x=Swish()(x)
    x=effcient_wise(x,filters=filters)
    x = Conv3D(filters, (5, 3, 3), strides=(1, 1, 1),
               kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization(axis=-1, momentum=0.99,epsilon=1e-3)(x)
    return x
def efficent_ADD(input,filters,stride=(1,1,1),deep=True):
    first=efficient_block(inputs=input,filters=filters,stride=stride)
    x=efficient_block(inputs=first,filters=filters*2,stride=(1,1,1))
    x = Conv3D(filters, (5, 3, 3), strides=(1, 1, 1),
               kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization(axis=-1, momentum=0.99,epsilon=1e-3)(x)
    x=DropConnect3D(0.5)(x)  #添加进去
    x=Add()([first,x])
    return x
def Efficient(**params):
    inputs=Input((24,112,112,3))
    x=efficent_ADD(input=inputs,filters=32,stride=(1,2,2))
    x=TTL(x,filters=32,time_size=[1,3,6],ttl=1,**params)
    x=efficent_ADD(input=x,filters=64,stride=(1,2,2))
    x = TTL(x, filters=64, time_size=[1, 3, 4], ttl=2, **params)
    x = efficent_ADD(input=x, filters=96, stride=(1, 2, 2))
    x=Conv3D(96, (5, 3, 3), strides=(1,1,1),
           kernel_initializer='he_normal', padding='same')(x)
    x=BatchNormalization(axis=-1, momentum=0.99,epsilon=1e-3)(x)
    x=Swish()(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x=Bidirectional(LSTM(256,return_sequences=True,kernel_initializer='Orthogonal', name='lstm1'),
                           merge_mode='concat',name='bid1')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2'),
                          merge_mode='concat', name='bid2')(x)
    x = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2'),
                      merge_mode='concat', name='gru1')(x)
    x = Flatten()(x)
    # x=Dropout(0.2)(x)
    dense1 = Dense(313, kernel_initializer=EfficientNetDenseInitializer(), name='dense1')(x)
    y_pred = Activation('softmax', name='softmax')(dense1)
    models=Model(inputs,y_pred)
    models.summary()
    return models
def score_loss(y_true, y_pred):
    loss = 0
    for i in np.eye(313):
        y_true_ = K.constant([list(i)]) * y_true
        y_pred_ = K.constant([list(i)]) * y_pred
        loss += 0.5 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
    return - K.log(loss + K.epsilon())

    # return 0.5*K.sqrt(K.mean(K.square(y_true-y_pred)))+(1-K.mean(K.sum(y_pred*y_true,axis=1)))