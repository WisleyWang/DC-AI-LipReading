from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split
from keras.optimizers import *

import sys
from tensorflow import keras
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

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

##胶囊网络
class Capsule(Layer):
    def __init__(self, num_capsule=24, dim_capsule=128, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule  # cap的数量
        self.dim_capsule = dim_capsule  # cap的维度
        self.routings = routings        # 动态路由的轮数
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1] # 输入的维度
        if self.share_weights:  # 比如输入的是（None,12,200）
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule, # kernel shape=(1,200,n*dim)
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule, # shape=(12,200,n*dim)
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs,**kwge):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1]) # shape=(None,input_n_capsule,num_capsule*dim_capsule)
        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3)) # shape=[None,num_capsule,input_num_capsule,dim_capsule
        # ]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        #动态路由部分
        for i in range(self.routings):

            c = K.softmax(b,1)# (None,num_capsule, input_num_capsule)
       #output_shape: (None, 24, 128)
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2])) # (None,num_capsule, input_num_capsule) .
                                                                    # [None,num_capsule,input_num_capsule,dim_capsule]
                                                        #[None,num_capsule,num_capsule]
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    def get_config(self):
        config = {
            'routings': self.routings,
             ' num_capsule':self.num_capsule,
            ' dim_capsule':self.dim_capsule,
        }
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





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

##定义BN层，主要是加了激活函数和drop 但是drop一般不用了
def _bn_relu(layer, dropout=0, **params):
    # from keras.layers import BatchNormalization


    layer = BatchNormalization(axis=-1)(layer)
    layer =Activation('relu')(layer) ##激活的函数conv_activation，一般是relu

    if dropout > 0:
        layer = SpatialDropout3D(params["conv_dropout"])(layer)

    return layer

def resblock(inputs,filters,strides,name):
    shortcut=inputs
    x=Conv1D(filters,kernel_size=5,strides=strides,padding='same',name='blockcon1_'+name)(inputs)
    x=_bn_relu(x)
    x=Conv1D(filters,kernel_size=5,strides=1,padding='same',activation='relu',name='blockconv2_'+name)(x)
    if strides!=1:
        shortcut=MaxPool1D(2,padding='same')(shortcut)
    if int(inputs.shape[-1]) != filters:
        shortcut = Conv1D(filters, 1, padding='SAME',name='shortcut_'+name)(shortcut)
    x=Add()([shortcut,x])
    return x
def resconv1D(inputs,start_filters,s_list):
    x=Conv1D(start_filters,kernel_size=5,strides=1,padding='same',name='first_1d')(inputs)
    for i,s in enumerate(s_list):
        rate=2**(i//4)
        x=resblock(x,filters=start_filters*rate,strides=s,name=str(i))
        if i==len(s_list)-1:
            x=Conv1D(start_filters*rate,5,padding='same',activation='relu',name='last_conv')(x)

    return x
def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
##---------------------------------------------------------------------

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

def INC(inputs,filters):
    x1=Conv3D(filters,kernel_size=1,strides=1,padding='same',kernel_initializer='he_normal')(inputs)
    x1=Conv3D(filters,kernel_size=3,strides=1,padding='same',kernel_initializer='he_normal')(x1)
    x2 = Conv3D(filters, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x2 = Conv3D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(x2)
    x3=MaxPool3D((3,3,3),strides=1,padding='same')(inputs)
    x3=Conv3D(filters, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(x3)
    x4=Conv3D(filters,kernel_size=1,strides=1,padding='same',kernel_initializer='he_normal')(inputs)
    x=concatenate([x1,x2,x3,x4],axis=-1)
    return x
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
               kernel_initializer='he_normal', padding='same',dilation_rate=(1,2,2) )(inputs)
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
def seq_BLOCK(seq, filters): # 定义网络的Block
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(filters, 1, padding='SAME')(seq)
    seq = add([seq, cnn])
    return seq

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
    x = Conv3D(16, (7, 7, 7), strides=2, padding='same',name='start')(inputs)
    x=efficent_ADD(input=inputs,filters=32,stride=(1,2,2))
    x=TTL(x,filters=32,time_size=[1,3,6],ttl=1,**params)
    x=efficent_ADD(input=x,filters=64,stride=(1,2,2))
    x = TTL(x, filters=64, time_size=[1, 3, 4], ttl=2, **params)
    x = efficent_ADD(input=x, filters=96, stride=(1, 2, 2))
    x=Conv3D(96, (5, 3, 3), strides=(1,1,1),
           kernel_initializer='he_normal', padding='same')(x)
    x=BatchNormalization(axis=-1, momentum=0.99,epsilon=1e-3)(x)
    x=Swish()(x)
    # x = Bidirectional(ConvLSTM2D(96, kernel_size=3, strides=2, padding='same',return_sequences=True))(x)
    feature=TimeDistributed(GlobalAveragePooling2D())(x)
    x=Bidirectional(LSTM(256,return_sequences=True,kernel_initializer='Orthogonal', name='lstm11'),
                           merge_mode='concat',name='bid1')(feature)
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm12'),
                          merge_mode='concat', name='bid2')(x)


    x = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2'),
                      merge_mode='concat', name='gru1')(x)
    # x = resconv1D(x, 64, s_list=[1, 1, 2, 1, 1, 1, 2,1,2])
    x=seq_BLOCK(x,64)
    x = MaxPooling1D(2)(x)
    x = seq_BLOCK(x, 128)
    x = MaxPooling1D(2)(x)
    x = seq_BLOCK(x, 256)
    # x = GlobalMaxPooling1D()(x)
    x=Flatten()(x)
    # x=Dropout(0.2)(x)
    dense1 = Dense(313, kernel_initializer=EfficientNetDenseInitializer(), name='dense1')(x)
    y_pred = Activation('softmax', name='softmax')(dense1)


    models=Model(inputs,y_pred)
    models.summary()
    return models

def score_loss(y_true, y_pred):

    return 0.5*K.sqrt(K.mean(K.square(y_true-y_pred)))+(1-K.mean(K.sum(y_pred*y_true,axis=1)))

    # loss += K.sum(0.5 * K.sum(y_true_ * y_pred_,axis=0) / (K.sum(y_true_ + y_pred_,axis=0) + K.epsilon()))
    # return - K.log(loss + K.epsilon())

def mycrossentropy(y_true, y_pred, e=0.1):   # 交叉熵+均匀分布的的正则
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/313, y_pred)
    return (1-e)*loss1 + e*loss2
#----------I3D-------------
def I3D():
    inputs=Input((24,112,112,3))
    x=Conv3D(16,(7,7,7),strides=2,padding='same')(inputs)
    x=MaxPool3D((1,3,3),strides=(1,2,2),padding='same')(x)
    x=Conv3D(32,1,strides=1,padding='same')(x)
    x=Conv3D(32,3,strides=1,padding='same')(x)
    x=MaxPool3D((1,3,3),strides=(1,2,2),padding='same')(x)
    x=efficient_block(inputs=x, filters=32, stride=(1,1,1))
    x=INC(x,32)
    x=INC(x,32)
    x=MaxPool3D((3,3,3),strides=(2,2,2),padding='same')(x)
    x = INC(x, 64)
    x = INC(x, 64)
    x = INC(x, 64)
    x = INC(x, 64)
    x = INC(x, 64)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = INC(x, 128)
    x = INC(x, 128)

    x=Conv3D(512,1,strides=1,padding='same')(x)
    # x=Flatten()(x)
    x = GlobalAveragePooling3D()(x)
    dense1 = Dense(313, kernel_initializer=EfficientNetDenseInitializer(), name='dense1')(x)
    y_pred = Activation('softmax', name='softmax')(dense1)
    models=Model(inputs,y_pred)
    models.summary()
    return models
def EfficientCONvlstm(**params):
    inputs=Input((24,112,112,3))
    x = Conv3D(16, (7, 7, 7), strides=2, padding='same',name='start')(inputs)
    x=efficent_ADD(input=x,filters=32,stride=(1,2,2))
    x=TTL(x,filters=32,time_size=[1,3,6],ttl=1,**params)

    x=efficent_ADD(input=x,filters=64,stride=(1,1,1))
    x = TTL(x, filters=64, time_size=[1, 3, 4], ttl=2, **params)
    x = Bidirectional(ConvLSTM2D(64, kernel_size=3, strides=1, padding='same', activation='relu', return_sequences=True))(x)
    x = efficent_ADD(input=x, filters=96, stride=(1, 2, 2))
    x=Conv3D(96, (5, 3, 3), strides=(1,1,1),
           kernel_initializer='he_normal', padding='same')(x)
    x=BatchNormalization(axis=-1, momentum=0.99,epsilon=1e-3)(x)
    x=Swish()(x)
    # x=Bidirectional(ConvLSTM2D(96,kernel_size=3,strides=1,padding='same',activation='relu',return_sequences=True))(x)
    # x = Bidirectional(ConvLSTM2D(128, kernel_size=3, strides=1, padding='same', activation='relu', return_sequences=True))(x)
    feature = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm1'),
                      merge_mode='concat', name='bid1')(feature)
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2'),
                      merge_mode='concat', name='bid2')(x)
    x=Conv1D(256,12,strides=2,padding='valid',activation='relu')(x)
    x=BatchNormalization()(x)
    # x = Conv1D(256, 12, strides=1, padding='valid', activation='relu')(x)
    # x = BatchNormalization()(x)

    x=Flatten()(x)
    dense1 = Dense(313, kernel_initializer=EfficientNetDenseInitializer(), name='dense1')(x)
    y_pred = Activation('softmax', name='softmax')(dense1)
    model=Model(inputs,y_pred)
    model.summary()
    return model
