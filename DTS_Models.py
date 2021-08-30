import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint


def DTS_Net( dataset = 'NYUV2'):
    if dataset == 'VOC':
        shape= (480,480,3)
        cls_cnt = 20
    elif dataset == 'CITYSCAPES':
        shape= (512,1024,3)
        cls_cnt = 19
    elif dataset == 'NYUV2':
        shape= (480,640,3)
        cls_cnt = 13
        
    base = tf.keras.applications.Xception(input_shape = shape, weights= None,include_top=False, pooling= None)
    out1 = Conv2D(1024, (1,1), activation = 'linear')(base.output)
    out_f = tf.nn.depth_to_space(out1, 32)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out2 = Dense(cls_cnt, activation = 'sigmoid')(avg)
    out11 = tf.keras.layers.Activation('linear')(out_f)
    model = Model(base.input, [out11, out2])
    return model
    
    
def DTS_Net_Lite(dataset = 'NYUV2'):
    if dataset == 'VOC':
        shape= (256,256,3)
        cls_cnt = 20
    elif dataset == 'CITYSCAPES':
        shape= (352,704,3)
        cls_cnt = 33
    elif dataset == 'NYUV2':
        shape= (256,320,3)
        cls_cnt = 13
        
    base = tf.keras.applications.MobileNetV2(input_shape = shape, weights= None,include_top=False, pooling= None)
    out1 = Conv2D(1024, (1,1), activation = 'linear')(base.output)
    out_f = tf.nn.depth_to_space(out1, 32)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out2 = Dense(cls_cnt, activation = 'sigmoid')(avg)
    out11 = tf.keras.layers.Activation('linear')(out_f)
    model = Model(base.input, [out11, out2])
    return model
    
def Joint_DTS_Net(dataset = 'NYUV2'):
    if dataset == 'CITYSCAPES':
        shape= (512,1024,3)
        cls_cnt = 33
    elif dataset == 'NYUV2':
        shape= (480,640,3)
        cls_cnt = 13
        
    base = tf.keras.applications.Xception(input_shape = shape, weights= None,include_top=False, pooling= None)
    out1 = Conv2D(1024, (1,1), activation = 'linear')(base.output)
    out_f = tf.nn.depth_to_space(out1, 32)
    out2 = Conv2D(1024, (1,1), activation = 'linear')(base.output)
    out_f2 = tf.nn.depth_to_space(out2, 32)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out3 = Dense(cls_cnt, activation = 'sigmoid')(avg)
    model = Model(base.input, [out_f, out_f2, out3])
    return model
