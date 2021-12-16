import tensorflow as tf
from tensorflow.python.keras.models import Model, model_from_json
import os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
import math


#base = tf.keras.applications.Xception(input_shape = (704,704,3), weights= 'imagenet',include_top=False, pooling= None)
base = tf.keras.applications.MobileNetV2(input_shape = (512,512,3), weights= 'imagenet',include_top=False, pooling= None)
out1 = Conv2D(1024, (1,1), activation = 'sigmoid')(base.output)
out_f = tf.nn.depth_to_space(out1, 32)
model = Model(base.input, out_f)
#model.load_weights('weights/Xception_deepGlobe.h5')     #Xception backbone weights
model.load_weights('weights/MobileNetV2_deepGlobe.h5')   #MobileNetV2 backbone weights
model.summary()

img_name = '280611_sat.jpg'
image = cv2.imread('E:/DeepGlobe/valid/'+img_name)
image = cv2.resize(image, (512,512))
img = image/255.
img = np.expand_dims(img, axis = 0)

seg_map = model.predict(img)
seg_map = seg_map[0,:,:,0]

# seg_map[seg_map< 0.1] = 0          #0.1 threshould works better for Xception
# seg_map[seg_map>=0.1] = 1

seg_map[seg_map< 0.2] = 0         #0.2 threshould works better for MobileNetV2
seg_map[seg_map>=0.2] = 1


plt.figure('predicted')
plt.imshow(seg_map, vmin = 0, vmax = 1, cmap='gray')
plt.show()

#plt.imsave('MN2_predicted_'+img_name,seg_map, vmin = 0, vmax = 1,cmap='gray' )
