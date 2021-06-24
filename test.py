import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from DTS_Models import DTS_Net, DTS_Net_Lite 
import numpy as np
import sys
from Filteration import NLF
import matplotlib.pyplot as plt

#choose dataset to load the corresponding weights 
#'NYUV2', 'VOC', and 'CITYSCAPES'
dataset = 'NYUV2'
lite = False
joint = True
#datasets weights and image size options
if dataset == 'VOC':
    if lite:
        shape= (256,256)
    else:
        shape= (480,480)
    cls_cnt = 21
elif dataset == 'CITYSCAPES':
    if lite:
        shape= (704,352)
    else:
        shape= (1024,512)
    cls_cnt = 34
elif dataset == 'NYUV2':
    if lite:
        shape= (320,256)
    else:
        shape= (640,480)
    cls_cnt = 14
#the weights path of the model
weights_path = 'weights/weights_DTS_Net_VOC2012.h5' 
#load model choose between DTS_Net and DTS_Net_Lite
if lite:
    model = DTS_Net_Lite(dataset= dataset)
elif joint:
    model = Joint_DTS_Net(dataset= dataset)
else:
    model = DTS_Net(dataset= dataset)
#load weights   
model.load_weights(weights_path)

img_path = 'E:/NYUv2 semantic segmentation/images/42.png'
image = cv2.imread(img_path)
image = cv2.resize(image, shape)
img = image/255.
img = np.expand_dims(img, axis = 0)

if joint:
    seg_map, depth_map, class_pred = model.predict(img)
    depth_map = depth_map[0,:,:,0]
else:
     seg_map, class_pred = model.predict(img)
seg_map = seg_map[0,:,:,0]
class_pred = class_pred[0]
seg_map = np.rint(seg_map)
class_pred = np.rint(class_pred)
seg_map[seg_map<0] = 0
seg_map[seg_map>(cls_cnt-1)] = cls_cnt - 1
class_pred= np.insert(class_pred, 0, 1)

gl = np.where(class_pred==1)    #global labels
gl = gl[0]
#unfiltered_labels = np.unique(seg_map)

plt.figure('predicted_seg_no_filteration')
plt.imshow(seg_map, vmin = 0, vmax = cls_cnt-1)

seg_map = NLF(seg_map , gl ,dataset = dataset, lite = lite, cls_cnt= cls_cnt)

plt.figure('filtered_seg')
plt.imshow(seg_map, vmin = 0, vmax = cls_cnt-1)
if joint:
    plt.figure('predicted_depth')
    plt.imshow(depth_map)
plt.show()

