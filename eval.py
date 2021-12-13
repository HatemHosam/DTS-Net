import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from DTS_Models import DTS_Net, DTS_Net_Lite, Joint_DTS_Net 
import numpy as np
import sys
from Filteration import NLF
from depth_metrics import compute_errors
import time
import matplotlib.pyplot as plt

#choose dataset to load the corresponding weights 
#'NYUV2', 'VOC', and 'CITYSCAPES'
dataset = 'CITYSCAPES'
lite = False
joint = True

errors = []

if dataset == 'VOC':
    images_path = 'pascal VOC/JPEGImages/'
    seg_maps_path = 'pascal VOC/segClass/'
    val_file = 'val_VOC.txt'
    if lite:
        shape= (256,256)
    else:
        shape= (480,480)
    cls_cnt = 21
elif dataset == 'CITYSCAPES':
    images_path = 'leftImg8bit_trainvaltest/leftImg8bit/val/'
    seg_maps_path = 'gtFine/seg/val/'
    depth_path = 'disparity_trainvaltest/disparity/val/'
    val_file = 'val_CITYSCAPES.txt'
    if lite:
        shape= (704,352)
    else:
        shape= (1024,512)
    cls_cnt = 19
elif dataset == 'NYUV2':
    images_path = 'NYUv2 semantic segmentation/images/'
    seg_maps_path = 'NYUv2 semantic segmentation/images_seg/'
    depth_path = 'NYU depthv2/depth2/'
    val_file = 'val_NYUV2.txt'
    if lite:
        shape= (320,256)
    else:
        shape= (640,480)
    cls_cnt = 13

#load model choose between DTS_Net and DTS_Net_Lite and load weights
if lite:
    model = DTS_Net_Lite(dataset= dataset)
elif joint:
    model = Joint_DTS_Net(dataset= dataset)
else:
    model = DTS_Net(dataset= dataset)

weights_path = 'weights/weights_DTS_Net_VOC2012.h5'   #change line for corresponding weights 
model.load_weights(weights_path)


with open(val_file,'r') as f:
    val_data = f.readlines()
val_list = [] 

for data in val_data:
    m = tf.keras.metrics.MeanIoU(num_classes=cls_cnt)
    old = time.time()
    if dataset == 'NYUV2':
        img_name = str(int(data.split('_')[-1].split('.png')[0]))+'.png'
        image = cv2.imread(images_path+img_name)
    if dataset == 'VOC':
        img_name = data.split('\n')[0]+'.png'
        image = cv2.imread(images_path+img_name)
    if dataset == 'CITYSCAPES':
        img_name = data.split('\n')[0]
        image = cv2.imread(images_path+img_name)
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

    seg_map = NLF(seg_map , gl ,dataset = dataset, lite = lite, cls_cnt= cls_cnt)
    
    now = time.time()
    print(now-old)
    
    if dataset == 'NYUV2':
        seg = cv2.imread(seg_maps_path+data.split('\n')[0], cv2.IMREAD_GRAYSCALE)
        if joint:
            depth = np.load(depth_path+str(int(data.split('_')[-1].split('.png')[0]))+'.npy')*10.0
            pred = depth_map
            gt = depth
            mask = np.logical_and(pred, gt)
            errors.append(compute_errors(gt[mask], pred[mask]))
    if dataset == 'VOC':
        seg = cv2.imread(seg_maps_path+data.split('\n')[0]+'.png', cv2.IMREAD_GRAYSCALE)
    if dataset == 'CITYSCAPES':
        seg = cv2.imread(seg_maps_path+img_name.replace('leftImg8bit.png','gtFine_labelIds.png'), cv2.IMREAD_GRAYSCALE)
        seg[seg<0] = 0
        seg[seg==1] = 0
        seg[seg==2] = 0
        seg[seg==3] = 0
        seg[seg==4] = 0
        seg[seg==5] = 0
        seg[seg==6] = 0
        seg[seg==7] = 1
        seg[seg==8] = 2
        seg[seg==9] = 0
        seg[seg==10] = 0
        seg[seg==11] = 3
        seg[seg==12] = 4
        seg[seg==13] = 5
        seg[seg==14] = 0
        seg[seg==15] = 0
        seg[seg==16] = 0
        seg[seg==17] = 6
        seg[seg==18] = 0
        seg[seg==19] = 7
        seg[seg==20] = 8
        seg[seg==21] = 9
        seg[seg==22] = 10
        seg[seg==23] = 11
        seg[seg==24] = 12
        seg[seg==25] = 13
        seg[seg==26] = 14
        seg[seg==27] = 15
        seg[seg==28] = 16
        seg[seg==29] = 0
        seg[seg==30] = 0
        seg[seg==31] = 17
        seg[seg==32] = 18
        seg[seg==33] = 19  
        seg[seg>33] = 19
        if joint:
            depth = cv2.imread(depth_path+img_name.replace('leftImg8bit.png','disparity.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth = cv2.resize(depth, dsize=shape, interpolation=cv2.INTER_LINEAR)
            depth[depth > 0] = (depth[depth > 0] - 1) / 256
            pred = depth_map
            gt = depth
            mask = np.logical_and(pred, gt)
            errors.append(compute_errors(gt[mask], pred[mask]))
    seg = cv2.resize(seg, dsize=shape, interpolation=cv2.INTER_NEAREST)
    if dataset == 'VOC':
        m.update_state(seg, seg_map)
    else:
        m.update_state(seg[seg!=0], seg_map[seg_map!=0])
    miou_val = m.result().numpy()
    val_list.append(miou_val)

print('mean IOU: ', sum(val_list)/len(val_list))
if joint:
    mean_errors = np.array(errors).mean(0)  
    print('mean depth errors: ',mean_errors)
