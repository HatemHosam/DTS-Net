import numpy as np

def NLF(seg_map , global_labels, dataset = 'NYUV2', lite = True, cls_cnt= 14):
    if dataset == 'VOC':
        if lite:
            img_w, img_h = 256, 256
            size1, size2 = 32, 32
        else:
            img_w, img_h = 480, 480
            size1, size2 = 48, 48
    elif dataset == 'CITYSCAPES':
        if lite:
            img_w, img_h = 512, 256
            size1, size2 = 64, 32
        else:
            img_w, img_h = 1024, 512
            size1, size2 = 128, 64
    elif dataset == 'NYUV2':
        if lite:
            img_w, img_h = 320, 240
            size1, size2 = 40, 30
        else:
            img_w, img_h = 640, 480
            size1, size2 = 64, 48
        
    for i in range(int(img_h/size2)):
        for j in range(int(img_w/size1)):
            a = seg_map[i*size2:(i+1)*size2,j*size1:(j+1)*size1]
            local_labels = np.unique(a)
            real_labels = []
            for label in local_labels:
                if label in global_labels:
                    real_labels.append(label)
            for label in local_labels:
                if label not in real_labels:
                    min_dist = cls_cnt
                    for cls in real_labels:
                        dist = np.abs(label - cls)
                        if dist < min_dist:
                            min_dist = dist
                            filtered_label = cls
                    #map = seg_map[i*size2:(i+1)*size2,j*size1:(j+1)*size1]
                    a[a == label] = filtered_label
    return seg_map