from PIL import Image
from matplotlib import pyplot as plt  
import pandas as pd  
import cv2
import numpy as np
from pathlib import PurePath
import os

with open('/home/manli/3rd_trim_ultrasounds/data/CUB_200_2011/images.txt', 'r') as f:
    image_list = f.read()

with open('/home/manli/3rd_trim_ultrasounds/data/CUB_200_2011/parts/part_locs.txt', 'r') as f:
    partloc_list = f.read()

image_list = image_list.split('\n')[:-1]
partloc_list = partloc_list.split('\n')[:-1]
part_label_dict = {
    # 1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:7,12:8,13:9,14:11,15:12
    1:1,2:2,3:1,4:1,5:2,6:2,7:2,8:1,9:1,10:2,11:2,12:1,13:1,14:1,15:2
}

for kaokao in range(len(image_list)):
    parts = []
    for i in range(kaokao*15, (kaokao+1)*15):
        partloc = partloc_list[i].split(' ')
        assert eval(partloc[0]) == kaokao+1
        part_label = eval(partloc[1])
        # marge parts
        part_label = part_label_dict[part_label]
        part_y = eval(partloc[2])
        part_x = eval(partloc[3])
        stat = eval(partloc[4])
        if stat:
            parts.append([part_label, part_x, part_y])

    part_label = np.array([x[0] for x in parts])
    part_x = np.array([x[1] for x in parts])
    part_y = np.array([x[2] for x in parts])

    img_dir = '/home/manli/3rd_trim_ultrasounds/data/CUB_200_2011/images/' + image_list[kaokao].split(' ')[1]
    img = cv2.imread(img_dir)
    seg_dir = '/home/manli/3rd_trim_ultrasounds/data/CUB_200_2011/segmentations/' + image_list[kaokao].split(' ')[1].replace('jpg', 'png')
    seg = cv2.imread(seg_dir, 0)
    # seg = seg > 0
    seg = seg / 255.

    # x = np.array(range(seg.shape[0]))
    # y = np.array(range(seg.shape[1]))
    # x, y = np.meshgrid(x, y, indexing='ij')

    # diff_x = np.expand_dims(x, axis=-1) - np.expand_dims(np.expand_dims(part_x, axis=0), axis=0)
    # diff_y = np.expand_dims(y, axis=-1) - np.expand_dims(np.expand_dims(part_y, axis=0), axis=0)
    # dist = diff_x**2 + diff_y**2
    # allign = np.argmin(dist, axis=-1)

    # allign_copy = allign.copy()
    # for n in range(len(part_label)):
    #     allign_copy[allign==n] = part_label[n]
    # allign = allign_copy
    # seg = seg > 0.5
    # seg = seg * allign
    # seg = np.expand_dims(seg, axis=-1)
    # seg = np.concatenate((seg, seg, seg), axis=-1)
    # for i in range(allign.max()+1):
    #     seg[..., i] = seg[...,i]*(allign==i)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.subplot(1,4,2)
    # plt.imshow(seg[...,0])
    # plt.subplot(1,4,3)
    # plt.imshow(seg[...,1])
    # plt.subplot(1,4,4)
    # plt.imshow(seg[...,2])
    # plt.subplot(1,2,2)
    # plt.imshow(seg)
    # plt.show()

    seg_path = "/home/manli/3rd_trim_ultrasounds/data/CUB_200_2011/processed_segmentations/" + image_list[kaokao].split(' ')[1].replace('jpg', 'npy')

    # seg = np.asarray(seg, dtype=np.int32)
    # seg = Image.fromarray(seg)
    if not os.path.exists('/'.join(seg_path.split('/')[:-1])):
        os.mkdir('/'.join(seg_path.split('/')[:-1]))
    # seg.save(seg_path)
    np.save(seg_path, seg)
    # print(np.unique(seg))
    # seg = np.load(seg_path, allow_pickle=True)
    # print(np.unique(seg))
    # plt.figure()
    # plt.subplot(1,4,1)
    # plt.imshow(img)
    # plt.subplot(1,4,2)
    # plt.imshow(seg[...,0])
    # plt.subplot(1,4,3)
    # plt.imshow(seg[...,1])
    # plt.subplot(1,4,4)
    # plt.imshow(seg[...,2])
    # plt.show()
    print(kaokao)