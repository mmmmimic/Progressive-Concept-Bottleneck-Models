'''
Fetal data, 3rd trim
'''
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import albumentations as A
from torchvision import transforms as T
from PIL import Image
import cv2
import sys
from pathlib import PurePath
import sys
from tqdm import tqdm

class Barcelona(Dataset):
    def __init__(self):
        self.df = pd.read_csv('/home/manli/src/datasets/Barcelona/FETAL_PLANES_DB_data.csv')
        values = self.df.iloc[:,0]
        self.image_names = list(map(lambda x: x.split(';')[0], values))

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_dir = f"/home/manli/src/datasets/Barcelona/Images/{image_name}.png"
        image = cv2.imread(image_dir)
        if len(image.shape) == 2:
            tf = T.Compose(
                [
                    T.ToTensor(),
                    T.Resize((224, 288)),
                    T.Normalize((0.5), (0.5))
                ]
            )
            image = Image.fromarray(image)
            image = tf(image)

        elif len(image.shape) == 3:
            if image.shape[-1] > 3:
                image = image[...,:3]
            image = Image.fromarray(image)
            tf = T.Compose(
                [
                    T.ToTensor(),
                    T.Resize((224, 288)),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
            image = tf(image)
            image = torch.mean(image, dim=0, keepdims=True)
        else:
            print(image.shape)
            raise ValueError

        return image

    def __len__(self):
        return len(self.df)


class BarcelonaHead(Dataset):
    def __init__(self, split='train'):
        self.df = pd.read_csv('/home/manli/src/datasets/Barcelona/FETAL_PLANES_DB_data.csv')
        values = self.df.iloc[:,0]
        s = list(map(lambda x: int(x.split(';')[-1]), values))
        s = np.array(s)
        self.split = split

        if split == 'train':
            index = s==1
        else:
            index = s==0

        image_names = list(map(lambda x: x.split(';')[0], values))
        image_names = np.array(image_names)
        image_names = image_names[index]

        labels = list(map(lambda x: x.split(';')[2], values))
        labels = np.array(labels)
        labels = labels[index]

        brain_labels = list(map(lambda x: x.split(';')[3], values))
        brain_labels = np.array(brain_labels)
        brain_labels = brain_labels[index]

        index = (brain_labels != 'Not A Brain') * (brain_labels != 'Other')
        self.image_names = image_names[index]
        brain_labels = brain_labels[index]

        self.labels = []
        # for l, b in zip(labels, brain_labels):
        #     if l == 'Fetal femur':
        #         self.labels.append(0)
        #     elif l == 'Fetal abdomen':
        #         self.labels.append(1)
        #     elif l == 'Fetal brain':
        #         if b == 'Trans-thalamic':
        #             self.labels.append(2)
        #         else:
        #             self.labels.append(3)
        #     elif l == 'Maternal cervix':
        #         self.labels.append(4) 
        #     else:
        #         self.labels.append(5)
        for b in brain_labels:
            if b == 'Trans-thalamic':
                self.labels.append(0)
            # elif b == 'Trans-ventricular':
            #     self.labels.append(2)
            # elif b == 'Trans-cerebellum':
            #     self.labels.append(1)
            else:
                self.labels.append(1)

        # index = brain_labels!= 'Not A Brain'
        # brain_labels = brain_labels[index]
        # self.image_names = image_names[index]
        # self.brain_labels = list(map(lambda x: 6 if x!='Trans-thalamic' else 2, brain_labels))

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_dir = f"/home/manli/src/datasets/Barcelona/Images/{image_name}.png"
        image = cv2.imread(image_dir)
        label = self.labels[index]
        if len(image.shape) == 2:
            if self.split == 'train':
                tf = T.Compose(
                    [
                        T.ToTensor(),
                        T.RandomVerticalFlip(0.5),
                        T.RandomRotation(15),
                        T.Resize((224, 288)),
                        T.Normalize(0.5, 0.5)
                    ]
                )
            else:
                tf = T.Compose(
                    [
                        T.ToTensor(),
                        T.Resize((224, 288)),
                        T.Normalize((0.5), (0.5))
                    ]
                )
            image = Image.fromarray(image)
            image = tf(image)

        elif len(image.shape) == 3:
            if image.shape[-1] > 3:
                image = image[...,:3]
            image = Image.fromarray(image)
            if self.split == 'train':
                tf = T.Compose(
                    [
                        T.ToTensor(),
                        T.RandomVerticalFlip(0.5),
                        T.RandomRotation(15),
                        T.Resize((224, 288)),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )
            else:
                tf = T.Compose(
                    [
                        T.ToTensor(),
                        T.Resize((224, 288)),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )
            image = tf(image)
            image = torch.mean(image, dim=0, keepdims=True)
        else:
            print(image.shape)
            raise ValueError

        return image, label

    def __len__(self):
        return len(self.image_names)

    def get_labels(self):
        return self.labels

if __name__ == "__main__":

    traindata = BarcelonaHead('train')
    print(len(traindata))
    traindata = BarcelonaHead('test')
    print(len(traindata))    
    loader = DataLoader(traindata, batch_size=32, drop_last=False, shuffle=True)
    from collections import Counter
    print(Counter(traindata.labels))
    plt.figure()
    for image, label in tqdm(loader):
        # print(image.shape)
        print(label)
        plt.imshow((image[10,...].squeeze().detach().cpu().numpy()+1)/2)
        plt.show()