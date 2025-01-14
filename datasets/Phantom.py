'''
Fetal data, 3rd trim
'''
import os
import numpy as np
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
from glob import glob

# images = glob('/home/manli/src/datasets/FPU/Dataset_Plane/*/*.png')
# plane = list(map(lambda x: PurePath(x).parts[-2], images))
# df = {'images': images, 'Plane':plane}
# df = pd.DataFrame(df)
# df.to_csv('/home/manli/src/datasets/FPU/data.csv', index=False)
# sys.exit()

class Phantom(Dataset):
    def __init__(self):
        self.df = pd.read_csv('/home/manli/src/datasets/FPU/data.csv')
        self.image_names = self.df['images'].values

    def __getitem__(self, index):
        image_dir = self.image_names[index]
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

class PhantomPlane(Dataset):
    def __init__(self, split):
        self.df = pd.read_csv('/home/manli/src/datasets/FPU/data.csv')
        image_names = self.df['images'].values
        labels = self.df['Plane'].values

        if split == 'train':
            image_names = image_names[::2]
            labels = labels[::2]
        else:
            image_names = image_names[1::2]
            labels = labels[1::2]            

        self.image_names = image_names
        self.labels = labels


    def __getitem__(self, index):
        image_dir = self.image_names[index]
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

        label = self.labels[index]
        if label == 'AC_PLANE':
            label = 0
        elif label == 'FL_PLANE':
            label = 1
        elif label == 'NO_PLANE':
            label = 2
        elif label == 'BPD_PLANE':
            label = 3

        return image, label

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

if __name__ == "__main__":
    traindata = PhantomPlane('train')
    loader = DataLoader(traindata, batch_size=32, drop_last=False, shuffle=True)
