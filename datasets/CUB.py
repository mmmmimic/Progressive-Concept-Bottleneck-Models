'''
CUB data
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
import torchvision

class CUBSeg(Dataset):
    def __init__(self, transforms, split, meta_dir, binary, **kwargs):
        super().__init__()

        assert split in ['train', 'vali', 'test']
        self.split = split
        
        if self.split == 'train':
            self.data = np.load("/home/manli/3rd_trim_ultrasounds/data/class_attr_data_10/train.pkl", allow_pickle=True)
        elif self.split == 'vali':
            self.data = np.load("/home/manli/3rd_trim_ultrasounds/data/class_attr_data_10/val.pkl", allow_pickle=True)
        elif self.split == 'test':
            self.data = np.load("/home/manli/3rd_trim_ultrasounds/data/class_attr_data_10/test.pkl", allow_pickle=True)

        if self.split == 'train':
            self.pred_concept = np.load("/home/manli/3rd_trim_ultrasounds/data/ConceptBottleneck/newbee/train.pkl", allow_pickle=True)
        elif self.split == 'vali':
            self.pred_concept = np.load("/home/manli/3rd_trim_ultrasounds/data/ConceptBottleneck/newbee/val.pkl", allow_pickle=True)
        elif self.split == 'test':
            self.pred_concept = np.load("/home/manli/3rd_trim_ultrasounds/data/ConceptBottleneck/newbee/test.pkl", allow_pickle=True)

        self.meta_dir = meta_dir

        self.transforms = transforms

    def __getitem__(self, index):
        # read image
        data = self.data[index]
        image_dir = data['img_path']
        image_dir = '/'.join(image_dir.split('/')[-2:])

        label = data['class_label']
        concept = torch.tensor(data['attribute_label']).float()
        # concept = concept*0.98 + 0.01

        pred_concept = torch.tensor(self.pred_concept[index]['attribute_label']).float()
        pred_concept = torch.sigmoid(pred_concept)

        image = Image.open(os.path.join(self.meta_dir, 'images', image_dir)).convert('RGB')
        image = np.asarray(image)

        mask = np.load(os.path.join(self.meta_dir, 'processed_segmentations', image_dir.replace('.jpg', '.npy')))

        data = self.transforms(image=image, mask=mask)
        image, mask = data['image'], data['mask']

        image = Image.fromarray(image)

        if self.split == 'train':
            image = T.Compose([
            T.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            # T.RandomHorizontalFlip(),
            T.ToTensor()
            ])(image)
        else:
            image = T.ToTensor()(image)
        
        # image = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image) # Imagenet normalization
        image = T.Normalize(mean=(0.5, 0.5, 0.5), std=(2, 2, 2))(image) # Imagenet normalization
        mask = torch.from_numpy(mask)

        if np.random.rand() < 0.5:
            image = torchvision.transforms.functional.hflip(image)
            mask = torchvision.transforms.functional.hflip(mask)

        data = {}
        data['image'] = image
        data['mask'] = mask.float()#.long()
        data['label'] = label
        data['concept'] = concept
        data['pred_concept'] = pred_concept

        return data

    def __len__(self):
        return len(self.data)

class CUBSeg1(Dataset):
    def __init__(self, transforms, split, meta_dir, binary, **kwargs):
        super().__init__()

        assert split in ['train', 'vali', 'test']
        self.split = split

        index = np.load('/home/manli/3rd_trim_ultrasounds/data/cub-200-2010/split.npy')
        concepts = np.load('/home/manli/3rd_trim_ultrasounds/data/cub-200-2010/concepts.npy')

        # read file name list
        with open(os.path.join('/home/manli/3rd_trim_ultrasounds/data/cub-200-2010/lists/files.txt'), 'r') as f:
            image_list = f.read()      

        image_list = np.array(image_list.split('\n'))
        index = np.array(index, dtype=np.int64)
        self.image_list = image_list[index > 0]
        self.concept_list = concepts[index > 0, :]

        self.meta_dir = meta_dir

        self.transforms = transforms

    def __getitem__(self, index):
        image_dir = self.image_list[index]
        image = Image.open(os.path.join('/home/manli/3rd_trim_ultrasounds/data/cub-200-2010', 'images', image_dir)).convert('RGB')
        image = np.asarray(image)

        mask = np.load(os.path.join('/home/manli/3rd_trim_ultrasounds/data/cub-200-2010', 'annotations-mat', image_dir.replace('jpg', 'npy')))

        data = self.transforms(image=image, mask=mask)
        image, mask = data['image'], data['mask']

        image = Image.fromarray(image)
        if self.split == 'train':
            image = T.Compose([
            # T.ColorJitter(),
            # T.RandomHorizontalFlip(),
            T.ToTensor()
            ])(image)
        else:
            image = T.ToTensor()(image)
        image = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image) # Imagenet normalization
        # image = T.Normalize(mean=[0.5,0.5,0.5], std=[2, 2, 2])(image)
        mask = torch.from_numpy(mask)


        concept = self.concept_list[index, :]
        index = np.array([1,4,6,7,10,14,15,20,21,23,25,29,30,35,36,38,40,44,45,50,51,53,54,56,57,59,63,64,69,70,72,75,80,84,90,91,
                            93,99,101,106,110,111,116,117,119,125,126,131,132,134,145,149,151,152,153,157,158,163,164,168,172,178,179,181,
                            183,187,188,193,194,196,198,202,203,208,209,211,212,213,218,220,221,225,235,236,238,239,240,242,243,244,249,253,
                            254,259,260,262,268,274,277,283,289,292,293,294,298,299,304,305,308,309,310,311])
        concept = concept[index]
        concept = torch.from_numpy(concept)
        concept = concept*0.8 + 0.1
        

        label = image_dir.split('.')[0]
        while label.startswith('0'):
            label = label[1:]
        label = int(eval(label)) - 1

        data = {}
        data['image'] = image
        data['mask'] = mask.float()#.long()
        data['label'] = label
        data['concept'] = concept.float()

        return data

    def __len__(self):
        return len(self.image_list)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return int(dataset.data[idx]['attribute_label'][0] > 0.5)

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    tfs = A.Compose(
    [
        A.Resize(342, 342),
        # A.RandomCrop(448, 448),
        A.CenterCrop(299, 299)
    ]
    )
    dataset = CUBSeg(transforms=tfs, split='train', meta_dir="/home/manli/3rd_trim_ultrasounds/data/CUB_200_2011/", binary=False)
    
    data = dataset[2]
    image = data['image']
    image = image / 2 + 0.5
    mask = data['mask']
    concept = data['concept']
    label = data['label']   
    pred_concept = data['pred_concept']
    print(concept, label, mask.min(), mask.max(), image.min(), image.max())
    print(pred_concept)
    print(torch.nn.L1Loss()(pred_concept, concept))
    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     image = data['image']
    #     mask = data['mask']
    #     concept = data['concept']
    #     label = data['label']
    #     print(concept)
    #     print(label)
    #     assert image.shape[0] == 3
    #     assert image.shape[1] == 224
    #     print(image.min(), image.max(), mask.min(), mask.max(), torch.unique(mask))
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(image.permute(1,2,0).numpy())
    # plt.subplot(1,2,2)
    # plt.imshow(mask.numpy())
    plt.subplot(1,3,2)
    plt.imshow(mask[...].numpy())
    plt.subplot(1,3,3)
    plt.imshow(mask[...].numpy())
    plt.show()