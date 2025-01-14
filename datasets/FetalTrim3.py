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

class FetalBase(Dataset):
    def __init__(self, split, csv_dir, meta_dir, split_index, keep_trim3=False, plane=[], remove_calipers=True, **kwargs):
        super().__init__()

        assert split in ['train', 'vali', 'test', 'trainval', 'traintest', 'testval', 'all']
        assert isinstance(plane, list)
        for p in plane:
            assert p in ['Other', 'Femur', 'Cervix', 'Abdomen', 'Head']
        
        self.planes = plane

        self.split = split
        self.remove_calipers = remove_calipers

        # read file name list
        csv = pd.read_csv(csv_dir)

        if len(plane):
            # only use images from specific planes
            csv = csv[csv.apply(lambda x: x['plane'] in plane, axis=1)]
        else:
            # use all planes
            pass

        if split == 'trainval':
            vali_csv = csv[csv[split_index]=='vali']
            csv = csv[csv[split_index]=='train']
            csv = pd.concat((vali_csv, csv), axis=0)
        elif split == 'traintest':
            test_csv = csv[csv[split_index]=='test']
            csv = csv[csv[split_index]=='train']
            csv = pd.concat((test_csv, csv), axis=0)
        elif split == 'testval':
            test_csv = csv[csv[split_index]=='test']
            csv = csv[csv[split_index]=='val']
            csv = pd.concat((test_csv, csv), axis=0)
        elif split == 'all':
            pass
        else:
            csv = csv[csv[split_index]==split]

        if remove_calipers:
            # just remove all the gray images
            csv = csv[csv['is_rgb']]
        else:
            csv = csv[csv['is_rgb']]
            # Warning('Note that you are now training a model with calipers, which leads to potential bias.')
            # input('You are using images with calipers. Are you aware of the potential bias? Press Enter to continue.')
            # print('You are using images with calipers. Are you aware of the potential bias? Press Enter to continue.')


        if keep_trim3: # only use 3-trim data, this is important for training a concept bottleneck model
            csv = csv[csv.apply(lambda x: x['trimester']==3 or x['plane']=='Other', axis=1)]
            # csv = csv[csv.apply(lambda x: x['trimester']==3, axis=1)]

        # csv = csv.sample(frac=1) # shuffle the data
        self.csv = csv.reset_index(drop=True)

        # read meta information
        with open(meta_dir, 'r') as f:
            self._meta = yaml.load(f, yaml.FullLoader)
    
    def _get_attr(self, index, attr):
        return self.csv.loc[index, attr]

    def __getitem__(self, index):
        # read image
        image_dir = self._get_attr(index, 'image_dir')

        # read clean image
        if self.remove_calipers:
            clean_image_dir = image_dir.split('.')
            clean_image_dir[-2] += '_clean'
            clean_image_dir = '.'.join(clean_image_dir)
            if (not os.path.isfile(clean_image_dir)) and self._get_attr(index, 'plane') == 'Other':
                # many other class images do not have the clean version
                clean_image_dir = image_dir
        else:
            clean_image_dir = image_dir

        raw_image = Image.open(image_dir)
        raw_image = np.asarray(raw_image)

        image = Image.open(clean_image_dir)
        image = np.asarray(image)

        # read mask
        mask_dir = self._get_attr(index, 'mask_dir')
        if mask_dir is not np.nan:
            # mask = Image.open(mask_dir)
            mask = np.load(mask_dir.replace('.tif', '.npy'))
        else:
            # wrong image, should be cleaned
            mask = np.zeros_like(image)[...,0]
            image *= 0
            raw_image *= 0

        mask = np.asarray(mask, dtype=np.int64)

        image = cv2.resize(image, (960, 720))
        raw_image = cv2.resize(raw_image, (960, 720))
        mask = cv2.resize(mask, (960, 720), interpolation=cv2.INTER_NEAREST)

        image = image[60:734,81:874,...]
        raw_image = raw_image[60:734,81:874,...]
        mask = mask[60:734,81:874,...]

        return raw_image, image, mask

    def __len__(self):
        return len(self.csv)

    @property
    def meta(self):
        return self._meta


class FetalConceptAug(FetalBase):
    def __getitem__(self, index):
        # return ground truth property concept and image category
        # pick a category
        cat = np.random.randint(low=0, high=8)
        csv = self.csv
        if cat < 4:
            # acceptable
            csv = csv[csv['acceptable']]
            plane = cat
        else:
            csv = csv[~csv['acceptable']]
            plane = cat - 4

        concept_mask = torch.zeros(27)
        if plane == 0:
            csv = csv[csv['plane']=='Femur']
            concept_mask[:4] = 1
        elif plane == 1:
            csv = csv[csv['plane']=='Abdomen']
            concept_mask[4:13] = 1
        elif plane == 2:
            csv = csv[csv['plane']=='Head']
            concept_mask[13:22] = 1
        elif plane == 3:
            csv = csv[csv['plane']=='Cervix']
            concept_mask[22:] = 1
        else:
            raise ValueError()

        concepts = csv['concept'].values
        concepts = list(map(eval, concepts))
        concepts = np.array(concepts)
        concepts[:, [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]] *= 10

        values = []
        for i in range(27):
            values.append(list(set(concepts[:,i])))
        synthetic_concept = [np.random.choice(v)+(np.random.rand()-0.5)*2*0.3 for v in values]
        # synthetic_concept = [np.random.choice(v) for v in values]
        synthetic_concept = torch.tensor(synthetic_concept)

        # range
        synthetic_concept[[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]] = torch.clamp(synthetic_concept[[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]], min=0, max=10)
        synthetic_concept[[2, 3, 4, 8, 13, 17, 22, 24]] = torch.clamp(synthetic_concept[[2, 3, 4, 8, 13, 17, 22, 24]], min=0, max=1)
        synthetic_concept *= concept_mask

        data = {}
        data['concept'] = synthetic_concept.float()
        data['label'] = cat

        return data

    def __len__(self):
        return 30000

class FetalSeg(FetalBase):
    def __init__(self, transforms, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms

    def __getitem__(self, index):
        raw_image, image, mask = super().__getitem__(index)

        # augmentations
        data = self.transforms(image=raw_image, mask=mask)
        raw_image, raw_mask = data['image'], data['mask']

        data = self.transforms(image=image, mask=mask)
        image, mask = data['image'], data['mask']

        
        color_tf = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        assert len(image.shape) == 3
        image = color_tf(np.array(image))
        gray_image = torch.mean(image, dim=0, keepdim=True)

        raw_image = color_tf(np.array(raw_image))
        gray_raw_image = torch.mean(raw_image, dim=0, keepdim=True)


        # image = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image) # Imagenet normalization

        mask = torch.from_numpy(np.array(mask))

        # read plane category and convert it into integer
        plane = self._meta['PLANE'][self._get_attr(index, 'plane')]
        accp = self._get_attr(index, 'acceptable')

        # whether symmetric
        symm = self._get_attr(index, 'symmetric')

        if not accp:
            plane = plane + 4 #
        
        if self._get_attr(index, 'plane') == 'Other':
            plane = len(self.planes)*2 - 2

        # wrapup
        data = {}

        # custom dataset only need image/gray image, mask, concept and label
        data['image'] = image
        data['gray_image'] = gray_image
        data['raw_image'] = raw_image
        data['gray_raw_image'] = gray_raw_image
        data['mask'] = mask.long()
        data['label'] = plane
        
        #------------
        data['symm'] = symm
        data['acceptable'] = accp
        data['raw_mask'] = raw_mask

        concept = torch.tensor(eval(self._get_attr(index, 'concept')))
        concept[9] = torch.min(concept[[9,10,11,12]], dim=0, keepdim=False)[0] # 9, 10, 11, 12: min, max, mean, std
        concept[10] = torch.max(concept[[9,10,11,12]], dim=0, keepdim=False)[0] # 9, 10, 11, 12: min, max, mean, std
        concept[11] = torch.mean(concept[[9,10,11,12]], dim=0, keepdim=False) # 9, 10, 11, 12: min, max, mean, std
        concept[12] = torch.median(concept[[9,10,11,12]], dim=0, keepdim=False)[0] # 9, 10, 11, 12: min, max, mean, median

        concept[[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]] *= 10

        data['concept'] = concept

        return data
    
    def _get_label(self, index):
        plane = self._meta['PLANE'][self._get_attr(index, 'plane')]
        accp = self._get_attr(index, 'acceptable')
        if not accp:
            plane = plane+4
        return plane

    def get_labels(self):
        label = np.array([self._get_label(l) for l in range(len(self.csv))])
        return label

if __name__ == "__main__":
    args = {     
        'split':'train', 
            'csv_dir': 'metas/trim3_sp.csv',
            'meta_dir': 'trim3_sp.yaml', 
            'split_index': 'split1',
            'plane': ['Head', 'Femur', 'Abdomen', 'Cervix'],
            'remove_calipers': True,
            'keep_trim3': True
            }
    traindata = FetalConceptAug(**args)
    print(traindata[0]['concept'])
