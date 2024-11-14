import torch
import torch.nn as nn
import torchvision
from ._modules import Conv1x1
from sononet import SonoNet
from .backbones import ResNet
from .unet_family import RegUNet

'''
NB: According to arxiv.org/abs/2105.04289, among the three variants of CBM (joint, sequential, independent), 
only the independent variant has the three designed property. 
'''

class SonoNets(nn.Module):
    def __init__(self, config, num_labels, weights,
             features_only, in_channels):
        super().__init__()
        self.net = SonoNet(config, num_labels, weights,
             features_only, in_channels)
    
    def forward(self, x):
        image = x['image']
        logit = self.net(image)
        return {'logit': logit}    


def act(x, sigmoid=True):
    quality_index = [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]
    binary_index = [2, 3, 4, 8, 13, 17, 22, 24]
    quality_names = ['femur left', 'femur_right', 'stomach', 'umv', 'kidney', 'ada1', 'ada2', 'adb1', 'adb2', 'thalamus', 'csp', 'fossa', 'bpd_near', 'bpd_far', 'ofd_occ', 'ofd_fro', 'bladder', 'orif_inner', 'orif_ext']


    concept = x['concept']
    seg_mask = x['mask']
    assign_mtx = x['seg_logit']
    x = x['concept_logit']

    x[:,quality_index] = torch.relu(x[:, quality_index])
    x[:,quality_index] = 10 - torch.relu((10-x[:,quality_index]))

    if sigmoid:
        x[:, binary_index] = torch.sigmoid(x[:, binary_index])
    # x = torch.sigmoid(x)

    # use segmentation for masking
    mask = torch.ones_like(x).detach() 
    for b in range(seg_mask.size(0)):
        # cervix
        if ((1 not in seg_mask[b,...]) + (2 not in seg_mask[b,...]) + (3 not in seg_mask[b,...])) >= 3:
                mask[b, [22, 24, 25, 26]] = 0
        if ((1 not in seg_mask[b,...]) + (3 not in seg_mask[b,...])) >= 2:
            mask[b, 25] = 0
        if ((1 not in seg_mask[b,...]) + (2 not in seg_mask[b,...])) >= 2:
            mask[b, 26] = 0
        if 4 not in seg_mask[b,...]:
            mask[b, 23] = 0
        
        # femur
        if 5 not in seg_mask[b,...]:
                mask[b, [0,1,2,3]] = 0

        # abdomen
        if 6 not in seg_mask[b,...]:
                mask[b, 5] = 0
        if 8 not in seg_mask[b,...]:
                mask[b, 6] = 0
        if 9 not in seg_mask[b,...]:
                mask[b, 7] = 0
        if 7 not in seg_mask[b, ...]:
                mask[b, [9, 10, 11, 12]] = 0
                mask[b, 8] = 0

        if ((6 not in seg_mask[b,...]) + (7 not in seg_mask[b,...]) + (8 not in seg_mask[b,...])) >= 3:
                mask[b, [4, 8]] = 0
                mask[b, [9, 10, 11, 12]] = 0

        # head
        if 10 not in seg_mask[b,...]:
                mask[b, 14] = 0
        if 11 not in seg_mask[b,...]:
                mask[b, 16] = 0
        if 12 not in seg_mask[b,...]:
                mask[b, 15] = 0
        if 13 not in seg_mask[b, ...]:
                mask[b, [18, 19, 20, 21]] = 0
                mask[b, 17] = 0
        else:
            bone = seg_mask[b,...] == 13
            if torch.sum(assign_mtx[b, -4, ...] * bone) <= 1: # bpd far
                mask[b, 19] = 0  
            if torch.sum(assign_mtx[b, -3, ...] * bone) <= 1: # bpd near
                mask[b, 18] = 0

        # cervix
        if 4 not in seg_mask[b,...]:
            mask[b, 23] = 0
        if 1 not in seg_mask[b, ...] or 2 not in seg_mask[b, ...]:
            mask[b, 26] = 0
        if 1 not in seg_mask[b, ...] or 3 not in seg_mask[b, ...]:
            mask[b, 25] = 0
        if ((0 not in seg_mask[b,...]) + (1 not in seg_mask[b,...]) + (2 not in seg_mask[b,...])) >= 3:
            mask[b, [22,23,24,25,26]] = 0

        if ((10 not in seg_mask[b,...]) + (12 not in seg_mask[b,...]) + (13 not in seg_mask[b,...])) >= 3:
                mask[b, [13, 17]] = 0
                mask[b, [18, 19, 20, 21]] = 0

    # mask = torch.ones_like(x).detach() 
    x = mask*x
    

    return x, mask

class MTLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = RegUNet(1, 14).cuda()
        self.unet.load_state_dict(torch.load('logs/dtunet_iccv/unet.t7'))

        self.shared_mlp = nn.Sequential(
            nn.Linear(888, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )

        self.concept_head = nn.Sequential(
            # nn.LeakyReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 27)   
        )

        self.cls_head = nn.Sequential(
            # nn.LeakyReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 8)   
        )

    def forward(self, x):
        seg_result = self.unet(x)
        seg_emb = seg_result['emb']
        seg_logit = seg_result['logit']
        feat = self.shared_mlp(seg_emb)
        concept_logit = self.concept_head(feat)
        cls_logit = self.cls_head(feat)
        x['seg_logit'] = seg_logit
        x['concept_logit'] = concept_logit
        x['logit'] = cls_logit
        # concept_pred, conecpt_mask = act(x)

        # x['concept_pred'] = concept_pred
        # x['concept_mask'] = conecpt_mask

        return x

if __name__ == "__main__":
    # model = SonoNets(config='SN32', num_labels=8, weights=False, features_only=False, in_channels=3)

    # model.train()
    # data = {
    #     'image': torch.rand(2, 3, 224, 224).float(),
    #     'concept': torch.ones(2, 27).float(),
    #     'mask': torch.ones(2, 224, 224).long()
    # }
    # out = model(data)
    # print(out['logit'].shape)
    # print(out['logit'])
    # print(out['concept_logit'].shape)
    model = MTLNet().cuda()
    model.train()
    data = {'image':torch.rand(2,3,224, 288).float().cuda()}
    print(mode(data))
    






