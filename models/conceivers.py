import torch
import torch.nn as nn
from .backbones import ResNet, Inception, SparseConv2d, VGG
import numpy as np
from .activations import fetal_caliper_concept
from .predictors import Predictor
from collections import Counter
from .geometry_tools import get_abdomen_concept, get_femur_concept, get_head_concept, get_cervix_concept

# conceiver of the origional CBM (baseline)
class CBMConceiver(nn.Module):
    def __init__(self, in_channels, global_index, local_index, backbone='ResNet18'):
        super().__init__()
        if backbone != 'ResNet18':
            raise NotImplementedError()

        # self.encoder = ResNet(in_channels=in_channels, out_channels=len(sum(global_index+local_index, [])), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
                # construct global conceivers for each group
        if not isinstance(global_index[0], list):
            global_index = [global_index] # only one group

        self.conceivers = nn.ModuleDict()
        for i, global_group_list in enumerate(global_index, 0):
            model = ResNet(in_channels=in_channels , out_channels=len(global_group_list), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
            self.conceivers[f'global_conceiver{i}'] = model

        # construct local conceivers for each group
        if not isinstance(local_index[0], list):
            local_index = [local_index] # only one group

        for i, local_group_list in enumerate(local_index, 0):
            model = ResNet(in_channels=in_channels, out_channels=len(local_group_list), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
            self.conceivers[f'local_conceiver{i}'] = model

        self.global_index = global_index
        self.local_index = local_index

# base class for a conceiver
class Conceiver(nn.Module):
    def __init__(self, in_channels, seg_concept_num, global_index, local_index, relationship, backbone='ResNet18'):
        '''
        in_channels (int): image channel number
        seg_concept_num (int): number of segmentation concept
        global_index (list[list]): indices of the global property concepts in groups, e.g., [[global_index_group1], [global_index_group2], ...]
        local_index (list[list]): indices of the local property concepts in groups
        relationship: a dictionary storing the relationship of segmentation concepts and property concepts, for example:
        {
            property_concept_id: (segmentation_concept_id0, segmentation_concept_id1, ...)
        }
        note that one property concept can be related to multiple segmentation concepts
        backbone (str)
        '''
        super().__init__()
        if backbone != 'ResNet18':
            raise NotImplementedError()

        # construct global conceivers for each group
        if not isinstance(global_index[0], list):
            global_index = [global_index] # only one group

        self.conceivers = nn.ModuleDict()
        for i, global_group_list in enumerate(global_index, 0):
            model = ResNet(in_channels=in_channels + seg_concept_num, out_channels=len(global_group_list), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
            self.conceivers[f'global_conceiver{i}'] = model

        # construct local conceivers for each group
        if not isinstance(local_index[0], list):
            local_index = [local_index] # only one group

        for i, local_group_list in enumerate(local_index, 0):
            model = ResNet(in_channels=in_channels + 1, out_channels=len(local_group_list), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
            self.conceivers[f'local_conceiver{i}'] = model

        self.local_index = local_index
        self.global_index = global_index
        self.relationship = relationship

    def local_forward(self, index, group_id, image, assign_mtx):
        seg_index = self.relationship[index]
        seg_mask = assign_mtx[:, seg_index, ...]
        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask.unsqueeze(1)
        seg_mask = torch.sum(seg_mask, dim=1, keepdims=True)
        x = torch.cat((image, seg_mask), dim=1)
        x = self.conceivers[f'local_conceiver{group_id}']({'image':x})['logit']
        return x

    def forward(self, x):
        '''
        input: assign_mtx(B, C, H, W) segmentation probability mask / gt onthot encoding
        image: image(B, 1, H, W), 1 is the image channel number, can be 3
        '''
        image = x['image']
        assign_mtx = x['assign_mtx']

        # number of global and local concepts
        index = self.global_index + self.local_index
        index = sum(index, [])
        concept = torch.zeros((image.size(0), len(index))).to(image.device)
        # global conceivers
        for i, global_group_list in enumerate(self.global_index, 0):
            concept[:,np.array(global_group_list)] = self.conceivers[f'global_conceiver{i}']({'image': torch.cat((image, assign_mtx), dim=1)})['logit']

        for group_id, local_group_list in enumerate(self.local_index, 0):
            for i, index in enumerate(local_group_list):
                concept[:, index] = self.local_forward(index, group_id, image, assign_mtx)[:,i,...]

        concept_logit, concept_mask, concept_pred = self.seg_based_rule(concept, assign_mtx)
        x['concept_logit'] = concept_logit
        x['concept_mask'] = concept_mask
        x['concept_pred'] = concept_pred

        return x

    def seg_based_rule(self, concept, assign_mtx):
        # apply some pre-defined rules here according to the segmentation mask
        # concept_logit: predicted property concept values
        # concept_mask: binary mask indicating whether the property concept applicable for an instance
        mask = torch.ones_like(concept).detach()
        return concept, mask, concept

class FetalConceiver(Conceiver):
    def __init__(self, in_channels, seg_concept_num, global_index, local_index, relationship):
        super().__init__(in_channels, seg_concept_num, global_index, local_index, relationship)

    def forward(self, x):
        assign_mtx = x['assign_mtx']
        seg_mask = torch.argmax(assign_mtx, dim=1)
        plane = []
        self.x = x

        # segmentation based plane pre-classification: divide plane into femur, head, abdomen, and cervix -> (0, 1, 2, 3)
        for b in range(seg_mask.size(0)):
            # plane recognition based on segmentation
            counts = Counter(seg_mask[b, ...].flatten().detach().cpu().numpy()) # time consuming operation
            counts = list(sorted(list(counts.keys()), key=lambda x: counts[x], reverse=True))
            if len(counts)>=3:
                counts = counts[1:4]
            else:
                counts = counts[1:]
            ## if head
            if ((10 in counts) + (12 in counts) + (13 in counts) + (11 in counts))>=2:
                plane.append(2)
            ## if abdomen
            elif ((6 in counts) + (7 in counts) + (8 in counts) + (9 in counts))>=2:
                plane.append(1)
            ## if cervix:
            elif ((1 in counts) + (2 in counts) + (3 in counts))>=2:
                # cervix
                plane.append(3)
            ## if femur
            elif 5 in counts:
                plane.append(0)
            elif counts[0] in [10, 11, 12, 13]:
                plane.append(2)
            elif counts[0] in [6, 7, 8, 9]:
                plane.append(1)
            elif counts[0] in [1, 2, 3, 4]:
                plane.append(3)
            
        x['assign_mtx'] = fetal_caliper_concept({'assign_mtx':assign_mtx, 'seg_mask':seg_mask, 'plane':plane})['assign_mtx']
        # x = super().forward(x)
        image = x['image']
        assign_mtx = x['assign_mtx']
        # number of global and local concepts
        index = self.global_index + self.local_index
        index = sum(index, [])
        concept = torch.zeros((image.size(0), len(index))).to(image.device)
        # global conceivers
        for i, global_group_list in enumerate(self.global_index, 0):
            concept[:,np.array(global_group_list)] = self.conceivers[f'global_conceiver{i}']({'image': torch.cat((image, assign_mtx), dim=1)})['logit']

        for group_id, local_group_list in enumerate(self.local_index, 0):
            for i, index in enumerate(local_group_list):
                concept[:, index] = self.local_forward(index, group_id, image, assign_mtx)[:,i,...]

        concept_logit, concept_mask, concept_pred = self.seg_based_rule(concept, assign_mtx, seg_mask, plane, image)
        x['concept_logit'] = concept_logit
        x['concept_mask'] = concept_mask
        x['concept_pred'] = concept_pred
        return x

    # overide pre-defined rules for different datasets
    def seg_based_rule(self, concept, assign_mtx, seg_mask, plane, image):
        quality_index = np.array(sum(self.local_index, [])) # organ quality
        binary_index = np.array(self.global_index[0]) # binary concept

        # quality should in [0, 10]
        # concept[:, quality_index] = torch.clamp(concept[:, quality_index], min=0, max=10)
        concept[:,quality_index] = torch.relu(concept[:, quality_index])
        concept[:,quality_index] = 10 - torch.relu((10-concept[:,quality_index]))

        # use segmentation for masking
        mask = torch.ones_like(concept).detach() 

        for b in range(seg_mask.size(0)):
            ## if head
            if plane[b] == 2:
                mask[b, 22:] = 0
                mask[b, :13] = 0
            ## if abdomen
            elif plane[b] == 1:
                mask[b, :4] = 0
                mask[b, 13:] = 0
            ## if cervix:
            elif plane[b] == 3:
                # cervix
                mask[b, :22] = 0
                # occ = get_cervix_concept(image[b,0,...], seg_mask[b,...])
                # concept[b, 22] = float(occ)
                # binary_index = binary_index[binary_index!=22]
            ## if femur
            elif plane[b] == 0:
                mask[b, 4:] = 0
                # angle, occ = get_femur_concept(image[b,0,...], seg_mask[b,...])
                # concept[b, 3] = float(occ)
                # # concept[b, 3] = self.x['concept'][b, 3]
                # binary_index = binary_index[binary_index!=3]

            # cervix
            if ((1 not in seg_mask[b,...]) + (3 not in seg_mask[b,...])) >= 1:
                mask[b, 25] = 0
            if ((1 not in seg_mask[b,...]) + (2 not in seg_mask[b,...])) >= 1:
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

        concept_logit = mask*concept
        concept_pred = concept_logit.detach().clone()
        
        concept_pred[:, binary_index] = (torch.sigmoid(concept_pred[:,binary_index]) >= 0.5).float()
        concept_pred *= mask

        return concept_logit, mask, concept_pred

class FetalCBMConceiver(CBMConceiver):
    def __init__(self, in_channels, global_index, local_index):
        super().__init__(in_channels, global_index, local_index)

    def forward(self, x):
        quality_index = np.array(sum(self.local_index, [])) # organ quality
        binary_index = np.array(self.global_index[0]) # binary concept
        image = x['image']

        # number of global and local concepts
        index = self.global_index + self.local_index
        index = sum(index, [])
        concept = torch.zeros((image.size(0), len(index))).to(image.device)


        for i, global_group_list in enumerate(self.global_index, 0):
            concept[:,np.array(global_group_list)] = self.conceivers[f'global_conceiver{i}']({'image': image})['logit']
        for i, local_group_list in enumerate(self.local_index, 0):
            concept[:,np.array(local_group_list)] = self.conceivers[f'local_conceiver{i}']({'image': image})['logit']

        concept[:,quality_index] = torch.relu(concept[:, quality_index])
        concept[:,quality_index] = 10 - torch.relu((10-concept[:,quality_index]))
        concept_pred = concept.clone()
        concept_pred[:, binary_index] = (torch.sigmoid(concept_pred[:,binary_index]) >= 0.5).float()
        concept_mask = torch.ones_like(concept) # only used for computing the cost function

        x['concept_pred'] = concept_pred
        x['concept_logit'] = concept
        x['concept_mask'] = concept_mask
        
        return x


if __name__ == "__main__":
    model = FetalConceiver(in_channels=1, 
                        seg_concept_num=21, 
                        global_index=[[2, 3, 4, 8, 13, 17, 22, 24]], 
                        local_index=[[5,6,7,14,15,16,23], # quality 
                                    [0,1,9,10,11,12,18,19,20,21,25,26] # caliper
                                    ], 
                        relationship=
                        {
                           0:(14),
                           1:(15),
                           5:(6),
                           6:(8),
                           7:(9),
                           9:(16),
                           10:(16),
                           11:(16),
                           12:(16),
                           14:(10),
                           15:(12),
                           16:(11),
                           18:(18),
                           19:(17),
                           20:(20),
                           21:(19),
                           23:(4),
                           25:(1,3),
                           26:(1,2)
                        })
    image = torch.rand(2, 1, 224, 288)
    seg = torch.rand(2, 21, 224, 288)
    print(model({'image':image, 'assign_mtx':seg})['concept_pred'])