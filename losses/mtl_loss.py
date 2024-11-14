import torch.nn as nn
from monai.losses import DiceFocalLoss
from torch.nn import functional as F
import torch
import numpy as np

quality_index = [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]
binary_index = [2, 3, 4, 8, 13, 17, 22, 24]
quality_names = ['femur left', 'femur_right', 'stomach', 'umv', 'kidney', 'ada1', 'ada2', 'adb1', 'adb2', 'thalamus', 'csp', 'fossa', 'bpd_near', 'bpd_far', 'ofd_occ', 'ofd_fro', 'bladder', 'orif_inner', 'orif_ext']



class DFLoss:
    def __init__(self, weight, **kwargs):
        if 'sigmoid' in kwargs.keys():
            self.criterion = DiceFocalLoss(
                                        include_background=True, 
                                        # softmax=True, 
                                        sigmoid=True,
                                        focal_weight=weight, 
                                        # to_onehot_y=True,
                                        to_onehot_y=False,
                                        lambda_dice=1., lambda_focal=1.,
                                        smooth_dr=1e-5, smooth_nr=1e-5,
                                        squared_pred=False, 
                                        )
            self.sigmoid = True
        else:
            self.criterion = DiceFocalLoss(
                                        include_background=True, 
                                        softmax=True, 
                                        focal_weight=weight, 
                                        to_onehot_y=True,
                                        lambda_dice=1., lambda_focal=1.,
                                        smooth_dr=1e-5, smooth_nr=1e-5,
                                        squared_pred=False, 
                                        )
            self.sigmoid=False           

    def __call__(self, x):
        logit, mask = x['seg_logit'], x['mask']
        # to onehot
        if not self.sigmoid:
            mask = mask.unsqueeze(1)
        loss = self.criterion(logit, mask)
        return loss

## build losses
class Loss:
    def __init__(self, **kwargs):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # self.bce = nn.BCELoss(reduction='none')
        # self.mse = nn.SmoothL1Loss(reduction='none') # less sensitive to outliers than mse
        # self.mse = nn.BCELoss(reduction='none')
        self.mse = nn.HuberLoss(reduction='none')
        # self.mse = nn.MSELoss(reduction='none')
    
    def __call__(self, outs):
        plane = outs['label']
        epoch = outs['epoch']

        concept_logits = outs['concept_pred'] 
        concept_gt = outs['concept']
        mask = outs['mask']
        concept_mask = outs['concept_mask']

        batch_size = concept_logits.size(0)

        count = np.load('count.npy')
        count = torch.from_numpy(count).to(mask.device)

        loss = nn.BCEWithLogitsLoss()(concept_logits[:,binary_index], concept_gt[:,binary_index])
        # loss = loss + nn.SmoothL1Loss()(concept_logits[:,quality_index][concept_mask[:,quality_index]==1], concept_gt[:,quality_index][concept_mask[:,quality_index]==1]) if not apply_sigmoid else nn.SmoothL1Loss()(concept_logits[:,quality_index][concept_mask[:,quality_index]==1], concept_gt[:,quality_index][concept_mask[:,quality_index]==1]*10)
        # loss = torch.tensor([0.]).to(concept_logits.device)
        # loss = torch.nn.BCELoss()(concept_logits, concept_gt)
        bin_loss = torch.tensor([0.]).to(concept_logits.device)
        cnter = 0
        densities = []
        losses = []
        for i in binary_index:
            index = concept_mask[:,i]==1
            if not torch.sum(index):
                continue
            else:
                concept = concept_gt[index,i].clone() # [64]
                concept_pred = concept_logits[index,i].clone() # [64]

            batch_loss = self.bce(concept_pred, concept)
            count_1 = torch.sum(concept)
            count_0 = batch_size - count_1
            weight_0 = (1/(count_0+1)) / (1/(count_0+1) + 1/(count_1+1))
            weight_1 = (1/(count_1+1)) / (1/(count_0+1) + 1/(count_1+1))
            weight = (1-concept)*weight_0 + concept*weight_1

            weight_sum = torch.sum(weight)
            if weight_sum < 1e-9:
                weight_sum = 1.
            weight = weight / weight_sum
            batch_loss = batch_loss * weight
            batch_loss = torch.sum(batch_loss)

            bin_loss += batch_loss
            cnter += 1
        
        if cnter:
            bin_loss = bin_loss / cnter

        concept_loss = torch.tensor([0.]).to(concept_logits.device)
        cnter = 0
        # densities = []
        # losses = []
        # # ranking_loss = [] # add ranking loss
        ranking_loss = torch.tensor([0.]).to(concept_logits.device)
        # range_loss = 0.
        for i in quality_index: 
            index = concept_mask[:,i]==1
            if not torch.sum(index):
                continue
            else:
                concept = concept_gt[index,i].clone()*10
                concept_pred = concept_logits[index,i].clone()
            # print(concept_pred, concept)
            # oversampling according to the density

            # count density
            density = torch.zeros_like(concept) # [N]
            for c in range(density.size(0)):
                density[c] = torch.sum(concept == concept[c])
                # if concept[c] <= 2:
                #     density[c] = density[c]/10 # if it is < 2, be aware that the weight should be larger

            # other labels
            # concept1 = torch.relu(concept-0.8)
            # concept2 = torch.clamp(concept+0.8, min=None, max=10)
            
            # batch_loss = 0.
            # batch_cnter = 0
            # weight = 1 / (count[i,:] + 1)
            # weight = weight / torch.sum(weight)
            # for value in range(0, 11):
            #     if torch.sum(concept==value):
            #         l1 = self.mse(concept_pred[concept==value], concept[concept==value])
            #         l2 = self.mse(concept_pred[concept==value], concept[concept==value])
            #         l3 = self.mse(concept_pred[concept==value], concept[concept==value])
            #         l = torch.min(torch.cat((l1.unsqueeze(1), l2.unsqueeze(1), l3.unsqueeze(1)), dim=1), dim=1)[0]
            #         # l = self.mse(concept_pred[concept==value], concept[concept==value])
            #         batch_loss += l.mean()*weight[value]
            #         batch_cnter += 1
            # concept_loss += batch_loss / batch_cnter if batch_cnter else 0.
            batch_loss = self.mse(concept_pred, concept) # [64]
            # batch_loss1 = self.mse(concept_pred, concept1) # [64]
            # batch_loss2 = self.mse(concept_pred, concept2) # [64]
            # batch_loss = torch.min(torch.cat((batch_loss.unsqueeze(1), batch_loss1.unsqueeze(1), batch_loss2.unsqueeze(1)), dim=1), dim=1)[0]

            weight = 1/density
            weight_sum = torch.sum(weight)
            weight = weight / weight_sum
            # weight[concept==1] = 1.
            # weight[concept==2] = 1.
            weight = torch.clamp(weight, min=1e-3, max=None)
            concept_loss += torch.sum(batch_loss*weight/weight_sum)

            # # losses.append(batch_loss)
            # # densities.append(1/density)

            # # # limit the range, we allow at maximum difference +-1
            # # [2, 5] traffic light?
            # penealty_factor = 1
            # # red
            # index = concept<2
            # if torch.sum(index):
            #     range_loss += torch.relu(concept_pred[index] - 2 + 0.5).mean()/2*penealty_factor
            # index = ((concept<5)*(concept>=2))>0
            # if torch.sum(index):
            #     range_loss += (torch.relu(concept_pred[index] - 5 + 0.5).mean()/2*penealty_factor + torch.relu(2 - concept_pred[index] - 0.5).mean()/2*penealty_factor)
            # index = concept>=5
            # if torch.sum(index):
            #     range_loss += torch.relu(5 - concept_pred[index] - 0.5).mean()/2*penealty_factor

            # if not torch.sum(concept_gt[:,i]):
            #     continue # if all the ground truth labels are the same, do not compute rank loss

            # # ranking loss (hinge)
            label_diff = concept.unsqueeze(1) - concept.unsqueeze(0)
            diff_sign = -torch.sign(label_diff)
            diff = concept_pred.unsqueeze(1) - concept_pred.unsqueeze(0)
            diff = diff_sign*diff
            diff_ = diff.flatten()[diff_sign.flatten()!=0]
            if len(diff_):
                ranking_loss += (torch.relu(diff_+1)/2).mean()
                # ranking_loss += torch.exp(-torch.abs(diff_)).mean()/2
            # diff_ = diff.flatten()[diff_sign.flatten()==0]
            # if len(diff_):
            #     ranking_loss += (diff_/2).mean()
            # loss from ranknet (consider the ranking task as a binary classification problem)
            # diff_label = (torch.sign(concept.unsqueeze(0) - concept.unsqueeze(1))+1)*0.5
            # diff = torch.sigmoid((concept_pred.unsqueeze(0) - concept_pred.unsqueeze(1))*6)
            # batch_ranking_loss += torch.nn.BCELoss(reduction='none')(diff, diff_label).mean()/2

            cnter += 1

        ranking_loss = ranking_loss / cnter if cnter else 0
        # range_loss = range_loss / len(quality_index) if len(quality_index) else 0
        concept_loss = concept_loss / len(quality_index) if len(quality_index) else 0

        # if len(densities):
        #     s = torch.cat(densities).sum()
        #     for d, l in zip(densities, losses):
        #         weight = d/s
        #         weight = torch.clamp(weight, min=1e-3)
        #         concept_loss += torch.sum(weight*l)
        # concept_loss = concept_loss / cnter
        # for i, j in zip(concept, weight):
        #     print(i, j)
        loss = loss + ranking_loss + concept_loss + bin_loss

        return loss




class MTLLoss:
    def __init__(self, **kwargs):
        self.cse = nn.CrossEntropyLoss()
        self.seg_loss = DFLoss(weight=torch.tensor([9.9460e-05, 1.0123e-01, 3.5893e-02, 6.8795e-02, 6.9788e-02, 4.5338e-02,
        7.1722e-02, 3.8831e-02, 1.6281e-01, 
        9.0414e-01, 1.1600e-01, 6.2635e-01, # kidney, _, fossa
        1.1064e-01, 2.5810e-02], requires_grad=False).cuda())
    
    def __call__(self, outs):
        logits = outs['logit']
        gt = outs['label']
        seg_loss = self.seg_loss(outs)
        cls_loss = self.cse(logits, gt)

        # concept loss
        concept_loss = Loss()(outs)

        return cls_loss + concept_loss + seg_loss



        


