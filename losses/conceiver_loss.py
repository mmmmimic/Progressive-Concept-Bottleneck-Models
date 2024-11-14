import torch
import numpy as np
import torch.nn as nn

## build losses
class ConceiverLoss:
    def __init__(self, reg_index, cat_index):
        self.density_bce = nn.BCEWithLogitsLoss(reduction='none')
        self.overall_bce = nn.BCEWithLogitsLoss()
        self.overall_reg = nn.HuberLoss()
        self.reg = nn.HuberLoss(reduction='none')
        self.reg_index = reg_index
        self.cat_index = cat_index
    
    def __call__(self, x):
        concept_logit = x['concept_logit'] # concept prediction
        concept_mask = x['concept_mask'] # concept mask
        concept_gt = x['concept_gt'] # concept ground truth

        batch_size = concept_logit.size(0)

        overall_bce_loss = self.overall_bce(concept_logit[:,self.cat_index], concept_gt[:,self.cat_index])
        overall_reg_loss = self.overall_reg(concept_logit[:,self.reg_index], concept_gt[:,self.reg_index])

        density_bce_loss = torch.tensor([0.]).to(concept_logit.device)
        cnter = 0
        densities = []
        losses = []
        for i in self.cat_index:
            index = concept_mask[:,i]==1
            if not torch.sum(index):
                continue
            else:
                concept = concept_gt[index,i].clone() # [64]
                concept_pred = concept_logit[index,i].clone() # [64]

            batch_loss = self.density_bce(concept_pred, concept)
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

            density_bce_loss += batch_loss
            cnter += 1
        
        if cnter:
            density_bce_loss = density_bce_loss / cnter

        reg_loss = torch.tensor([0.]).to(concept_logit.device)
        cnter = 0
        ranking_loss = torch.tensor([0.]).to(concept_logit.device)
        for i in self.reg_index: 
            index = concept_mask[:,i]==1
            if not torch.sum(index):
                continue
            else:
                concept = concept_gt[index,i].clone()
                concept_pred = concept_logit[index,i].clone()


            # count density
            density = torch.zeros_like(concept) # [N]
            for c in range(density.size(0)):
                count = torch.sum(concept == concept[c])
                density[c] = count
            batch_loss = self.reg(concept_pred, concept) # [64]

            weight = 1/density
            weight_sum = torch.sum(weight)
            weight = weight / weight_sum
            weight = torch.clamp(weight, min=1e-3, max=None)
            reg_loss += torch.sum(batch_loss*weight/weight_sum)

            # # ranking loss (hinge)
            label_diff = concept.unsqueeze(1) - concept.unsqueeze(0)
            diff_sign = -torch.sign(label_diff)
            diff = concept_pred.unsqueeze(1) - concept_pred.unsqueeze(0)
            diff = diff_sign*diff
            diff_ = diff.flatten()[diff_sign.flatten()!=0]
            if len(diff_):
                ranking_loss += (torch.relu(diff_+ 0.5)/2).mean()
            cnter += 1

        ranking_loss = ranking_loss / cnter if cnter else 0
        reg_loss = reg_loss / len(self.reg_index) if len(self.reg_index) else 0

        # loss = overall_bce_loss + overall_reg_loss + density_bce_loss + reg_loss + ranking_loss
        loss = overall_bce_loss + density_bce_loss + reg_loss + ranking_loss

        return {
                'loss':loss, 
                'overall_bce_loss':overall_bce_loss, 
                'density_bce_loss':density_bce_loss, 
                'overall_reg_loss':overall_reg_loss,
                'reg_loss':reg_loss,
                'ranking_loss':ranking_loss
                }