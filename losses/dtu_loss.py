from .dice_loss import DiceLoss, CSECLDiceLoss, CLDiceLoss
from .focal_loss import FocalLoss
from .dicefocal_loss import DFLoss, DiceFocalLoss
import torch.nn as nn
from .cse_loss import CSELoss
from .dicecse_loss import DCLoss, DiceCSELoss
import torch
from monai.losses import DiceLoss, FocalLoss

class DTULoss:
    def __init__(self, weight, **kwargs):
        self.dicefocal = DFLoss(weight=weight, lambda_dice=1, lambda_focal=1)
        # self.dicefocal = DFLoss(weight=None, lambda_dice=1, lambda_focal=1)
        self.cl_dice = CLDiceLoss(act=False)

        # self.bce = nn.BCELoss()
        self.bce = nn.NLLLoss(weight=torch.FloatTensor([0.3, 0.7]).cuda())
        self.fine_dice = DiceLoss(to_onehot_y=True, include_background=False, softmax=False, sigmoid=False)
        self.weight = weight

    def __call__(self, outs):
        coarse_logit, mask, logit, topo_mask, triplet_loss = outs['coarse_logit'], outs['mask'], outs['logit'], outs['topo_mask'], outs['triplet_loss']  
        dicefocal_loss_coarse = self.dicefocal({'logit':coarse_logit, 'mask':mask})
        # dicefocal_loss_fine = self.fine_dice(logit, mask.unsqueeze(1)) + self.cl_dice({'logit':logit, 'mask':mask})
        # dicefocal_loss_fine = self.cl_dice({'logit':logit, 'mask':mask})
        dicefocal_loss_fine = self.fine_dice(logit, mask.unsqueeze(1))
        # dicefocal_loss_fine += nn.NLLLoss(weight=self.weight.float())(torch.log(logit+1e-12), mask)
        dicefocal_loss_fine += nn.NLLLoss(weight=None)(torch.log(logit+1e-12), mask)

        # bce_loss = self.bce(topo_mask, (mask>0).float())

        bin_label = (mask > 0).long()
        topo_mask = topo_mask.unsqueeze(1)
        topo_mask = torch.cat((1-topo_mask, topo_mask), dim=1)
        topo_mask_ = torch.log(topo_mask + 1e-12)
        bce_loss = self.bce(topo_mask_, bin_label)

        loss = triplet_loss + dicefocal_loss_coarse + dicefocal_loss_fine + bce_loss

        return loss