from monai.losses import DiceCELoss
from torch.nn import functional as F

class DiceCSELoss:
    def __init__(self, weight, **kwargs):
        self.criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1., lambda_ce=1.,
            smooth_dr=1e-5, smooth_nr=1e-5,
            squared_pred=False,       
            ce_weight=weight      
        )

    def __call__(self, x):
        logit, mask = x['logit'], x['mask']
        # to onehot
        mask = mask.unsqueeze(1)
        loss = self.criterion(logit, mask)
        return loss

class DCLoss:
    def __init__(self, weight, **kwargs):
        self.criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=False,
            lambda_dice=1., lambda_ce=1.,
            smooth_dr=1e-5, smooth_nr=1e-5,
            squared_pred=False,       
            ce_weight=weight      
        )

    def __call__(self, logit, mask):
        # to onehot
        # mask = mask.unsqueeze(1)
        loss = self.criterion(logit, mask)
        return loss