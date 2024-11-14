from metrics import get_dice, get_cldice
import torch.nn as nn

class DiceLoss:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, outs):
        logit, mask = outs['logit'], outs['mask']  
        dice_score = get_dice(logit, mask, **self.kwargs)

        return 1. - dice_score

class CLDiceLoss:
    def __init__(self, class_num=14, **kwargs):
        self.kwargs = kwargs
        self.class_num = class_num

    def __call__(self, outs):
        logit, mask = outs['logit'], outs['mask']  
        dice_score = get_cldice(logit, mask, class_num=self.class_num, **self.kwargs)        

        return dice_score

class CSECLDiceLoss:
    def __init__(self, weight, **kwargs):
        self.kwargs = kwargs
        self.cse = nn.CrossEntropyLoss(weight=weight)

    def __call__(self, outs):
        logit, mask = outs['logit'], outs['mask']  
        dice_loss = get_cldice(logit, mask, class_num=14, **self.kwargs) 

        cse_loss = self.cse(logit, mask)       

        return dice_loss + cse_loss