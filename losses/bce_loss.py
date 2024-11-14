import torch.nn as nn
import torch

class BCELoss:
    def __init__(self, **kwargs):
        self.bce = nn.BCELoss(**kwargs)
    
    def __call__(self, outs):
        logits = outs['binary_logit']
        mask = outs['mask']

        logits = torch.sigmoid(logits)
        mask = (mask > 0).float()

        return self.bce(logits, mask)

