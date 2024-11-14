import torch.nn as nn
import torch

class BCLoss:
    def __init__(self, **kwargs):
        self.bce = nn.BCELoss(**kwargs)
    
    def __call__(self, outs):
        logits = outs['logit']
        mask = outs['mask']

        logits = torch.sigmoid(logits)

        return self.bce(logits.squeeze(1), mask)

