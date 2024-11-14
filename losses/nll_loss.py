import torch.nn as nn

class NLLLoss:
    def __init__(self, weight, **kwargs):
        self.nll = nn.NLLLoss(weight=weight, **kwargs)
    
    def __call__(self, outs):
        logits = outs['logit']
        mask = outs['mask']

        return self.nll(logits, mask)