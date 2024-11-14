from .contrast_topo_loss import ContrastTopoLoss
from .focal_loss import FocalLoss, get_weight
from .dice_loss import DiceLoss, CLDiceLoss
from .cse_loss import CSELoss
from .bce_loss import BCELoss
from .nll_loss import NLLLoss
from .dtu_loss import DTULoss
from .dicefocal_loss import DFLoss
from .dicecse_loss import DiceCSELoss
from .cbm_loss import PCBMLoss
from .bc_loss import BCLoss
from .mtl_loss import MTLLoss
from .conceiver_loss import ConceiverLoss

class Criterion:
    def __init__(self, loss_dict, loss_configs, weights=None):
        self.loss_modules = []
        self.loss_weights = []
        self.loss_names = []

        for loss, weight in loss_dict.items():
            if loss in loss_configs.keys():
                kwargs = loss_configs[loss]
            else:
                kwargs = {}
            if loss == "dice loss":
                self.loss_modules.append(DiceLoss(**kwargs))
            if loss == "bc loss":
                self.loss_modules.append(BCLoss(**kwargs))
            elif loss == "focal loss":
                self.loss_modules.append(FocalLoss(weight=weights, **kwargs))
            elif loss == "crossentropy":
                self.loss_modules.append(CSELoss(weight=weights, **kwargs))
            elif loss == "nll loss":
                self.loss_modules.append(NLLLoss(weight=weights, **kwargs))
            elif loss == "contrast topo loss":
                self.loss_modules.append(ContrastTopoLoss(**kwargs))
            elif loss == "cldice loss":
                self.loss_modules.append(CLDiceLoss(**kwargs))
            elif loss == "bce loss":
                self.loss_modules.append(BCELoss(**kwargs))
            elif loss == "dtu loss":
                self.loss_modules.append(DTULoss(weight=weights, **kwargs))
            elif loss == "dicefocal loss":
                self.loss_modules.append(DFLoss(weight=weights, **kwargs))
            elif loss == "dicecse loss":
                self.loss_modules.append(DiceCSELoss(weight=weights, **kwargs))
            elif loss == 'pcbm loss':
                self.loss_modules.append(PCBMLoss(**kwargs))
            elif loss == 'mtl loss':
                self.loss_modules.append(MTLLoss(**kwargs))
            else:
                raise NotImplementedError()
            self.loss_weights.append(weight)
            self.loss_names.append(loss)

    def __call__(self, outs):
        loss = {}
        for loss_criterion, weight, name in zip(self.loss_modules, self.loss_weights, self.loss_names):
            loss[name] = weight*loss_criterion(outs)

        loss['loss'] = sum(loss.values())
        return loss

    def __repr__(self):
        return str(self.loss_names)