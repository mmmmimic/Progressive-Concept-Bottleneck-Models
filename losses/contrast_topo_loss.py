import torch
import torch.nn as nn
from models.triplet_modules import TripletModule
import yaml

class ContrastTopoLoss:
    def __init__(self, model_path, device, meta_path, patch_size=32, stride=20, norm_factor=32):
        self.tripnet = TripletModule().to(device)
        self.tripnet.load_state_dict(torch.load(model_path))
        self.tripnet.eval()        
        self.patch_size = patch_size
        self.stride = stride
        self.norm_factor = norm_factor
        
        with open(meta_path, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        if 'ANATOMY' in meta.keys():
            anatomy = meta['ANATOMY']
            linstrips = list(filter(lambda x: anatomy[x]['SHAPE']=='linestrip', anatomy.keys()))
            self.inds = list(map(lambda x: anatomy[x]['INDEX'], linstrips))
        else:
            self.inds = []
    
    def __call__(self, outs):
        logit = outs['logit']
        mask = outs['mask']

        pred = torch.argmax(logit, dim=1) # (B, H, W)

        if len(self.inds):
            iter_pred = torch.zeros_like(pred)
            for i in self.inds:
                if i in pred:
                    iter_pred[pred==i] = 1
            pred = iter_pred.float()
            
            iter_mask = torch.zeros_like(mask)
            for i in self.inds:
                if i in mask:
                    iter_mask[mask==i] = 1
            mask = iter_mask.float()

        mask = mask.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        mask = mask.flatten(0,2).unsqueeze(1) # (BHW/P**2, 1, P, P)
        pred = pred.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        pred = pred.flatten(0,2).unsqueeze(1) # (BHW/P**2, 1, P, P)

        # resample the zero patches, for data balance
        value = mask.flatten(-3, -1).sum(-1)
        index = torch.arange(0, value.size(0), step=1, device=mask.device)
        nonzero_index = index[value > 0]
        zero_index = index[value <= 0]
        if zero_index.size(0) > nonzero_index.size(0):
            zero_index = zero_index[:nonzero_index.size(0)]
        index = torch.cat((nonzero_index, zero_index))

        if not index.size(0):
            return torch.tensor([0]).to(mask.device)

        pred, mask = pred[index, ...], mask[index, ...]

        pred_emb = self.tripnet(pred)
        with torch.no_grad():
            mask_emb = self.tripnet(mask)

        loss = nn.functional.pairwise_distance(pred_emb, mask_emb, p=2).mean() / self.norm_factor

        return loss