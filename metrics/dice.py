import torch
import torch.nn as nn
from torch.nn import functional as F

def one_hot_embedding(y, num_classes):
    '''
    Embedding labels to one-hot form.

    Args:
        y: (LongTensor) class labels, sized [N, W, H]

    Returns:
        (tensor) encoded labels, sized [N, C, W, H].
    '''
    y = y.long()
    b, w, h = y.shape # [B, W, H]
    y = y.flatten() # [B*W*H]

    D = torch.eye(num_classes).to(y.device) # [C, C]

    y = D[y] # [N, C]

    y = y.view(b, w, h, -1).permute(0, 3, 1, 2) # [B, C, W, H]

    return y.float()

def soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3,1), stride=(1,1), padding=(1,0))
    p2 = -F.max_pool2d(-img, kernel_size=(1,3), stride=(1,1), padding=(0,1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=(3,3), stride=(1,1), padding=(1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_close(img):
    return soft_erode(soft_dilate(img))

def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)

    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta-skel*delta)
    
    return skel

def get_dice(logit, mask, class_num, smooth=1e-5, reduction='micro', exclude_background=True):
    # mask = nn.functional.one_hot(mask, class_num).permute(0,3,1,2)
    mask = one_hot_embedding(mask, num_classes=class_num)
    if smooth is not None:
        pred = torch.argmax(logit, dim=1)
        # pred = nn.functional.one_hot(pred, class_num).permute(0,3,1,2)
        pred = one_hot_embedding(pred, num_classes=class_num)
        if exclude_background:
            pred = pred[:, 1:, ...]
            mask = mask[:, 1:, ...]
        if reduction == 'macro':
            intersection = (pred*mask).flatten(-2, -1).sum(-1) # (B, C)
            union = (pred + mask).flatten(-2, -1).sum(-1)
            nonzero_num = torch.sum(union>0, dim=-1)
            dice_score = (2*intersection+smooth) / (union+smooth)
            dice_score = (dice_score.mean(-1)*intersection.size(-1) - (intersection.size(-1) - nonzero_num))/nonzero_num
            dice_score = dice_score.mean(-1)
        elif reduction == 'micro':
            intersection = torch.sum(pred*mask)
            dice_score = (2.*intersection + smooth) / (torch.sum(pred) + torch.sum(mask) + smooth)
        dice_score.requires_grad = True
    else:
        pred = torch.softmax(logit, dim=1)
        if exclude_background:
            pred = pred[:, 1:, ...]
            mask = mask[:, 1:, ...]
        if reduction == 'macro':
            intersection = (pred*mask).flatten(-2, -1).sum(-1) # (B, C)
            union = pred.flatten(-2, -1).sum(-1) + mask.flatten(-2, -1).sum(-1)
            nonzero_num = torch.sum(mask.flatten(-2,-1).sum(-1)>0, dim=-1)
            dice_score = (2*intersection / union)
            dice_score = dice_score.mean(-1)*intersection.size(1)/nonzero_num
            dice_score = dice_score.mean(-1)   
        elif reduction == 'micro':
            intersection = (pred*mask).sum()
            dice_score = (2.*intersection / (pred.sum() + mask.sum()))             

    return dice_score


def get_cldice(logit, mask, class_num, smooth=1e-5, iter_=3, alpha=1, cls_ = [1,2,3,5,7,10,13], act=True):
    mask = one_hot_embedding(mask, num_classes=class_num)
    if act:
        pred = torch.softmax(logit, dim=1)
    else:
        pred = logit

    # pred = torch.argmax(logit, dim=1)
    # pred = one_hot_embedding(pred, num_classes=class_num)

    # pred = pred[:, 1:, ...]
    # mask = mask[:, 1:, ...]
    # intersection = (pred*mask).sum()
    # dice_score = 1. - (2.0*intersection + smooth) / (pred.sum() + mask.sum() + smooth)
    
    # intersection = torch.sum((pred*mask).flatten(2), dim=-1)
    # union = pred + mask
    # union = torch.sum(union.flatten(2), dim=-1)
    # dice_score = 1. - (2.0*intersection + smooth) / (union + smooth)
    # dice_score = dice_score[:, 1:, ...]
    # dice_score = dice_score.mean(-1).mean(-1)

    skel_pred = soft_skel(pred, iter_)
    skel_true = soft_skel(mask, iter_)

    skel_pred = skel_pred.flatten(2)
    skel_true = skel_true.flatten(2)

    tprec = (torch.sum(skel_pred*mask.flatten(2), dim=-1)+smooth)/(torch.sum(skel_pred, dim=-1)+smooth)
    tsens = (torch.sum(skel_true*pred.flatten(2), dim=-1)+smooth)/(torch.sum(skel_true, dim=-1)+smooth)
    # tprec = (torch.sum(torch.multiply(skel_pred, mask)[:,1:,...])+smooth)/(torch.sum(skel_pred[:,1:,...])+smooth)
    # tsens = (torch.sum(torch.multiply(skel_true, pred)[:,1:,...])+smooth)/(torch.sum(skel_true[:,1:,...])+smooth)        
    cl_dice = 1. - 2.0*(tprec*tsens)/(tprec + tsens)   
    # cl_dice = cl_dice[:, 1:]
    cl_dice = cl_dice[:, cls_]
    cl_dice = cl_dice.mean(-1).mean(-1)

    loss = alpha*cl_dice# + (1.0-alpha)*dice_score

    # loss.requires_grad=True   

    return loss