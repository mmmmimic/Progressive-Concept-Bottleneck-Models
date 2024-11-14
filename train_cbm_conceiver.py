import torch
from tqdm import tqdm
import metrics
import datasets
import albumentations as A
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml
from misc import Logger
import os
import random
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torchsampler import ImbalancedDatasetSampler
import math
import models
import torch.nn as nn
import matplotlib.pyplot as plt
from misc import fit_line
import timm
import yaml
from pathlib import PurePath
from losses import ConceiverLoss

#%%
## load configs
parser = argparse.ArgumentParser(description="Conceiver")
parser.add_argument('--config', type=str, default="./configs/exp.yaml", metavar='-c')
parser.add_argument('--eval', type=bool, default=False, metavar='-e')
parser.add_argument('--checkpoint', type=str, metavar='-ckp', default='')
parser.add_argument('--model_path', type=str, metavar='-m', default='')

config_args = parser.parse_args()

with open(config_args.config, 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

data_cfg = args['DATA']
train_cfg = args['TRAINING']

device = "cuda" if torch.cuda.is_available() else "cpu"
if not train_cfg['UseCUDA']:
    device = "cpu" 

epochs = train_cfg['Epochs']
batch_size = train_cfg['BatchSize']
lr = train_cfg['LearningRate']
weight_decay = train_cfg['WeightDecay']

seg_channel = data_cfg['SegChannel']
in_channels = data_cfg['ImageChannel']

#%%
# # fix random seed
seed = train_cfg['Seed']
torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

reg_index = data_cfg['RegIndex']
cat_index = data_cfg['CatIndex']
global_index = data_cfg['GlobalIndex']
local_index = data_cfg['LocalIndex']
relationship = data_cfg['Relationship']

# set up model and dataset here
if data_cfg['DataSet'] == 'FetalTrim3':
    # fetal dataset

    ## build datasets
    tfs = []
    tfs.append(A.Resize(*train_cfg['TrainSize']))
    augs = train_cfg['TrainAugmentations']
    for a in augs.keys():
        aug = eval("A.%s(**%s)"%(a, augs[a]))
        tfs.append(aug)

    tfs.append(A.OneOf([
            A.RandomGamma(gamma_limit=(60, 120), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),
        ]))
    train_transforms = A.Compose(
        tfs
    )

    tfs = []
    tfs.append(A.Resize(*train_cfg['EvalSize']))
    augs = train_cfg['EvalAugmentations']
    for a in augs.keys():
        aug = eval("A.%s(**%s)"%(a, augs[a]))
        tfs.append(aug)
    eval_transforms = A.Compose(
        tfs
    )
    dataset_cfg = data_cfg['Configs']
    trainset = datasets.FetalSeg(train_transforms, split='train', **dataset_cfg)
    valset = datasets.FetalSeg(eval_transforms, split='vali', **dataset_cfg)
    testset = datasets.FetalSeg(eval_transforms, split='test', **dataset_cfg)

    ## build the model
    model = models.FetalCBMConceiver(in_channels, global_index, local_index)
else:
    raise NotImplementedError() # add your dataset here

model = model.to(device)

exp_name = PurePath(config_args.config).parts[-2] + PurePath(config_args.config).parts[-1].split('.')[0] + 'cbm'
exp_folder = os.path.join("./logs", exp_name)
model_path = os.path.join(exp_folder, "model.t7")

if not os.path.exists(exp_folder):
    os.system(f"mkdir {exp_folder}")
os.system(f"cp {config_args.config} {os.path.join(exp_folder, 'config.yaml')}")
ckp_folder = os.path.join(exp_folder, 'checkpoints')
if not os.path.exists(ckp_folder):
    os.system(f"mkdir {ckp_folder}")

# initialize logger
logger = Logger(os.path.join(exp_folder, 'logs.log'), 'a')
logger.fprint(f"Start experiment {exp_name}")
logger.fprint(f'Fix random seed at {seed}')
logger.fprint("Model")
#%%

## setup optimisers
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, momentum=0.9)
logger.fprint(f"Using SGD, lr is {lr}, momentum is {0.9}, weight decay is {weight_decay}")

#%%
## setup schedulers
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs)

trainloader = DataLoader(trainset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(trainset), drop_last=False, num_workers=np.min([batch_size, 32]))
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))


criterion = ConceiverLoss(reg_index, cat_index)


def train_one_epoch(loader):
    model.train()
    loss_meter = metrics.AverageMeter()
    cls_meter = metrics.ClassMeter()
    concept_preds = []
    concept_gts = []
    epoch_loss = 0
    for x in tqdm(loader):
        image, mask, concept_gt = x['gray_image'], x['mask'], x['concept']
        image, mask, concept_gt = image.to(device), mask.to(device), concept_gt.to(device)

        assign_mtx = torch.nn.functional.one_hot(mask, num_classes=seg_channel).permute(0,3,1,2)

        x['image'] = image
        x['assign_mtx'] = assign_mtx
        x['concept_gt'] = concept_gt

        optimizer.zero_grad()
        x = model(x)

        loss = criterion(x)['loss']
        loss.backward()

        optimizer.step()

        batch_size = image.size(0)
        epoch_loss += loss.item()

        concept_preds.append(x['concept_pred'].detach().cpu().numpy())
        concept_gts.append(concept_gt.detach().cpu().numpy())

    epoch_metrics = {'loss': epoch_loss/len(loader)}
    
    concept_preds = np.concatenate(concept_preds, axis=0)
    concept_gts = np.concatenate(concept_gts, axis=0)

    cat_pred = concept_preds[:, np.array(cat_index)]
    cat_gt = concept_gts[:, np.array(cat_index)]
    reg_pred = concept_preds[:, np.array(reg_index)]
    reg_gt = concept_gts[:, np.array(reg_index)]

    epoch_metrics['mse nonzero'] = math.sqrt(np.mean((reg_pred[reg_gt!=0] - reg_gt[reg_gt!=0])**2))
    epoch_metrics['mse'] = math.sqrt(np.mean((reg_pred - reg_gt)**2))
    acc = balanced_accuracy_score(cat_gt.flatten(), cat_pred.flatten())
    epoch_metrics['acc'] = acc
    return epoch_metrics

def validate_one_epoch(loader):
    model.eval()
    loss_meter = metrics.AverageMeter()
    cls_meter = metrics.ClassMeter()
    concept_preds = []
    concept_gts = []
    epoch_loss = 0

    for x in tqdm(loader):
        image, mask, concept_gt = x['gray_image'], x['mask'], x['concept']
        image, mask, concept_gt = image.to(device), mask.to(device), concept_gt.to(device)
        
        assign_mtx = torch.nn.functional.one_hot(mask, num_classes=seg_channel).permute(0,3,1,2)

        x['image'] = image
        x['assign_mtx'] = assign_mtx
        x['concept_gt'] = concept_gt

        with torch.no_grad():
            x = model(x)

        loss = criterion(x)['loss']

        batch_size = image.size(0)
        epoch_loss += loss.item()

        concept_preds.append(x['concept_pred'].detach().cpu().numpy())
        concept_gts.append(concept_gt.detach().cpu().numpy())

    epoch_metrics = {'loss': epoch_loss/len(loader)}
    
    concept_preds = np.concatenate(concept_preds, axis=0)
    concept_gts = np.concatenate(concept_gts, axis=0)

    cat_pred = concept_preds[:, np.array(cat_index)]
    cat_gt = concept_gts[:, np.array(cat_index)]
    reg_pred = concept_preds[:, np.array(reg_index)]
    reg_gt = concept_gts[:, np.array(reg_index)]

    epoch_metrics['mse nonzero'] = math.sqrt(np.mean((reg_pred[reg_gt!=0] - reg_gt[reg_gt!=0])**2))
    epoch_metrics['mse'] = math.sqrt(np.mean((reg_pred - reg_gt)**2))
    acc = balanced_accuracy_score(cat_gt.flatten(), cat_pred.flatten())
    epoch_metrics['acc'] = acc

    return epoch_metrics

def train():
    best_score = 100
    best_epoch = 0

    ## load checkpoints
    if len(config_args.checkpoint):
        if os.path.exists(config_args.checkpoint):
            try:
                ckp = torch.load(config_args.checkpoint)
                # load model statedict
                model.load_state_dict(ckp['model_state_dict'])
            except:
                logger.fprint("model parameters might not be loaded correctly")

            # load optimiser statedict
            optimizer.load_state_dict(ckp['optimizer_state_dict'])
            # load scheduler statedict
            scheduler.load_state_dict(ckp['scheduler_state_dict'])
            # load epoch
            start_epoch = ckp['epoch'] + 1
            # best epoch and best score
            best_epoch, best_score = ckp['best_epoch'], ckp['best_score']
            logger.fprint(f"Loaded checkpoint {config_args.checkpoint}, start at epoch {start_epoch}. ")
            del ckp
        else:
            logger.fprint(f"checkpoint {config_args.checkpoint} does not exist")
            raise NameError()
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(trainloader)
        log_info = f"epoch: {epoch: d}"
        for k, v in train_metrics.items():
            log_info += f", train_{k}: {v: .4f}"
        logger.fprint(log_info)

        val_metrics = validate_one_epoch(valloader)
        log_info = f"epoch: {epoch: d}"
        for k, v in val_metrics.items():
            log_info += f", eval_{k}: {v: .4f}"
        logger.fprint(log_info)       

        scheduler.step()
        
        val_score = val_metrics['mse nonzero']
        
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print(f'Model is saved at {model_path}!')
    
        logger.fprint('Best %s: %.4f at epoch %d'%('mse nonzero', best_score, best_epoch))

        ## save checkpoints as a dictionary
        if not (epoch % 100):
            ckp_name = os.path.join(ckp_folder, f"checkpoint_{epoch}.t7")
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'loss': train_metrics['loss'],
            'epoch': epoch,
            'device': device,
            'best_score': best_score,
            'best_epoch': best_epoch
            },
            ckp_name)
            logger.fprint(f"checkpoint saved at {ckp_name}")

def test():
    try:
        model.load_state_dict(torch.load(config_args.model_path if os.path.exists(config_args.model_path) else model_path))
    except RuntimeError:
        logger.fprint(f"The given model '{model_path}' is not valid.")
    val_metrics = validate_one_epoch(testloader)
    log_info = f"Test on Testset"
    for k, v in val_metrics.items():
        if k == 'confusion_matrix' or k == "classification_report":
            log_info += f", eval_{k}: {v}"
        else:
            log_info += f", eval_{k}: {v: .4f}"
    logger.fprint(log_info)  

if __name__ == "__main__":
    if config_args.eval:
        logger.fprint("Start Testing")
        test()
    else:
        logger.fprint("Start Training")
        train()
