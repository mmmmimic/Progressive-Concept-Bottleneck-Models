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
parser.add_argument('--conceiver_path', type=str, default='logs/conceiver3/model.t7')
parser.add_argument('--observer_path', type=str, default='logs/nnn3/model.t7')

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
    tfs.append(A.Resize(*train_cfg['EvalSize']))
    augs = train_cfg['EvalAugmentations']
    for a in augs.keys():
        aug = eval("A.%s(**%s)"%(a, augs[a]))
        tfs.append(aug)
    eval_transforms = A.Compose(
        tfs
    )
    dataset_cfg = data_cfg['Configs']
    testset = datasets.FetalSeg(eval_transforms, split='test', **dataset_cfg)

    ## build the model
    model = models.FetalConceiver(
                                    in_channels=in_channels, 
                                    seg_concept_num=seg_channel+7, # 7 for caliper concepts
                                    global_index=global_index, 
                                    local_index=local_index, 
                                    relationship=relationship
                                )

    seg_model = models.DTUNet(in_channels, seg_channel)
else:
    raise NotImplementedError() # add your dataset here

model = model.to(device)
seg_model = seg_model.to(device)

model.load_state_dict(torch.load(config_args.conceiver_path))
seg_model.load_state_dict(torch.load(config_args.observer_path))

exp_name = PurePath(config_args.config).parts[-2] + PurePath(config_args.config).parts[-1].split('.')[0] + '_eval'
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
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))

def validate_one_epoch(loader):
    model.eval()
    seg_model.eval()
    loss_meter = metrics.AverageMeter()
    cls_meter = metrics.ClassMeter()
    concept_preds = []
    concept_gts = []

    for x in tqdm(loader):
        image, concept_gt = x['gray_image'], x['concept']
        image, concept_gt = image.to(device), concept_gt.to(device)
        
        with torch.no_grad():
            assign_mtx = seg_model({'image':image})['logit']
            # seg_mask = torch.argmax(assign_mtx, dim=1)
            # seg_mask = x['mask'].to(device)
            # assign_mtx = torch.nn.functional.one_hot(seg_mask, num_classes=seg_channel).permute(0,3,1,2)

        x['image'] = image
        x['assign_mtx'] = assign_mtx
        x['concept_gt'] = concept_gt

        with torch.no_grad():
            x = model(x)

        batch_size = image.size(0)

        concept_preds.append(x['concept_pred'].detach().cpu().numpy())
        concept_gts.append(concept_gt.detach().cpu().numpy())

    epoch_metrics = {}
    
    concept_preds = np.concatenate(concept_preds, axis=0)
    concept_gts = np.concatenate(concept_gts, axis=0)

    cat_pred = concept_preds[:, np.array(cat_index)]
    cat_gt = concept_gts[:, np.array(cat_index)]

    for i in range(len(cat_index)):
        print(cat_index[i]+1, accuracy_score(cat_gt[:,i], cat_pred[:,i]))

    reg_pred = concept_preds[:, np.array(reg_index)]
    reg_gt = concept_gts[:, np.array(reg_index)]

    loss = []
    for i in range(10):
        if np.sum(reg_gt==i):
            loss.append(math.sqrt(np.mean((reg_pred[reg_gt==i] - reg_gt[reg_gt==i])**2)))

    # for i in range(len(reg_index)):
    #     pred = reg_pred[:, i]
    #     gt = reg_gt[:, i]
    #     loss = []
    #     for j in range(11):
    #         if np.sum(gt==j):
    #             loss.append(math.sqrt(np.mean((pred[gt==j] - gt[gt==j])**2)))
    #     print(reg_index[i]+1, np.mean(loss))

    epoch_metrics['avg_mse'] = np.mean(loss)
    epoch_metrics['mse nonzero'] = math.sqrt(np.mean((reg_pred[reg_gt!=0] - reg_gt[reg_gt!=0])**2))
    epoch_metrics['mse'] = math.sqrt(np.mean((reg_pred - reg_gt)**2))
    epoch_metrics['acc'] = accuracy_score(cat_gt.flatten(), cat_pred.flatten())
    epoch_metrics['avg_acc'] = balanced_accuracy_score(cat_gt.flatten(), cat_pred.flatten())
    return epoch_metrics

def test():
    val_metrics = validate_one_epoch(testloader)
    log_info = f"Test on Testset"
    for k, v in val_metrics.items():
        if k == 'confusion_matrix' or k == "classification_report":
            log_info += f", eval_{k}: {v}"
        else:
            log_info += f", eval_{k}: {v: .4f}"
    logger.fprint(log_info)  

if __name__ == "__main__":
    test()