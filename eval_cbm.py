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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from torchsampler import ImbalancedDatasetSampler
import math
import models
import torch.nn as nn
import matplotlib.pyplot as plt
import timm
import yaml
from pathlib import PurePath

#%%
## load configs
parser = argparse.ArgumentParser(description="Conceiver")
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--conceiver_path', type=str, default=f'logs/conceiver1cbm/model.t7')
parser.add_argument('--predictor_path', type=str, default=f'logs/plain_predictor1/model.t7')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 32

seg_channel = 14
in_channels = 1
num_classes = 8
concept_num = 27
expand_dim = 256
head_num = 0

#%%
# # fix random seed
seed = 42
torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

reg_index = [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]
cat_index = [2, 3, 4, 8, 13, 17, 22, 24]
global_index = [[2, 3, 4, 8, 13, 17, 22, 24]]
local_index = [
                [5,6,7,14,15,16,23], # local concept indices, group 1 (organ quality) 
                [0,1,9,10,11,12,18,19,20,21,25,26] # local concept indices, group 2, (caliper quality)
                ]


# fetal dataset
## build datasets
tfs = []
tfs.append(A.Resize(224, 288))
eval_transforms = A.Compose(
    tfs
)
testset = datasets.FetalSeg(
                            eval_transforms, 
                            split='test',
                            csv_dir= '/home/manli/progressive-concept-bottleneck-models/metas/trim3_sp.csv', # folder containing the data or name of the csv file 
                            meta_dir= '/data/proto/Zahra_Study1_Trials/trim3_sp.yaml', # name of the meta file
                            split_index= f'split{args.split}',
                            keep_trim3=True,
                            plane=['Femur', 'Head', 'Abdomen', 'Cervix'],
                            remove_calipers=False
                            )

## build the model
conceiver = models.FetalCBMConceiver(in_channels, global_index, local_index)
predictor = models.Predictor(
                                num_classes=num_classes, 
                                concept_num=concept_num,
                                expand_dim=expand_dim,
                                head_num=head_num,
                                cat_index=cat_index 
                                )


conceiver = conceiver.to(device)
predictor = predictor.to(device)


conceiver.load_state_dict(torch.load(args.conceiver_path))
predictor.load_state_dict(torch.load(args.predictor_path))

exp_name = f'cbm_fetal_eval{args.split}'
exp_folder = os.path.join("./logs", exp_name)
model_path = os.path.join(exp_folder, "model.t7")

if not os.path.exists(exp_folder):
    os.system(f"mkdir {exp_folder}")

# initialize logger
logger = Logger(os.path.join(exp_folder, 'logs.log'), 'a')
logger.fprint(f"Start experiment {exp_name}")
logger.fprint(f'Fix random seed at {seed}')
logger.fprint("Model")
#%%
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))

def validate_one_epoch(loader):
    conceiver.eval()
    predictor.eval()
    preds = []
    gts = []
    logits = []
    concept_preds = []
    concept_gts = []

    for x in tqdm(loader):
        image= x['gray_image'].to(device)
        label = x['label'].to(device)

        x['image'] = image
        concept_gt = x['concept'].to(device)

        with torch.no_grad():
            x = conceiver(x)
            concept_logit = x['concept_logit']
            concept_logit[:, [2, 3, 4, 8, 13, 17, 22, 24]] = torch.sigmoid(concept_logit[:, [2, 3, 4, 8, 13, 17, 22, 24]])
            concept_logit[:, 23] = 0 # cervix should not be considered according to ISUOG
            concept_logit = concept_logit * x['concept_mask']
            x['concept_logit'] = concept_logit
            x = predictor(x)
        
        logit = x['logit']
        batch_size = image.size(0)

        pred = torch.argmax(logit, dim=1)
        preds.append(pred.detach().cpu().numpy())
        gts.append(label.cpu().numpy())
        logits.append(logit.detach().cpu().numpy())
        concept_preds.append(concept_logit.detach().cpu().numpy())
        concept_gts.append(concept_gt.cpu().numpy())

    epoch_metrics = {}
    
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    concept_preds = np.concatenate(concept_preds, axis=0)
    concept_gts = np.concatenate(concept_gts, axis=0)
    logits = np.concatenate(logits, axis=0)

    epoch_metrics['acc'] = accuracy_score(gts, preds)
    epoch_metrics['avg_acc'] = balanced_accuracy_score(gts, preds)

    np.save('pred.npy', preds)
    np.save('gt.npy', gts)
    np.save('concept_preds.npy', concept_preds)
    np.save('concept_gts.npy', concept_gts)
    np.save('logits.npy', logits)

    # print(classification_report(gts, preds, zero_division=0)) 
    print(confusion_matrix(gts, preds))   

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