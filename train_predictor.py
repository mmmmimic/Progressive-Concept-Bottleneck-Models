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
from pathlib import PurePath

#%%
## load configs
parser = argparse.ArgumentParser(description="Predictor")
parser.add_argument('--config', type=str, default="./configs/exp.yaml", metavar='-c')
parser.add_argument('--eval', type=bool, default=False, metavar='-e')
parser.add_argument('--checkpoint', type=str, metavar='-ckp', default='')
parser.add_argument('--model_path', type=str, metavar='-m', default='')

config_args = parser.parse_args()

with open(config_args.config, 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

data_cfg = args['DATA']
train_cfg = args['TRAINING']
model_cfg = args['MODEL']

device = "cuda" if torch.cuda.is_available() else "cpu"
if not train_cfg['UseCUDA']:
    device = "cpu" 

epochs = train_cfg['Epochs']
batch_size = train_cfg['BatchSize']
lr = train_cfg['LearningRate']
weight_decay = train_cfg['WeightDecay']

concept_num = data_cfg['ConceptNum']
cat_index = data_cfg['CatIndex']

class_num = data_cfg['ClassNum']
expand_dim = model_cfg['ExpandDim']
head_num = model_cfg['HeadNum']

#%%
# # fix random seed
seed = train_cfg['Seed']
torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# build the predictor
model = models.Predictor(
                        num_classes=class_num, 
                        concept_num=concept_num,
                        expand_dim=expand_dim,
                        head_num=head_num,
                        cat_index=cat_index
                        )
model = model.to(device)

# set up model and dataset here
if data_cfg['DataSet'] == 'FetalTrim3':
    # fetal dataset
    ## build datasets
    eval_transforms = A.Resize(224, 288)
    dataset_cfg = data_cfg['Configs']
    trainset = datasets.FetalSeg(eval_transforms, split='train', **dataset_cfg)
    valset = datasets.FetalSeg(eval_transforms, split='vali', **dataset_cfg)
    testset = datasets.FetalSeg(eval_transforms, split='test', **dataset_cfg)
else:
    raise NotImplementedError() # add your dataset here

exp_name = PurePath(config_args.config).parts[-2] + PurePath(config_args.config).parts[-1].split('.')[0]
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

trainloader = DataLoader(trainset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(trainset), drop_last=False, num_workers=np.min([batch_size, 32]))
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))

criterion = torch.nn.CrossEntropyLoss()

def train_one_epoch(loader):
    model.train()
    preds = []
    gts = []
    epoch_loss = 0
    for x in tqdm(loader):
        concept_logit = x['concept']
        concept_logit = concept_logit.to(device)
        # concept_logit[:,23] = 0
        label = x['label']
        label = label.to(device)
        x['concept_logit'] = concept_logit

        optimizer.zero_grad()
        x = model(x)
        logit = x['logit']

        loss = criterion(logit, label)
        loss.backward()

        optimizer.step()

        batch_size = concept_logit.size(0)
        epoch_loss += loss.item()

        pred = torch.argmax(logit, dim=1)
        preds.append(pred.flatten().detach().cpu().numpy())
        gts.append(label.cpu().numpy())

    epoch_metrics = {'loss': epoch_loss/len(loader)}
    
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    acc = accuracy_score(gts, preds)
    avg_acc = balanced_accuracy_score(gts, preds)
    epoch_metrics['acc'] = acc
    epoch_metrics['avg_acc'] = avg_acc
    return epoch_metrics

def validate_one_epoch(loader):
    model.eval()
    preds = []
    gts = []
    epoch_loss = 0
    for x in tqdm(loader):
        concept_logit = x['concept']
        concept_logit = concept_logit.to(device)
        # concept_logit[:,23] = 0
        label = x['label']
        label = label.to(device)
        x['concept_logit'] = concept_logit

        with torch.no_grad():
            x = model(x)
        logit = x['logit']

        loss = criterion(logit, label)

        batch_size = concept_logit.size(0)
        epoch_loss += loss.item()

        pred = torch.argmax(logit, dim=1)
        preds.append(pred.flatten().detach().cpu().numpy())
        gts.append(label.cpu().numpy())

    epoch_metrics = {'loss': epoch_loss/len(loader)}
    
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    acc = accuracy_score(gts, preds)
    avg_acc = balanced_accuracy_score(gts, preds)
    epoch_metrics['acc'] = acc
    epoch_metrics['avg_acc'] = avg_acc
    return epoch_metrics

def train():
    best_score = 0
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
        
        key = 'avg_acc'
        val_score = val_metrics[key]
        
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print(f'Model is saved at {model_path}!')
    
        logger.fprint('Best %s: %.4f at epoch %d'%(key, best_score, best_epoch))

        ## save checkpoints as a dictionary
        if not (epoch % 50):
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
