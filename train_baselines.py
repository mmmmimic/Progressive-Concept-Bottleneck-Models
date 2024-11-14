import torch
import models
import losses
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
from sklearn.metrics import accuracy_score
from torchsampler import ImbalancedDatasetSampler
import math
from tqdm import tqdm
from pathlib import PurePath

#%%
## load configs
parser = argparse.ArgumentParser(description="Classification")
parser.add_argument('--config', type=str, default="./configs/classification/cbm_exp1.yaml", metavar='-c')
parser.add_argument('--eval', type=bool, default=False, metavar='-e')
parser.add_argument('--checkpoint', type=str, metavar='C', default='')
parser.add_argument('--model_path', type=str, metavar='-m', default='')

config_args = parser.parse_args()

with open(config_args.config, 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

data_cfg = args['DATA']
train_cfg = args['TRAINING']
model_cfg = args['MODEL']
metric_cfg = args['METRICS']

device = "cuda" if torch.cuda.is_available() else "cpu"
if not train_cfg['UseCUDA']:
    device = "cpu" 
print(f"Using device {device}")

epochs = train_cfg['Epochs']
batch_size = train_cfg['BatchSize']

lr = train_cfg['LearningRate']
weight_decay = train_cfg['WeightDecay']

class_num = data_cfg['ClassNum']
img_channel = data_cfg['ImageChannel']

#%%
# fix random seed
seed = train_cfg['Seed']
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#%%
## build model
if model_cfg['ModelType'] == 'ResNet18':
    model = models.ResNet(in_channels=img_channel, 
                        out_channels=class_num, depth=18, weights='ResNet18_Weights.IMAGENET1K_V1')
elif model_cfg['ModelType'] == 'ResNet50':
    model = models.ResNet(in_channels=img_channel, 
                        out_channels=class_num, depth=50, weights='ResNet50_Weights.IMAGENET1K_V1')
elif model_cfg['ModelType'] == 'SonoNet':
    model = models.SonoNets(config='SN32', num_labels=class_num, weights=False, features_only=False, in_channels=img_channel)
elif model_cfg['ModelType'] == 'SASceneNet':
    model = models.SASNet(arch='ResNet-18', scene_classes=class_num, semantic_classes=model_cfg['SegConceptNum']-1, in_channel=img_channel, seg_model_path=model_cfg['SegModelPath'])
elif model_cfg['ModelType'] == 'MTLNet':
    model = models.MTLNet(model_cfg['ModelPath'])
elif model_cfg['ModelType'] == 'StandardModel':
    model = models.StandardModel()
else:
    raise NotImplementedError()

model = model.to(device)
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
logger.fprint(f"Save the checkpoint every {train_cfg['CheckRate']} epochs")

#%%
## setup optimisers
if train_cfg['UseSGD']:
    lr = lr*10
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, momentum=train_cfg['Momentum'])
    logger.fprint(f"Using SGD, lr is {lr}, momentum is {train_cfg['Momentum']}, weight decay is {weight_decay}")
else:
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    logger.fprint(f"Using AdamW, lr is {lr}, weight decay is {weight_decay}")

logger.fprint("Training settings")
logger.fprint(train_cfg)
#%%
## setup schedulers
if train_cfg['Scheduler'] == "ReduceOnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
elif train_cfg['Scheduler'] == "Cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
else:
    print(f"{train_cfg['Scheduler']} has not been implemented")
    raise NotImplementedError()

#%%
## build datasets
tfs = []
tfs.append(A.Resize(*train_cfg['TrainSize']))
augs = train_cfg['TrainAugmentations']
for a in augs.keys():
    aug = eval("A.%s(**%s)"%(a, augs[a]))
    tfs.append(aug)
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
dataset_cfg['keep_trim3'] = True # just to make sure    
if data_cfg['DataSet'] == 'FetalTrim3':
    trainset = datasets.FetalSeg(train_transforms, split='train', **dataset_cfg)
    valset = datasets.FetalSeg(eval_transforms, split='vali', **dataset_cfg)
    testset = datasets.FetalSeg(eval_transforms, split='test', **dataset_cfg)
else:
    logger.fprint(f"Dataset {data_cfg['DataSet']} has not been implemented.")
    raise NotImplementedError()

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=np.min([batch_size, 32]), drop_last=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))
if data_cfg['DataSet'] == 'FetalTrim3':
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(trainset), drop_last=False, num_workers=np.min([batch_size, 32]))

#%%
## build losses
criterion = losses.Criterion(train_cfg['Loss'], train_cfg['LossConfigs'])

def train_one_epoch(loader, epoch):
    model.train()
    loss_meter = metrics.AverageMeter()
    cls_meter = metrics.ClassMeter()

    for x in tqdm(loader):
        label = x['label'].to(device)
        if img_channel == 3:
            image = x['image'].to(device)
        elif img_channel == 1:
            image = x['gray_image'].to(device)
        else:
            raise ValueError

        x['image'], x['label'] = image, label 

        optimizer.zero_grad()
        x = model(x)
            
        x['label'] = label
        x['epoch'] = epoch

        loss_unred = criterion(x)
        loss = loss_unred['loss']
        loss.backward()

        optimizer.step()

        batch_size = image.size(0)
        loss_metrics = dict(zip(metric_cfg['TrainLossMetrics'], list(map(lambda x: loss_unred[x], metric_cfg['TrainLossMetrics'])))) 
        loss_meter.add(loss_metrics, batch_size)

        logit = x['logit']
        cls_meter.add(logit.transpose(0,1), 
                        torch.argmax(logit, dim=1).flatten(),
                        x['label'].flatten())

    loss_metrics = list(map(lambda x: loss_meter.avg[x], metric_cfg['TrainLossMetrics']))
    loss_metrics = dict(zip(metric_cfg['TrainLossMetrics'], loss_metrics))

    cls_metrics = []
    cls_meter.gather()
    for m in metric_cfg['TrainClassMetrics']:
        cls_metrics.append(eval(f"cls_meter.get_{m}()"))
    
    cls_metrics = dict(zip(metric_cfg['TrainClassMetrics'], cls_metrics))

    epoch_metrics = {**loss_metrics, **cls_metrics}

    return epoch_metrics

def validate_one_epoch(loader, epoch):
    model.eval()
    loss_meter = metrics.AverageMeter()
    cls_meter = metrics.ClassMeter()

    with torch.no_grad():
        for x in tqdm(loader):
            label = x['label'].to(device)
            if img_channel == 3:
                image = x['image'].to(device)
            elif img_channel == 1:
                image = x['gray_image'].to(device)
            else:
                raise ValueError

            x['image'], x['label'] = image, label

            x = model(x) 
             
            x['label'] = label
            x['epoch'] = epoch       

            loss_unred = criterion(x)

            batch_size = image.size(0)
            loss_metrics = dict(zip(metric_cfg['EvalLossMetrics'], list(map(lambda x: loss_unred[x], metric_cfg['EvalLossMetrics'])))) 
            loss_meter.add(loss_metrics, batch_size)

            logit = x['logit']
            cls_meter.add(logit.transpose(0,1), 
                        torch.argmax(logit, dim=1).flatten(),
                        x['label'].flatten())

    loss_metrics = list(map(lambda x: loss_meter.avg[x], metric_cfg['EvalLossMetrics']))
    loss_metrics = dict(zip(metric_cfg['EvalLossMetrics'], loss_metrics))

    cls_metrics = []
    cls_meter.gather()

    for m in metric_cfg['EvalClassMetrics']:
        cls_metrics.append(eval(f"cls_meter.get_{m}()"))
    
    cls_metrics = dict(zip(metric_cfg['EvalClassMetrics'], cls_metrics))

    epoch_metrics = {**loss_metrics, **cls_metrics}

    return epoch_metrics

def train():
    best_score = 0.0 if train_cfg['MonitorPattern'] != 'min' else 100
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
            # best_epoch, best_score = ckp['best_epoch'], ckp['best_score']
            logger.fprint(f"Loaded checkpoint {config_args.checkpoint}, start at epoch {start_epoch}. ")
            del ckp
        else:
            logger.fprint(f"checkpoint {config_args.checkpoint} does not exist")
            raise NameError()
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(trainloader, epoch)
        log_info = f"epoch: {epoch: d}"
        for k, v in train_metrics.items():
            log_info += f", train_{k}: {v: .4f}"
        logger.fprint(log_info)

        val_metrics = validate_one_epoch(valloader, epoch)
        log_info = f"epoch: {epoch: d}"
        for k, v in val_metrics.items():
            log_info += f", eval_{k}: {v: .4f}"
        logger.fprint(log_info)       

        if train_cfg['Scheduler'] == "ReduceOnPlateau":
            scheduler.step(train_metrics['loss'])
        else:
            scheduler.step()
        
        assert train_cfg['MonitorMetric'] in val_metrics.keys(),f"The monitored metric {train_cfg['MonitorMetric']} is not saved. "
        
        val_score = val_metrics[train_cfg['MonitorMetric']]
        
        if train_cfg['MonitorPattern'] == 'min':
            if val_score < best_score:
                best_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(f'Model is saved at {model_path}!')
        else:
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(f'Model is saved at {model_path}!')            
        
        logger.fprint('Best %s: %.4f at epoch %d'%(train_cfg['MonitorMetric'], best_score, best_epoch))

        ## save checkpoints as a dictionary
        if not (epoch % train_cfg['CheckRate']):
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
        torch.save(model.state_dict(), model_path.replace('model.t7', 'last_model.t7'))

def test():
    # model_path = config_args.eval_model if os.path.exists(config_args.eval_model) else model_path 
    try:
        model.load_state_dict(torch.load(config_args.model_path if os.path.exists(config_args.model_path) else model_path))
    except RuntimeError:
        logger.fprint(f"The given model '{model_path}' is not valid.")
    

    val_metrics = validate_one_epoch(testloader, epoch=0)
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