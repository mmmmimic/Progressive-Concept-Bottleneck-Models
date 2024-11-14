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
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm
import cv2
import pytorch_warmup as warmup
from pathlib import PurePath
from sklearn import metrics as skm

#%%
## load configs
parser = argparse.ArgumentParser(description="Segmentation")
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
metric_cfg = args['METRICS']

device = "cuda" if torch.cuda.is_available() else "cpu"
if not train_cfg['UseCUDA']:
    device = "cpu" 

epochs = train_cfg['Epochs']
batch_size = train_cfg['BatchSize']
lr = train_cfg['LearningRate']
weight_decay = train_cfg['WeightDecay']

class_num = data_cfg['ClassNum']
in_channels = data_cfg['ImageChannel']

#%%
# # fix random seed
seed = train_cfg['Seed']
torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#%%
## build model
if model_cfg['Backbone'] == 'DTUNet':
    train_cfg['Loss'] = {'dtu loss': 1} # can only use dtu loss

if model_cfg['Backbone'] == 'RegUnet':
    model = models.RegUNet(in_channels=in_channels, num_classes=class_num, 
                                 model_size='016', model_type='y', 
                                interpolation=model_cfg['Interpolation'],
                                extra_fc=True
                                ).to(device)
elif model_cfg['Backbone'] == 'Unet':
    model = models.FlexUNet(in_channels=in_channels, out_channels=class_num, pair_num=4).to(device)
elif model_cfg['Backbone'] == 'ResUnet':
    model = models.ResUNet(in_channels=in_channels, num_classes=class_num, model_size='resnet18', dropout=0.2).to(device)
elif model_cfg['Backbone'] == 'DTUNet':
    model = models.DTUNet(in_channels=in_channels, out_channels=class_num, interpolation=model_cfg['Interpolation']).to(device)
elif model_cfg['Backbone'] == 'SMPRegUnet':
    model = models.UNet(encoder_name='timm-regnety_016', 
                                    encoder_weights='imagenet', 
                                    decoder_attention_type='scse', 
                                    classes=class_num, 
                                    activation=None).to(device)
elif model_cfg['Backbone'] == 'SMPResUnet':
    model = models.UNet(encoder_name='resnet50', 
                                    encoder_weights='imagenet', 
                                    decoder_attention_type='scse', 
                                    classes=class_num,
                                    encoder_depth=5,
                                    decoder_channels=[512, 256, 128, 64, 32], 
                                    activation=None
    ).to(device)
elif model_cfg['Backbone'] == 'SMPDeepLab':
    model = models.SMPDeepLab(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        decoder_channels=256,
        classes=class_num,
        in_channels=3
    ).to(device)
elif model_cfg['Backbone'] == 'SMPPSPNet':
    model = models.SMPPSPNet(encoder_name='resnet18',
        encoder_weights='imagenet',
        classes=class_num,
        in_channels=3,
        encoder_depth=3).to(device)
else:
    raise NotImplementedError()

exp_name = PurePath(config_args.config).parts[-2] + PurePath(config_args.config).parts[-1].split('.')[0]
# exp_name = train_cfg['ExpName']
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
logger.fprint("Model")
# logger.fprint(model)
#%%
## setup optimisers
if train_cfg['UseSGD']:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=train_cfg['Momentum'])
    logger.fprint(f"Using SGD, lr is {lr}, momentum is {train_cfg['Momentum']}, weight decay is {weight_decay}")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    logger.fprint(f"Using AdamW, lr is {lr}, weight decay is {weight_decay}")

logger.fprint("Training settings")
logger.fprint(train_cfg)
#%%
## setup schedulers
if train_cfg['Scheduler'] == "ReduceOnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
elif train_cfg['Scheduler'] == "Cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
else:
    print(f"{train_cfg['Scheduler']} has not been implemented")
    raise NotImplementedError()

# warm up
# warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=20)

#%%
## build datasets
tfs = []
# tfs.append(A.Resize(256, 256))
# tfs.append(A.CenterCrop(224, 224))
# tfs.append(A.RandomCrop(160, 160))
tfs.append(A.Resize(*train_cfg['TrainSize']))
# do not scale without changing the pixel size
augs = train_cfg['TrainAugmentations']
for a in augs.keys():
    aug = eval("A.%s(**%s)"%(a, augs[a]))
    tfs.append(aug)
# 0.2, 0.2, 45
# revise augmentation here
tfs.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=1, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR))
tfs.append(A.OneOf([
            A.RandomGamma(gamma_limit=(60, 120), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),
        ]))
tfs.append(A.ElasticTransform(alpha_affine=10, p=0.5, border_mode=cv2.BORDER_CONSTANT))
train_transforms = A.Compose(
    tfs
)

tfs = []
# tfs.append(A.Resize(256, 256))
# tfs.append(A.CenterCrop(224, 224))
tfs.append(A.Resize(*train_cfg['EvalSize']))
augs = train_cfg['EvalAugmentations']
for a in augs.keys():
    aug = eval("A.%s(**%s)"%(a, augs[a]))
    tfs.append(aug)
eval_transforms = A.Compose(
    tfs
)

dataset_cfg = data_cfg['Configs']
if data_cfg['DataSet'] == 'FetalTrim3':
    trainset = datasets.FetalSeg(train_transforms, split='train', remove_calipers=False,**dataset_cfg)
    valset = datasets.FetalSeg(eval_transforms, split='vali', remove_calipers=False,**dataset_cfg)
    testset = datasets.FetalSeg(eval_transforms, split='test', remove_calipers=False, **dataset_cfg)
else:
    logger.fprint(f"Dataset {data_cfg['DataSet']} has not been implemented.")
    raise NotImplementedError()

trainloader = DataLoader(trainset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(trainset), drop_last=False, num_workers=np.min([batch_size, 32]))
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=np.min([batch_size, 32]))
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))
# valloader = DataLoader(valset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(valset), drop_last=False, num_workers=np.min([batch_size, 32]))
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=np.min([batch_size, 32]))

#%%
## build losses
# get weight for cse and focal loss
if train_cfg['UseWeight']:
    # if not config_args.eval:
    weight = losses.get_weight(trainloader)
    # weight = weight / weight.sum()
    # else:
    #     weight = torch.tensor(
    #     [1.3555e-04, 
    #     1.2190e-01, 4.5482e-02, 8.8302e-02, 1.2709e-01, 
    #     3.6402e-02, 
    #     5.4992e-02, 3.1640e-02, 9.2790e-02, 6.2810e-02, 
    #     1.1635e-01, 4.9432e-02, 1.3794e-01, 3.4726e-02]
    #     , requires_grad=False)
    # weight = torch.tensor(
    #     [0.00005,  # background
    #     0.5/4, 0.8/4, 0.8/4, 0.8/4, # cervix
    #     0.005,  # femur
    #     0.2/4, 0.5/4, 0.2/4, 0.8/4, # abdomen
    #     0.8/4, 0.8/4, 1/4, 0.5/4, # head
    #     ], requires_grad=False
    # )
    # weight = torch.tensor([9.9460e-05, 1.0123e-01, 3.5893e-02, 6.8795e-02, 6.9788e-02, 4.5338e-02,
    #     7.1722e-02, 3.8831e-02, 1.6281e-01, 
    #     9.0414e-01, 1.1600e-01, 6.2635e-01, # kidney, _, fossa
    #     1.1064e-01, 2.5810e-02], requires_grad=False)
    # for cervix
    # weight = torch.tensor([0.0014, 0.3686, 0.1304, 0.2503, 0.2493], requires_grad=False)
    weight = weight.to(device)

else:
    weight = None

logger.fprint(f"use weight: {weight}")

criterion = losses.Criterion(train_cfg['Loss'], train_cfg['LossConfigs'], weights=weight)

def train_one_epoch(loader, epoch):
    model.train()
    loss_meter = metrics.AverageMeter()
    cls_meter = metrics.ClassMeter()
    for x in tqdm(loader):
        image, mask = x['gray_image'], x['mask']
        image, mask = image.to(device), mask.to(device)

        x['image'], x['mask'] = image, mask
        x['epoch'] = epoch 
        
        optimizer.zero_grad()
        outs = model(x)

        outs['mask'] = mask
        outs['label'] = mask
        logit = outs['logit']
        loss_unred = criterion(outs)
        loss = loss_unred['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        batch_size = image.size(0)
        loss_metrics = dict(zip(metric_cfg['TrainLossMetrics'], list(map(lambda x: loss_unred[x], metric_cfg['TrainLossMetrics'])))) 
        loss_meter.add(loss_metrics, batch_size)

        logit = outs['logit']
        cls_meter.add(logit.permute(1,0,2,3).flatten(1), 
                        torch.argmax(logit, dim=1).flatten(),
                        outs['mask'].flatten())
        del logit

    loss_metrics = list(map(lambda x: loss_meter.avg[x], metric_cfg['TrainLossMetrics']))
    loss_metrics = dict(zip(metric_cfg['TrainLossMetrics'], loss_metrics))


    if epoch < 250:
        cls_metrics = {}
    else:
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
    preds = []
    gts = []
    with torch.no_grad():
        for x in tqdm(loader):
            image, mask = x['gray_image'], x['mask']
            image, mask = image.to(device), mask.to(device)

            x['image'], x['mask'] = image, mask
            x['epoch'] = epoch 
    
            outs = model(x)

            outs['mask'] = mask
            outs['label'] = mask
            logit = outs['logit']   
            
            loss_unred = criterion(outs)

            batch_size = image.size(0)
            loss_metrics = dict(zip(metric_cfg['EvalLossMetrics'], list(map(lambda x: loss_unred[x], metric_cfg['EvalLossMetrics'])))) 
            loss_meter.add(loss_metrics, batch_size)

            logit = outs['logit']
            cls_meter.add(logit.permute(1,0,2,3).flatten(1), 
                        torch.argmax(logit, dim=1).flatten(),
                        outs['mask'].flatten())
            preds.append(logit.detach().cpu().numpy())
            gts.append(outs['mask'].detach().cpu().numpy())
            del logit

    loss_metrics = list(map(lambda x: loss_meter.avg[x], metric_cfg['EvalLossMetrics']))
    loss_metrics = dict(zip(metric_cfg['EvalLossMetrics'], loss_metrics))

    cls_metrics = []
    cls_meter.gather()

    # if config_args.eval:
    #     preds = np.concatenate(preds, axis=0)
    #     preds = np.argmax(preds, axis=1)
    #     gts = np.concatenate(gts, axis=0)
    #     # print(skm.classification_report(gts, preds))
    #     ious = []
    #     for i in range(14):
    #         if i in gts:
    #             ious.append(skm.jaccard_score(gts[gts==i], preds[gts==i], average='micro'))
    #         else:
    #             ious.append(1)
    #     ious = np.array(ious)
    #     print(ious)
    #     print(ious[1:].mean())
        # np.save(f"{exp_folder}/ious.npy", ious)

    for m in metric_cfg['EvalClassMetrics']:
        cls_metrics.append(eval(f"cls_meter.get_{m}()"))
    
    cls_metrics = dict(zip(metric_cfg['EvalClassMetrics'], cls_metrics))
    epoch_metrics = {**loss_metrics, **cls_metrics}

    return epoch_metrics

def train():
    best_score = 0.0
    best_epoch = 0
    ## load checkpoints
    if len(config_args.checkpoint):
        if os.path.exists(config_args.checkpoint):
            ckp = torch.load(config_args.checkpoint)
            # load model statedict
            try:
                model.load_state_dict(ckp['model_state_dict'])  
            except:
                print('failed to loead model statedict')
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
            # with warmup_scheduler.dampening():
                scheduler.step(train_metrics['loss'])
        else:
            # with warmup_scheduler.dampening():
                scheduler.step()
        
        assert train_cfg['MonitorMetric'] in val_metrics.keys(),f"The monitored metric {train_cfg['MonitorMetric']} is not saved. "
        
        val_score = val_metrics[train_cfg['MonitorMetric']]

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
    try:
        model.load_state_dict(torch.load(config_args.model_path if os.path.exists(config_args.model_path) else model_path))
    except RuntimeError:
        logger.fprint(f"The given model '{model_path}' is not valid.")
    # model.load_state_dict(torch.load(config_args.model_path if os.path.exists(config_args.model_path) else model_path))
    val_metrics = validate_one_epoch(testloader, 0)
    log_info = f"Test on Testset"
    for k, v in val_metrics.items():
        log_info += f", eval_{k}: {v: .4f}"
    logger.fprint(log_info)  

if __name__ == "__main__":
    if config_args.eval:
        logger.fprint("Start Testing")
        test()
    else:
        logger.fprint("Start Training")
        train()