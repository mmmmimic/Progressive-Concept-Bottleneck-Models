import torch.nn as nn
from ._modules import Conv1x1, Conv3x3, DenseLayer
from .unet_family import DTUNet, UNet
from .backbones import ResNet, Inception, IdendityMapping
import torch
from .activations import fetal_caliper_concept
from .geometry_tools import get_femur_concept, get_abdomen_concept, get_head_concept
from copy import copy


quality_index = [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]
# quality_index = [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21]
binary_index = [2, 3, 4, 8, 13, 17, 22, 24]
quality_names = ['femur_left', 'femur_right', 'stomach', 'umv', 'kidney', 'ada1', 'ada2', 'adb1', 'adb2', 'thalamus', 'csp', 'fossa', 'bpd_near', 'bpd_far', 'ofd_occ', 'ofd_fro','bladder','orif_inner', 'orif_ext']
# quality_names = ['femur_left', 'femur_right', 'stomach', 'umv', 'kidney', 'ada1', 'ada2', 'adb1', 'adb2', 'thalamus', 'csp', 'fossa', 'bpd_near', 'bpd_far', 'ofd_occ', 'ofd_fro']

class QualityNetv1(nn.Module):
    def __init__(self):
        super().__init__()
        # a resnet for classification
        model = ResNet(in_channels=21, out_channels=len(binary_index), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.resnet = model

        self.quality_heads = nn.ModuleDict()
        model = ResNet(in_channels=2, out_channels=6, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.quality_heads['organ'] = model
        model = ResNet(in_channels=2, out_channels=10, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.quality_heads['caliper'] = model

    def forward_once(self, x, name, index, seg, seg_mask):
        x = x.unsqueeze(1) # B, 1, H, W

        # dilation = 20
        # mask = torch.zeros_like(seg_mask)
        # for i in range(mask.size(0)):
        #     if torch.sum(seg_mask[i,...]):
        #         points = torch.nonzero(seg_mask[i,...])
        #         min_x, min_y, max_x, max_y = torch.min(points[:,0]), torch.min(points[:,1]), torch.max(points[:,0]), torch.max(points[:,1])
        #         min_x, min_y = max([0, min_x-dilation]), max([0, min_y - dilation])
        #         max_x, max_y = min([seg_mask.size(1), max_x + dilation]), min([seg_mask.size(2), max_y + dilation])
        #         mask[i, min_x:max_x, min_y:max_y] = 1
        # x = x*mask.unsqueeze(1)
        
        x = torch.cat((x, seg.unsqueeze(1)), dim=1)

        if name in ['bpd_near', 'bpd_far', 'ofd_occ', 'ofd_fro', 'femur_left', 'femur_right', 'ada1', 'ada2', 'adb1', 'adb2']:
            name = 'caliper'
        else:
            name = 'organ'
        x = self.quality_heads[name]({'image':x})['logit'][:,index]
        return x


    def forward(self, x):
        image = x['image']
        full_image = torch.sum(image[:,:14,...], dim=1)
        assign_mtx = x['assign_mtx']
        seg_mask = x['seg_mask']
        binary_concept = self.resnet({'image': image})['logit']
        concept = torch.zeros((image.size(0), 27)).to(image.device)
        concept[:,0] = self.forward_once(full_image, quality_names[0], 0, assign_mtx[:,14,...], assign_mtx[:,14,...])
        concept[:,1] = self.forward_once(full_image, quality_names[1], 1, assign_mtx[:,15,...], assign_mtx[:,15,...])
        concept[:,2] = binary_concept[:,0]
        concept[:,3] = binary_concept[:,1]
        concept[:,4] = binary_concept[:,2]
        concept[:,5] = self.forward_once(full_image, quality_names[2], 0,  assign_mtx[:,6,...], seg_mask==6)
        concept[:,6] = self.forward_once(full_image, quality_names[3], 1,  assign_mtx[:,8,...], seg_mask==8)
        concept[:,7] = self.forward_once(full_image, quality_names[4], 2,  assign_mtx[:,9,...], seg_mask==9)
        concept[:,8] = binary_concept[:,3]
        concept[:,9] = self.forward_once(full_image, quality_names[5], 2, assign_mtx[:,16,...], assign_mtx[:,16,...])
        concept[:,10] = self.forward_once(full_image, quality_names[6], 3, assign_mtx[:,16,...], assign_mtx[:,16,...])
        concept[:,11] = self.forward_once(full_image, quality_names[7], 4, assign_mtx[:,16,...], assign_mtx[:,16,...])
        concept[:,12] = self.forward_once(full_image, quality_names[8], 5, assign_mtx[:,16,...], assign_mtx[:,16,...])
        concept[:,13] = binary_concept[:,4]
        concept[:,14] = self.forward_once(full_image, quality_names[9], 3, assign_mtx[:,10,...], seg_mask==10)
        concept[:,15] = self.forward_once(full_image, quality_names[10], 4, assign_mtx[:,12,...], seg_mask==12)
        concept[:,16] = self.forward_once(full_image, quality_names[11], 5, assign_mtx[:,11,...], seg_mask==11)
        concept[:,17] = binary_concept[:,5]
        concept[:,18] = self.forward_once(full_image, quality_names[12], 6, assign_mtx[:,18,...], assign_mtx[:,18,...])
        concept[:,19] = self.forward_once(full_image, quality_names[13], 7, assign_mtx[:,17,...], assign_mtx[:,17,...])
        concept[:,20] = self.forward_once(full_image, quality_names[14], 8, assign_mtx[:,20,...], assign_mtx[:,20,...])
        concept[:,21] = self.forward_once(full_image, quality_names[15], 9, assign_mtx[:,19,...], assign_mtx[:,19,...])


        return {'logit':concept}    

class QualityNetv2(nn.Module):
    def __init__(self):
        super().__init__()
        # a resnet for classification
        model = ResNet(in_channels=1, out_channels=len(binary_index), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.resnet = model

        self.quality_heads = nn.ModuleDict()
        model = ResNet(in_channels=1, out_channels=7, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.quality_heads['organ'] = model
        model = ResNet(in_channels=1, out_channels=12, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.quality_heads['caliper'] = model

    def forward_once(self, x, name, index, seg=None):
        x = x.unsqueeze(1) # B, 1, H, W,
        if seg is not None:
            seg = seg.unsqueeze(1)
            x = torch.cat((x, seg), dim=1)
        if name in ['bpd_near', 'bpd_far', 'ofd_occ', 'ofd_fro', 'femur_left', 'femur_right', 'ada1', 'ada2', 'adb1', 'adb2','orif_inner', 'orif_ext']:
            name = 'caliper'
        else:
            name = 'organ'
        x = self.quality_heads[name]({'image':x})['logit'][:,index]
        return x

    def forward(self, x):
        image = x['image']
        full_image = torch.sum(image[:,:14,...], dim=1, keepdim=True)
        assign_mtx = x['assign_mtx']
        binary_concept = self.resnet({'image': full_image})['logit']
        concept = torch.zeros((image.size(0), 27)).to(image.device)

        organ_quality = self.quality_heads['organ']({'image': full_image})['logit']
        caliper_quality = self.quality_heads['caliper']({'image': full_image})['logit']

        concept[:,0] = caliper_quality[:,0]
        concept[:,1] = caliper_quality[:,1]
        concept[:,2] = binary_concept[:,0]
        concept[:,3] = binary_concept[:,1]
        concept[:,4] = binary_concept[:,2]
        concept[:,5] = organ_quality[:,0]
        concept[:,6] = organ_quality[:,1]
        concept[:,7] = organ_quality[:,2]
        concept[:,8] = binary_concept[:,3]
        concept[:,9] = caliper_quality[:,2]
        concept[:,10] = caliper_quality[:,3]
        concept[:,11] = caliper_quality[:,4]
        concept[:,12] = caliper_quality[:,5]
        concept[:,13] = binary_concept[:,4]
        concept[:,14] = organ_quality[:,3]
        concept[:,15] = organ_quality[:,4]
        concept[:,16] = organ_quality[:,5]
        concept[:,17] = binary_concept[:,5]
        concept[:,18] = caliper_quality[:,6]
        concept[:,19] = caliper_quality[:,7]
        concept[:,20] = caliper_quality[:,8]
        concept[:,21] = caliper_quality[:,9]
        concept[:,22] = binary_concept[:,6]
        concept[:,23] = organ_quality[:,6]
        concept[:,24] = binary_concept[:,7]
        concept[:,25] = caliper_quality[:,10]
        concept[:,26] = caliper_quality[:,11]

        return {'logit':concept}    

class QualityNetv3(nn.Module):
    def __init__(self):
        super().__init__()
        # a resnet for classification
        model = ResNet(in_channels=22, out_channels=len(binary_index), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.resnet = model

        self.quality_heads = nn.ModuleDict()
        model = ResNet(in_channels=2, out_channels=7, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.quality_heads['organ'] = model
        model = ResNet(in_channels=2, out_channels=12, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.quality_heads['caliper'] = model

    def forward_once(self, x, name, index, seg=None):
        # seg = (seg>0.1).float()
        if seg is not None:
            seg = seg.unsqueeze(1)
            # dilation = 0
            # mask = torch.zeros_like(seg)
            # for i in range(mask.size(0)):
            #     if torch.sum(seg[i,...]):
            #         points = torch.nonzero(seg[i,...])
            #         min_x, min_y, max_x, max_y = torch.min(points[:,0]), torch.min(points[:,1]), torch.max(points[:,0]), torch.max(points[:,1])
            #         min_x, min_y = max([0, min_x-dilation]), max([0, min_y - dilation])
            #         max_x, max_y = min([seg.size(1), max_x + dilation]), min([seg.size(2), max_y + dilation])
            #         mask[i, min_x:max_x, min_y:max_y] = 1
            # seg = mask
            x = torch.cat((x, seg), dim=1)
        if name in ['bpd_near', 'bpd_far', 'ofd_occ', 'ofd_fro', 'femur_left', 'femur_right', 'ada1', 'ada2', 'adb1', 'adb2', 'orif_inner', 'orif_ext']:
            name = 'caliper'
        else:
            name = 'organ'
        x = self.quality_heads[name]({'image':x})['logit'][:,index]
        return x

    def forward(self, x):
        image = x['image']
        full_image = torch.sum(image[:,:14,...], dim=1, keepdim=True)
        assign_mtx = x['assign_mtx']
        # assign_mtx = (assign_mtx > 0.1).float()
        binary_concept = self.resnet({'image': torch.cat((full_image, assign_mtx), dim=1)})['logit']
        # binary_concept = self.resnet(image)
        concept = torch.zeros((image.size(0), 27)).to(image.device)
        concept[:,0] = self.forward_once(full_image, quality_names[0], 0, assign_mtx[:,14,...])
        concept[:,1] = self.forward_once(full_image, quality_names[1], 1, assign_mtx[:,15,...])
        concept[:,2] = binary_concept[:,0]
        concept[:,3] = binary_concept[:,1]
        concept[:,4] = binary_concept[:,2]
        concept[:,5] = self.forward_once(full_image, quality_names[2], 0,  assign_mtx[:,6,...])
        concept[:,6] = self.forward_once(full_image, quality_names[3], 1,  assign_mtx[:,8,...])
        concept[:,7] = self.forward_once(full_image, quality_names[4], 2,  assign_mtx[:,9,...])
        concept[:,8] = binary_concept[:,3]
        concept[:,9] = self.forward_once(full_image, quality_names[5], 2, assign_mtx[:,16,...])
        concept[:,10] = self.forward_once(full_image, quality_names[6], 3, assign_mtx[:,16,...])
        concept[:,11] = self.forward_once(full_image, quality_names[7], 4, assign_mtx[:,16,...])
        concept[:,12] = self.forward_once(full_image, quality_names[8], 5, assign_mtx[:,16,...])
        concept[:,13] = binary_concept[:,4]
        concept[:,14] = self.forward_once(full_image, quality_names[9], 3, assign_mtx[:,10,...])
        concept[:,15] = self.forward_once(full_image, quality_names[10], 4, assign_mtx[:,12,...])
        concept[:,16] = self.forward_once(full_image, quality_names[11], 5, assign_mtx[:,11,...])
        concept[:,17] = binary_concept[:,5]
        concept[:,18] = self.forward_once(full_image, quality_names[12], 6, assign_mtx[:,18,...])
        concept[:,19] = self.forward_once(full_image, quality_names[13], 7, assign_mtx[:,17,...])
        concept[:,20] = self.forward_once(full_image, quality_names[14], 8, assign_mtx[:,20,...])
        concept[:,21] = self.forward_once(full_image, quality_names[15], 9, assign_mtx[:,19,...])
        concept[:,22] = binary_concept[:,6]
        concept[:,23] = self.forward_once(full_image, quality_names[16], 6, assign_mtx[:,4,...])
        concept[:,24] = binary_concept[:,7]
        concept[:,25] = self.forward_once(full_image, quality_names[17], 10, assign_mtx[:,3,...] + assign_mtx[:,1,...])
        concept[:,26] = self.forward_once(full_image, quality_names[18], 11, assign_mtx[:,2,...] + assign_mtx[:,1,...])

        return {'logit':concept}    

class QualityNetv4(nn.Module):
    def __init__(self):
        super().__init__()
        # a resnet for classification
        model = ResNet(in_channels=21, out_channels=len(binary_index), depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.resnet = model

        self.quality_heads = nn.ModuleDict()
        model = ResNet(in_channels=1, out_channels=7, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.quality_heads['organ'] = model
        model = ResNet(in_channels=1, out_channels=12, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        self.quality_heads['caliper'] = model

    def forward_once(self, x, name, index, seg=None):
        # x = x.unsqueeze(1) # B, 1, H, W,
        # seg = (seg > 0.1).float()
        if seg is not None:
            seg = seg.unsqueeze(1).float()
            x = seg
        if name in ['bpd_near', 'bpd_far', 'ofd_occ', 'ofd_fro', 'femur_left', 'femur_right', 'ada1', 'ada2', 'adb1', 'adb2', 'orif_inner', 'orif_ext']:
            name = 'caliper'
        else:
            name = 'organ'
        x = self.quality_heads[name]({'image':x})['logit'][:,index]
        # x = x.squeeze(-1)
        return x

    def forward(self, x):
        image = x['image']
        full_image = torch.sum(image[:,:14,...], dim=1, keepdim=True)
        assign_mtx = x['assign_mtx']
        assign_mtx = (assign_mtx > 0.1).float()
        binary_concept = self.resnet({'image': assign_mtx.float()})['logit']

        concept = torch.zeros((image.size(0), 27)).to(image.device)
        concept[:,0] = self.forward_once(full_image, quality_names[0], 0, assign_mtx[:,14,...])
        concept[:,1] = self.forward_once(full_image, quality_names[1], 1, assign_mtx[:,15,...])
        concept[:,2] = binary_concept[:,0]
        concept[:,3] = binary_concept[:,1]
        concept[:,4] = binary_concept[:,2]
        concept[:,5] = self.forward_once(full_image, quality_names[2], 0,  assign_mtx[:,6,...])
        concept[:,6] = self.forward_once(full_image, quality_names[3], 1,  assign_mtx[:,8,...])
        concept[:,7] = self.forward_once(full_image, quality_names[4], 2,  assign_mtx[:,9,...])
        concept[:,8] = binary_concept[:,3]
        concept[:,9] = self.forward_once(full_image, quality_names[5], 2, assign_mtx[:,16,...])
        concept[:,10] = self.forward_once(full_image, quality_names[6], 3, assign_mtx[:,16,...])
        concept[:,11] = self.forward_once(full_image, quality_names[7], 4, assign_mtx[:,16,...])
        concept[:,12] = self.forward_once(full_image, quality_names[8], 5, assign_mtx[:,16,...])
        concept[:,13] = binary_concept[:,4]
        concept[:,14] = self.forward_once(full_image, quality_names[9], 3, assign_mtx[:,10,...])
        concept[:,15] = self.forward_once(full_image, quality_names[10], 4, assign_mtx[:,12,...])
        concept[:,16] = self.forward_once(full_image, quality_names[11], 5, assign_mtx[:,11,...])
        concept[:,17] = binary_concept[:,5]
        concept[:,18] = self.forward_once(full_image, quality_names[12], 6, assign_mtx[:,18,...])
        concept[:,19] = self.forward_once(full_image, quality_names[13], 7, assign_mtx[:,17,...])
        concept[:,20] = self.forward_once(full_image, quality_names[14], 8, assign_mtx[:,20,...])
        concept[:,21] = self.forward_once(full_image, quality_names[15], 9, assign_mtx[:,19,...])
        concept[:,22] = binary_concept[:,6]
        concept[:,23] = self.forward_once(full_image, quality_names[16], 6, assign_mtx[:,4,...])
        concept[:,24] = binary_concept[:,7]
        concept[:,25] = self.forward_once(full_image, quality_names[17], 10, assign_mtx[:,3,...] + assign_mtx[:,1,...])
        concept[:,26] = self.forward_once(full_image, quality_names[18], 11, assign_mtx[:,2,...] + assign_mtx[:,1,...])

        return {'logit':concept}    

def create_observer(in_channels, out_channels, model_name, state_dict_dir=None, **kwargs):
    # create segmentation model
    if model_name == 'DTUNet':
        model = DTUNet(in_channels, out_channels, **kwargs)
    elif model_name == 'UNet':
        model = UNet(in_channels, out_channels, **kwargs)
    elif model_name == 'Identity':
        model = IdendityMapping()
    else:
        print(f'Given model name {model_name} is not supported')
        print(f'Please pick a model from [DTUNet, UNet, Identity]')
        raise KeyError
    print(f'model {model_name} is created as the observer')
    
    # load statedict
    if state_dict_dir is not None and model_name != 'Identity':
        try:
            model.load_state_dict(torch.load(state_dict_dir))
            print(f'load observer weight from {state_dict_dir}')
        except RuntimeError:
            print(f'observer weight dir {state_dict_dir} is not valid, hence not loaded')

    return model

def create_conceiver(in_channels, out_channels, model_name, state_dict_dir=None, imagenet_init=True, **kwargs):
    # create conceiver model
    if model_name == 'ResNet18':
        if imagenet_init:
            model = ResNet(in_channels=in_channels, out_channels=out_channels, depth=18, 
                            weights="ResNet18_Weights.IMAGENET1K_V1", **kwargs)
        else:
            model = ResNet(in_channels=in_channels, out_channels=out_channels, depth=18, 
                            weights=None, **kwargs)
    elif model_name == 'ResNet50':
        if imagenet_init:
            model = ResNet(in_channels=in_channels, out_channels=out_channels, depth=50, 
                            weights="ResNet50_Weights.IMAGENET1K_V1", **kwargs)
        else:
            model = ResNet(in_channels=in_channels, out_channels=out_channels, depth=50, 
                            weights=None, **kwargs)
    elif model_name == 'Inception':
        if imagenet_init:
            model = Inception(in_channels=in_channels, out_channels=out_channels, 
                            weights='Inception_V3_Weights.IMAGENET1K_V1', **kwargs)
        else:
            model = Inception(in_channels=in_channels, out_channels=out_channels, 
                            weights=None, **kwargs)
    elif model_name == 'QualityNetv1':
        model = QualityNetv1()
    elif model_name == 'QualityNetv2':
        model = QualityNetv2()
    elif model_name == 'QualityNetv3':
        model = QualityNetv3()
    elif model_name == 'QualityNetv4':
        model = QualityNetv4()
    else:
        print(f'Given model name {model_name} is not supported')
        print(f'Please pick a model from [ResNet18, ResNet50, Inception]')
        raise KeyError
    
    print(f'model {model_name} is created as the conceiver')
    
    if state_dict_dir is not None:
        try:
            model.load_state_dict(torch.load(state_dict_dir))
            print(f'load conceiver weight from {state_dict_dir}')
        except RuntimeError:
            print(f'conceiver weight dir {state_dict_dir} is not valid')
    
    return model

class FetalPlaneClassifier:
    '''
    A classifier recognize fetal planes according to the segmentation prediction
    '''
    def __init__(self, **kwargs):
        self.include_cervix = kwargs['include_cervix'] if 'include_cervix' in kwargs.keys() else True
        self.include_other = kwargs['include_other'] if 'include_other' in kwargs.keys() else True
        self.auto_measure = kwargs['auto_measure'] if 'auto_measure' in kwargs.keys() else False

    def __call__(self, x):
        # make a rough prediction on the plane type
        seg_mask = x['seg_mask']
        assign_mtx = x['assign_mtx']
        batch_size = seg_mask.size(0)
        prediction = torch.ones(batch_size)*4
        if not self.include_cervix:
            # not including any cervix information
            x['concept_logit'][:, 22:27] = 0

        for b in range(batch_size):
            batch_mask = seg_mask[b, ...].detach()
            batch_image = x['image'][b, 0, ...].detach()
            # iterate over the batch
            # Head and Abdomen are the most important ones
            if (torch.sum(batch_mask==7)>=100) and ((6 in batch_mask) and (8 in batch_mask)): # or (9 in batch_mask), all the organs MUST be presented
                # if we have skin boundary and either stomach bubble or umv appears
                prediction[b] = 1 # 1 is abdomen sp

                if self.auto_measure:
                    occ = get_abdomen_concept(batch_image, batch_mask)
                    # if 'concept' in x.keys():
                    #     if int(occ >= 0.5)==x['concept'][b, 8]:
                    #         x['concept_logit'][b, 8] = occ
                    #         x['concept_pred'][b, 8] = int(occ >= 0.5)
                    # else:
                    #         x['concept_logit'][b, 8] = occ
                    #         x['concept_pred'][b, 8] = int(occ >= 0.5)                    
                    x['concept_logit'][b, 8] = occ
                    x['concept_pred'][b, 8] = int(occ >= 0.5)  

                x['concept_logit'][b, :4] = 0
                x['concept_pred'][b, :4] = 0
                x['concept_logit'][b, 13:] = 0
                x['concept_pred'][b, 13:] = 0
            elif ((torch.sum(batch_mask==7)>=100) and ((6 in batch_mask) or (8 in batch_mask))) or ((torch.sum(batch_mask==7)>=100) and (((6 in batch_mask) or (8 in batch_mask)) and (9 in batch_mask))):
                prediction[b] = 6 # 6 is abdomen nsp


                if self.auto_measure:
                    occ = get_abdomen_concept(batch_image, batch_mask)
                    # if 'concept' in x.keys():
                    #     if int(occ >= 0.5)==x['concept'][b, 8]:
                    #         x['concept_logit'][b, 8] = occ
                    #         x['concept_pred'][b, 8] = int(occ >= 0.5)
                    # else:
                    #         x['concept_logit'][b, 8] = occ
                    #         x['concept_pred'][b, 8] = int(occ >= 0.5)   
                    x['concept_logit'][b, 8] = occ
                    x['concept_pred'][b, 8] = int(occ >= 0.5)  

                x['concept_logit'][b, :4] = 0
                x['concept_pred'][b, :4] = 0
                x['concept_logit'][b, 13:] = 0
                x['concept_pred'][b, 13:] = 0                

            elif (torch.sum(batch_mask==13)>=100) and ((10 in batch_mask) and (12 in batch_mask)): #  or (11 in batch_mask)
                # if we have bone boundary and either thalamus or csp
                prediction[b] = 2 # 2 is head

                if self.auto_measure:
                    occ = get_head_concept(batch_image, batch_mask)
                    # if 'concept' in x.keys():
                    #     if int(occ >= 0.5)==x['concept'][b, 17]:
                    #         x['concept_logit'][b, 17] = occ
                    #         x['concept_pred'][b, 17] = int(occ >= 0.5)
                    # else:
                    #         x['concept_logit'][b, 17] = occ
                    #         x['concept_pred'][b, 17] = int(occ >= 0.5)   

                    x['concept_logit'][b, 17] = occ
                    x['concept_pred'][b, 17] = int(occ >= 0.5)   

                x['concept_logit'][b, :13] = 0
                x['concept_pred'][b, :13] = 0
                x['concept_logit'][b, 22:] = 0
                x['concept_pred'][b, 22:] = 0

            elif ((torch.sum(batch_mask==13)>=100) and ((10 in batch_mask) or (12 in batch_mask))) or ((torch.sum(batch_mask==13)>=100) and (((10 in batch_mask) or (12 in batch_mask)) and (11 in batch_mask))):
                prediction[b] = 7 # 7 is head nsp

                if self.auto_measure:
                    occ = get_head_concept(batch_image, batch_mask)
                    # if 'concept' in x.keys():
                    #     if int(occ >= 0.5)==x['concept'][b, 17]:
                    #         x['concept_logit'][b, 17] = occ
                    #         x['concept_pred'][b, 17] = int(occ >= 0.5)
                    # else:
                    #         x['concept_logit'][b, 17] = occ
                    #         x['concept_pred'][b, 17] = int(occ >= 0.5)   

                    x['concept_logit'][b, 17] = occ
                    x['concept_pred'][b, 17] = int(occ >= 0.5)   

                x['concept_logit'][b, :13] = 0
                x['concept_pred'][b, :13] = 0
                x['concept_logit'][b, 22:] = 0
                x['concept_pred'][b, 22:] = 0

            elif torch.sum(batch_mask==5)>=100:
                # if there is femur bone
                prediction[b] = 0 # 0 is femur

                if self.auto_measure:
                    angle, occ = get_femur_concept(batch_image, batch_mask)
                    x['concept_logit'][b, 2] = angle
                    x['concept_pred'][b, 2] = not int(45/180 < angle < 135/180)
                    # if 'concept' in x.keys():
                    #     if int(occ >= 0.5)==x['concept'][b, 3]:
                    #         x['concept_logit'][b, 3] = occ
                    #         x['concept_pred'][b, 3] = int(occ >= 0.5)
                    # else:
                    #         x['concept_logit'][b, 3] = occ
                    #         x['concept_pred'][b, 3] = int(occ >= 0.5)   

                    # x['concept_logit'][b, 3] = occ
                    # x['concept_pred'][b, 3] = int(occ >= 0.5)   

                x['concept_logit'][b, 4:] = 0
                x['concept_pred'][b, 4:] = 0

            elif self.include_cervix and ((1 in batch_mask) or (2 in batch_mask) or (3 in batch_mask)):
                prediction[b] = 3 # 3 is cervix
            elif self.include_other:
                prediction[b] = 4
            else:
                if 7 in batch_mask:
                    prediction[b] = 1
                elif 13 in batch_mask:
                    prediction[b] = 2
                else:
                    raise ValueError

            if (torch.sum(batch_mask==7)<100):
                    x['concept_logit'][b, [4,8,9,10,11,12]] *= 1e-5
                    x['concept_pred'][b, [4,8,9,10,11,12]] *= 1e-5

            if torch.sum(batch_mask==5)<100:
                x['concept_logit'][b, [0,1,2,3]] *= 1e-5
                x['concept_pred'][b, [0,1,2,3]] *= 1e-5
            
            if torch.sum(batch_mask==13)<100:
                x['concept_logit'][b, [13,17,18.19,20,21]] *= 1e-5
                x['concept_pred'][b, [13,17,18.19,20,21]] *= 1e-5


        prediction = prediction.to(seg_mask.device)
        x['plane_pred_seg'] = prediction

        return x

class SegProbe(nn.Module):
    '''
    Progressive Concept Bottleneck Models

    Image -> Observer -> Conceiver -> Predictor
    x -> s -> c -> y
    '''
    def __init__(self,  
                        out_channels, # number of categories to recognize
                        seg_concept_num, # number of segmentation concepts 
                        prop_concept_num, # number of property concepts
                        observer_cfgs={}, # configs for creating the Observer (x -> s)
                        include_bg = True, # whether include the background as visual clues 
                        conceiver_cfgs={}, # configs for creating the Conceiver (s -> c)
                        concept_act = nn.Identity, # activation function of concepts
                        expand_dim=1024, # latent dimension of the Predictor (c -> y)
                        extra_concept_num=8, # number of new concepts generated by concept interaction
                        bin_concept_ind=[], # the indices of binary concepts, which would be used for concept interaction 
                        seg_intervention=False, # whether activate segmentation-level intervention
                        prop_intervention=False, # whether activate property-level intervention
                        rule_cfgs = {}, # rule-based operation configs, if any
                        ):
        super().__init__()

        if not include_bg:
            seg_concept_num -= 1
        self.include_bg = include_bg

        if 'allow_texture' in observer_cfgs.keys():
            self.allow_texture = observer_cfgs['allow_texture']
            del observer_cfgs['allow_texture']
        else:
            self.allow_texture = False

        # create the Conceiver (s -> c)
        if self.allow_texture:
            self.conceiver = create_conceiver(in_channels=seg_concept_num, out_channels=prop_concept_num, 
                                    **conceiver_cfgs)
        else:
            self.conceiver = create_conceiver(in_channels=seg_concept_num, out_channels=prop_concept_num, 
                                    **conceiver_cfgs)
        self.concept_act = concept_act
        
        if extra_concept_num:
            # if not 0, activate concept interaction
            bin_concept_num = len(bin_concept_ind)
            assert bin_concept_num > 0,'When concept interaction is activated, there should be binary concepts'
            self.bin_concept_ind = bin_concept_ind
            self.grouping = nn.Sequential(
                DenseLayer(bin_concept_num, bin_concept_num*2),
                DenseLayer(bin_concept_num*2, bin_concept_num*2),
                DenseLayer(bin_concept_num*2, extra_concept_num*bin_concept_num)
            )
        self.extra_concept_num = extra_concept_num

        if expand_dim:
            # if not 0, then a 2 layer mlp, otherwise a linear layer
            self.projector = DenseLayer((prop_concept_num+extra_concept_num), expand_dim, bn=False)
        self.expand_dim = expand_dim
        
        self.fc = DenseLayer(expand_dim, out_channels, activation=None, bn=False) if expand_dim else DenseLayer(prop_concept_num+extra_concept_num, out_channels, activation=None, bn=False)
        # self.fc = DenseLayer(expand_dim*2, out_channels, activation=None, bn=False) if expand_dim else DenseLayer(prop_concept_num+extra_concept_num, out_channels, activation=None, bn=False)

        self.seg_interv = seg_intervention
        self.prop_interv = prop_intervention
        self.seg_concept_num = seg_concept_num
        self.prop_concept_num = prop_concept_num

        if 'FetalPlaneCls' in rule_cfgs.keys():
            self.cls_rule = FetalPlaneClassifier(**rule_cfgs['FetalPlaneCls'])
        else:
            self.cls_rule = None

        self.seg_rule = fetal_caliper_concept
        # self.seg_rule = lambda x: x

    def forward(self, x, mask_cervix=False):
        # Run the Observer
        assign_mtx = x['mask'].float()
        seg_mask = torch.argmax(assign_mtx, dim=1)
        if not self.include_bg:
            assign_mtx = assign_mtx[:,1:,...]
            if len(assign_mtx.shape)==3:
                assign_mtx = assign_mtx.unsqueeze(1)

        x['seg_mask'] = seg_mask
        x['assign_mtx'] = assign_mtx
        x = self.seg_rule(x)
        assign_mtx = x['assign_mtx']

        # Run the Conceiver
        # assign the image into the segmentation mask
        if self.allow_texture:
            # allow the texture information flow from the image to the segmentation concepts
            seg_concepts = x['image'].unsqueeze(1)*assign_mtx.unsqueeze(2)
            seg_concepts = seg_concepts.flatten(1, 2)
        else:
            seg_concepts = assign_mtx
        
        if (not self.training) and self.prop_interv: 
            # with property-level human intervention during test time
            concept_pred = x['concept'].clone()
        # elif self.training:
        #     concept_pred = x['concept'].clone()
        #     concept_pred[:,quality_index] *= 10
        #     concept_logit = x['concept'].clone()
        #     concept_logit[:,23] = 0
        #     x['concept_logit'] = concept_logit
        else:
            with torch.no_grad():
                self.conceiver.eval()
                concept_logit = self.conceiver({'image': seg_concepts, 'assign_mtx':assign_mtx, 'seg_mask':seg_mask})['logit']
            x['concept_logit'] = concept_logit
            x = self.concept_act(x)
            concept_logit, concept_pred = x['concept_logit'], x['concept_pred']

        x['concept_pred'] = concept_pred

        if self.cls_rule is not None:
            x = self.cls_rule(x)
        # plane_pred = x['plane_pred_seg']
        concept = x['concept_logit']
        
        # Predictor (c -> y)
        if self.extra_concept_num:
            raw_concept = concept.clone()
            concept = concept[:,self.bin_concept_ind]
            square_concept = concept**2

            groups = self.grouping(concept)
            groups = nn.functional.relu(groups.reshape(concept.size(0), self.extra_concept_num, -1))

            concept = concept.unsqueeze(-1)
            groups = groups.expand(concept.size(0), self.extra_concept_num, square_concept.shape[-1])
            concept = (groups@concept).flatten(-2) # B, H, G, 1

            # group squared concepts
            square_concept = square_concept.unsqueeze(-1)
            square_concept = (groups**2@square_concept).flatten(-2)
            concept = (concept**2 - square_concept) / 2 # (x1+x2)**2 - x1**2 - x2**2
            # (a+b)
            norminal = (groups.sum(-1))**2
            # (a2+b2)
            square_norm = (groups**2).sum(-1)
            norm = (norminal - square_norm)/2

            norm = torch.clamp(norm, min=1e-9)
            concept = concept / norm
            concept = torch.relu(concept)
            concept = torch.sqrt(concept)
            concept = torch.cat((raw_concept, concept), dim=-1)
            x['full_concept'] = concept.detach()
        else:
            x['full_concept'] = concept_pred.clone()

        if self.expand_dim:
            emb = self.projector(concept)
        else:
            emb = concept

        x['emb'] = emb.detach()
        logit = self.fc(emb)
        
        x['logit'] = logit
        return x      

class SBM(nn.Module):
    '''
    Semantic Bottleneck Models

    Image -> Observer -> Predictor
    x -> s -> y
    '''
    def __init__(self,  
                        in_channels, # input channel of the image
                        out_channels, # number of categories to recognize
                        seg_concept_num, # number of segmentation concepts 
                        observer_cfgs={}, # configs for creating the Observer (x -> s)
                        include_bg = True, # whether include the background as visual clues 
                        seg_intervention=False, # whether activate segmentation-level intervention
                        ):
        super().__init__()

        if not include_bg:
            seg_concept_num -= 1
        self.include_bg = include_bg

        if 'allow_texture' in observer_cfgs.keys():
            self.allow_texture = observer_cfgs['allow_texture']
            del observer_cfgs['allow_texture']
        else:
            self.allow_texture = False

        # create the Observer (x -> s)
        self.observer = create_observer(in_channels=in_channels, out_channels=seg_concept_num, 
                                    **observer_cfgs)
        # create the Predictor (s -> y)
        if self.allow_texture:
            self.predictor = ResNet(in_channels=in_channels*seg_concept_num, out_channels=out_channels, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        else:
            self.predictor = ResNet(in_channels=in_channels*seg_concept_num, out_channels=out_channels, depth=18, weight="ResNet18_Weights.IMAGENET1K_V1")

        self.seg_interv = seg_intervention

    def forward(self, x, mask_cervix=False):
        # Run the Observer
        if (not self.training) and self.seg_interv: 
            # with segmentation-level human intervention during test time
            seg_mask = x['mask'].float()
            assign_mtx = nn.functional.one_hot(seg_mask, num_classes=self.seg_concept_num).permute(0,3,1,2)
        else:
            self.observer.eval()
            with torch.no_grad():
                assign_mtx = self.observer({'image':x['raw_image']})['logit']
                seg_mask = torch.argmax(assign_mtx, dim=1)
        if not self.include_bg:
            assign_mtx = assign_mtx[:,1:,...]
            if len(assign_mtx.shape)==3:
                assign_mtx = assign_mtx.unsqueeze(1)
        
        x['seg_mask'] = seg_mask
        x['assign_mtx'] = assign_mtx

        # Run the Conceiver
        # assign the image into the segmentation mask
        if self.allow_texture:
            # allow the texture information flow from the image to the segmentation concepts
            seg_concepts = x['image'].unsqueeze(1)*assign_mtx.unsqueeze(2)
            seg_concepts = seg_concepts.flatten(1, 2)
        else:
            seg_concepts = assign_mtx
        
        logit = self.predictor({'image':seg_concepts})['logit']
        
        x['logit'] = logit

        return x         

class LinearProbe(nn.Module):
    '''
    Linear Probe

    Conceiver -> Predictor
    c -> y
    '''
    def __init__(self,  
                        out_channels,
                        prop_concept_num, # number of property concepts
                        expand_dim=256, # latent dimension of the Predictor (c -> y)
                        extra_concept_num=16, # number of new concepts generated by concept interaction
                        bin_concept_ind=[], # the indices of binary concepts, which would be used for concept interaction 
                        ):
        super().__init__()
        if extra_concept_num:
            # if not 0, activate concept interaction
            bin_concept_num = len(bin_concept_ind)
            assert bin_concept_num > 0,'When concept interaction is activated, there should be binary concepts'
            self.bin_concept_ind = bin_concept_ind
            self.grouping = nn.Sequential(
                DenseLayer(bin_concept_num, bin_concept_num*2),
                DenseLayer(bin_concept_num*2, bin_concept_num*2),
                DenseLayer(bin_concept_num*2, extra_concept_num*bin_concept_num)
            )
        self.extra_concept_num = extra_concept_num

        if expand_dim:
            # if not 0, then a 2 layer mlp, otherwise a linear layer
            self.projector = DenseLayer((prop_concept_num+extra_concept_num), expand_dim, bn=False)
        self.expand_dim = expand_dim
        
        self.fc = DenseLayer(expand_dim, out_channels, activation=None, bn=False) if expand_dim else DenseLayer(prop_concept_num+extra_concept_num, out_channels, activation=None, bn=False)

        self.prop_concept_num = prop_concept_num

    def forward(self, x):
        concept = x['concept_pred']
        
        # Predictor (c -> y)
        if self.extra_concept_num:
            raw_concept = concept.clone()
            concept = concept[:,self.bin_concept_ind]
            square_concept = concept**2

            groups = self.grouping(concept)
            groups = nn.functional.relu(groups.reshape(concept.size(0), self.extra_concept_num, -1))

            concept = concept.unsqueeze(-1)
            groups = groups.expand(concept.size(0), self.extra_concept_num, square_concept.shape[-1])
            concept = (groups@concept).flatten(-2) # B, H, G, 1

            # group squared concepts
            square_concept = square_concept.unsqueeze(-1)
            square_concept = (groups**2@square_concept).flatten(-2)
            concept = (concept**2 - square_concept) / 2 # (x1+x2)**2 - x1**2 - x2**2
            # (a+b)
            norminal = (groups.sum(-1))**2
            # (a2+b2)
            square_norm = (groups**2).sum(-1)
            norm = (norminal - square_norm)/2

            norm = torch.clamp(norm, min=1e-9)
            concept = concept / norm
            concept = torch.relu(concept)
            concept = torch.sqrt(concept)
            concept = torch.cat((raw_concept, concept), dim=-1)

        if self.expand_dim:
            emb = self.projector(concept)
        else:
            emb = concept

        x['emb'] = emb.detach()
        logit = self.fc(emb)
        
        x['logit'] = logit
        return x      

class PCBM(nn.Module):
    '''
    Progressive Concept Bottleneck Models

    Image -> Observer -> Conceiver -> Predictor
    x -> s -> c -> y
    '''
    def __init__(self,  
                        in_channels, # input channel of the image
                        out_channels, # number of categories to recognize
                        seg_concept_num, # number of segmentation concepts 
                        prop_concept_num, # number of property concepts
                        observer_cfgs={}, # configs for creating the Observer (x -> s)
                        include_bg = True, # whether include the background as visual clues 
                        conceiver_cfgs={}, # configs for creating the Conceiver (s -> c)
                        concept_act = nn.Identity, # activation function of concepts
                        expand_dim=1024, # latent dimension of the Predictor (c -> y)
                        extra_concept_num=8, # number of new concepts generated by concept interaction
                        bin_concept_ind=[], # the indices of binary concepts, which would be used for concept interaction 
                        seg_intervention=False, # whether activate segmentation-level intervention
                        prop_intervention=False, # whether activate property-level intervention
                        rule_cfgs = {}, # rule-based operation configs, if any
                        ):
        super().__init__()

        if not include_bg:
            seg_concept_num -= 1
        self.include_bg = include_bg

        if 'allow_texture' in observer_cfgs.keys():
            self.allow_texture = observer_cfgs['allow_texture']
            del observer_cfgs['allow_texture']
        else:
            self.allow_texture = False

        # create the Observer (x -> s)
        self.observer = create_observer(in_channels=in_channels, out_channels=seg_concept_num, 
                                    **observer_cfgs)
        # create the Conceiver (s -> c)
        if self.allow_texture:
            self.conceiver = create_conceiver(in_channels=in_channels*seg_concept_num, out_channels=prop_concept_num, 
                                    **conceiver_cfgs)
        else:
            self.conceiver = create_conceiver(in_channels=seg_concept_num, out_channels=prop_concept_num, 
                                    **conceiver_cfgs)
        self.concept_act = concept_act
        
        if extra_concept_num:
            # if not 0, activate concept interaction
            bin_concept_num = len(bin_concept_ind)
            assert bin_concept_num > 0,'When concept interaction is activated, there should be binary concepts'
            self.bin_concept_ind = bin_concept_ind
            self.grouping = nn.Sequential(
                DenseLayer(bin_concept_num, bin_concept_num*2),
                DenseLayer(bin_concept_num*2, bin_concept_num*2),
                DenseLayer(bin_concept_num*2, extra_concept_num*bin_concept_num)
            )
        self.extra_concept_num = extra_concept_num

        if expand_dim:
            # if not 0, then a 2 layer mlp, otherwise a linear layer
            self.projector = DenseLayer((prop_concept_num+extra_concept_num), expand_dim, bn=False)
        self.expand_dim = expand_dim
        
        self.fc = DenseLayer(expand_dim, out_channels, activation=None, bn=False) if expand_dim else DenseLayer(prop_concept_num+extra_concept_num, out_channels, activation=None, bn=False)

        self.seg_interv = seg_intervention
        self.prop_interv = prop_intervention
        self.seg_concept_num = seg_concept_num
        self.prop_concept_num = prop_concept_num

        if 'FetalPlaneCls' in rule_cfgs.keys():
            self.cls_rule = FetalPlaneClassifier(**rule_cfgs['FetalPlaneCls'])
        else:
            self.cls_rule = None

        self.seg_rule = fetal_caliper_concept
        # self.seg_rule = lambda x: x

    def forward(self, x, mask_cervix=False):
        # Run the Observer
        if (not self.training) and self.seg_interv: 
            # with segmentation-level human intervention during test time
            seg_mask = x['mask'].float()
            assign_mtx = nn.functional.one_hot(seg_mask.long(), num_classes=self.seg_concept_num).permute(0,3,1,2)
        else:
            self.observer.eval()
            with torch.no_grad():
                seg_outs = self.observer({'image':x['raw_image']})
                assign_mtx = seg_outs['logit']
                seg_mask = torch.argmax(assign_mtx, dim=1)
                # assign_mtx = nn.functional.one_hot(seg_mask.long(), num_classes=self.seg_concept_num).permute(0,3,1,2)
        if not self.include_bg:
            assign_mtx = assign_mtx[:,1:,...]
            if len(assign_mtx.shape)==3:
                assign_mtx = assign_mtx.unsqueeze(1)

        x['seg_mask'] = seg_mask
        x['assign_mtx'] = assign_mtx
        x = self.seg_rule(x)
        assign_mtx = x['assign_mtx']

        # Run the Conceiver
        # assign the image into the segmentation mask
        if self.allow_texture:
            # allow the texture information flow from the image to the segmentation concepts
            seg_concepts = x['image'].unsqueeze(1)*assign_mtx.unsqueeze(2)
            seg_concepts = seg_concepts.flatten(1, 2)
        else:
            seg_concepts = assign_mtx
        
        if (not self.training) and self.prop_interv: 
            # with property-level human intervention during test time
            concept_pred = x['concept'].clone()
            concept_pred[:,quality_index] *= 10
            concept_logit = x['concept'].clone()
            concept_logit[:,23] = 0
            x['concept_logit'] = concept_logit
        elif self.training: # independent training
            concept_pred = x['concept'].clone()
            concept_pred[:,quality_index] *= 10
            concept_logit = x['concept'].clone()
            concept_logit[:,23] = 0
            x['concept_logit'] = concept_logit
        else:
            with torch.no_grad():
                self.conceiver.eval()
                concept_logit = self.conceiver({'image': seg_concepts, 'assign_mtx':assign_mtx, 'seg_mask':seg_mask})['logit']
            x['concept_logit'] = concept_logit
            x = self.concept_act(x)
            concept_logit, concept_pred = x['concept_logit'], x['concept_pred']

        x['concept_pred'] = concept_pred

        if self.cls_rule is not None:
            x = self.cls_rule(x)
        plane_pred = x['plane_pred_seg']
        concept = x['concept_logit']
        
        # Predictor (c -> y)
        if self.extra_concept_num:
            raw_concept = concept.clone()

            concept = concept[:,self.bin_concept_ind]
            square_concept = concept**2

            groups = self.grouping(concept)
            groups = nn.functional.relu(groups.reshape(concept.size(0), self.extra_concept_num, -1))

            concept = concept.unsqueeze(-1)
            groups = groups.expand(concept.size(0), self.extra_concept_num, square_concept.shape[-1])
            concept = (groups@concept).flatten(-2) # B, H, G, 1

            # group squared concepts
            square_concept = square_concept.unsqueeze(-1)
            square_concept = (groups**2@square_concept).flatten(-2)
            concept = (concept**2 - square_concept) / 2 # (x1+x2)**2 - x1**2 - x2**2
            # (a+b)
            norminal = (groups.sum(-1))**2
            # (a2+b2)
            square_norm = (groups**2).sum(-1)
            norm = (norminal - square_norm)/2

            norm = torch.clamp(norm, min=1e-9)
            concept = concept / norm
            concept = torch.relu(concept)
            concept = torch.sqrt(concept)
            concept = torch.cat((raw_concept, concept), dim=-1)
            
            x['full_concept'] = concept.detach()
        else:
            x['full_concept'] = concept_pred.clone()

        if self.expand_dim:
            emb = self.projector(concept)
        else:
            emb = concept

        x['emb'] = emb.detach()
        logit = self.fc(emb)
        
        x['logit'] = logit
        return x    