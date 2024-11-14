import torch.nn as nn
import torchvision
import torch

def resnet(in_channels, out_channels, depth=18, weights=None):
    resnet = eval(f"torchvision.models.resnet{depth}(weights=weights, progress=True)")
    # for name, param in resnet.named_parameters():
    #     if 'fc' not in name: 
    #         param.requires_grad = False

    if in_channels != 3:
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if out_channels != 1000:
        resnet.fc = nn.Linear(resnet.fc.in_features, out_channels, bias=True)
    return resnet

def inception(in_channels, out_channels, weights="Inception_V3_Weights.IMAGENET1K_V1"):
    inception = torchvision.models.inception_v3(weights = weights)
    if in_channels != 3:
        inception.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    if out_channels != 1000:
        inception.fc = nn.Linear(2048, out_channels, bias=True)
        inception.AuxLogits.fc = nn.Linear(in_features=768, out_features=out_channels, bias=True)

    # for name, param in model.named_parameters():
    #     if 'fc' not in name:  # and 'Mixed_7c' not in name:
    #         param.requires_grad = False
    return inception

def vgg(in_channels, out_channels, weights="VGG11_BN_Weights.IMAGENET1K_V1"):
    vgg = torchvision.models.vgg11_bn(weights=weights)
    if in_channels != 3:
        vgg.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if out_channels != 1000:
        vgg.classifier[6] = nn.Linear(in_features=4096, out_features=out_channels)
    return vgg

class VGG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = vgg(in_channels, out_channels)
    
    def forward(self, x):
        image = x['image']
        logit = self.net(image)
        return {'logit': logit}

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = inception(in_channels, out_channels)
    
    def forward(self, x):
        x = self.model(x)

        return x

class ResNet(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.net = resnet(**args)
    
    def forward(self, x):
        image = x['image']
        logit = self.net(image)
        return {'logit': logit}

class IdendityMapping(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return {'logit': x['image']}


class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size).to_sparse())
        self.stride = stride
        self.padding = padding
    
    def to_sparse(self, x):
        indices = x.nonzero().transpose(0,1)
        values = x[indices[0], indices[1], indices[2], indices[3]]
        x = torch.sparse_coo_tensor(indices, values, x.shape)

    def forward(self, x):
        # Convert dense input to sparse representation
        x = x.to_sparse()
        
        # Perform sparse convolution
        out = torch.nn.functional.conv2d(x, self.weights, stride=self.stride, padding=self.padding)
        
        # Convert sparse output to dense representation
        out = out.to_dense()
        
        return out

if __name__=="__main__":
    model = vgg(1,2)
    tensor = torch.rand(8, 1, 300, 300)
    a = model(tensor)
    print(a)