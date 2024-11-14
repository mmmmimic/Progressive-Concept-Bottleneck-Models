import torch
import pandas as pd
from matplotlib import pyplot as plt    
import json
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from copy import deepcopy

def getMaskProperties(mask):
    """
    Uses image moments to estimate covariance matrix and center of mass of the masks

    Parameters
    ----------
    mask : torch.tensor [...,H,W]
        DESCRIPTION.

    Returns
    -------
    covs : covariance matrices [...,2x2]
    y_cent : y center coordinates 
    x_cent : x center coordinates

    """
    
    device = 'cpu' if mask.get_device()==-1 else 'cuda'
    
    m00 = torch.sum(mask,dim=[-2,-1])
    
    y_idx, x_idx = torch.meshgrid(torch.arange(mask.shape[-2],device=device),
                                  torch.arange(mask.shape[-1],device=device))

    m10 = torch.sum(mask * y_idx,           dim=[-2,-1])
    m01 = torch.sum(mask * x_idx,           dim=[-2,-1])
    m11 = torch.sum(mask * y_idx * x_idx,   dim=[-2,-1])
    m20 = torch.sum(mask * y_idx**2,        dim=[-2,-1])
    m02 = torch.sum(mask * x_idx**2,        dim=[-2,-1])
    
    y_cent = m10 / m00
    x_cent = m01 / m00
    
    u11 = (m11 - y_cent * m01) / m00
    u20 = (m20 - y_cent * m10) / m00
    u02 = (m02 - x_cent * m01) / m00
    
    covs = torch.zeros((*mask.shape[:-2],2,2),device=device)
    
    covs[...,0,0] = u20
    covs[...,0,1] = u11
    covs[...,1,0] = u11
    covs[...,1,1] = u02

    if torch.isnan(y_cent):
        y_cent = torch.tensor(mask.shape[-2]/2)
    if torch.isnan(x_cent):
        x_cent = torch.tensor(mask.shape[-1]/2)

    return covs, y_cent, x_cent

def getAngleFromCovariance_old(covs):
    
    _,_,V = torch.linalg.svd(covs)
    ang = torch.atan2(V[...,0,0],V[...,1,0])
    ang[ang>0] -= torch.pi
    ang = torch.abs(ang)*180.0/torch.pi

    return ang

def getAngleFromCovariance(covs):
    theta = 0.5*torch.atan2( 2*covs[...,0,1] , (covs[...,0,0]-covs[...,1,1]) )
    
    return theta*180.0/torch.pi + 90
    
def getEllipseAxis(covs):
    """
    Equation from raphael.candelier.fr blog (image moments)

    """
    
    u20 = covs[...,0,0]
    u02 = covs[...,1,1]
    u11 = covs[...,0,1]
    
    u20_u02 = u20+u02
    s = torch.sqrt(4*torch.square(u11)+torch.square(u20-u02))
    
    major = torch.sqrt( 8*(u20_u02+s) )
    minor = torch.sqrt( 8*(u20_u02-s) )
    
    return major,minor

def getBoxSize(mask):
    """
   Computes length and width of bounding boxes of the segmentataion masks
   
   Manxi's function with a slight modification

    Parameters
    ----------
    mask : TYPE
        DESCRIPTION.

    Returns
    -------
    length : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.

    """
    
    h,w = mask.shape[-2:]

    rows = torch.any(mask, dim=-1)  
    row_min = torch.argmax(rows.float(), dim=-1)
    row_last = torch.argmax(rows.float().flip(dims=[-1]), dim=-1)
    row_max = h - row_last
    ind = row_last==0
    row_max[ind] = row_min[ind]
    
    cols = torch.any(mask, dim=-2)
    col_min = torch.argmax(cols.float(), dim=-1)
    col_last = torch.argmax(cols.float().flip(dims=[-1]), dim=-1)
    col_max = w - col_last
    ind = col_last==0
    col_max[ind] = col_min[ind]
    
    width = row_max-row_min
    length = col_max-col_min
    return length, width, (row_min, row_max, col_min, col_max)

def centerMask(mask,yc,xc):   
    
    n_dim = len(mask.shape)   

    if n_dim==4:
        b,c,h,w = mask.shape
        _mask = mask.view((b*c,h,w))
        _yc = yc.view((b*c)).long()
        _xc = xc.view((b*c)).long()
    elif n_dim == 3:
        c,h,w = mask.shape
        _yc = yc.long()
        _xc = xc.long()
        _mask = mask
    else:
        raise Exception('mask can be 3 or 4 dimensional only')
        
    mask_centered = torch.zeros_like(_mask)    
    
    for i in range(_mask.shape[0]):
        sy = int(mask.shape[-2]/2 -_yc[i])
        sx = int(mask.shape[-1]/2 -_xc[i])
        
        mask_centered[i] = torch.roll(_mask[i],
                                      shifts=(sy,sx),dims=[0,1]).unsqueeze(0)
        
    if n_dim == 4:
        mask_centered = mask_centered.view(b,c,h,w)
    
    return mask_centered   

def rotateMask(mask,angle):
    """
    Rotates each individual mask using three shear method.
    it is assumed that masks are centered!

    Parameters
    ----------
    mask : torch.tensor B,C,H,W or C,H,W
        DESCRIPTION.
    angle : torch.tensor B,C or C
        Rotation angles per mask in degrees

    Returns
    -------
    mask_rotated : torch.tensor B,C,H,W or C,H,W
        mask rotated by angle. matches the size of input
        
        

    """

    
    n_dim = len(mask.shape)
    device = 'cpu' if mask.get_device()==-1 else 'cuda'
    
    if n_dim == 4:
        b,c,h,w = mask.shape
        _mask = mask.view((b*c,h,w))
        _angle = -(angle.view((b*c))+90) * torch.pi/180
        
    elif n_dim == 3:
        c,h,w = mask.shape
        _mask = mask
        _angle = -(angle+90) * torch.pi/180 
        
    else:
        raise Exception('Unsupported number of dimensions')
    
    # Fix aliasing problem
    _angle[(torch.abs(_angle)-0.9273)<1e-4] += 1e-4
    
    nz = torch.nonzero(_mask)
    N = nz.shape[0]
    
    
    
    y = (h//2-nz[:,1]) 
    x = (w//2-nz[:,2])
    nz_angle = torch.zeros(N,dtype=torch.float32,device=device)
    for i in range(_angle.shape[0]):
        nz_angle[nz[:,0]==i] = _angle[i].item()
    
    tan = torch.tan(nz_angle/2)
    sin = torch.sin(nz_angle)
    
    x = torch.round(x-y*tan)
    y = torch.round(x*sin+y)
    x = torch.round(x-y*tan)
    
    
    #print(tan,sin)
    
    pad = h//2
    mask_rotated = torch.zeros((_mask.shape[0],h+2*pad,w+2*pad),dtype=torch.long,device=device)
    mask_rotated[nz[:,0],y.long()+h//2+pad,x.long()+w//2+pad]=1
    
    mask_rotated = mask_rotated[:,pad:-pad,pad:-pad]
    
    if n_dim == 4:
        mask_rotated = mask_rotated.view((b,c,h,w))
    
    #return mask_rotated
    return torch.swapaxes(mask_rotated,-2,-1)

def getConcepts(Y,allow_nan_angle=False):
    """
    

    Parameters
    ----------
    Y : Output dictionary from the model.
        Assumed to contain: 'mask' torch.tensor of size [B,C,H,W]
                            'quality_segmentaion' of size [B,1,H,W]
                            
    allow_nan_angle : bool
                        Converts unreliable angle estimates to nan.
                        The angle estimation problem arises for masks that are
                        symmetric along short and long axes. E.g. circles, squares.

    Raises
    ------
    Exception
        Unsupported input shape.

    Returns
    -------
    C : dictionary of torch.tensors of size [B,C-1] 
        dictionary of concepts for the given segmentation mask. Channel 0 is
        assumed to be background and is skipped
        
    Concepts
    -------
    area   : area expressed as a fraction of the entire image area
              mask_area / (height*width)
            
    angle  : angle between the axis of maximum variance of the mask and x axis
              starting in 1st quadrant angle increases counter-clockwise.
              The angle is defined in range [0,180] degree mapped to [0,1]
            
            
    x_cent : x coordinate of the center of mass / width
    y_cent : y coordinate of the center of mass / height
            
    x_symm : 
    y_symm :
        
    ellipse_major : length of the major axis of an ellipse obtained from mask
                     covariance matrix
    ellipse_minor : length of the minor axis ...
        
    length : maximum distance along the axis of maximum variance 
    width  : maximum distance along the other axis
    
    quality_score : quality score based on quality mask
        
    """
    
    # Unpack and preprocess dict
    # num_cls = Y['mask'].shape[1]
    num_cls = 14
    # mask = Y['mask'].argmax(1) # [B, H, W]
    mask = Y['mask']
    mask = torch.nn.functional.one_hot(mask, num_cls) # [B, H, W] -> [B, C, H, W]
    mask = mask.permute(0,3,1,2)
    mask = mask[:,1:,...] # excluding the background
    
    quality_mask = Y['quality_mask']

    assert len(mask.shape)==4, "Segmentation mask 'Y['mask']' is expected to have 4 channels (B,C,H,W) "
    assert len(quality_mask.shape)==4, "Quality mask Y['quality_mask'] is expected to have 4 channels (B,1,H,W)"
    
    device = 'cpu' if mask.get_device()==-1 else 'cuda'
    
    b,c,h,w = mask.shape
    __mask = mask.float().clone()
    __mask[:,[0,1,2,6,11,12], ...] *= 0
    _mask = mask.reshape((b*c,h,w))

    
    # get areas and find non empty segmentation mask indices
    area = torch.sum(_mask,dim=[-2,-1])
    nz = torch.where(area>0)[0]
    N = _mask.shape[0]                  # number of segmentation masks
    
    
    # Compute quality scores
    quality_mask = quality_mask * __mask
    quality_score = torch.sum(quality_mask.flatten(-2),dim=-1).view(b*c)/torch.clamp(area,min=1)
 
    
    # Compute concepts only for the masks with non-zero areas
    _mask = _mask[nz] 

    covs,y_cent,x_cent = getMaskProperties(_mask)

    angle = getAngleFromCovariance(covs)
    
    major,minor = getEllipseAxis(covs)
    
    # Center and rotate masks. Axis of maximum variance corresponds to x
    mask_aligned = centerMask(_mask,y_cent,x_cent)
    mask_aligned = rotateMask(mask_aligned,-angle)
    
    length,width = getBoxSize(mask_aligned)
    
    y_symm = torch.sum(mask_aligned * torch.flip(mask_aligned, dims=[-2]),dim=[-2,-1])/area[nz]
    x_symm = torch.sum(mask_aligned * torch.flip(mask_aligned, dims=[-1]),dim=[-2,-1])/area[nz]


    if allow_nan_angle:
        angle_to_nan_idx = torch.abs(major-minor)<1 
        angle[angle_to_nan_idx] = torch.nan
    
    # Compose the output dictionary
    C = {}
    
    C['area'] = area / (h*w) # Area is expressed as a fraction of the entire image area
    C['quality_score'] = quality_score
    
    C['angle'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['angle'][nz] = angle / 180
    
    C['x_symm'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['x_symm'][nz] = x_symm
    C['y_symm'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['y_symm'][nz] = y_symm
    
    C['x_cent'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['x_cent'][nz] = x_cent / w
    C['y_cent'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['y_cent'][nz] = y_cent / h
    
    C['ellipse_major'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['ellipse_major'][nz] = major
    C['ellipse_minor'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['ellipse_minor'][nz] = minor
    
    # length is the "longer" axis of the mask, which is rotated to align with x axis,
    # also referred to as image width. This can be potentially confusing
    C['length'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['length'][nz] = length.float() / w 
    C['width'] = torch.zeros(N,dtype=torch.float32,device=device)
    C['width'][nz] = width.float() / h
    
    for key in C.keys():
        C[key] = C[key].view(b,c)
       
    return C

def get_ellipse(masks):
    mask = np.asarray(masks==7, dtype=np.float32) + np.asarray(masks==13, dtype=np.float32)
    points = np.nonzero(mask)
    points = np.array(points).T
    point_num = points.shape[0]
    if 0 < point_num < 5:
        print(f'point number is {point_num} which is less than 5, not enough for an ellipse')
        fixed_points = np.ones((5, 2), dtype=np.int64)
        fixed_points[:point_num, :] = points
        for i in range(5 - point_num):
            fixed_points[point_num+i, :] = np.mean(fixed_points[:point_num+i, :], axis=0, keepdims=True).astype(np.int64)
        points = fixed_points
        return masks
    elif point_num==0:
        return masks
    (xc, yc), (d1, d2), angle  = cv2.fitEllipse(points) # return ([centeroid coordinate], [length of the semi-major and semi-minor axis], [rotation angle])
    r1, r2 = d1/2, d2/2
    img = np.zeros_like(mask)
    img2 = cv2.ellipse(deepcopy(img), (int(yc), int(xc)), 
                    (int(r2), int(r1)), 
                    -angle, 0, 360, (1), thickness=5)
    masks = np.array(masks > 0, dtype=np.float32) + img2

    masks = masks > 0
    
    return masks

def get_femur_concept(image, mask):
    gray = image.detach().cpu().numpy()
    gray = (gray/2+0.5)*255. # (0-1)
    # angle
    _mask = (mask == 5)
    covs,x_cent,y_cent = getMaskProperties(_mask)
    angle = getAngleFromCovariance(covs).item()
    angle = angle / 180.
    # 0 - 45, 135 - 180 is OK, 45 - 135 is not ok

    # 3. get length
    width, height, corner = getBoxSize(_mask)
    width, height = width.item(), height.item()
    # gray[:80,:] = 0
    bins = 255 - cv2.inRange(gray, 0, 1)
    img_hr_idx = np.nonzero(bins[int(x_cent), ...])[0]
    img_vt_idx = np.nonzero(bins[...,int(y_cent)])[0]
    if len(img_hr_idx) and len(img_vt_idx):
        img_width = max(img_hr_idx) - min(img_hr_idx)
        img_height = max(img_vt_idx) - min(img_vt_idx)
    else:
        img_width = torch.tensor(gray.shape[1])
        img_height = torch.tensor(gray.shape[0])
        img_hr_idx = [0, gray.shape[1]]
        img_vt_idx = [0, gray.shape[0]]

    width = width / img_width if img_width else 1
    width_good =  width >= 0.5
    height = (min(img_vt_idx) - corner[1]) / (corner[1] - max(img_vt_idx)) if (corner[1] - max(img_vt_idx)) else 1
    height_good = height.item() >= 0.5
    occ = width_good and height_good

    # if occ < 0:
    #     occ = 0

    return angle, occ

def get_abdomen_concept(image, mask):
    gray = image.detach().cpu().numpy()
    gray = (gray/2+0.5)*255. # (0-1)
    # 5. length
    bdr = get_ellipse(mask.clone().cpu().squeeze(0).numpy())
    mask = torch.from_numpy(bdr).unsqueeze(0) 
    _mask = mask > 0
    covs,x_cent,y_cent = getMaskProperties(_mask)
    width, height,corner = getBoxSize(_mask)
    width, height = width.item(), height.item()
    # gray[:80,:] = 0
    # gray[:,:100] = 0
    bins = 255 - cv2.inRange(gray, 0, 1)
    img_hr_idx = np.nonzero(bins[int(x_cent), ...])[0]
    img_vt_idx = np.nonzero(bins[...,int(y_cent)])[0]

    if len(img_hr_idx) and len(img_vt_idx):
        img_width = max(img_hr_idx) - min(img_hr_idx)
        img_height = max(img_vt_idx) - min(img_vt_idx)
    else:
        img_width = torch.tensor(gray.shape[1])
        img_height = torch.tensor(gray.shape[0])
        img_hr_idx = [0, gray.shape[1]]
        img_vt_idx = [0, gray.shape[0]]

    img_width = gray.shape[1]
    width = width / img_width if img_width else 1
    height = (min(img_vt_idx) - corner[1]) / (min(img_vt_idx) - max(img_vt_idx)) if (corner[1] - max(img_vt_idx)) else 1
    occ = min([width, height])

    if occ < 0:
        occ = 0

    return occ

def get_head_concept(image, mask):
    gray = image.detach().cpu().numpy()
    gray = (gray/2+0.5)*255. # (0-1)
     # 5. length
    bdr = get_ellipse(mask.clone().cpu().squeeze(0).numpy())
    mask = torch.from_numpy(bdr).unsqueeze(0) 
    _mask = mask > 0
    covs,x_cent,y_cent = getMaskProperties(_mask)
    width, height,corner = getBoxSize(_mask)
    width, height = width.item(), height.item()
    # gray[:80,:] = 0
    # gray[:,:100] = 0
    bins = 255 - cv2.inRange(gray, 0, 1)
    img_hr_idx = np.nonzero(bins[int(x_cent), ...])[0]
    img_vt_idx = np.nonzero(bins[...,int(y_cent)])[0]
    if len(img_hr_idx) and len(img_vt_idx):
        img_width = max(img_hr_idx) - min(img_hr_idx)
        img_height = max(img_vt_idx) - min(img_vt_idx)
    else:
        img_width = gray.shape[1]
        img_height = gray.shape[0]
        img_hr_idx = [0, gray.shape[1]]
        img_vt_idx = [0, gray.shape[0]]

    img_width = gray.shape[1]
    width = width / img_width if img_width else 1
    height = ((min(img_vt_idx) - corner[1]) / (min(img_vt_idx) - max(img_vt_idx))).item() if (corner[1] - max(img_vt_idx)) else 1
    occ = min([width, height])

    if occ < 0:
        occ = 0

    return occ

def get_cervix_concept(image, mask):
    _mask = mask > 0
    covs,x_cent,y_cent = getMaskProperties(_mask)
    width, height,corner = getBoxSize(_mask)
    width, height = height.item(), width.item()
    gray = image.detach().cpu().numpy()
    gray = (gray/2+0.5)*255. # (0-1)
    bins = 255 - cv2.inRange(gray, 0, 1)
    img_hr_idx = np.nonzero(bins[int(x_cent), ...])[0]
    img_vt_idx = np.nonzero(bins[...,int(y_cent)])[0]
    img_width = max(img_hr_idx) - min(img_hr_idx)
    img_height = max(img_vt_idx) - min(img_vt_idx)
    scan_x_cent = min(img_hr_idx) + img_width/2
    scan_y_cent = min(img_vt_idx) + img_height/2
    if (np.abs(x_cent.item() - scan_x_cent) / mask.size(1)) <= 0.33:
        middle = True
    else:
        middle = False
    width = width / img_width if img_width else 1
    width_good = width >= 0.5
    height = ((min(img_vt_idx) - corner[1]) / (min(img_vt_idx) - max(img_vt_idx))).item() if (corner[1] - max(img_vt_idx)) else 1
    height_good = height >= 0.75

    length = height_good# and middle

    return length

if __name__ == "__main__":
    from torchvision import transforms as T
    from glob import glob
    import matplotlib.pyplot as plt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfs = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 288)),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    masks = glob('/data/proto/Zahra_Study1_Trials/Trial14/Head/*_mask.tif')
    images = list(map(lambda x: x.replace('_mask.tif', '.jpg'), masks))
    for image, mask in zip(images, masks):
        image, mask = Image.open(image), Image.open(mask)

        image = tfs(image)
        image = torch.mean(image, dim=0)
        mask = T.Resize((224, 288), interpolation=T.InterpolationMode.NEAREST)(mask)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        image, mask = image.to(device), mask.to(device)

        # angle, occ = get_femur_concept(image, mask)
        # occ = get_abdomen_concept(image, mask)
        occ = get_head_concept(image, mask)

        # print(angle, occ)
        print(occ)
