import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from copy import deepcopy
import math

def get_ellipse(mask:torch.Tensor, fill_value, dilation=5): # 7, 13
    device = mask.device
    mask = np.asarray(mask.squeeze(-1).detach().cpu().numpy(), dtype=np.float32)
    mask = cv2.medianBlur(mask, ksize=3)
    points = np.nonzero(mask)
    points = np.array(points).T
    point_num = points.shape[0]
    if point_num < 5:
        return torch.from_numpy(mask).unsqueeze(-1).to(device), [(0, 0), (0, 0), (0, 0), (0, 0)]
    (xc, yc), (d1, d2), angle  = cv2.fitEllipse(points) # return ([centeroid coordinate], [length of the semi-major and semi-minor axis], [rotation angle])
    r1, r2 = d1/2, d2/2
    img = np.zeros_like(mask)
    img2 = cv2.ellipse(deepcopy(img), (int(yc), int(xc)), 
                    (int(r2), int(r1)), 
                    -angle, 0, 360, (fill_value), thickness=dilation)
    bottom_x = int(xc + math.cos(math.radians(angle))*r1)
    bottom_y = int(yc + math.sin(math.radians(angle))*r1)
    top_x = int(xc + math.cos(math.radians(angle+180))*r1)
    top_y = int(yc + math.sin(math.radians(angle+180))*r1)
    angle = angle + 90
    left_x = int(xc + math.cos(math.radians(angle))*r2)
    left_y = int(yc + math.sin(math.radians(angle))*r2)
    right_x = int(xc + math.cos(math.radians(angle+180))*r2)
    right_y = int(yc + math.sin(math.radians(angle+180))*r2)

    x, y = [top_x, bottom_x, left_x, right_x], [top_y, bottom_y, left_y, right_y]
    bottom_x, bottom_y = x[x.index(min(x))], y[x.index(min(x))]
    top_x, top_y = x[x.index(max(x))], y[x.index(max(x))]
    left_x, left_y = x[y.index(min(y))], y[y.index(min(y))]
    right_x,right_y = x[y.index(max(y))], y[y.index(max(y))]

    mask = torch.from_numpy(img2).unsqueeze(-1).to(device)

    return mask, [(left_x, left_y), (right_x, right_y), (top_x, top_y), (bottom_x, bottom_y)]

def filling_values(img, x, y, patch_size, value):
    x1 = max([0, x-patch_size[0]])
    x2 = min([img.size(0), x+patch_size[1]])
    y1 = max([0, y-patch_size[2]])
    y2 = min([img.size(1), y+patch_size[3]])
    img[x1:x2, y1:y2, ...] = value
    return img

def fetal_caliper_concept(x):
    # add extra segmentation concepts for calipers
    seg_mask = x['seg_mask']
    assign_mtx = x['assign_mtx']
    plane = x['plane']
    # patch_size = [32, 32, 32, 32]
    patch_size = [40, 40, 40, 40]
    fill_value = 1.
    caliper_concepts = []

    for b in range(seg_mask.size(0)): # for each image in the batch
        # femur left and right end
        mask = seg_mask[b, ...].unsqueeze(-1)
        left_mask = torch.zeros_like(mask)
        right_mask = torch.zeros_like(mask)
        abdomen_mask = torch.zeros_like(mask)
        bpdn_mask = torch.zeros_like(mask) # top
        bpdf_mask = torch.zeros_like(mask) # bottom
        ofdf_mask = torch.zeros_like(mask) # left or near csp
        ofdo_mask = torch.zeros_like(mask) # right or near thalamus

        if plane[b] == 0 and torch.sum(mask==5) >= 100:
            points = torch.nonzero(mask==5)
            index_left, index_right = torch.argmin(points[:,1]), torch.argmax(points[:,1])
            x_left, x_right, y_left, y_right = points[index_left, 0], points[index_right, 0], points[index_left, 1], points[index_right, 1]
            
            length = y_right - y_left

            index = points[:,1]< y_left+length/10
            left_points = points[index, :]
            height = (torch.max(left_points[:,0].float()) + torch.min(left_points[:,0].float()))/2 
            index = points[:,1]< y_left+length/25
            left_points = points[index, :]
            index = torch.argmin(torch.abs(height - left_points[:, 0]))
            x_left, y_left, _ = left_points[index, :]

            index = points[:,1]> y_right-length/10
            right_points = points[index, :]
            height = (torch.max(right_points[:,0].float()) + torch.min(right_points[:,0].float()))/2
            index = points[:,1]> y_right-length/25
            right_points = points[index, :]
            index = torch.argmin(torch.abs(height - right_points[:, 0]))
            x_right, y_right, _ = right_points[index, :]

            x_left, x_right, y_left, y_right = x_left.item(), x_right.item(), y_left.item(), y_right.item()   

            left_mask = filling_values(left_mask, x_left, y_left, patch_size, fill_value)*(mask==5)
            right_mask = filling_values(right_mask, x_right, y_right, patch_size, fill_value)*(mask==5)
        
        elif plane[b] == 1 and torch.sum(mask==7) >= 100:
            abdomen_mask, _ = get_ellipse(mask==7, fill_value, dilation=30)
        elif plane[b] == 2 and torch.sum(mask==13) >= 100:
            ellipse, points = get_ellipse(mask==13, fill_value, dilation=30)
            left_point, right_point, top_point, bottom_point = points
            bpdf_mask = filling_values(bpdf_mask, top_point[0], top_point[1], patch_size, fill_value)*ellipse
            bpdn_mask = filling_values(bpdn_mask, bottom_point[0], bottom_point[1], patch_size, fill_value)*ellipse
            # near, far
            left_patch = [50, 50, 32, 50]
            right_patch = [50, 50, 50, 32]
            if torch.sum(mask==12) >= 100:
                points = torch.nonzero(mask==12)
                y_c = torch.median(points[:,1])
                if torch.abs(left_point[1] - y_c) < torch.abs(right_point[1] - y_c):
                    tmp = right_point
                    right_point = left_point
                    left_point = tmp  
                    tmp = right_patch
                    right_patch = left_patch
                    left_patch = tmp
            elif torch.sum(mask==10) >= 100:
                points = torch.nonzero(mask==10)
                y_c = torch.median(points[:,1])
                if torch.abs(left_point[1] - y_c) > torch.abs(right_point[1] - y_c):
                    tmp = right_point
                    right_point = left_point
                    left_point = tmp
                    tmp = right_patch
                    right_patch = left_patch
                    left_patch = tmp

            ofdo_mask = filling_values(ofdo_mask, left_point[0], left_point[1], left_patch, fill_value)*ellipse
            ofdf_mask = filling_values(ofdf_mask, right_point[0], right_point[1], right_patch, fill_value)*ellipse            
        caliper_concepts.append(torch.cat((left_mask, right_mask, abdomen_mask, bpdf_mask, bpdn_mask, ofdf_mask, ofdo_mask), dim=-1).unsqueeze(0))
    caliper_concepts = torch.cat(caliper_concepts, dim=0)
    assign_mtx = torch.cat((assign_mtx, caliper_concepts.permute(0,3,1,2)), axis=1)
    
    x['assign_mtx'] = assign_mtx
    return x