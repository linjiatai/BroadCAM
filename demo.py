# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from utils.tools_for_broadcam import *
from core.networks import *
from utils.tools.ai.torch_utils import *
from utils.tools.ai.augment_utils import *
from utils.tools.ai.randaugment import *
from utils.torchcam import methods
from utils.torchcam.methods import *
from PIL import Image, ImageDraw, ImageFont
from utils.broadcam_final import BroadCAM
def show_cam_on_image(img, mask):
    img = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = (0.7*np.float32(heatmap) + 0.3*img) / 255
    cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main(args, is_fit_bls=False):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    ori_image = Image.open('examples/img/' + args.image_id + '.jpg').convert('RGB')
    mask = Image.open('examples/mask/' + args.image_id + '.png')
    mask_np = np.array(mask)
    
    labels = np.unique(mask_np)[1:-1]
    ##
    scales = [0.5, 1.0, 1.5, 2.0]
    ori_w, ori_h = ori_image.size
    ttf = ImageFont.truetype("./utils/arialuni.ttf", 40)
    n_labels = labels.shape[0]
    n_proportions = len(args.proportions)
    n_cam_methods = len(args.cam_methods)
    for k in range(n_labels):
        label = labels[k]
        class_idx = [label-1, label-1]
        concat = Image.new('RGB', (ori_w*(n_proportions+1), ori_h*n_cam_methods))
        concat.paste(ori_image, (ori_w*0, ori_h*0))
        concat.paste(mask, (ori_w*0, ori_h*1))
        for i in range(n_proportions):
            proportion = args.proportions[i]
            ## Load model
            os.makedirs('checkpoints/checkpoints_of_deep_model/', exist_ok=True)
            model_path = 'checkpoints/checkpoints_of_deep_model/resnest101@train_aug_'+proportion+'@optimal.pth'
            load_model(model, model_path)
    
            for j in range(n_cam_methods):
                cam_method = args.cam_methods[j]
                strided_cams_list = []
                if cam_method=='BroadCAM' and is_fit_bls==True:
                    model.eval()
                    args.alpha = 50
                    args.n_class=20
                    args.n_level=2
                    broadcam = BroadCAM(alpha=args.alpha, 
                                        n_level=args.n_level, 
                                        model=model, target_layer='[0+1]', n_class=args.n_class)
                    broadcam.fit_from_feature_one_path(path='checkpoints/features_and_labels/resnest101@train_aug_'+proportion+'@optimal_features_and_labels.pth')
                    os.makedirs('checkpoints/parameters_of_BroadCAM/', exist_ok=True)
                    dump_pickle('checkpoints/parameters_of_BroadCAM/'+'BroadCAM_parameter_train_aug_'+proportion+'.pkl', broadcam)
                for l in range(4):
                    scale = scales[l]
                    image = copy.deepcopy(ori_image)
                    image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                    image = normalize_fn(image)
                    image = image.transpose((2, 0, 1))
                    image = torch.from_numpy(image)
                    flipped_image = image.flip(-1)
                    images = torch.stack([image, flipped_image])
                    # inferenece
                    if cam_method=='CAM':
                        model.eval()
                        _, features = model(images.cuda(), with_cam=True)
                        cams = F.relu(features)
                        
                        cams = cams[0] + cams[1].flip(-1)
                        cams = F.interpolate(cams.unsqueeze(0), (ori_h, ori_w), mode='bilinear', align_corners=False)[0]
                        cam_tmp = cams[label-1].detach().cpu()
                        test = 1
                    elif cam_method=='GradCAM':
                        model.eval()
                        extractor = GradCAM(model, 'stage5')
                        scores = model(images.cuda())
                        cams = extractor(class_idx=class_idx, scores=scores)[0]
                        cams = (cams[0] + cams[1].flip(-1)).unsqueeze(0).unsqueeze(0)
                        cam_tmp = F.interpolate(cams, (ori_h, ori_w), mode='bilinear', align_corners=False)[0][0].detach().cpu()
                    elif cam_method=='LayerCAM':
                        model.eval()
                        extractor = LayerCAM(model, ['stage2', 'stage3', 'stage4', 'stage5'])
                        scores = model(images.cuda())
                        cams = extractor(class_idx=class_idx, scores=scores)
                        cams = extractor.fuse_cams(cams)
                        cams = (cams[0] + cams[1].flip(-1)).unsqueeze(0).unsqueeze(0)
                        cam_tmp = F.interpolate(cams, (ori_h, ori_w), mode='bilinear', align_corners=False)[0][0].detach().cpu()
                    elif cam_method=='BroadCAM':
                        model.eval()
                        broadcam = load_pickle('checkpoints/parameters_of_BroadCAM/'+'BroadCAM_parameter_train_aug_'+proportion+'.pkl')
                        tmpcam = broadcam.gen_broadcam(images.cuda(),is_flip=True, WSSS_or_WSOL='WSSS')
                        cams = torch.from_numpy(tmpcam)
                        cams = F.interpolate(cams.unsqueeze(0), (ori_h, ori_w), mode='bilinear', align_corners=False)[0]
                        cam_tmp= cams[label-1].detach().cpu()

                    strided_cams_list.append(cam_tmp)
                cam_seed = torch.sum(torch.stack(strided_cams_list), dim=0)
                cam_seed = cam_seed/(cam_seed.max()+1e-10)
                cam_seed = show_cam_on_image(cv2.cvtColor(np.array(ori_image), cv2.COLOR_RGB2BGR), cam_seed)
                cam_seed = Image.fromarray(cv2.cvtColor(cam_seed, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(cam_seed)
                draw.text((0, 0), cam_method, font=ttf, fill=(255,0,0), width=20)
                draw.text((0, 40), 'Proportion : '+str(int(proportion))+'%', font=ttf, fill=(255,0,0), width=20)
                concat.paste(cam_seed, (ori_w*(i+1), ori_h*j))
        concat.show()
        concat.save('examples/CAM_seeds/'+args.image_id+'_'+str(label)+ '.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cam_methods', default=['CAM', 'GradCAM', 'LayerCAM', 'BroadCAM'], type=str)
    parser.add_argument('--proportions', default=['001', '002', '005', '008', '010', '020', '050', '080', '100'], type=str)
    parser.add_argument('--image_id', default='2007_001595', type=str) ## ['2007_001595', '2007_001698', '2007_001834']
    args = parser.parse_args()
    model = Classifier('resnest101', 20, mode='normal')
    model = model.cuda()
    main(args)