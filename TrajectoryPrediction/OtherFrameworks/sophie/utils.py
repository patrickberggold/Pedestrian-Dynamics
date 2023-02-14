import os
import math
import random
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import h5py
from torchvision import transforms 
from torchvision.models.vgg import vgg19_bn

from constants import *

def get_dset_path(dset_name, dset_type):
    return os.path.join('C:\\Users\\Remotey\\Documents\\Pedestrian-Dynamics\\TrajectoryPrediction\\sophie\\datasets', dset_name, dset_type)

def get_dset_path_floorplans(dset_name):
    return os.path.join('C:\\Users\\Remotey\\Documents\\Datasets\\SIMPLE_FLOORPLANS\\CSV_SIMULATION_DATA_numAgents_50', dset_name)

def relative_to_abs(rel_traj, start_pos):
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def l2_loss(pred_traj, pred_traj_gt, mode='average'):
    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / (seq_len * batch)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(
    pred_pos, pred_pos_gt, mode='sum'
):
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


class Backbone():
    def __init__(self) -> None:
        ckpt_path = '\\'.join(['C:','Users','Remotey','Documents','Pedestrian-Dynamics','TrajectoryPrediction','sophie','feature_extractor','checkpoints', 'model_vgg_img2img_epoch=54-step=8580.ckpt'])
        state_dict = OrderedDict([(key.replace('net.0.', ''), tensor) for key, tensor in torch.load(ckpt_path)['state_dict'].items() if key.startswith('net.0')])

        self.model = vgg19_bn(pretrained=True, progress=True).features
        self.model.load_state_dict(state_dict)
        self.model.to(f'cuda:{CUDA_DEVICE}')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward_pass(self, txt_path):

        filename = txt_path.split("\\")[-1].replace("txt", "h5")
        img_path = os.path.join(
            '\\'.join(txt_path.split('\\')[:-3]),
            'HDF5_INPUT_IMAGES_resolution_800_800',
            txt_path.split('\\')[-2],
            f'HDF5_floorplan_zPos_0.0_roomWidth_0.24_numRleft_2.0_numRright_2.0_{filename}'
        )
        assert os.path.isfile(img_path), f'Img file {img_path} does not exist!'
        img = np.array(h5py.File(img_path, 'r').get('img'))
        
        # transform image to tensor
        img = transforms.ToTensor()(img).unsqueeze(0).to(f'cuda:{CUDA_DEVICE}')

        # forward pass
        features = self.model.forward(img)
        features = nn.AdaptiveAvgPool2d((15, 15))(features)
        # features = features.detach().cpu().numpy()
        features = features.transpose(1,2).transpose(2,3)

        # requires_grad indicates whether a variable is trainable
        # retain_grad() is used to signify that we should store the gradient on non-"leaf" variables to the "grad" attribute
        return features