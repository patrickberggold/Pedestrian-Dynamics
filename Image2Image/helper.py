import torch
from torch.utils import data
import numpy as np
import os
import cv2
import h5py
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy import ndimage
import platform

OpSys = platform.system()
SEP = '\\' if OpSys == 'Windows' else '/'

CLASS_NAMES = {
    # get color values from https://doc.instantreality.org/tools/color_calculator/
    # Format RGBA
    'unpassable': np.array([0, 0, 0, 1]), 
    'walkable area': np.array([1, 1, 1, 1]), 
    'spawn_zone': np.array([0.501, 0.039, 0, 1]), #np.array([1, 0.0, 0, 1]) 
    'destination': np.array([0.082, 0.847, 0.054, 1]), # np.array([0.0, 1, 0, 1]), 
    # 'path': np.array([0, 0, 1, 1])
    }

def get_customized_colormap():

    len_classes = len(CLASS_NAMES.keys())
    
    viridis = cm.get_cmap('viridis', len_classes)
    custom_colors = viridis(np.linspace(0, 1, len_classes)) 
    # define colors
    for idx, key in enumerate(CLASS_NAMES):
        custom_colors[idx] = CLASS_NAMES[key]
    customed_colormap = ListedColormap(custom_colors)
    return customed_colormap
   
def get_color_from_array(index, max_index, return_in_cv2: bool = False):

    # colors from https://doc.instantreality.org/tools/color_calculator/
    # [143, 255, 211] -> [143, 255, 255] -> [0, 187, 255] -> [0, 0, 255] -> [255, 0, 255]
    b_val0 = 211
    r_val_1, g_val_1 = 143, 225
    r_val_2, g_val_2 = 0, 187
    r_val_3, g_val_3 = 0, 0
    r_val_4, g_val_4 = 255, 0
    b_val = 255 
    color_range_0_bval = np.arange(211, 255)
    color_range_1_rval = np.arange(143, 0, -1)
    color_range_2_gval = np.arange(187, 0, -1)
    color_range_3_rval = np.arange(255)

    color_range_len = len(color_range_0_bval) + len(color_range_1_rval) + len(color_range_2_gval) + len(color_range_3_rval)

    fraction0 = len(color_range_0_bval)/color_range_len # 0.06995230524642289
    fraction1 = (len(color_range_0_bval) + len(color_range_1_rval))/color_range_len # 0.2972972972972973
    fraction2 = (len(color_range_0_bval) + len(color_range_1_rval) + len(color_range_2_gval))/color_range_len # 0.5945945945945946

    fr = index / max_index
    if fr <= fraction0:
        b_val = round(b_val0 + index / (fraction0*max_index) * (255 - b_val0))
        r_val, g_val = r_val_1, g_val_1
    elif fraction0 < fr <= fraction1:
        r_val = round(r_val_1 - index / (fraction1*max_index) * (r_val_1 - r_val_2))
        g_val = round(g_val_1 - index / (fraction1*max_index) * (g_val_1 - g_val_2))
    elif fraction1 < fr <= fraction2:
        ll = (index-fraction1*max_index) / (fraction2*max_index-fraction1*max_index)
        r_val = 0
        g_val = round(g_val_2 - (index-fraction1*max_index) / (fraction2*max_index-fraction1*max_index) * (g_val_2 - g_val_3))
    elif fraction2 <= fr <= 1:
        ll = (index-fraction2*max_index) / (max_index-fraction2*max_index)
        g_val = 0
        r_val = round(r_val_3 + (index-fraction2*max_index) / (max_index-fraction2*max_index) * (r_val_4 - r_val_3))
    else:
        raise ValueError

    color_array = np.array([r_val, g_val, b_val])

    return color_array.astype('float64')

def get_color_from_pedId(id, num_agents = 40):
    id = int(id)
    if id < 5:
        color = np.array([138, 255, 0])
    elif 5 <= id < 10:
        color = np.array([0, 255, 171])
    elif 10 <= id < 15:
        color = np.array([0, 255, 255])
    elif 15 <= id < 20:
        color = np.array([0, 160, 255])
    elif 20 <= id < 25:
        color = np.array([0, 57, 255])
    elif 25 <= id < 30:
        color = np.array([109, 0, 255])
    elif 30 <= id < 35:
        color = np.array([185, 0, 255])
    elif 35 <= id <= 40:
        color = np.array([255, 0, 213])
    else:
        raise NotImplementedError

    return color.astype('float64')

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
