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

    # colors = [get_color_from_array(i, 179) for i in range(179)]
    # [143, 225, 255] -> [0, 187, 255] -> [0, 0, 255] -> [180, 0, 255]
    r_val_1, g_val_1 = 143, 225
    r_val_2, g_val_2 = 0, 187
    r_val_3, g_val_3 = 0, 0
    r_val_4, g_val_4 = 180, 0
    b_val = 255 
    color_range_1_rval = np.arange(143, 0, -1)
    color_range_2_gval = np.arange(187, 0, -1)
    color_range_3_rval = np.arange(181)

    color_range_len = len(color_range_1_rval) + len(color_range_2_gval) + len(color_range_3_rval)

    fraction1 = len(color_range_1_rval)/color_range_len # 0.27984344422700586
    fraction2 = (len(color_range_1_rval) + len(color_range_2_gval))/color_range_len # 0.6457925636007827

    fr = index / max_index
    if fr <= fraction1:
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
