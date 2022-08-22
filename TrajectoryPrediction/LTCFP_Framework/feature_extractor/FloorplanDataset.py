from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import random
import h5py
from helper import SEP
import matplotlib.pyplot as plt
from src.m_transforms import m_RandomRotation, m_RandomHorizontalFlip, m_RandomVerticalFlip

class img2img_dataset(Dataset):
    def __init__(self, mode: str, img_paths: list, transform = None):

        self.transform = transform
        self.img_paths = img_paths
        self.mode = mode
        self.final_resolution = None
        self.max_floorplan_meters = None

        assert mode in ['img2img'], 'Unknown mode setting!'

    def _image_preprocessing(self, img_path):

        floorplan_size_x = float(img_path.split('siteX_')[-1].split('_')[0])
        floorplan_size_y = float(img_path.split('siteY_')[-1].split('_')[0])

        img = transforms.ToTensor()(np.array(h5py.File(img_path, 'r').get('img')))

        scale_x = floorplan_size_x / self.max_floorplan_meters
        scale_y = floorplan_size_y / self.max_floorplan_meters

        assert 0.0 <= scale_x <= 1.0 and 0.0 <= scale_y <= 1.0

        scaled_resolution_x = int(self.final_resolution * scale_x)
        scaled_resolution_y = int(self.final_resolution * scale_y)

        # Scale image down
        img = transforms.Resize(size=(scaled_resolution_y, scaled_resolution_x))(img)
        # Rotate the image by an angle
        transform_rot = m_RandomRotation((0, 360), expand=True, fill=0)
        rotated_img = transform_rot(img)
        rotation_angle = transform_rot.angle
        # Make sure that the rotated image doesnt exceed the final resolution
        if rotated_img.size()[-2] > self.final_resolution or rotated_img.size()[-1] > self.final_resolution:
            rotation_angle = random.choice([0,90,180,270])
            rotated_img = transforms.RandomRotation((rotation_angle, rotation_angle), expand=False, fill=0)(img)

        # pad the rest of the image with zeros
        width = rotated_img.size()[-1]
        height = rotated_img.size()[-2]

        padd_width_x0 = (self.final_resolution - width)//2
        padd_width_x1 = self.final_resolution - width - padd_width_x0

        padd_width_y0 = (self.final_resolution - height)//2
        padd_width_y1 = self.final_resolution - height - padd_width_y0

        assert width+padd_width_x0+padd_width_x1 == self.final_resolution
        assert height+padd_width_y0+padd_width_y1 == self.final_resolution

        # left, top, right and bottom borders respectively
        rotated_img = transforms.Pad([padd_width_x0, padd_width_y0, padd_width_x1, padd_width_y1], fill=0., padding_mode='constant')(rotated_img)

        # include random flipping
        transform_hf = m_RandomHorizontalFlip(p=0.5)
        rotated_img = transform_hf(rotated_img)
        h_flip = transform_hf.h_flip
        transform_vf = m_RandomVerticalFlip(p=0.5)
        rotated_img = transform_vf(rotated_img)
        v_flip = transform_vf.v_flip

        # plt.imshow(rotated_img.permute(1, 2, 0))
        assert rotated_img.size()[-2] == self.final_resolution and rotated_img.size()[-1] == self.final_resolution
        
        return rotated_img, [rotation_angle, h_flip, v_flip]

    def __len__(self):
        return(len(self.img_paths))
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # quick check if paths are consistent

        img, [rotation_angle, h_flip, v_flip] = self._image_preprocessing(img_path)
        
        # img = np.array(h5py.File(img_path, 'r').get('img'))
        # plt.imsave(f'test_input_img_{idx}.jpeg', img, vmin=0, vmax=255)

        # if self.transform:
        #     img = self.transform(img)

        return img
