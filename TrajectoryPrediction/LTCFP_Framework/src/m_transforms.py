from torchvision import transforms as T
from torch import Tensor
import torchvision.transforms.functional as F
import torch
from torchvision import transforms
import numpy as np
import h5py
import random

class m_RandomRotation(T.RandomRotation):
    def __init__(self, degrees, interpolation=F.InterpolationMode.BILINEAR, expand=False, fill=0):
        super(m_RandomRotation, self).__init__(degrees, interpolation=interpolation, expand=expand, fill=fill)
        self.angle = self.get_params(self.degrees)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        return F.rotate(img, self.angle, self.resample, self.expand, self.center, fill)

class m_RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p):
        super(m_RandomHorizontalFlip, self).__init__(p=p)
        if torch.rand(1) < self.p:
            self.h_flip = True
        else:
            self.h_flip = False
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if self.h_flip:
            return F.hflip(img)
        return img

class m_RandomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, p):
        super(m_RandomVerticalFlip, self).__init__(p=p)
        if torch.rand(1) < self.p:
            self.v_flip = True
        else:
            self.v_flip = False
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if self.v_flip:
            return F.vflip(img)
        return img


def image_and_traj_preprocessing(self, img_path, abs_pixel_coord):

    import matplotlib.pyplot as plt

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
    transform_rot = m_RandomRotation((0, 360), F.InterpolationMode.BILINEAR, expand=True, fill=0)
    rotated_img = transform_rot(img)
    rotation_angle = transform_rot.angle

    # Make sure that the rotated image doesnt exceed the final resolution
    if rotated_img.size()[-2] > self.final_resolution or rotated_img.size()[-1] > self.final_resolution:
        rotation_angle = random.choice([0,90,180,270])
        rotated_img = transforms.RandomRotation((rotation_angle, rotation_angle), expand=False, fill=0)(img)

    # pad the rest of the image with zeros
    width = rotated_img.size()[-1]
    height = rotated_img.size()[-2]

    padd_width_x0 = 3 #(self.final_resolution - width)//2
    padd_width_x1 = self.final_resolution - width - padd_width_x0

    padd_width_y0 = 3# (self.final_resolution - height)//2
    padd_width_y1 = self.final_resolution - height - padd_width_y0

    assert width+padd_width_x0+padd_width_x1 == self.final_resolution
    assert height+padd_width_y0+padd_width_y1 == self.final_resolution

    # left, top, right and bottom borders respectively
    rotated_img = transforms.Pad([padd_width_x0, padd_width_y0, padd_width_x1, padd_width_y1], fill=0., padding_mode='constant')(rotated_img)
    # plt.imshow(rotated_img.permute(1, 2, 0))

    # include random flipping
    # transform_hf = m_RandomHorizontalFlip(p=0.5)
    # rotated_img = transform_hf(rotated_img)
    # h_flip = transform_hf.h_flip
    # transform_vf = m_RandomVerticalFlip(p=0.5)
    # rotated_img = transform_vf(rotated_img)
    # v_flip = transform_vf.v_flip

    # plt.imshow(rotated_img.permute(1, 2, 0))
    assert rotated_img.size()[-2] == self.final_resolution and rotated_img.size()[-1] == self.final_resolution
    
    return rotated_img, [padd_width_x0, padd_width_y0, scale_x, scale_y, rotation_angle, True, True]