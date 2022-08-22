from typing import Tuple
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad
from torchvision.transforms.functional import resize, rotate

def rotate_and_rescale(input_img: torch.Tensor, angle: float, final_resolution: int, floorplan_sizes_meters: Tuple, max_size_meters: float):

    scale_x = floorplan_sizes_meters[0] / max_size_meters
    scale_y = floorplan_sizes_meters[1] / max_size_meters

    assert 0.0 <= scale_x <= 1.0 and 0.0 <= scale_y <= 1.0

    scaled_resolution_x = int(final_resolution * scale_x)
    scaled_resolution_y = int(final_resolution * scale_y)

    # Scale image down
    input_img = resize(input_img, (scaled_resolution_y, scaled_resolution_x))
    # Rotate the image by an angle
    rotated_img = rotate(input_img, angle, expand=True, fill=0)
    # Make sure that the rotated image doesnt exceed the final resolution
    if rotated_img.size()[-2] > final_resolution or rotated_img.size()[-1] > final_resolution:
        rotated_img = input_img

    # pad the rest of the image with zeros
    width = rotated_img.size()[-1]
    height = rotated_img.size()[-2]

    padd_width_x0 = (final_resolution - width)//2
    padd_width_x1 = final_resolution - width - padd_width_x0

    padd_width_y0 = (final_resolution - height)//2
    padd_width_y1 = final_resolution - height - padd_width_y0

    assert width+padd_width_x0+padd_width_x1 == final_resolution
    assert height+padd_width_y0+padd_width_y1 == final_resolution

    rotated_img = pad(rotated_img, (padd_width_x0, padd_width_x1, padd_width_y0, padd_width_y1), mode='constant', value=0.)
    # plt.imshow(rotated_img.squeeze().permute(1, 2, 0))

    return rotated_img


if __name__ == '__main__':
    test_path = "C:\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\INPUT\\train_station\\2__floorplan_siteX_65_siteY_25_numEscX_4_numEscY_2_SET_together\\HDF5_floorplan_siteX_65_siteY_25_numEscX_4_numEscY_2_SET_together_variation_5.h5"
    test_img = np.array(h5py.File(test_path, 'r').get('img'))

    # Transform to tensor
    test_img = torch.tensor(test_img).permute(2,0,1).unsqueeze(0)

    # plt.imshow(test_img)
    test_img = rotate_and_rescale(test_img, 45, 1000, (65, 25), 70)

    # plt.imshow(test_img.squeeze().permute(1, 2, 0))
    # h = 1