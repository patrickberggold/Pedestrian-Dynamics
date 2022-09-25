import math

import torch
import torchvision.transforms as TT

def normalize_prob_map(x):
    """Normalize a probability map of shape (B, T, H, W) so
    that sum over H and W equal ones"""
    assert len(x.shape) == 4
    sums = x.sum(-1, keepdim=True).sum(-2, keepdim=True)
    x = torch.divide(x, sums)
    return x


def un_normalize_prob_map(x):
    """Un-Normalize a probability map of shape (B, T, H, W) so
    that each pixel has value between 0 and 1"""
    assert len(x.shape) == 4
    (B, T, H, W) = x.shape
    maxs, _ = x.reshape(B, T, -1).max(-1)
    assert all(maxs.bool().tolist()), 'Not all elements are bigger zero'
    x = torch.divide(x, maxs.unsqueeze(-1).unsqueeze(-1))
    return x

def make_gaussian_map_patches(gaussian_centers,
                              width,
                              height,
                              norm=False,
                              gaussian_std=None):
    """
    gaussian_centers.shape == (T, 2)
    Make a PyTorch gaussian GT map of size (1, T, height, width)
    centered in gaussian_centers. The coordinates of the centers are
    computed starting from the left upper corner.
    """
    assert isinstance(gaussian_centers, torch.Tensor)

    if not gaussian_std:
        gaussian_std = min(width, height) / 64
    gaussian_var = gaussian_std ** 2

    x_range = torch.arange(0, height, 1)
    y_range = torch.arange(0, width, 1)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    pos = torch.stack((grid_y, grid_x), dim=2)
    pos = pos.unsqueeze(2)
    # create a Gaussian map for each coordinate in the trajectory, for each trajectory
    gaussian_map = (1. / (2. * math.pi * gaussian_var)) * \
                   torch.exp(-torch.sum((pos - gaussian_centers) ** 2., dim=-1)
                             / (2 * gaussian_var))

    # from (H, W, T) to (1, T, H, W)
    gaussian_map = gaussian_map.permute(2, 0, 1).unsqueeze(0)

    if norm:
        # normalised prob: sum over coordinates equals 1
        gaussian_map = normalize_prob_map(gaussian_map)
    else:
        # un-normalize probabilities (otherwise the network learns all zeros)
        # each pixel has value between 0 and 1
        gaussian_map = un_normalize_prob_map(gaussian_map)

    return gaussian_map


def create_tensor_image(big_numpy_image,
                        down_factor=1):
    img = TT.functional.to_tensor(big_numpy_image)
    C, H, W = img.shape
    new_heigth = int(H / down_factor)
    new_width = int(W / down_factor)
    tensor_image = TT.functional.resize(img, (new_heigth, new_width),
                                        interpolation=TT.InterpolationMode.BILINEAR)
    return tensor_image


def create_CNN_inputs_loop(batch_abs_pixel_coords,
                           tensor_image):

    num_agents = batch_abs_pixel_coords.shape[1]
    C, H, W = tensor_image.shape
    input_traj_maps = list()

    # loop over agents
    for agent_idx in range(num_agents):
        trajectory = batch_abs_pixel_coords[:, agent_idx, :].detach()\
            .clone().to(torch.device("cpu"))

        traj_map_cnn = make_gaussian_map_patches(
            gaussian_centers=trajectory,
            height=H,
            width=W)
        # append
        input_traj_maps.append(traj_map_cnn)

    # list --> tensor
    input_traj_maps = torch.cat(input_traj_maps, dim=0)

    return input_traj_maps
