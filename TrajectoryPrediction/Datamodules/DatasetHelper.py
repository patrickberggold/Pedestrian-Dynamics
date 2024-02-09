import math

import torch
import torchvision.transforms as TT
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_occupancy_map(image_input, obs_coords, size=64, resize_factor=None, gaussian_std=None):
    """
    Pre-compute occupancy maps used by the goal modules and add them to data_dict
    """
    if isinstance(image_input, np.ndarray):
        np_image = image_input
        H, W, C = np_image.shape
        new_height = int(H / resize_factor)
        new_width = int(W / resize_factor)
        # tensor_image = TT.functional.resize(img, (new_heigth, new_width),
        #                                     interpolation=TT.InterpolationMode.NEAREST)
        if resize_factor != 1.0:
            np_image = cv2.resize(np_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    elif isinstance(image_input, tuple):
        H, W = image_input[:2]
        new_height = int(H / resize_factor)
        new_width = int(W / resize_factor)
        np_image = None
    
    if gaussian_std is None:
        gaussian_std = min(new_width, new_height) / 64
        gaussian_std = 2 # made smaller
    gaussian_var = gaussian_std ** 2

    x_range = np.arange(0, new_height, 1) # torch.arange(0, height, 1)
    y_range = np.arange(0, new_width, 1) # torch.arange(0, width, 1)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    pos = np.stack((grid_y, grid_x), axis=2)
    pos = np.expand_dims(pos, axis=2) # pos.unsqueeze(2)
    # create a Gaussian map for each coordinate in the trajectory, for each trajectory
    # convert all finite obs_coords to int
    obs_coords_int = np.floor(obs_coords)
    gaussian_map = np.exp(-np.sum((pos - obs_coords_int) ** 2., axis=-1) / (2 * gaussian_var))

    assert np.all(gaussian_map[:,:,np.isinf(obs_coords[:,0])]==0)
    
    # normalize each agent to 1
    # maxima_01 = np.max(gaussian_map, axis=(0, 1))
    gaussian_map = gaussian_map.max(axis=-1)
    # obs_coords_int = obs_coords.astype(int)
    # agent_values = gaussian_map[obs_coords_int[:,0], obs_coords_int[:,1]]
    # assert that all agent_values are equal
    # assert np.all(agent_values == agent_values[0]) and agent_values[0]==gaussian_map.max()

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the 3D surface
    # surface = ax.plot_surface(grid_x, grid_y, gaussian_map, cmap='viridis')
    
    return np_image, gaussian_map.transpose(1,0)



def create_occupancy_map_torch(image_input, obs_coords, size=64, resize_factor=None, gaussian_std=None):
    """
    Pre-compute occupancy maps used by the goal modules and add them to data_dict
    """
    if isinstance(image_input, torch.Tensor):
        np_image = image_input
        H, W, C = np_image.shape
        new_height = int(H / resize_factor)
        new_width = int(W / resize_factor)
        # tensor_image = TT.functional.resize(img, (new_heigth, new_width),
        #                                     interpolation=TT.InterpolationMode.NEAREST)
        if resize_factor != 1.0:
            np_image = cv2.resize(np_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    elif isinstance(image_input, tuple):
        H, W = image_input[:2]
        np_image = None
        new_height = int(H / resize_factor)
        new_width = int(W / resize_factor)
    
    if gaussian_std is None:
        gaussian_std = min(new_width, new_height) / 64
        gaussian_std = 2 # made smaller
    gaussian_var = gaussian_std ** 2

    x_range = torch.arange(0, new_height, 1) # torch.arange(0, height, 1)
    y_range = torch.arange(0, new_width, 1) # torch.arange(0, width, 1)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    pos = torch.stack((grid_y, grid_x), axis=2)
    pos = torch.unsqueeze(pos, dim=2).to(obs_coords.device)
    # create a Gaussian map for each coordinate in the trajectory, for each trajectory
    # convert all finite obs_coords to int
    obs_coords_int = torch.floor(obs_coords)
    gaussian_map = torch.exp(-torch.sum((pos - obs_coords_int) ** 2., dim=-1) / (2 * gaussian_var))

    assert torch.all(gaussian_map[:,:,torch.isinf(obs_coords[:,0])]==0)
    
    # normalize each agent to 1
    # maxima_01 = np.max(gaussian_map, axis=(0, 1))
    gaussian_map, _ = torch.max(gaussian_map, dim=-1)
    # obs_coords_int = obs_coords.astype(int)
    # agent_values = gaussian_map[obs_coords_int[:,0], obs_coords_int[:,1]]
    # assert that all agent_values are equal
    # assert np.all(agent_values == agent_values[0]) and agent_values[0]==gaussian_map.max()

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the 3D surface
    # surface = ax.plot_surface(grid_x, grid_y, gaussian_map, cmap='viridis')
    
    return np_image, gaussian_map#.permute(1,0)



def add_cnn_maps(np_image, abs_pixel_coord, down_factor):
    """
    Pre-compute CNN maps used by the goal modules and add them to data_dict
    """
    H, W, C = np_image.shape
    new_height = int(H / down_factor)
    new_width = int(W / down_factor)
    # tensor_image = TT.functional.resize(img, (new_heigth, new_width),
    #                                     interpolation=TT.InterpolationMode.NEAREST)
    np_image = cv2.resize(np_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    input_traj_maps = create_CNN_inputs_loop(
        batch_abs_pixel_coords= abs_pixel_coord / down_factor,
        tensor_image=np_image)

    return np_image, input_traj_maps


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
    assert isinstance(gaussian_centers, np.ndarray) # torch.Tensor)

    if not gaussian_std:
        gaussian_std = min(width, height) / 64
    gaussian_var = gaussian_std ** 2

    x_range = np.arange(0, height, 1) # torch.arange(0, height, 1)
    y_range = np.arange(0, width, 1) # torch.arange(0, width, 1)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    pos = np.stack((grid_y, grid_x), axis=2)
    pos = np.expand_dims(pos, axis=2) # pos.unsqueeze(2)
    # create a Gaussian map for each coordinate in the trajectory, for each trajectory
    gaussian_map = (1. / (2. * math.pi * gaussian_var)) * \
                   np.exp(-np.sum((pos - gaussian_centers) ** 2., axis=-1)
                             / (2 * gaussian_var))

    # from (H, W, T) to (1, T, H, W)
    # gaussian_map = gaussian_map.permute(2, 0, 1).unsqueeze(0)
    gaussian_map = np.expand_dims(np.transpose(gaussian_map, (2, 0, 1)), axis=0)
    
    # make sure all infs become zeros in Gaussian maps
    inf_idx = np.isinf(gaussian_centers[:,0])
    if np.any(inf_idx):
        vals = gaussian_map[:,inf_idx,:,:]
        assert np.all(vals == 0)

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
    img = big_numpy_image
    H, W, C = img.shape
    new_height = int(H / down_factor)
    new_width = int(W / down_factor)
    # tensor_image = TT.functional.resize(img, (new_heigth, new_width),
    #                                     interpolation=TT.InterpolationMode.NEAREST)
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return resized_image


def create_CNN_inputs_loop(batch_abs_pixel_coords,
                           tensor_image):

    num_agents = batch_abs_pixel_coords.shape[0]
    H, W, C = tensor_image.shape
    input_traj_maps = list()

    # loop over agents
    for agent_idx in range(num_agents):
        trajectory = batch_abs_pixel_coords[agent_idx, :, :] # .detach().clone().to(torch.device("cpu"))

        traj_map_cnn = make_gaussian_map_patches(
            gaussian_centers=trajectory,
            height=H,
            width=W)
        # append
        input_traj_maps.append(traj_map_cnn)

    # list --> tensor
    input_traj_maps = np.concatenate(input_traj_maps, axis=0)

    return input_traj_maps


def normalize_prob_map(x):
    """Normalize a probability map of shape (B, T, H, W) so
    that sum over H and W equal ones"""
    assert len(x.shape) == 4
    # sums = x.sum(-1, keepdim=True).sum(-2, keepdim=True)
    # x = torch.divide(x, sums)
    sums = np.sum(x, axis=-1, keepdims=True).sum(axis=-2, keepdims=True)
    sums[sums == 0] = 1e-5 # avoid division by zero
    x = np.divide(x, sums)
    
    return x


def un_normalize_prob_map(x):
    """Un-Normalize a probability map of shape (B, T, H, W) so
    that each pixel has value between 0 and 1"""
    assert len(x.shape) == 4
    (B, T, H, W) = x.shape # (1, 20, 320, 320)
    # maxs, _ = x.reshape(B, T, -1).max(-1)
    # x = torch.divide(x, maxs.unsqueeze(-1).unsqueeze(-1))
    maxs = x.reshape(B, T, -1).max(axis=-1)
    maxs[maxs == 0] = 1e-5 # avoid division by zero
    x = np.divide(x, np.expand_dims(np.expand_dims(maxs, axis=-1), axis=-1))
    
    return x


def select_coordinate_pairs(abs_pixel_coord, pred_length, resize_factor=1.0):
    # select obs_coords
    # get earliest finite indices for all agents
    finite_idx = np.isfinite(abs_pixel_coord[:,:,0])
    first_finite_idx = np.argmax(finite_idx, axis=1)
    # select obs_coords
    obs_coords = abs_pixel_coord[np.arange(abs_pixel_coord.shape[0]), first_finite_idx, :]
    # get latest finite indices for all agents
    last_finite_idx = abs_pixel_coord.shape[1] - np.argmax(finite_idx[:,::-1], axis=1) - 1
    # if difference between first and last non-inf index is larger than pred_length, use pred_length
    last_finite_idx[last_finite_idx - first_finite_idx >= pred_length] = first_finite_idx[last_finite_idx - first_finite_idx >= pred_length] + pred_length

    # select goal_coords
    goal_coords = abs_pixel_coord[np.arange(abs_pixel_coord.shape[0]), last_finite_idx, :]

    return (obs_coords / resize_factor).round(), (goal_coords / resize_factor).round()