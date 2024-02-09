import platform
from skimage.draw import line
from prettytable import PrettyTable
import os
import pytorch_lightning as pl
import torch
from collections import OrderedDict
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import os

OPSYS = platform.system()
SEP = '\\' if OPSYS == 'Windows' else '/'
PREFIX = '/mnt/c' if OPSYS == 'Linux' else 'C:'

# Check intermediate layers and sizes
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    table.reversesort = True
    layers = model.named_parameters()
    for name, parameter in layers:
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table.get_string(sortby="Parameters"))
    # print(table)
    table.sortby = "Parameters"
    print(f"Total Trainable Params: {total_params}")
    return total_params


def linemaker(p_start, p_end, thickness=1):
    x_start, x_end, y_start, y_end = p_start[0], p_end[0], p_start[1], p_end[1]
    
    x_diff, y_diff = x_end - x_start, y_end -y_start
    m = y_diff / x_diff if x_diff != 0 else 10.

    lines = []
    or_line = [coord for coord in zip(*line(*(p_start[0], p_start[1]), *(p_end[0], p_end[1])))]
    lines += [[coord] for coord in zip(*line(*(p_start[0], p_start[1]), *(p_end[0], p_end[1])))]

    line_factors = []
    for i in range(thickness-1):
        sign = 1
        if i%2 != 0:
            sign = -1
        i = int(i/2.)+1
        # th=2: +1, th=3: [+1,-1], th=4: [+1,-1,+2]
        line_factors.append(sign*i)

    for factor in line_factors:

        if abs(m) > 1:
            extra_line = list(zip(*line(*(p_start[0]+factor, p_start[1]), *(p_end[0]+factor, p_end[1]))))
            # extra_line = list(zip(*_line_profile_coordinates((x_start+1, y_start), (x_end+1, y_end), linewidth=1)))
        else:
            extra_line = list(zip(*line(*(p_start[0], p_start[1]+factor), *(p_end[0], p_end[1]+factor))))
            # extra_line = list(zip(*_line_profile_coordinates((x_start, y_start+1), (x_end, y_end+1), linewidth=1)))
        # lines += extra_line
        for idx in range(len(lines)):
            lines[idx].append(extra_line[idx])

        # check if all points are offsetted correctly
        for c_line_or, c_line_ex in zip(or_line, extra_line):
            if sum(c_line_ex) != sum(c_line_or)+factor:
                hi = 1

    return lines


class TQDMBytesReader(object):
    """ from https://stackoverflow.com/questions/30611840/pickle-dump-with-progress-bar """
    def __init__(self, fd, **kwargs):
        self.fd = fd
        from tqdm import tqdm
        self.tqdm = tqdm(**kwargs)

    def read(self, size=-1):
        bytes = self.fd.read(size)
        self.tqdm.update(len(bytes))
        return bytes

    def readline(self):
        bytes = self.fd.readline()
        self.tqdm.update(len(bytes))
        return bytes

    def __enter__(self):
        self.tqdm.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self.tqdm.__exit__(*args, **kwargs)


def load_module_from_checkpoint(module: pl.LightningModule, ckpt_folder, ckpt_base=SEP.join(['TrajectoryPrediction', 'Modules', 'coca', 'checkpoints'])):
    CKPT_PATH = os.sep.join([ckpt_base, ckpt_folder])
    model_file_path = [file for file in os.listdir(CKPT_PATH) if file.endswith('.ckpt') and not file.startswith('last')]
    assert len(model_file_path) == 1
    CKPT_PATH = SEP.join([CKPT_PATH, model_file_path[0]])
    state_dict = torch.load(CKPT_PATH)['state_dict']
    module_state_dict = module.state_dict()

    mkeys_missing_in_loaded = [module_key for module_key in list(module_state_dict.keys()) if module_key not in list(state_dict.keys())]
    lkeys_missing_in_module = [loaded_key for loaded_key in list(state_dict.keys()) if loaded_key not in list(module_state_dict.keys())]
    assert len(mkeys_missing_in_loaded) < 10 or len(lkeys_missing_in_module) < 10, 'Checkpoint loading went probably wrong...'

    load_dict = OrderedDict()
    for key, tensor in module_state_dict.items():
        if key in state_dict.keys() and tensor.size()==state_dict[key].size():
            load_dict[key] = state_dict[key]
        else:
            # if key == 'model.model.classifier.classifier.weight':
            #     load_dict[key] = state_dict['model.model.classifier.weight']
            # else:
            #     load_dict[key] = tensor
            load_dict[key] = tensor

    module.load_state_dict(load_dict)
    return module


def visualize_trajectories(gt_trajectories, pred_trajectories, np_image, obs_length: int, save_image: bool = False, save_path = None):
    # plt.imshow(np_image) # weg
    gt_trajectories_np = np.array(gt_trajectories)
    pred_trajectories_np = np.array(pred_trajectories) if pred_trajectories is not None else np.zeros_like(gt_trajectories_np)
    np_image_draw = np_image.copy()
    # traj = [t for t in traj]
    if len(gt_trajectories_np.shape) == 3 and gt_trajectories_np.shape[-1]==2:
        gt_trajectories_list, pred_trajectories_list = [], []
        obs_trajectories_list = []
        # append all observed trajectories
        for t in gt_trajectories_np[:, :obs_length]:
            finite_mask = np.isfinite(t[:, 0])
            obs_trajectories_list.append(np.array([t[finite_mask, 0], t[finite_mask, 1]]).transpose(1,0))
        
        # append all future and predicted trajectories
        assert len(gt_trajectories_np[:, obs_length:]) == len(pred_trajectories_np)
        for t, p in zip(gt_trajectories_np[:, obs_length:], pred_trajectories_np):
            finite_mask_gt = np.isfinite(t[:, 0])
            gt_trajectories_list.append(np.array([t[finite_mask_gt, 0], t[finite_mask_gt, 1]]).transpose(1,0))
            finite_mask_pred = np.isfinite(p[:, 0])
            pred_trajectories_list.append(np.array([p[finite_mask_pred, 0], p[finite_mask_pred, 1]]).transpose(1,0))

        # trajs = [np.array([t[np.isfinite(t[:, 0]), 0], t[np.isfinite(t[:, 0]), 1]]).transpose(1,0) for t in gt_trajectories_np[:, :obs_length]]
    else:
        raise NotImplementedError

    gt_color = (0., 0., 1.0) # blue is ground truth
    pred_color = (0., 1.0, 1.0) # purple is prediction
    # for t_id in traj:
    for obs_traj, gt_traj, pred_traj in zip(obs_trajectories_list, gt_trajectories_list, pred_trajectories_list):
        assert len(pred_traj)==len(gt_traj)

        # draw observed trajectory
        for i in range(obs_traj.shape[0] - 1):
            start_point_i = (round(obs_traj[i][0]), round(obs_traj[i][1]))
            end_point_i = (round(obs_traj[i + 1][0]), round(obs_traj[i + 1][1]))
            cv2.line(np_image_draw, start_point_i, end_point_i, (0,0,0), thickness=1)

        # draw ground truth connection
        if len(gt_traj) > 0: cv2.line(np_image_draw, (round(obs_traj[-1][0]), round(obs_traj[-1][1])), (round(gt_traj[0][0]), round(gt_traj[0][1])), gt_color, thickness=1)
        # # draw gt trajectory
        if gt_traj.shape[0] > 1:
            for k in range(gt_traj.shape[0] - 1):
                start_point_k = (round(gt_traj[k][0]), round(gt_traj[k][1]))
                end_point_k = (round(gt_traj[k + 1][0]), round(gt_traj[k + 1][1]))
                cv2.line(np_image_draw, start_point_k, end_point_k, gt_color, thickness=1)
            
        if pred_trajectories is not None:
            # draw prediction connection
            if len(pred_traj) > 0: cv2.line(np_image_draw, (round(obs_traj[-1][0]), round(obs_traj[-1][1])), (round(pred_traj[0][0]), round(pred_traj[0][1])), pred_color, thickness=2)
            # draw predicted trajectory
            if pred_traj.shape[0] > 1:
                for l in range(pred_traj.shape[0] - 1):
                    start_point_l = (round(pred_traj[l][0]), round(pred_traj[l][1]))
                    end_point_l = (round(pred_traj[l + 1][0]), round(pred_traj[l + 1][1]))
                    cv2.line(np_image_draw, start_point_l, end_point_l, pred_color, thickness=2)

    if save_image:
        plt.imsave(save_path, np_image_draw)
    else:
        return np_image_draw


def visualize_3d(input):
    if isinstance(input, torch.Tensor):
        input = input.detach().cpu().numpy()
    assert isinstance(input, np.ndarray)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the 3D surface
    x_range = np.arange(0, input.shape[1], 1) # torch.arange(0, height, 1)
    y_range = np.arange(0, input.shape[0], 1) # torch.arange(0, width, 1)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    surface = ax.plot_surface(grid_x, grid_y, input, cmap='viridis')
    plt.close('all')


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    # rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj#.permute(1, 0, 2)


def dir_maker(store_folder_path, description_log, config, train_config):
    if os.path.isdir(store_folder_path):
        print('Path already exists!')
        quit()
    else:
        os.mkdir(store_folder_path)
        with open(os.path.join(store_folder_path, 'description.txt'), 'w') as f:
            f.write(description_log)
            f.write("\n\nCONFIG: {\n")
            for k in config.keys():
                f.write("'{}':'{}'\n".format(k, str(config[k])))
            f.write("}")
            f.write("\n\nTRAIN_CONFIG: {\n")
            for k in train_config.keys():
                f.write("'{}':'{}'\n".format(k, str(train_config[k])))
            f.write("}\n\n")
        f.close()