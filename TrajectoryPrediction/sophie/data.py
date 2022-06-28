import os
import math
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, Dataset

from constants import *
from utils import Backbone

def read_file(path, delim='\t'):
    data = []
    assert os.path.isfile(path)
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def collate(data):
    obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, vgg_list = zip(*data)
    vgg_list = [vgg_per_seq.repeat(obs_seq[idx].size(0), 1, 1, 1) for idx, vgg_per_seq in enumerate(vgg_list)]
    obs_seq = torch.cat(obs_seq, dim=0).permute(3, 0, 1, 2) # IN: list(torch.tensor(n_ped_per_seq, 64, 2, 8)) of len = batch size, OUT: torch.tensor(8, batch_size*n_ped_per_seq, 64, 2)
    pred_seq = torch.cat(pred_seq, dim=0).permute(2, 0, 1) # IN: list(torch.tensor(n_ped_per_seq, 2, 12)) of len = batch size, OUT: torch.tensor(12, batch_size*n_ped_per_seq, 2)
    obs_seq_rel = torch.cat(obs_seq_rel, dim=0).permute(3, 0, 1, 2)
    pred_seq_rel = torch.cat(pred_seq_rel, dim=0).permute(2, 0, 1)
    vgg_list = torch.cat(vgg_list, dim=0)
    # vgg_list = vgg_list.repeat(obs_seq.size(1), 1, 1) # IN: torch.tensor(batch_size*225x512), OUT: torch.tensor(batch_size*n_ped_per_seq x batch_size*225 x 512), I guess for each ped the same env.
    return tuple([obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, vgg_list])

def data_loader(path, split):
    dset = TrajDataset(path, split)
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)
    return dset, loader

class TrajDataset(Dataset):
    def __init__(self, data_dir, split):

        super(TrajDataset, self).__init__()
        if split=='train':
            all_files = [os.path.join(data_dir, f'variation_{idx}.txt') for idx in range(5)]
            all_files = [os.path.join(data_dir, f'variation_{idx}.txt') for idx in range(1)]
            delim = ','
        elif split=='val':
            all_files = [os.path.join(data_dir, 'variation_5.txt')]
            delim = ','
        elif split=='old':
            all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path[0] != "." and path.endswith(".txt")]
            delim = '\t'
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_len = OBS_LEN + PRED_LEN
        fet_map = {}
        fet_list = []
        backbone = Backbone()

        for path in all_files:
            data = read_file(path, delim=delim) # frame, pedId, x, y
            frames = np.unique(data[:, 0]).tolist()

            hkl_path = os.path.splitext(path)[0] + ".pkl"
            # with open(hkl_path, 'rb') as handle:
            #     new_fet = pickle.load(handle, encoding='bytes')
            # fet_map[hkl_path] = torch.from_numpy(new_fet)
            if split == 'old':
                fet_map[hkl_path] = torch.rand(1, 15, 15, 512)
            else:
                fet_map[hkl_path] = backbone.forward_pass(path)

            frame_data = [data[frame == data[:, 0], :] for frame in frames] # restructure data into list with frame indices
            num_sequences = len(frames) - seq_len + 1
            for frame_idx in range(0, num_sequences+1):

                curr_seq_data = np.concatenate(frame_data[frame_idx:frame_idx + seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, seq_len)) # to save (x,y)-changes of all considered pedestrians in the current sequence
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len)) # to save (x,y) of all considered pedestrians in the current sequence
                
                num_peds_considered = 0
                # Iterate over all pedestrians in current sequence (sequence = observed + predicted frames)
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # Get start and end frames of pedestrians within current sequence
                    pad_front = frames.index(curr_ped_seq[0, 0]) - frame_idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - frame_idx + 1
                    if pad_end - pad_front != seq_len:
                        # Ignore pedestrians that are not present throughout current sequence length
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) # np.array 2x20: (x, y)
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # Get relative position changes by subtracting previous from subsequent locations
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    curr_seq[num_peds_considered, :, pad_front:pad_end] = curr_ped_seq # curr_seq = np.array: (num_ped_in_curr_seq, 2, len_seq) == coord trajectories of considered pedestrians
                    curr_seq_rel[num_peds_considered, :, pad_front:pad_end] = rel_curr_ped_seq
                    num_peds_considered += 1
                # TODO check if MAX_PED = 64 is really necessary, how often are 64 pedestriance per sequence really reached? -> maybe even variable size for actual peds in sequence can be used?
                if num_peds_considered > 1:
                    num_peds_in_seq.append(num_peds_considered)

                    curr_seq_exp = np.zeros((num_peds_considered, MAX_PEDS, 2, seq_len))
                    curr_seq_rel_exp = np.zeros((num_peds_considered, MAX_PEDS, 2, seq_len))
                    for i in range(num_peds_considered):
                        curr_seq_exp[i, 0, :, :] = curr_seq[i] # get entry of current considered ped
                        curr_seq_exp[i, 1:i+1, :, :] = curr_seq[0:i] # get all entries before considered ped
                        curr_seq_exp[i, i+1:num_peds_considered, :, :] = curr_seq[i+1:num_peds_considered] # get all entries after considered ped

                        dists = (curr_seq_exp[i, :] - curr_seq_exp[i, 0]) ** 2 # calc distance array for each considered ped to current considered ped's locations, ATTENTION: non-considered peds in current sequence have zero values and thus a distance too!
                        dists = np.sum(np.sum(dists, axis=2), axis=1) # calc sum of distance for each frame in current sequence for each considered ped to current considered ped's locations
                        idxs = np.argsort(dists) # TODO what if zero values are closer to traj than other peds? Then zeros are stored after current ped's coords, and other ped's trajectories are stored last... 
                        curr_seq_exp[i, :] = curr_seq_exp[i, :][idxs] # sort according to increasing distance from current considered ped
                        # Do the same for relative distance changes
                        curr_seq_rel_exp[i, 0, :, :] = curr_seq_rel[i]
                        curr_seq_rel_exp[i, 1:i+1, :, :] = curr_seq_rel[0:i]
                        curr_seq_rel_exp[i, i+1:num_peds_considered, :, :] = curr_seq_rel[i+1:num_peds_considered]
                        curr_seq_rel_exp[i, :] = curr_seq_rel_exp[i, :][idxs]

                    seq_list.append(curr_seq_exp[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel_exp[:num_peds_considered])
                    fet_list.append(hkl_path)

        # seq_list = list of coordinates of considered peds per sequence repeated in different orders sorted by distance to each considered ped trajectory
        # Example in one sequence with three considered agents: append to seq_list: [(x1,y1), (x2,y2), (x3,y3)], [(x2,y2), (x1,y1), (x3,y3)], [(x3,y3), (x2,y2), (x1,y1)] --> appending each list entry sorted by distance to first traj/coords
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0) # Each entry starts with the current ped's coords and then the other considered ped's coords in the sequence (until zeros) sorted by their distance to the considered ped
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :, :OBS_LEN]).type(torch.float) # considering also the neighboring peds in the sequence
        self.pred_traj = torch.from_numpy(
            seq_list[:, 0, :, OBS_LEN:]).type(torch.float) # not considering also the neighboring peds in the sequence
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :, :OBS_LEN]).type(torch.float) # considering also the neighboring peds in the sequence
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, 0, :, OBS_LEN:]).type(torch.float) # not considering also the neighboring peds in the sequence
        # IMPORTANT: this picks observed and future trajectory in each sequence automatically ==> [........ |<- OBS, FUTURE ->| ............], and selects current point in time thus automatically 

        self.fet_map = fet_map
        self.fet_list = fet_list

        # number of considered pedestrians per sequence indexing
        cumulative_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cumulative_start_idx, cumulative_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index] # number of considered peds in indexed sequence
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.fet_map[self.fet_list[index]]
        ] # [(end-start)x64x2x8, (end-start)x2x12, (end-start)x64x2x8, (end-start)x2x12, 1x15x15x512]
        return out
