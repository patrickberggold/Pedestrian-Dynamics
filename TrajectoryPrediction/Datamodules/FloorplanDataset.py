import os
from torch.utils.data import Dataset
import numpy as np
from helper import SEP, OPSYS, PREFIX
import matplotlib.pyplot as plt
import albumentations as A
import random
import torch
from Datamodules.SceneLoader import Scene_floorplan
import warnings
import torchvision.transforms.functional as F
from tqdm import tqdm
import sparse
import time

def is_legitimate_traj(traj_df, step):
    agent_id = traj_df.agent_id.values
    # check if I only have 1 agent (always the same)
    if not (agent_id[0] == agent_id).all():
        print("not same agent")
        return False
    frame_ids = traj_df.frame_id.values
    equi_spaced = np.arange(frame_ids[0], frame_ids[-1] + 1, step, dtype=int)
    if len(frame_ids) != len(equi_spaced):
        raise ValueError
    # check that frame rate is evenly-spaced
    if not (frame_ids == equi_spaced).all():
        print("not equi_spaced")
        return False
    # if checks are passed
    return True


class Dataset_Seq2Seq(Dataset):

    def __init__(self, config: dict, img_list: list, csv_list: list, split: str, transforms = {}):
        # TODO implement more sophisticated data loading from AgentFormer
        
        self.arch = config['arch']

        self.data_format = config['data_format']
        self.traj_quantity = config['traj_quantity']
        assert self.data_format in ['random', 'by_frame', 'tokenized']
        assert self.traj_quantity in ['pos', 'vel']
        
        self.batch_size = 16 if self.arch == 'goal' else 1000 # 203 vs 1482 can I use the same batch size in the original tf? if not, some mistake... --> uses 10 GB for BS=1000
        self.dataset_limit = None
        
        self.seq_length = config['seq_length']
        
        self.split = split
        
        # self.data_augmentation = True if split == 'train' else False
        self.data_augmentation = True

        # self.dataset_folder = self.dataset_folder_rgb # os.path.join(self.dataset_folder_rgb, scene_name.split(SEP)[0], scene_name.split(SEP)[1])
        # self.scene_folder = os.path.join(self.dataset_folder, self.name)

        if not config['read_from_pickle']:
            self.scenes = {i: Scene_floorplan(img_path, csv_path, verbose=False) for  i, (img_path, csv_path) in enumerate(tqdm(zip(img_list, csv_list), desc=f'[stage=={self.split}]\tLoading scenes...', total=len(img_list)))}# i, scene_name in enumerate(img_list)}

        self.transforms = transforms['transforms']
        self.img_size = transforms['feature_extractor_size']
        
        assert self.img_size == 640, 'Resizing is necessary, either via torchvision (in Datamodule) or Albumentations (here in Dataset)!'
        
        # handcrafted for now...
        self.overall_mean = 320
        self.overall_std = 75
        self.normalize_dataset = config['normalize_dataset']

        # Normalize coordinate system
        # assert self.scenes[0].image_res_x == self.scenes[0].image_res_y
        # self.shift_cs = self.scenes[0].image_res_x // 2

        # self.final_resolution = 640
        self.max_floorplan_meters = 70
        
        self.dataset_statistics = {
            'corr_e2e': {
                'mean': [300.6625 , 319.4248],
                'std': [103.7613 , 23.8261],
            },
            'corr_cross': {
                'mean': [318.6957 , 319.7076],
                'std': [116.2994 , 27.9643],
            },
            'train_station': {
                'mean': [315.6758, 311.8410],
                'std': [111.2399, 65.4355],
            }
        }
        means = [stat_dict['mean'] for stat_dict in self.dataset_statistics.values()]
        stds = [stat_dict['std'] for stat_dict in self.dataset_statistics.values()]

        # self.overall_mean = np.stack(means).mean(0).mean() # not accounted yet for rotations...
        # self.overall_std = np.stack(stds).mean(0).mean() # not accounted yet for rotations...

        self.tokens = ['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[TRAJ]', '[UNK]']
        self.max_pads_per_seq = 5
        self.fill_value = np.inf 

        self.numerical_tokenization = False # REGRESSION TRANSFORMER: CONCURRENT CONDITIONAL GENERATION AND REGRESSION BY BLENDING NUMERICAL AND TEXTUAL TOKENS --> x & y to one or two tokens

        if config['read_from_pickle']:
            from pickle import Unpickler
            from helper import TQDMBytesReader
            store_path = SEP.join(['TrajectoryPrediction', 'Datamodules', f'dataset_{split}_whole_{self.data_format}_seq{self.seq_length}.pickle'])
            print(f'Loading {split} dataset from pickle ({store_path})...')
            with open(store_path, 'rb') as handle:
                total = os.path.getsize(store_path)
                # self.sequence_list = pickle.load(handle)
                with TQDMBytesReader(handle, total=total) as pbhandle:
                    self.sequence_list =  Unpickler(pbhandle).load()
                print(f'Loaded {split} dataset!')
            if OPSYS=='Linux':
                for item in self.sequence_list:
                    path_tokens = item['scene_path'].split('\\')
                    path_tokens = [PREFIX]+path_tokens[1:]
                    item['scene_path'] = SEP.join(path_tokens) 
        elif self.data_format == 'by_frame':
            self.read_by_frame_trajectory_loader()
        elif self.data_format == 'random':
            self.read_random_trajectory_loader()
        elif self.data_format == 'tokenized':
            self.read_tokenized_trajectory_loader()
        else:
            raise NotImplementedError

        MAX_TRAJ_PER_BATCH = 3000
        # Chunking batches to comply with maximum trajectories per batch (OOM issues...)
        if MAX_TRAJ_PER_BATCH:
            # print('S')
            sequence_list = []
            for item in tqdm(self.sequence_list, desc=f'[Chunking batches to maximum {MAX_TRAJ_PER_BATCH} sequence length]...'):
                num_sequences = len(item['batch_of_sequences'])
                if num_sequences <= MAX_TRAJ_PER_BATCH:
                    sequence_list += [item]
                else:
                    sequences_per_scene = []
                    if self.data_format == 'tokenized': tokens_per_scene = []
                    
                    for x in range(0, num_sequences, MAX_TRAJ_PER_BATCH):
                        sequences_per_scene += [item['batch_of_sequences'][x:x+MAX_TRAJ_PER_BATCH]]
                        if self.data_format == 'tokenized': tokens_per_scene += [item['tokenized_sequences'][x:x+MAX_TRAJ_PER_BATCH]]
                    
                    for ids, sequence_chunk in enumerate(sequences_per_scene):
                        add_item = [{
                            'batch_of_sequences': sequence_chunk,
                            'scene_path': item['scene_path']
                        }]
                        if self.data_format == 'tokenized': add_item[0]['tokenized_sequences'] = tokens_per_scene[ids]
                        sequence_list += add_item

            self.sequence_list = sequence_list
        
        warnings.filterwarnings("ignore")


    def tokenize(self, token):
        return self.tokens.index(token)


    def read_by_frame_trajectory_loader(self):
        # Based on SoPhie
        seq_list_pos = []
        seq_list_vel = []
        self.sequence_list = []
        considered_peds_per_scene = []
        total_num_sequences = 0

        self.statistics_per_scene = {}

        for sc_id, scene in tqdm(self.scenes.items(), desc=f'[stage=={self.split}]\tPreprocessing trajectories...'):
            raw_pixel_data = scene.raw_pixel_data.to_numpy(dtype=object)

            frames = np.unique(raw_pixel_data[:, 0]).tolist()
            frame_data = [raw_pixel_data[frame == raw_pixel_data[:, 0], :] for frame in frames] # restructure data into list with frame indices
            # quick check whether loaded correctly
            assert raw_pixel_data.shape[0] == sum([frame_datum.shape[0] for frame_datum in frame_data])
            num_sequences = max(frames) - self.seq_length + 1

            trajectory_data = {'x': [], 'y': []}
            considered_peds_per_frame = []

            for frame_idx in range(0, num_sequences + 1):
                curr_seq_data = np.concatenate(frame_data[frame_idx:frame_idx + self.seq_length], axis=0)
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])

                agent_data_per_frame = []
                # considered_peds_per_frame = []

                num_agents_considered = 0
                # Iterate over all pedestrians in current sequence (sequence = observed + predicted frames)
                for _, agent_id in enumerate(agents_in_curr_seq):
                    seq_data_per_agent = curr_seq_data[curr_seq_data[:, 1] == agent_id, :]
                    # seq_data_per_agent = np.around(seq_data_per_agent, decimals=4)
                    # Get start and end frames of pedestrians within current sequence
                    first_frame_in_seq = frames.index(seq_data_per_agent[0, 0]) - frame_idx
                    last_frame_in_seq = frames.index(seq_data_per_agent[-1, 0]) - frame_idx + 1
                    if last_frame_in_seq - first_frame_in_seq != self.seq_length:
                        # Ignore pedestrians that are not present throughout current sequence length
                        continue
                    # agent_data_per_frame.append(seq_data_per_agent)
                    agent_traj_data = np.array(seq_data_per_agent[:, -2:], dtype=np.float32)

                    # apply mean shift and std
                    # if self.traj_quantity == 'pos':
                    #     agent_traj_data = (agent_traj_data - self.overall_mean) / self.overall_std
                    
                    if self.traj_quantity == 'pos':
                        agent_data_per_frame.append({
                            'abs_pixel_coord': agent_traj_data
                        })
                        trajectory_data['x'].append(np.array(seq_data_per_agent[:, -2]).astype(np.float32))
                        trajectory_data['y'].append(np.array(seq_data_per_agent[:, -1]).astype(np.float32))
                    elif self.traj_quantity == 'vel':
                        seq_vel_per_agent = np.zeros(agent_traj_data.shape)
                        seq_vel_per_agent[0, :] = agent_traj_data[0, :]
                        # TODO prepend zeros somewhere?
                        seq_vel_per_agent[1:, :] = agent_traj_data[1:, :] - agent_traj_data[:-1, :]

                        agent_data_per_frame.append({
                            'abs_pixel_coord': seq_vel_per_agent
                        })
                        trajectory_data['x'].append(np.array(seq_vel_per_agent[1:, -2]).astype(np.float32))
                        trajectory_data['y'].append(np.array(seq_vel_per_agent[1:, -1]).astype(np.float32))

                    # seq_pos_per_agent = np.transpose(seq_data_per_agent[:, 2:]) # np.array 2x20: (x, y)
                    # seq_vel_per_agent = np.zeros(seq_pos_per_agent.shape)
                    # # Get relative position changes by subtracting previous from subsequent locations
                    # seq_vel_per_agent[:, 1:] = seq_pos_per_agent[:, 1:] - seq_pos_per_agent[:, :-1]
                    # curr_seq_pos[num_agents_considered, :, first_frame_in_seq:last_frame_in_seq] = seq_pos_per_agent # curr_seq = np.array: (num_ped_in_curr_seq, 2, len_seq) == coord trajectories of considered pedestrians
                    # curr_seq_vel[num_agents_considered, :, first_frame_in_seq:last_frame_in_seq] = seq_vel_per_agent
                    num_agents_considered += 1
                    total_num_sequences += 1
                considered_peds_per_frame.append(num_agents_considered)

                if len(agent_data_per_frame) > 0:
                    self.sequence_list += [{
                        'batch_of_sequences': agent_data_per_frame,
                        'scene_path': scene.RGB_image_path,
                        'scene_data': {
                            'scene_id': sc_id,
                            # 'image_res_x': scene.image_res_x,
                            # 'image_res_y': scene.image_res_y,
                            # 'floorplan_min_x': scene.floorplan_min_x,
                            # 'floorplan_min_y': scene.floorplan_min_y,
                            # 'floorplan_max_x': scene.floorplan_max_x,
                            # 'floorplan_max_y': scene.floorplan_max_y,
                            },
                        'type': self.traj_quantity
                    }]
            # Calculate mean and std per x & y per scene
            statistics_in_scene = {}
            for key, arrays_list in trajectory_data.items():
                traj_data = np.concatenate(arrays_list, axis=0)
                mean = traj_data.mean()
                std = traj_data.std()
                statistics_in_scene.update({key: {'mean': mean, 'std': std}})
            
            self.statistics_per_scene.update({scene.RGB_image_path.split(SEP)[-1]: statistics_in_scene})
        
            considered_peds_per_scene.append(considered_peds_per_frame)
        # considered_peds.append(considered_peds_per_scene)
        if self.dataset_limit != None: 
            self.sequence_list = self.sequence_list[:self.dataset_limit]
    
    
    def read_random_trajectory_loader(self):
        """ Based on Goal-SAR """
        sequences_per_scene = []
        self.sequence_list = []

        sequence_id = 0
        for sc_id, scene in tqdm(self.scenes.items(), desc=f'[stage=={self.split}]\tPreprocessing trajectories...'):
            raw_pixel_data = scene.raw_pixel_data

            sequences_per_scene = []
            for agent_i in set(raw_pixel_data.agent_id):
                raw_agent_data = raw_pixel_data[raw_pixel_data.agent_id == agent_i]
                # downsample frame rate happens here, at the single agent level
                raw_agent_data = raw_agent_data.iloc[::1]

                for start_t in range(0, len(raw_agent_data)):
                    candidate_traj = raw_agent_data.iloc[start_t:start_t + self.seq_length]
                    if len(candidate_traj) == self.seq_length:
                        if is_legitimate_traj(candidate_traj, step=1):

                            agent_traj = np.array(candidate_traj[["x_coord", "y_coord"]].values, dtype=np.float32)
                            
                            # # apply mean shift and std
                            # if self.traj_quantity == 'pos':
                            #     agent_traj = (agent_traj - self.overall_mean) / self.overall_std

                            if self.traj_quantity=='pos':
                                agent_seq_data = agent_traj
                            elif self.traj_quantity=='vel':
                                agent_seq_data = np.zeros_like(agent_traj)
                                agent_seq_data[0, :] = agent_traj[0, :]
                                # TODO prepend zeros somewhere?
                                agent_seq_data[1:, :] = agent_traj[1:, :] - agent_traj[:-1, :]
                            else:
                                raise NotImplementedError

                            sequences_per_scene.append(agent_seq_data)

                            sequence_id += 1

            random.shuffle(sequences_per_scene)

            self.sequence_list += [{
                'batch_of_sequences': sequences_per_scene,
                'scene_path': scene.RGB_image_path,
                }]

            """ self.sequence_list += [{
                'batch_of_sequences': sequences_per_scene[x:x+self.batch_size],
                'scene_path': scene.RGB_image_path,
                'scene_data': {
                    'scene_id': sc_id,
                    # 'image_res_x': scene.image_res_x,
                    # 'image_res_y': scene.image_res_y,
                    # 'floorplan_min_x': scene.floorplan_min_x,
                    # 'floorplan_min_y': scene.floorplan_min_y,
                    # 'floorplan_max_x': scene.floorplan_max_x,
                    # 'floorplan_max_y': scene.floorplan_max_y,
                    },
                'type': 'pos',
                } for x in range(0, len(sequences_per_scene), self.batch_size)] """

        if self.dataset_limit != None: 
            self.sequence_list = self.sequence_list[:self.dataset_limit]


    def read_tokenized_trajectory_loader(self):
        
        self.sequence_list = []

        sequence_id = 0
        for sc_id, scene in tqdm(self.scenes.items(), desc=f'[stage=={self.split}]\tPreprocessing trajectories...'):
            raw_pixel_data = scene.raw_pixel_data

            frame_ids = np.sort(np.unique(raw_pixel_data.frame_id.values))
            start_frame_id = frame_ids[0]
            end_frame_id = frame_ids[-1]

            # expand frames from both sides for BOS and EOS tokens
            start_frame_id -= 1
            end_frame_id += 1
            all_frames = np.arange(start_frame_id, end_frame_id+1)

            sequences_per_scene = []
            # TODO store entire trajectories of variable length

            for agent_i in set(raw_pixel_data.agent_id):
                raw_agent_data = raw_pixel_data[raw_pixel_data.agent_id == agent_i]
                # downsample frame rate happens here, at the single agent level
                raw_agent_data = raw_agent_data.to_numpy(dtype=np.float32)

                agent_presence_frames = raw_agent_data[:, 0].astype(int)
                coord_data = raw_agent_data[:, 2:].astype(np.float32)

                agent_token_frames = []
                bos_it = 0
                eos_it = 0
                for index in all_frames:
                    if index in agent_presence_frames:
                        agent_token_frames.append(self.tokenize('[TRAJ]'))
                    elif index==(agent_presence_frames[0]-1):
                        agent_token_frames.append(self.tokenize('[BOS]'))
                        bos_it += 1
                    elif index==(agent_presence_frames[-1]+1):
                        agent_token_frames.append(self.tokenize('[EOS]'))
                        eos_it += 1
                    elif index < (agent_presence_frames[0]-1) or index > (agent_presence_frames[-1]+1):
                        agent_token_frames.append(self.tokenize('[PAD]'))
                    else:
                        raise IndexError('Bug or trajectory is interrupted!')

                bos_index = np.where(np.array(agent_token_frames) == self.tokenize('[BOS]'))[0]
                eos_index = np.where(np.array(agent_token_frames) == self.tokenize('[EOS]'))[0]

                coord_data = np.pad(coord_data, ((int(bos_index)+1, len(agent_token_frames)-int(eos_index)), (0, 0)), 'constant', constant_values=(self.fill_value, self.fill_value))

                # Some checking
                assert bos_it == 1 and eos_it == 1, f'bos_it == {bos_it} and eos_it == {eos_it}, both need to be one!'
                assert len(all_frames) == len(agent_token_frames) == len(coord_data)
                # assert np.all(coord_data[np.argwhere(np.array(agent_token_frames) != self.tokenize('[TRAJ]')).squeeze()] == self.fill_value), f'Assertion that all non-[TRAJ] are assigned with {self.fill_value} is violated!'
                # assert np.all(coord_data[np.argwhere(np.array(agent_token_frames) == self.tokenize('[TRAJ]')).squeeze()] != self.fill_value), 'Assertion that all [TRAJ] are assigned with true coords is violated!'

                for start_t in range(0, len(all_frames)-self.seq_length):
                    seq_tokens = agent_token_frames[start_t:start_t + self.seq_length]                  
                    if np.count_nonzero(np.array(seq_tokens) == self.tokenize('[PAD]')) > self.max_pads_per_seq:
                        # sequences cannot contain more than 3 PADS
                        continue

                    seq_coords = coord_data[start_t:start_t + self.seq_length]
                    sequences_per_scene.append([seq_tokens, seq_coords])

            random.shuffle(sequences_per_scene)

            sequence_tokens =  [agent_data[0] for agent_data in sequences_per_scene]
            sequences_coords = [agent_data[1] for agent_data in sequences_per_scene]

            self.sequence_list += [{
                'batch_of_sequences': sequences_coords,
                'tokenized_sequences': sequence_tokens,
                'scene_path': scene.RGB_image_path,
                }]

        hello = 1


    def __len__(self):
        return len(self.sequence_list)


    def augment_traj_and_images_sparse(self, trajectories, np_image, augmentation):

        # keypoints = list(map(tuple, trajectories.reshape(-1, 2)))
        keypoints = trajectories.reshape(-1, 2)
        all_finites = np.isfinite(keypoints[:,0])
        # keypoint_mask = keypoints[:, 0] != np.nan

        # flipping, transposing, random 90 deg rotations
        transform = A.Compose([
            A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
            A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
            A.augmentations.geometric.transforms.Transpose(p=0.5),
            A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
            ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        #################### TEST START ####################
        # self.visualize_traj(keypoints, np_image)
        #################### TEST END ####################
        
        transformed = transform(image=np_image, keypoints=keypoints)

        transformed_trajectories = np.array(transformed['keypoints'])
        # apply mean shift and std
        if self.traj_quantity == 'pos' and self.normalize_dataset:
            transformed_trajectories = (transformed_trajectories - self.overall_mean) / self.overall_std

        # #################### TEST START ####################
        # self.visualize_traj(np.array(transformed['keypoints']), transformed['image'])
        # #################### TEST END ####################

        image = torch.tensor(transformed['image']).permute(2, 0, 1).float()
        trajectories_t = torch.tensor(transformed_trajectories).float()
        trajectories_t = trajectories_t.view(trajectories.shape)

        return [image, trajectories_t]

    
    def visualize_traj(self, trajectories_nx2, np_image, max_traj_vis: int = 20):
        from skimage.draw import line
        # plt.imshow(np_image) # weg
        traj = np.array(trajectories_nx2)
        np_image_draw = np_image.copy()
        # traj = [t for t in traj]
        trajs = [traj[x:x+self.seq_length] for x in range(0, len(traj), self.seq_length)]
        # for t_id in traj:
        for traj in trajs[:max_traj_vis]:
            old_point = None
            r_val = random.uniform(0.4, 1.0)
            b_val = random.uniform(0.7, 1.0)
            for point in traj:
                if np.isinf(point[0]) and np.isinf(point[1]):
                    continue
                x_n, y_n = round(point[0]), round(point[1])
                assert 0 <= x_n <= np_image.shape[1] and 0 <= y_n <= np_image.shape[0], f'{x_n} > {np_image.shape[1]} or {y_n} > {np_image.shape[0]}'
                if old_point != None:
                    # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
                    c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, y_n)))]
                    for c in c_line:
                        np_image_draw[c[1], c[0]] = np.array([r_val, 0., b_val])
                    # plt.imshow(img_np)
                old_point = (x_n, y_n)

        plt.imshow(np_image_draw)
        plt.close('all')


    def __getitem__(self, idx):
        
        # LOAD CSV DATA
        sequence = self.sequence_list[idx] # 'scene_path': corr_cross\\0__floorplan_siteX_35_siteY_20_CORRWIDTH_3_SIDECORRWIDTH_20_numCross_2\\variation_56\\variation_56_num_agents_15.npz'

        abs_pixel_coord = np.stack(sequence['batch_of_sequences'], axis=0)
        if self.data_format == 'tokenized':
            tokenized_sequences = np.stack(sequence['tokenized_sequences'], axis=0)
       
        np_image = sparse.load_npz(sequence['scene_path']).todense()
        np_image = np_image.astype(np.float32) / 255. # normalization takes place down below....
        # TODO only way with this dataset for quick float 0. - 1. conversion, but light or dark area do not matter at all, only [1 0 0] or [0 1 0]
        
        # batch_data = self.augment_traj_and_create_traj_maps(batch_data, np_image, self.data_augmentation)
        batch_data = self.augment_traj_and_images_sparse(abs_pixel_coord, np_image, self.data_augmentation)

        if self.transforms:
            batch_data[0] = self.transforms(batch_data[0])

        if self.data_format == 'tokenized':
            batch_data.append(tokenized_sequences)
        
        return batch_data