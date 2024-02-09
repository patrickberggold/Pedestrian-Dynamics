import os
from torch.utils.data import Dataset
import numpy as np
from helper import SEP, OPSYS, PREFIX, visualize_trajectories
import matplotlib.pyplot as plt
import albumentations as A
import random
import torch
from Datamodules.SceneLoader import Scene_floorplan
import warnings
import torchvision.transforms.functional as F
from tqdm import tqdm
import sparse
import cv2
import time
from skimage.transform import resize
from skimage.util.dtype import img_as_bool
from Datamodules.DatasetHelper import add_cnn_maps, create_occupancy_map, select_coordinate_pairs

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

def destination_check(coord_per_agent, destinations):
    if coord_per_agent[0, 0] != np.inf and coord_per_agent[-1, 0] == np.inf:
        # find the last coordinate that is not inf
        last_coord = coord_per_agent[~np.isinf(coord_per_agent[:, 0])][-1]
        # check if it is in the destination
        dest_found = 0
        dest_found_yInv = 0
        assert destinations is not None
        x_coord = last_coord[0]
        y_coord = last_coord[1]
        for dest in destinations:
            if dest[0] <= x_coord <= dest[2] and dest[1] <= y_coord <= dest[3]:
                dest_found += 1
            if dest[0] <= x_coord <= dest[2] and 640-dest[3] <= y_coord <= 640-dest[1]:
                dest_found_yInv += 1

        if dest_found == 0:
            pix_lim = 3
            for dest in destinations:
                if abs(dest[0] - x_coord) <= pix_lim or abs(dest[2] - x_coord) <= pix_lim or abs(dest[1] - y_coord) <= pix_lim or abs(dest[3] - y_coord) <= pix_lim:
                    dest_found += 1
                    # break
        # if dest_found_yInv == 1:
        #     print('one inverted found')
        # elif dest_found_yInv > 1:
        #     print('more than one inverted found')
        # else: 
        #     print('none inverted found')
        assert dest_found == 1, f'Agent ends in destination but not in any of the destinations!'

class Dataset_Seq2Seq(Dataset):

    def __init__(self, config: dict, img_list: list, csv_list: list, split: str, semanticMap_list = None, globalSemMap_list = None, transforms = None):
        # TODO implement more sophisticated data loading from AgentFormer
        
        self.arch = config['arch']
        self.mode = config['mode']
        self.data_format = config['data_format']
        self.traj_quantity = config['traj_quantity']
        assert self.data_format in ['random', 'by_frame', 'by_frame_masked', 'partial_tokenized_random', 'full_tokenized_random', 'full_tokenized_by_frame', 'partial_tokenized_by_frame']
        assert self.traj_quantity in ['pos', 'vel']
        self.pred_length = config['pred_length']
        self.obs_length = config['num_obs_steps']
        self.seq_length = self.pred_length + self.obs_length
        is_masked = True if self.data_format == 'by_frame_masked' else False
        
        self.split = split
        self.p_augms = 0.5 # 0.0
        self.p_trans = 0.0 # bool(self.p_augms) * 1.0

        self.transforms = transforms['transforms'] if transforms is not None else None
        self.img_size = transforms['feature_extractor_size'] if transforms is not None else None
        # assert self.img_size == 640, 'Resizing is necessary, either via torchvision (in Datamodule) or Albumentations (here in Dataset)!'

        if not config['read_from_pickle']:
            self.scenes = {i: Scene_floorplan(img_path, csv_path, semanticMap_path, glSemMap_path, find_dsts=is_masked, verbose=False) for  i, (img_path, csv_path, semanticMap_path, glSemMap_path) in enumerate(tqdm(zip(img_list, csv_list, semanticMap_list, globalSemMap_list), desc=f'[stage=={self.split}]\tLoading scenes...', total=len(img_list)))}# i, scene_name in enumerate(img_list)}
        
        # handcrafted for now...
        # self.overall_mean = 320
        # self.overall_std = 75
        self.normalize_dataset = config['normalize_dataset']
        self.max_floorplan_meters = 70
        if self.mode == 'GOAL_PRED': 
            self.normalize_dataset = False
        self.newDS = True
        self.original_img_max_size = (800, 1280)
        self.resize_factor = config['resize_factor']
        self.img_max_size = (self.original_img_max_size[0]//self.resize_factor, self.original_img_max_size[1]//self.resize_factor)
        if self.newDS:
            # self.overall_mean, self.overall_std = 1280 // self.resize_factor / 2, 150 / self.resize_factor
            self.overall_mean, self.overall_std = config['ds_mean'], config['ds_std']
            self.pix_to_real = 0.03125
        self.pad_size = 64 // self.resize_factor
        self.window_size = 2*self.pad_size
        self.gaussian_std = 14 / self.resize_factor # (0.44 meters * 32 pix / meter) where 0.44 meters = torso size
        
        self.max_pads_per_seq = 9
        self.mtm_mask_prob = 0.15
        self.fill_value = np.inf

        self.max_traj_per_batch = 1000

        if config['read_from_pickle']:
            from pickle import Unpickler
            from helper import TQDMBytesReader
            store_path = SEP.join(['TrajectoryPrediction', 'Datamodules', f'dataset_{split}_whole_{self.data_format}_seq{self.seq_length}.pickle'])
            print(f'Loading {split} dataset from pickle ({store_path})...')
            with open(store_path, 'rb') as handle:
                total = os.path.getsize(store_path)
                # self.sequence_list = pickle.load(handle)
                assert self.mode == 'TRAJ_PRED' and self.data_format != 'tokenized', 'batch chunking with self.max_traj_per_batch not implemented for tokens yet!'
                with TQDMBytesReader(handle, total=total) as pbhandle:
                    self.sequence_list =  Unpickler(pbhandle).load()
                sequences = []
                for sequence in self.sequence_list:
                    assert isinstance(sequence['batch_of_sequences'], np.ndarray), '__getitem__() expects arrays now...'
                    if len(sequence['batch_of_sequences']) <= self.max_traj_per_batch:
                        sequences += [sequence]
                    else:
                        for x in range(0, len(sequence['batch_of_sequences']), self.max_traj_per_batch):
                            sequences += [{
                                'batch_of_sequences': sequence['batch_of_sequences'][x:x+self.max_traj_per_batch],
                                'scene_path': sequence['scene_path'],
                                }]
                self.sequence_list = sequences
                print(f'Loaded {split} dataset!')
            if OPSYS=='Linux':
                for item in self.sequence_list:
                    path_tokens = item['scene_path'].split('\\')
                    path_tokens = [PREFIX]+path_tokens[1:]
                    item['scene_path'] = SEP.join(path_tokens) 
        elif self.data_format in ['by_frame', 'by_frame_masked']:
            assert self.mode in ['TRAJ_PRED', 'GOAL_PRED', 'SEGMENTATION'], 'by_frame loading is only suitable for TRAJ_PRED mode!'
            self.read_by_frame_trajectory_loader(is_masked='masked' in self.data_format)
        elif self.data_format == 'random':
            assert self.mode == 'TRAJ_PRED', 'random loading is only suitable for TRAJ_PRED mode!'
            self.read_random_trajectory_loader()
        elif self.data_format == 'partial_tokenized_random':
            self.minimum_tokens = ['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[TRAJ]', '[UNK]']
            self.read_partial_tokenized_trajectory_loader(seq_mode='random')
        elif self.data_format == 'partial_tokenized_by_frame':
            self.minimum_tokens = ['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[TRAJ]', '[UNK]']
            # self.coord_tokens = [i for i in range(2*50)]
            self.read_partial_tokenized_trajectory_loader(seq_mode='by_frame')
            # self.read_full_tokenized_by_frame_trajectory_loader()
            # self.normalize_dataset = False
            print('\nIn full tokenization, no need for normalizing the coordinates!\n')
        else:
            raise NotImplementedError
        
        warnings.filterwarnings("ignore")


    def minimum_tokenization(self, token):
        return self.minimum_tokens.index(token)


    def full_tokenization(self, coords):
        is_torch = False
        if isinstance(coords, torch.Tensor):
            is_torch = True
            coords = coords.numpy()
        # REGRESSION TRANSFORMERR: CONCURRENT CONDITIONAL GENERATION AND REGRESSION BY BLENDING NUMERICAL AND TEXTUAL TOKEN
        assert np.all(coords >= 0.) and np.all(coords < 1000.), 'assuming that all coordinate values are larger equal zero for now...'
        coords_copy = np.round(coords.copy(), 2)

        # Extract the highest order decimals
        highest_orders = np.floor_divide(coords_copy, 100)
        x_tokens = highest_orders[:, 0] + 40
        y_tokens = highest_orders[:, 1] + 40

        # Extract the next highest order decimals
        coords_copy -= highest_orders * 100
        second_orders = np.floor_divide(coords_copy, 10)
        x_tokens = np.stack((x_tokens, second_orders[:,0] + 30), axis=-1)
        y_tokens = np.stack((y_tokens, second_orders[:,1] + 30), axis=-1)

        # Reduce the original numbers
        coords_copy -= second_orders * 10
        third_orders = np.floor(coords_copy)
        x_tokens = np.concatenate((x_tokens, third_orders[:, 0, None] + 20), axis=1)
        y_tokens = np.concatenate((y_tokens, third_orders[:, 1, None] + 20), axis=1)

        # 
        coords_copy -= third_orders
        coords_copy = np.round(100*coords_copy)
        fourth_orders = np.floor_divide(coords_copy, 10)
        x_tokens = np.concatenate((x_tokens, fourth_orders[:, 0, None] + 10), axis=1)
        y_tokens = np.concatenate((y_tokens, fourth_orders[:, 1, None] + 10), axis=1)

        fifth_orders = coords_copy - fourth_orders*10
        x_tokens = np.concatenate((x_tokens, fifth_orders[:, 0, None]), axis=1)
        y_tokens = np.concatenate((y_tokens, fifth_orders[:, 1, None]), axis=1)

        x_tokens_split = np.hsplit(x_tokens, 5)
        y_tokens_split = np.hsplit(y_tokens, 5)
        coord_reconstr = np.array([[0., 0.]]*x_tokens.shape[0], dtype=np.float32)
        for idx, (x_r, y_r) in enumerate(zip(x_tokens_split, y_tokens_split)):
            cx = np.floor_divide(x_r, 10)
            cy = np.floor_divide(y_r, 10)
            digits_x = x_r - cx*10
            digits_y = y_r - cy*10
            if idx == 0: factor = 100.
            elif idx == 1: factor = 10.
            elif idx == 2: factor = 1.
            elif idx == 3: factor = 0.1
            elif idx == 4: factor = 0.01
            add_vec = np.concatenate([digits_x, digits_y], axis=-1)*factor
            coord_reconstr += add_vec

        # aaaaaaa, bbbbbbb = np.round(coord_reconstr, 2), np.round(coords, 2)
        assert np.all(np.abs(np.round(coord_reconstr - coords, 2)) <= 0.01)
        # fucking_args =  np.argwhere(np.abs(np.round(coord_reconstr - coords, 2)) > 0.01)
        # aaaaaaa_fuck, bbbbbbb_fuck = coord_reconstr[fucking_args[:,0], fucking_args[:,1]], coords[fucking_args[:,0], fucking_args[:,1]]

        tokens = np.stack((x_tokens + len(self.minimum_tokens), y_tokens + len(self.minimum_tokens) + 50), axis=-1).astype(np.int64)
        if is_torch:
            tokens = torch.tensor(tokens)
        return tokens


    def read_by_frame_trajectory_loader(self, is_masked = True):
        # Based on SoPhie
        seq_list_pos = []
        seq_list_vel = []
        self.sequence_list = []
        considered_peds_per_scene = []
        total_num_sequences = 0
        global_start_frame = 1

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
            all_scene_destinations = scene.all_scene_destinations
            destinations_per_agent = scene.destinations_per_agent if hasattr(scene, 'destinations_per_agent') else None
            semanticMap_path = scene.semanticMap_path if hasattr(scene, 'semanticMap_path') else None
            globalSemMap_path = scene.global_semanticMap_path if hasattr(scene, 'global_semanticMap_path') else None

            for frame_idx in range(0, num_sequences + 1):
                curr_seq_data = np.concatenate(frame_data[frame_idx:frame_idx + self.seq_length], axis=0)
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])

                agent_data_per_frame = []
                agent_ids_per_frame = []
                destinations_per_agent_per_frame = []
                # considered_peds_per_frame = []
                num_agents_in_curr_seq = len(agents_in_curr_seq)

                num_agents_considered = 0
                # Iterate over all pedestrians in current sequence (sequence = observed + predicted frames)
                for _, agent_id in enumerate(agents_in_curr_seq):
                    seq_data_per_agent = curr_seq_data[curr_seq_data[:, 1] == agent_id, :]
                    if is_masked:
                        # Apply mask where agent is present in the sequence
                        coord_per_agent = np.full((self.seq_length, 2), fill_value=np.inf)
                        coord_per_agent[seq_data_per_agent[:, 0].astype(np.int32) - frame_idx - global_start_frame, :] = seq_data_per_agent[:, 2:]
                        assert np.isinf(coord_per_agent[:, 0]).sum() == self.seq_length - len(seq_data_per_agent)
                        assert np.all(np.isinf(coord_per_agent[:, 0]) == np.isinf(coord_per_agent[:, 1]))
                        # check if agent ends in destination
                        destination_check(coord_per_agent, all_scene_destinations)

                    # seq_data_per_agent = np.around(seq_data_per_agent, decimals=4)
                    # Get start and end frames of pedestrians within current sequence
                    else:
                        first_frame_in_seq = frames.index(seq_data_per_agent[0, 0]) - frame_idx
                        last_frame_in_seq = frames.index(seq_data_per_agent[-1, 0]) - frame_idx + 1
                        if last_frame_in_seq - first_frame_in_seq != self.seq_length:
                            # Ignore pedestrians that are not present throughout current sequence length
                            continue
                        coord_per_agent = seq_data_per_agent[:, 2:]
                    # agent_data_per_frame.append(seq_data_per_agent)
                    # fill all False value in the mask with inf
                    agent_traj_data = np.array(coord_per_agent, dtype=np.float32)
                    destinations_per_agent[agent_id]

                    # apply mean shift and std
                    # if self.traj_quantity == 'pos':
                    #     agent_traj_data = (agent_traj_data - self.overall_mean) / self.overall_std
                    
                    destinations_per_agent_per_frame.append(destinations_per_agent[agent_id])
                    if self.traj_quantity == 'pos':
                        agent_data_per_frame.append(agent_traj_data)
                        agent_ids_per_frame.append(agent_id)
                        # trajectory_data['x'].append(np.array(seq_data_per_agent[:, -2]).astype(np.float32))
                        # trajectory_data['y'].append(np.array(seq_data_per_agent[:, -1]).astype(np.float32))
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

                if is_masked:
                    assert len(agent_data_per_frame) == num_agents_in_curr_seq
                    if np.all(np.isinf(np.array(agent_data_per_frame, dtype=np.float32)[:, self.obs_length-1, 0])):
                        assert np.all(np.isinf(np.array(agent_data_per_frame, dtype=np.float32)[:, self.obs_length-1:, 1]))
                        continue # continue if all future frames are inf
                if len(agent_data_per_frame) > 0:
                    sequence_chunk = {
                        'batch_of_sequences': np.array(agent_data_per_frame, dtype=np.float32),
                        'scene_path': scene.RGB_image_path,
                        'semanticMap_path': semanticMap_path,
                        'globalSemMap_path': globalSemMap_path,
                        'agent_ids': np.array(agent_ids_per_frame),
                    }
                    if all_scene_destinations is not None: sequence_chunk.update({'all_scene_destinations': np.array(all_scene_destinations, dtype=np.float32)})
                    if len(destinations_per_agent_per_frame) > 0: sequence_chunk.update({'destinations_per_agent': np.array(destinations_per_agent_per_frame, dtype=np.float32)})
                    self.sequence_list += [sequence_chunk]
            # Calculate mean and std per x & y per scene
            # statistics_in_scene = {}
            # for key, arrays_list in trajectory_data.items():
            #     traj_data = np.concatenate(arrays_list, axis=0)
            #     mean = traj_data.mean()
            #     std = traj_data.std()
            #     statistics_in_scene.update({key: {'mean': mean, 'std': std}})
            
            # self.statistics_per_scene.update({scene.RGB_image_path.split(SEP)[-1]: statistics_in_scene})
        
            considered_peds_per_scene.append(considered_peds_per_frame)
    
    
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

            if len(sequences_per_scene) <= self.max_traj_per_batch:
                self.sequence_list += [{
                    'batch_of_sequences': np.array(sequences_per_scene, dtype=np.float32),
                    'scene_path': scene.RGB_image_path,
                    }]
            else:
                for x in range(0, len(sequences_per_scene), self.max_traj_per_batch):
                    self.sequence_list += [{
                        'batch_of_sequences': np.array(sequences_per_scene[x:x+self.max_traj_per_batch], dtype=np.float32),
                        'scene_path': scene.RGB_image_path,
                        }]


    def read_full_tokenized_by_frame_trajectory_loader(self):

        self.sequence_list = []
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
            all_agent_token_frames = []
            all_agent_coords = []
            all_coord_data = []
            
            # TODO store entire trajectories of variable length

            for agent_i in set(raw_pixel_data.agent_id):
                raw_agent_data_np = raw_pixel_data[raw_pixel_data.agent_id == agent_i]
                # downsample frame rate happens here, at the single agent level
                raw_agent_data_np = raw_agent_data_np.to_numpy(dtype=np.float32)

                agent_presence_frames = raw_agent_data_np[:, 0].astype(int)
                coord_data = raw_agent_data_np[:, 2:].astype(np.float32)
                # coord_tokens = self.full_tokenization(coord_data)
                # agent_frame_offset = agent_presence_frames[0] - all_frames[0] # assuming that all agents have continuous trajectories
                
                agent_token_frames = []
                # agent_coords = []
                # agent_token_frames = [coord_tokens[i - agent_frame_offset] if i in agent_presence_frames else None for i in all_frames]
                bos_it = 0
                eos_it = 0
                # all_agent_coords.append(coord_data)
                # TODO filter out some useless frames....

                for index in all_frames:
                    if index in agent_presence_frames:
                        agent_token_frames.append(self.minimum_tokenization('[TRAJ]'))
                        # agent_coords.append(coord_data[index - agent_frame_offset])
                    elif index==(agent_presence_frames[0]-1):
                        agent_token_frames.append(self.minimum_tokenization('[BOS]'))
                        # agent_coords.append([-1, -1])
                        bos_it += 1
                    elif index==(agent_presence_frames[-1]+1):
                        agent_token_frames.append(self.minimum_tokenization('[EOS]'))
                        # agent_coords.append([-1, -1])
                        eos_it += 1
                    elif index < (agent_presence_frames[0]-1) or index > (agent_presence_frames[-1]+1):
                        # agent_coords.append([-1, -1])
                        agent_token_frames.append(self.minimum_tokenization('[PAD]'))
                    else:
                        raise IndexError('Bug or trajectory is interrupted!')
                
                bos_index = agent_token_frames.index(self.minimum_tokenization('[BOS]'))
                eos_index = agent_token_frames.index(self.minimum_tokenization('[EOS]'))

                coord_data = np.pad(coord_data, ((int(bos_index)+1, len(agent_token_frames)-int(eos_index)), (0, 0)), 'constant', constant_values=(self.fill_value, self.fill_value))
                # Some checking
                assert bos_it == 1 and eos_it == 1, f'bos_it == {bos_it} and eos_it == {eos_it}, both need to be one!'
                assert len(all_frames) == len(agent_token_frames) == len(coord_data)

                all_agent_token_frames.append(agent_token_frames)
                all_coord_data.append(coord_data)
                # all_agent_coords.append(agent_coords)

                # Some checking
                assert bos_it == 1 and eos_it == 1, f'bos_it == {bos_it} and eos_it == {eos_it}, both need to be one!'
                assert len(all_frames) == len(agent_token_frames)

            skipped_frames = 0
            # chop into frames
            for start_t in range(0, len(all_frames)-self.seq_length):
                tokenized_sequences = [agent_sequence[start_t:start_t + self.seq_length] for agent_sequence in all_agent_token_frames]
                # filter the frames
                tokenized_sequences = np.array(tokenized_sequences, dtype=np.int64)
                # any row start with eight zeros:
                starts_with_eight_zeros = np.all(tokenized_sequences[:, :self.obs_length] == self.minimum_tokenization('[PAD]'), axis=1)
                result = np.any(starts_with_eight_zeros)
                if result:
                    skipped_frames += 1
                    continue

                sequences_coords = [seq_coord[start_t:start_t + self.seq_length] for seq_coord in all_coord_data]
                sequences_coords = np.array(sequences_coords, dtype=np.float32)
                # all_coords_per_frame = [agent_coords_p_f[start_t:start_t + self.seq_length] for agent_coords_p_f in all_agent_coords]
                # all_coords_per_frame = np.array(all_coords_per_frame, dtype=np.float32)
                # sequences_per_scene.append(all_sequences_per_frame)

                sequence_chunk = {
                    'batch_of_sequences': sequences_coords,
                    'tokenized_sequences': tokenized_sequences,
                    # 'is_traj_mask': all_sequences_per_frame==-1,
                    'scene_path': scene.RGB_image_path,
                }
                if self.mode == 'MTM':
                    raise NotImplementedError
                    mtm_mask = np.random.choice(a=[False, True], size=(len(sequences_per_scene), self.seq_length), p=[1-self.mtm_mask_prob, self.mtm_mask_prob])
                    sequence_chunk['mtm_mask'] = sparse.COO.from_numpy(mtm_mask, fill_value=False)
            
                self.sequence_list += [sequence_chunk]
            
            print(f'\nSkipped {skipped_frames} out of {len(all_frames)-self.seq_length} frames...')
            lol = 2
            
            
    def read_partial_tokenized_trajectory_loader(self, seq_mode):
        
        self.sequence_list = []

        # coordinates are already in pixel format
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

            # for random
            sequences_per_scene = []

            # for by_frame
            all_coord_data = []
            all_agent_token_frames = []
            # TODO store entire trajectories of variable length

            for agent_i in set(raw_pixel_data.agent_id):
                raw_agent_data_np = raw_pixel_data[raw_pixel_data.agent_id == agent_i]
                # downsample frame rate happens here, at the single agent level
                raw_agent_data_np = raw_agent_data_np.to_numpy(dtype=np.float32)

                agent_presence_frames = raw_agent_data_np[:, 0].astype(int)
                coord_data = raw_agent_data_np[:, 2:].astype(np.float32)

                agent_token_frames = []
                bos_it = 0
                eos_it = 0
                for index in all_frames:
                    if index in agent_presence_frames:
                        agent_token_frames.append(self.minimum_tokenization('[TRAJ]'))
                    elif index==(agent_presence_frames[0]-1):
                        agent_token_frames.append(self.minimum_tokenization('[BOS]'))
                        bos_it += 1
                    elif index==(agent_presence_frames[-1]+1):
                        agent_token_frames.append(self.minimum_tokenization('[EOS]'))
                        eos_it += 1
                    elif index < (agent_presence_frames[0]-1) or index > (agent_presence_frames[-1]+1):
                        agent_token_frames.append(self.minimum_tokenization('[PAD]'))
                    else:
                        raise IndexError('Bug or trajectory is interrupted!')

                bos_index = np.where(np.array(agent_token_frames) == self.minimum_tokenization('[BOS]'))[0]
                eos_index = np.where(np.array(agent_token_frames) == self.minimum_tokenization('[EOS]'))[0]

                coord_data = np.pad(coord_data, ((int(bos_index)+1, len(agent_token_frames)-int(eos_index)), (0, 0)), 'constant', constant_values=(self.fill_value, self.fill_value))

                # Some checking
                assert bos_it == 1 and eos_it == 1, f'bos_it == {bos_it} and eos_it == {eos_it}, both need to be one!'
                assert len(all_frames) == len(agent_token_frames) == len(coord_data)
                # assert np.all(coord_data[np.argwhere(np.array(agent_token_frames) != self.minimum_tokenization('[TRAJ]')).squeeze()] == self.fill_value), f'Assertion that all non-[TRAJ] are assigned with {self.fill_value} is violated!'
                # assert np.all(coord_data[np.argwhere(np.array(agent_token_frames) == self.minimum_tokenization('[TRAJ]')).squeeze()] != self.fill_value), 'Assertion that all [TRAJ] are assigned with true coords is violated!'

                if seq_mode == 'random':
                    for start_t in range(0, len(all_frames)-self.seq_length):
                        seq_tokens = agent_token_frames[start_t:start_t + self.seq_length]                  
                        if np.count_nonzero(np.array(seq_tokens) == self.minimum_tokenization('[PAD]')) > self.max_pads_per_seq:
                            # sequences cannot contain more than 3 PADS
                            continue

                        seq_coords = coord_data[start_t:start_t + self.seq_length]
                        sequences_per_scene.append([seq_tokens, seq_coords])
                
                elif seq_mode == 'by_frame':
                    all_agent_token_frames.append(agent_token_frames)
                    all_coord_data.append(coord_data)

            if seq_mode == 'random':
                random.shuffle(sequences_per_scene)

                sequence_tokens =  [agent_data[0] for agent_data in sequences_per_scene]
                sequences_coords = [agent_data[1] for agent_data in sequences_per_scene]

                if self.mode == 'MTM':
                    mtm_mask = np.random.choice(a=[False, True], size=(len(sequence_tokens), self.seq_length), p=[1-self.mtm_mask_prob, self.mtm_mask_prob])
                    # TODO 80% of the time, replace with [MASK], 10% of the time, replace random word, 10% of the time, keep same
                    # TODO reconstruct entire trajectory or only masked tokens --> maybe concentration on only masks better?
                    # TODO what if encoder always deletes the first token and replaces it with generated one in last position -> no decoder necessary? 
                    # TODO also do GPT-2 (pre)training: always start from <BOS>, <init coord>?
                # chunking into max_seq_length pieces to prevent OOM issues
                if len(sequences_coords) <= self.max_traj_per_batch:
                    sequence_chunk = {
                        'batch_of_sequences': np.array(sequences_coords, dtype=np.float32),
                        'tokenized_sequences': np.array(sequence_tokens, dtype=np.int64),
                        'scene_path': scene.RGB_image_path,
                        }
                    if self.mode == 'MTM': sequence_chunk['mtm_mask'] = sparse.COO.from_numpy(mtm_mask, fill_value=False)
                    
                    self.sequence_list += [sequence_chunk]
                
                else:
                    for x in range(0, len(sequences_coords), self.max_traj_per_batch):
                        sequence_chunk = {
                            'batch_of_sequences': np.array(sequences_coords[x:x+self.max_traj_per_batch], dtype=np.float32),
                            'tokenized_sequences': np.array(sequence_tokens[x:x+self.max_traj_per_batch], dtype=np.int64),
                            'scene_path': scene.RGB_image_path,
                        }
                        if self.mode == 'MTM': sequence_chunk['mtm_mask'] = sparse.COO.from_numpy(mtm_mask[x:x+self.max_traj_per_batch], fill_value=False)
                        
                        self.sequence_list += [sequence_chunk]

            elif seq_mode == 'by_frame':
                skipped_frames = 0
                # chop into frames
                for start_t in range(0, len(all_frames)-self.seq_length):
                    tokenized_sequences = [agent_sequence[start_t:start_t + self.seq_length] for agent_sequence in all_agent_token_frames]
                    # filter the frames
                    tokenized_sequences = np.array(tokenized_sequences, dtype=np.int64)
                    # any row start with eight zeros:
                    starts_with_eight_zeros = np.all(tokenized_sequences[:, :self.obs_length] == self.minimum_tokenization('[PAD]'), axis=1)
                    result = np.any(starts_with_eight_zeros)
                    if result:
                        skipped_frames += 1
                        continue

                    sequences_coords = [seq_coord[start_t:start_t + self.seq_length] for seq_coord in all_coord_data]
                    sequences_coords = np.array(sequences_coords, dtype=np.float32)
                    # all_coords_per_frame = [agent_coords_p_f[start_t:start_t + self.seq_length] for agent_coords_p_f in all_agent_coords]
                    # all_coords_per_frame = np.array(all_coords_per_frame, dtype=np.float32)
                    # sequences_per_scene.append(all_sequences_per_frame)

                    sequence_chunk = {
                        'batch_of_sequences': sequences_coords,
                        'tokenized_sequences': tokenized_sequences,
                        # 'is_traj_mask': all_sequences_per_frame==-1,
                        'scene_path': scene.RGB_image_path,
                        'scene': scene,
                    }
                    if self.mode == 'MTM':
                        raise NotImplementedError
                        mtm_mask = np.random.choice(a=[False, True], size=(len(sequences_per_scene), self.seq_length), p=[1-self.mtm_mask_prob, self.mtm_mask_prob])
                        sequence_chunk['mtm_mask'] = sparse.COO.from_numpy(mtm_mask, fill_value=False)
                
                    self.sequence_list += [sequence_chunk]
                
                print(f'\nSkipped {skipped_frames} out of {len(all_frames)-self.seq_length} frames...')
                lol = 32

    def __len__(self):
        return len(self.sequence_list)


    def augment_traj_and_images_sparse(self, trajectories, np_image, **kwargs):

        input_traj_maps = kwargs["input_traj_maps"] if "input_traj_maps" in kwargs else None
        input_traj_maps_tf = input_traj_maps.reshape(input_traj_maps.shape[0], input_traj_maps.shape[1], -1) if input_traj_maps is not None else None
        
        destinations = kwargs["destinations"] if "destinations" in kwargs else None
        labels = np.ones(len(destinations), dtype=np.int64) if destinations is not None else None

        # add addtionals
        additional_targets = {}
        bbox_params = None
        if input_traj_maps is not None: additional_targets.update({'traj_map': 'image'})
        if destinations is not None: bbox_params = A.BboxParams(format='pascal_voc', label_fields=['class_labels'])

        # keypoints = list(map(tuple, trajectories.reshape(-1, 2)))
        keypoints = trajectories.reshape(-1, 2)
        keypoint_params = A.KeypointParams(format='xy', remove_invisible=False)

        if self.newDS:
            pad_value = 0 #if self.mode == 'TRAJ_PRED' else (1,1,1) ---> with this performance actually deteriorates
            cval = 0#0 if self.mode == 'TRAJ_PRED' else (1,1,1) ---> with this performance actually deteriorates
            transform = A.Compose([
                A.PadIfNeeded(self.img_max_size[0]+self.pad_size, self.img_max_size[1]+self.pad_size, value=pad_value, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                A.augmentations.geometric.transforms.Affine(translate_px={'x': (-self.pad_size//2, self.pad_size//2), 'y': (-self.pad_size//2, self.pad_size//2)}, p=self.p_trans, cval=cval, mode=cv2.BORDER_CONSTANT),
                A.augmentations.geometric.transforms.HorizontalFlip(p=self.p_augms),
                A.augmentations.geometric.transforms.VerticalFlip(p=self.p_augms),
                A.augmentations.geometric.transforms.Transpose(p=self.p_augms),
                A.augmentations.geometric.rotate.RandomRotate90(p=self.p_augms),
                ], 
                keypoint_params=keypoint_params, additional_targets=additional_targets, bbox_params=bbox_params)
        else:
            transform = A.Compose([
                A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
                A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
                A.augmentations.geometric.transforms.Transpose(p=0.5),
                A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
                ],
                keypoint_params=keypoint_params,
                additional_targets=additional_targets,
                bbox_params=bbox_params
            )

        #################### TEST START ####################
        # visualize_trajectories(keypoints, None, np_image, self.obs_length)
        #################### TEST END ####################
        
        if destinations is not None and input_traj_maps_tf is not None: transformed = transform(image=np_image, keypoints=keypoints, traj_map=input_traj_maps_tf, bboxes=destinations, class_labels=labels)
        elif destinations is None and input_traj_maps_tf is not None: transformed = transform(image=np_image, keypoints=keypoints, traj_map=input_traj_maps_tf)
        elif destinations is not None and input_traj_maps_tf is None: transformed = transform(image=np_image, keypoints=keypoints, bboxes=destinations, class_labels=labels)
        else: transformed = transform(image=np_image, keypoints=keypoints)

        transformed_trajectories = np.array(transformed['keypoints'])
        transformed_destinations = np.array(transformed['bboxes']) if 'bboxes' in transformed else None
        # apply mean shift and std
        # if self.traj_quantity == 'pos' and self.normalize_dataset:
        #     transformed_trajectories = (transformed_trajectories - self.overall_mean) / self.overall_std
        #     transformed_destinations = (transformed_destinations - self.overall_mean) / self.overall_std if destinations is not None else None

        # #################### TEST START ####################
        # visualize_trajectories(np.array(transformed['keypoints']), None, transformed['image'], self.obs_length)
        # origins, goals = np.split(np.array(transformed['keypoints']).astype(np.int32), 2)
        # for i in range(len(origins)):
        #     cv2.circle(transformed['image'], (origins[i,0], origins[i,1]), 2, (0, 255, 255), -1)
        #     cv2.circle(transformed['image'], (goals[i,0], goals[i,1]), 2, (255, 255, 0), -1)
        # plt.imshow(img_tf)
        # #################### TEST END ####################

        image = torch.tensor(transformed['image']).permute(2, 0, 1).float()
        transformed_trajectories = torch.tensor(transformed_trajectories).float().view(trajectories.shape)
        input_traj_maps = torch.tensor(transformed['traj_map']).float() if 'traj_map' in transformed else None
        transformed_destinations = torch.tensor(transformed_destinations).float() if destinations is not None else None

        return {'image': image, 'transformed_trajectories': transformed_trajectories, 'input_traj_maps': input_traj_maps, 'transformed_destinations': transformed_destinations}

    
    def __getitem__(self, idx):
        
        # LOAD CSV DATA
        sequence = self.sequence_list[idx] # 'scene_path': corr_cross\\0__floorplan_siteX_35_siteY_20_CORRWIDTH_3_SIDECORRWIDTH_20_numCross_2\\variation_56\\variation_56_num_agents_15.npz'

        abs_pixel_coord = sequence['batch_of_sequences']
       
        np_image = sparse.load_npz(sequence['scene_path']).todense()
        globalSemMap = sparse.load_npz(sequence['globalSemMap_path']).todense() if 'globalSemMap_path' in sequence else None
        semanticMaps_per_agent = sparse.load_npz(sequence['semanticMap_path']).todense() if 'semanticMap_path' in sequence else None
        agent_ids_present = sequence['agent_ids'] if 'agent_ids' in sequence else None
        semanticMaps_per_agent = semanticMaps_per_agent[agent_ids_present, :, :] if semanticMaps_per_agent is not None else None
        # import matplotlib.pyplot as plt
        # plt.imshow(np_image)
        np_image = np_image.astype(np.float32) / 255.

        all_scene_destinations = sequence['all_scene_destinations'] if 'all_scene_destinations' in sequence else None
        destinations_per_agent = sequence['destinations_per_agent'] if 'destinations_per_agent' in sequence else None
        batch_item = {}
        withGlobalSemMap = True
        
        # sequence prediction
        if self.mode == 'TRAJ_PRED':
            # Goal-SAR
            if self.arch == 'goal': 
                down_factor = 4
                np_image, input_traj_maps = add_cnn_maps(np_image, abs_pixel_coord, down_factor)
                abs_pixel_coord /= down_factor
                all_scene_destinations /= down_factor if all_scene_destinations is not None else None

                batch_data = self.augment_traj_and_images_sparse(abs_pixel_coord, np_image, input_traj_maps=input_traj_maps, destinations=all_scene_destinations)
                batch_item.update({'input_traj_maps': batch_data['input_traj_maps']})
                if all_scene_destinations is not None: batch_item.update({'all_scene_destinations': batch_data['transformed_destinations']})
            elif self.arch in ['coca_goal', 'simple_goal', 'gan_goal', 'adv_goal']:
                # apply resize factor
                abs_pixel_coord = abs_pixel_coord.copy() / self.resize_factor
                destinations_per_agent = destinations_per_agent.copy() / self.resize_factor
                all_scene_destinations = all_scene_destinations.copy() / self.resize_factor
                globalSemMap = resize(globalSemMap, (globalSemMap.shape[0]//self.resize_factor, globalSemMap.shape[1]//self.resize_factor), order=0, anti_aliasing=False, preserve_range=True)
                semanticMaps_per_agent = resize(semanticMaps_per_agent, (semanticMaps_per_agent.shape[0], semanticMaps_per_agent.shape[1]//self.resize_factor, semanticMaps_per_agent.shape[2]//self.resize_factor), order=0, anti_aliasing=False, preserve_range=True)
                np_image = cv2.resize(np_image, (np_image.shape[1]//self.resize_factor, np_image.shape[0]//self.resize_factor), interpolation=cv2.INTER_NEAREST)
                _, global_occupancy_map = create_occupancy_map(np_image, abs_pixel_coord[:, self.obs_length-1, :], resize_factor=1.0, gaussian_std=self.gaussian_std)
                globalSemMap[globalSemMap == 0] = global_occupancy_map[globalSemMap == 0]

                batch_data = self.augment_traj_and_images_sparse(abs_pixel_coord, np_image, input_traj_maps=np.concatenate((np.expand_dims(globalSemMap, axis=-1), semanticMaps_per_agent.transpose(1,2,0)), axis=-1), destinations=np.concatenate((destinations_per_agent, all_scene_destinations), axis=0))
                                
                # normalize
                transformed_destinations = batch_data['transformed_destinations'] # - (batch_data['transformed_destinations'] - self.overall_mean) / self.overall_std
                window_centers_l = batch_data['transformed_trajectories'][:, self.obs_length-1]
                transformed_trajectories = batch_data['transformed_trajectories'] # (batch_data['transformed_trajectories'] - self.overall_mean) / self.overall_std
                
                # provide velocities and destinations
                transformed_velocities = transformed_trajectories[:, 1:, :] - transformed_trajectories[:, :-1, :]
                batch_item.update({'transformed_velocities': transformed_velocities})
                batch_item.update({'transformed_agent_destinations': transformed_destinations[:destinations_per_agent.shape[0]]})
                batch_item.update({'all_scene_destinations': transformed_destinations[destinations_per_agent.shape[0]:]})
                
                # calculate semantic egocentric maps, and provide those + global occupancy map
                semanticMaps_per_agent = batch_data['input_traj_maps'][:, :, 1:]
                globalOccupancyMap = batch_data['input_traj_maps'][:, :, 0].squeeze()
                # if (np_image.shape[0] + self.pad_size) == batch_data['image'].shape[1] or (np_image.shape[1] + self.pad_size) == batch_data['image'].shape[2]:
                #     raise NotImplementedError('check first')
                #     globalOccupancyMap = np.pad(globalOccupancyMap, ((self.window_size//2, self.window_size//2), (self.window_size//2, self.window_size//2)), 'constant', constant_values=(0, 0))
                # else: 
                globalOccupancyMap_cp = globalOccupancyMap.clone()
                occupancyMaps_egocentric = []
                for i in range(abs_pixel_coord.shape[0]):
                    x_coord = window_centers_l[i, 1]
                    y_coord = window_centers_l[i, 0]
                    if torch.isfinite(x_coord):
                        x_coord_i, y_coord_i = int(x_coord), int(y_coord)
                        mid_value = globalOccupancyMap_cp[x_coord_i, y_coord_i].item()
                        if abs(mid_value) != 1.0:
                            assert abs(mid_value) < 1.0, f'absolute mid value cannot be larger than 1, but is at {mid_value}'
                            # print(f'mid value should be + or -1, but is at {mid_value}')
                            mid_value_neighbor_1, mid_value_neighbor_2, mid_value_neighbor_3, mid_value_neighbor_4 = globalOccupancyMap_cp[x_coord_i+1, y_coord_i].item(), globalOccupancyMap_cp[x_coord_i-1, y_coord_i].item(), globalOccupancyMap_cp[x_coord_i, y_coord_i+1].item(), globalOccupancyMap_cp[x_coord_i, y_coord_i-1].item()
                            # assert abs(mid_value_neighbor_1) == 1.0 or abs(mid_value_neighbor_2) == 1.0 or abs(mid_value_neighbor_3) == 1.0 or abs(mid_value_neighbor_4) == 1.0, f'neighbors are {mid_value_neighbor_1}, {mid_value_neighbor_2}, {mid_value_neighbor_3}, {mid_value_neighbor_4}'
                            if abs(mid_value) < 0.8: print(f'mid value should be + or -1, but is at {mid_value}. Neighbors are {mid_value_neighbor_1}, {mid_value_neighbor_2}, {mid_value_neighbor_3}, {mid_value_neighbor_4}')
                        window = create_window(globalOccupancyMap_cp, (x_coord_i, y_coord_i), self.window_size//2)
                        # window = globalOccupancyMap_cp[(x_coord_i-self.window_size//2):(x_coord_i+self.window_size//2), (y_coord_i-self.window_size//2):(y_coord_i+self.window_size//2)]
                        # window_np = window.numpy()
                        # assert window.shape == (self.window_size, self.window_size), f'window shape is {window.shape}'
                    else:
                        window = torch.zeros((self.window_size, self.window_size))
                    occupancyMaps_egocentric.append(window)
                
                batch_item.update({'globalOccupancyMap': globalOccupancyMap})
                batch_item.update({'occupancyMaps_egocentric': torch.stack(occupancyMaps_egocentric, dim=0)})
                batch_item.update({'semanticMaps_per_agent': semanticMaps_per_agent})
            else:
                batch_data = self.augment_traj_and_images_sparse(abs_pixel_coord, np_image, destinations=all_scene_destinations)
                if all_scene_destinations is not None: batch_item.update({'all_scene_destinations': batch_data['transformed_destinations']})

            batch_item.update({'image': batch_data['image'], 'coords': transformed_trajectories}) #, 'scene': sequence['scene']})
        
        # goal prediction
        elif self.mode == 'GOAL_PRED':
            all_scene_destinations = None
            # select origin and goal, and resize
            obs_coords, goal_coords = select_coordinate_pairs(abs_pixel_coord, self.pred_length, self.resize_factor)
            coord_pairs = np.concatenate([obs_coords, goal_coords], axis=0)           
            
            # resize numpy array and create occupancy map 
            np_image, global_occupancy_map = create_occupancy_map(np_image, obs_coords, resize_factor=self.resize_factor)
            
            
            if withGlobalSemMap:
                globalSemMap[globalSemMap == 0] = global_occupancy_map[globalSemMap == 0]
                batch_data = self.augment_traj_and_images_sparse(coord_pairs, np_image, input_traj_maps=np.expand_dims(globalSemMap, axis=-1), destinations=destinations_per_agent)
                window_size = self.pad_size
                obs_coords, goal_coords = torch.chunk(batch_data['transformed_trajectories'], chunks=2, dim=0)
                obs_coords_l = obs_coords.long()

                semanticMaps_egocentric = []
                for i in range(obs_coords.shape[0]):
                    mid_value = batch_data['input_traj_maps'][obs_coords_l[i, 1], obs_coords_l[i, 0], 0].item()
                    assert mid_value == 1 or mid_value == -1, f'mid value should be + or -1, but is at {mid_value}'
                    window = batch_data['input_traj_maps'][(obs_coords_l[i, 1]-window_size//2):(obs_coords_l[i, 1]+window_size//2), (obs_coords_l[i, 0]-window_size//2):(obs_coords_l[i, 0]+window_size//2), 0]
                    # window_np = window.numpy()
                    assert window.shape == (window_size, window_size), f'window shape is {window.shape}'
                    semanticMaps_egocentric.append(window)
                # batch_item.update({'globalSemMap': batch_data['input_traj_maps'].squeeze(-1)})
                batch_item.update({'transformed_agent_destinations': (batch_data['transformed_destinations'] - self.overall_mean) / self.overall_std})

            else:
                semanticMap = sparse.load_npz(sequence['semanticMap_path']).todense() if 'semanticMap_path' in sequence else None
                semanticMap = semanticMap[sequence['agent_ids']].astype(np.float32) if semanticMap is not None else None
                assert self.resize_factor > 1
                occupancy_maps = np.tile(global_occupancy_map, (obs_coords.shape[0], 1, 1))
                # resize the semantic map via skimage and insert occupancy maps into semantic map where semantic map is 0
                semanticMap = resize(semanticMap, (semanticMap.shape[0], semanticMap.shape[1]//self.resize_factor, semanticMap.shape[2]//self.resize_factor), order=0, anti_aliasing=False, preserve_range=True)
                semanticMap_destinations = semanticMap.copy()
                semanticMap[semanticMap == 0] = occupancy_maps[semanticMap == 0]
                
                agent_coords = semanticMap[:, obs_coords.astype(int)[:,1], obs_coords.astype(int)[:,0]]
                assert np.all(np.abs(agent_coords) == 1)
            
                batch_data = self.augment_traj_and_images_sparse(coord_pairs, np_image, input_traj_maps=np.concatenate((semanticMap, semanticMap_destinations), axis=0).transpose(1,2,0), destinations=all_scene_destinations)

                window_size = self.pad_size
                obs_coords, goal_coords = torch.chunk(batch_data['transformed_trajectories'], chunks=2, dim=0)
                obs_coords_l = obs_coords.long()

                semanticMaps_egocentric = []
                for i in range(obs_coords.shape[0]):
                    # mid_value = batch_data['input_traj_maps'][obs_coords_l[i, 1], obs_coords_l[i, 0], i].item()
                    # assert mid_value == 1 or mid_value == -1, f'mid value should be + or -1, but is at {mid_value}'
                    window = batch_data['input_traj_maps'][(obs_coords_l[i, 1]-window_size//2):(obs_coords_l[i, 1]+window_size//2), (obs_coords_l[i, 0]-window_size//2):(obs_coords_l[i, 0]+window_size//2), i]
                    # window_np = window.numpy()
                    assert window.shape == (window_size, window_size), f'window shape is {window.shape}'
                    semanticMaps_egocentric.append(window)
                batch_item.update({'goalMaps_egocentric': batch_data['input_traj_maps'][:,:, obs_coords.shape[0]:]})
            
            
            # normalize after creating egocentric maps
            obs_coords = (obs_coords - self.overall_mean) / self.overall_std
            goal_coords = (goal_coords - self.overall_mean) / self.overall_std
            # assert torch.all(torch.abs(goal_coords - obs_coords) <= 8.8832)

            batch_item.update({'image': batch_data['image'], 'obs_coords': obs_coords, 'goal_coords': goal_coords})
            batch_item.update({'semanticMaps_egocentric': torch.stack(semanticMaps_egocentric, dim=0)})
            
            # plt.imshow(batch_data['image'].permute(1,2,0).numpy())

            # control plot start
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # Plot the 3D surface
            # height, width, C = np_image.shape
            # x_range = np.arange(0, batch_data['input_traj_maps'].shape[1], 1) # torch.arange(0, height, 1)
            # y_range = np.arange(0, batch_data['input_traj_maps'].shape[0], 1) # torch.arange(0, width, 1)
            # grid_x, grid_y = np.meshgrid(x_range, y_range)
            # surface = ax.plot_surface(grid_x, grid_y, batch_data['input_traj_maps'][:,:, obs_coords.shape[0]].numpy(), cmap='viridis')
            # control plot end

            # are goalMaps_egocentric and image same?
            # black_img = batch_data['image'].permute(1,2,0).numpy()
            # black_img_mask = np.all(black_img == 0., axis=(2))
            # ones_goal = batch_data['input_traj_maps'][:,:, obs_coords.shape[0]].numpy()
            # ones_goal_mask = ones_goal == 1.
            # non_equals = black_img_mask != ones_goal_mask
            # sum_unequals = np.sum(non_equals)
            # assert sum_unequals

            # abs_pixel_coord /= down_factor
            # if all_scene_destinations is not None: batch_item.update({'all_scene_destinations': batch_data['transformed_destinations']})

        elif self.mode == 'SEGMENTATION':
            # apply resize factor
            abs_pixel_coord = abs_pixel_coord.copy() / self.resize_factor   
            globalSemMap = resize(globalSemMap, (globalSemMap.shape[0]//self.resize_factor, globalSemMap.shape[1]//self.resize_factor), order=0, anti_aliasing=False, preserve_range=True)
            semanticMaps_per_agent = resize(semanticMaps_per_agent, (semanticMaps_per_agent.shape[0], semanticMaps_per_agent.shape[1]//self.resize_factor, semanticMaps_per_agent.shape[2]//self.resize_factor), order=0, anti_aliasing=False, preserve_range=True)
            np_image = cv2.resize(np_image, (np_image.shape[1]//self.resize_factor, np_image.shape[0]//self.resize_factor), interpolation=cv2.INTER_NEAREST)
            _, global_occupancy_map = create_occupancy_map(np_image, abs_pixel_coord[:, self.obs_length-1, :], resize_factor=1.0, gaussian_std=self.gaussian_std)
            globalSemMap[globalSemMap == 0] = global_occupancy_map[globalSemMap == 0]

            """ batch_data = self.augment_traj_and_images_sparse(abs_pixel_coord, np_image, input_traj_maps=np.concatenate((np.expand_dims(globalSemMap, axis=-1), semanticMaps_per_agent.transpose(1,2,0)), axis=-1), destinations=None) """
            batch_data = self.augment_traj_and_images_sparse(abs_pixel_coord, np_image, input_traj_maps=np.expand_dims(globalSemMap, axis=-1), destinations=None)
            # normalize
            window_centers_l = batch_data['transformed_trajectories'][:, self.obs_length-1]
            transformed_trajectories = (batch_data['transformed_trajectories'] - self.overall_mean) / self.overall_std
                        
            # calculate semantic egocentric maps, and provide those + global occupancy map
            """ semanticMaps_per_agent = batch_data['input_traj_maps'][:, :, 1:]
            globalOccupancyMap = batch_data['input_traj_maps'][:, :, 0].squeeze()
            # if (np_image.shape[0] + self.pad_size) == batch_data['image'].shape[1] or (np_image.shape[1] + self.pad_size) == batch_data['image'].shape[2]:
            #     globalOccupancyMap = np.pad(globalOccupancyMap, ((self.window_size//2, self.window_size//2), (self.window_size//2, self.window_size//2)), 'constant', constant_values=(0, 0))
            #     raise NotImplementedError('check first')
            # else: 
            globalOccupancyMap_cp = globalOccupancyMap.clone()
            occupancyMaps_egocentric = []
            for i in range(abs_pixel_coord.shape[0]):
                x_coord = window_centers_l[i, 1]
                y_coord = window_centers_l[i, 0]
                if torch.isfinite(x_coord):
                    x_coord_i, y_coord_i = int(x_coord), int(y_coord)
                    mid_value = globalOccupancyMap_cp[x_coord_i, y_coord_i].item()
                    if abs(mid_value) != 1.0:
                        assert abs(mid_value) < 1.0, f'absolute mid value cannot be larger than 1, but is at {mid_value}'
                        # print(f'mid value should be + or -1, but is at {mid_value}')
                        mid_value_neighbor_1, mid_value_neighbor_2, mid_value_neighbor_3, mid_value_neighbor_4 = globalOccupancyMap_cp[x_coord_i+1, y_coord_i].item(), globalOccupancyMap_cp[x_coord_i-1, y_coord_i].item(), globalOccupancyMap_cp[x_coord_i, y_coord_i+1].item(), globalOccupancyMap_cp[x_coord_i, y_coord_i-1].item()
                        # assert abs(mid_value_neighbor_1) == 1.0 or abs(mid_value_neighbor_2) == 1.0 or abs(mid_value_neighbor_3) == 1.0 or abs(mid_value_neighbor_4) == 1.0, f'neighbors are {mid_value_neighbor_1}, {mid_value_neighbor_2}, {mid_value_neighbor_3}, {mid_value_neighbor_4}'
                        if abs(mid_value) < 0.8: print(f'mid value should be + or -1, but is at {mid_value}. Neighbors are {mid_value_neighbor_1}, {mid_value_neighbor_2}, {mid_value_neighbor_3}, {mid_value_neighbor_4}')
                    window = globalOccupancyMap_cp[(x_coord_i-self.window_size//2):(x_coord_i+self.window_size//2), (y_coord_i-self.window_size//2):(y_coord_i+self.window_size//2)]
                    # window_np = window.numpy()
                    assert window.shape == (self.window_size, self.window_size), f'window shape is {window.shape}'
                else:
                    window = torch.zeros((self.window_size, self.window_size))
                occupancyMaps_egocentric.append(window)
            
            batch_item.update({'semanticMaps_egocentric': torch.stack(occupancyMaps_egocentric, dim=0)})
            batch_item.update({'semanticMaps_per_agent': semanticMaps_per_agent}) """
            batch_item.update({'image': batch_data['image']})
            batch_item.update({'coords': transformed_trajectories})
            batch_item.update({'globalOccupancyMap': batch_data['input_traj_maps'].squeeze()})
        
        if self.data_format in ['partial_tokenized_random', 'partial_tokenized_by_frame']:
            # batch_data.append(np.stack(sequence['tokenized_sequences'], axis=0))
            batch_item.update({'tokens': sequence['tokenized_sequences']})
        if 'mtm_mask' in sequence:
            mtm_mask = sequence['mtm_mask'].todense()
            assert mtm_mask.shape == abs_pixel_coord[:,:,0].shape
            # batch_data.append(mtm_mask)
            batch_item.update({'mtm_mask': mtm_mask})
        
        return batch_item
    

def create_window(occMap, centers, window_size_2):
    x_i, y_i = centers
    if x_i-window_size_2 >= 0 and x_i+window_size_2 < occMap.shape[0] and y_i-window_size_2 >= 0 and y_i+window_size_2 < occMap.shape[1]:
        window = occMap[(x_i-window_size_2):(x_i+window_size_2), (y_i-window_size_2):(y_i+window_size_2)]
    else:
        x_min_pad = window_size_2 - x_i if x_i-window_size_2 < 0 else 0
        x_max_pad = window_size_2 - (occMap.shape[0] - x_i) if x_i+window_size_2 >= occMap.shape[0] else 0
        y_min_pad = window_size_2 - y_i if y_i-window_size_2 < 0 else 0
        y_max_pad = window_size_2 - (occMap.shape[1] - y_i) if y_i+window_size_2 >= occMap.shape[1] else 0
        occMap_n = torch.nn.functional.pad(occMap, ((y_min_pad, y_max_pad, x_min_pad, x_max_pad)), 'constant', value=0)
        window = occMap_n[(x_i+x_min_pad-window_size_2):(x_i+x_min_pad+window_size_2), (y_i+y_min_pad-window_size_2):(y_i+y_min_pad+window_size_2)]
    assert window.shape == (window_size_2*2, window_size_2*2)
    return window