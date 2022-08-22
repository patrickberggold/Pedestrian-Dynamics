import os
import pickle
import random
import albumentations as A

import torch
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from skimage.draw import line

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from src.models.model_utils.cnn_big_images_utils import create_tensor_image, create_CNN_inputs_loop
from src.data_src.scene_src.scene_floorplan import Scene_floorplan
from src.m_transforms import m_RandomHorizontalFlip, m_RandomVerticalFlip, m_RandomRotation
import warnings

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


class Dataset_LTCFP(Dataset):
    """
    Dataset class to load iteratively pre-made batches saved as pickle files.
    Apply data augmentation when needed.
    """

    def __init__(self, args, split='train', full_dataset=True):

        self.name = "floorplan"
        self.floorplan_root = 'C:\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS'
        self.dataset_folder_rgb = os.path.join(self.floorplan_root, "INPUT")
        self.dataset_folder_csv = os.path.join(self.floorplan_root, "CSV_GT_TRAJECTORIES")

        self.layout_types = ['corr_e2e', 'corr_cross']

        self.args = args
        self.data_augmentation = args.data_augmentation \
            if split == 'train' else False

        # self.dataset_folder = self.dataset_folder_rgb # os.path.join(self.dataset_folder_rgb, scene_name.split(SEP)[0], scene_name.split(SEP)[1])
        # self.scene_folder = os.path.join(self.dataset_folder, self.name)

        csv_path_list = [
            os.path.join(layout_type, flooplan_folder, variation_folder, csv_file) \
                for layout_type in self.layout_types \
                for flooplan_folder in os.listdir(os.path.join(self.dataset_folder_csv, layout_type)) \
                for variation_folder in os.listdir(os.path.join(self.dataset_folder_csv, layout_type, flooplan_folder)) \
                for csv_file in os.listdir(os.path.join(self.dataset_folder_csv, layout_type, flooplan_folder, variation_folder))
        ]

        len_rgb_dataset = len(csv_path_list)
        random.shuffle(csv_path_list)
        test_split_index = int(0.2 * len_rgb_dataset)

        if full_dataset:
            if split in ['test', 'valid']:
                # self.scenes = [Scene_floorplan(scene_name, verbose=False) for scene_name in [csv_path_list[i] for i in range(0, test_split_index)]]
                self.scenes = {i: Scene_floorplan(scene_name, verbose=False) for i in range(0, test_split_index) for scene_name in [csv_path_list[i]]}
            elif split == 'train':
                # self.scenes = [Scene_floorplan(scene_name, verbose=False) for scene_name in [csv_path_list[i] for i in range(test_split_index, len_rgb_dataset)]]
                self.scenes = {i: Scene_floorplan(scene_name, verbose=False) for i in range(test_split_index, len_rgb_dataset) for scene_name in [csv_path_list[i]]}
            else:
                raise ValueError
        else:
            if split in ['test', 'valid']:
                # self.scenes = [Scene_floorplan(scene_name, verbose=False) for scene_name in[csv_path_list[i] for i in range(0, 1)]]
                self.scenes = {i: Scene_floorplan(scene_name, verbose=False) for i in range(0, 1) for scene_name in[csv_path_list[i]]}
            elif split == 'train':
                # self.scenes = [Scene_floorplan(scene_name, verbose=False) for scene_name in [csv_path_list[i] for i in range(1, 3)]]
                self.scenes = {i: Scene_floorplan(scene_name, verbose=False) for i in range(1, 3) for scene_name in[csv_path_list[i]]}
            else:
                raise ValueError

        self.downsample_frame_rate = 1
        delta_frame = 1
        self.down_factor = 1 # resizing happens exclusively in the end
        self.final_resolution = 800
        self.max_floorplan_meters = 70

        ### TODO implement this more like AgentFormer, but try out this 'loose' approach too
        fragments_per_scene = []
        self.fragment_list = []

        fragment_id = 0
        for sc_id, scene in self.scenes.items():
            raw_pixel_data = scene.raw_pixel_data

            fragments_per_scene = []
            for agent_i in set(raw_pixel_data.agent_id):
                raw_agent_data = raw_pixel_data[raw_pixel_data.agent_id == agent_i]
                # downsample frame rate happens here, at the single agent level
                raw_agent_data = raw_agent_data.iloc[::self.downsample_frame_rate]

                for start_t in range(0, len(raw_agent_data), self.args.skip_ts_window):
                    candidate_traj = raw_agent_data.iloc[start_t:start_t + self.args.seq_length]
                    if len(candidate_traj) == self.args.seq_length:
                        if is_legitimate_traj(candidate_traj, step=self.downsample_frame_rate * delta_frame):

                            fragments_per_scene.append({
                                'fragemet_id': fragment_id,
                                'abs_pixel_coord': np.array(candidate_traj[["x_coord", "y_coord"]].values).astype(np.float32),
                                # 'starting_frame': candidate_traj.frame_id.iloc[0],
                                # 'agent_id': candidate_traj.agent_id.iloc[0],
                            })

                            fragment_id += 1

            random.shuffle(fragments_per_scene)

            self.fragment_list += [{
                'batch_of_fragments': fragments_per_scene[x:x+args.batch_size],
                'scene_path': os.path.join(scene.scene_folder, scene.RGB_image_name),
                'scene_id': sc_id
                } for x in range(0, len(fragments_per_scene), args.batch_size)]

    def __len__(self):
        return len(self.fragment_list)

    def augment_traj_and_create_traj_maps(self, batch_data, np_image, augmentation):

        image = np_image
        abs_pixel_coord = batch_data["abs_pixel_coord"]
        # input_traj_maps = batch_data["input_traj_maps"]
        site_x = self.scenes[batch_data['scene_id']].floorplan_max_x
        site_y = self.scenes[batch_data['scene_id']].floorplan_max_y

        scale_x = site_x / self.max_floorplan_meters
        scale_y = site_y / self.max_floorplan_meters

        assert 0.0 <= scale_x <= 1.0 and 0.0 <= scale_y <= 1.0

        scaled_resolution_x = int(self.final_resolution * scale_x)
        scaled_resolution_y = int(self.final_resolution * scale_y)

        # get old channels for safety checking
        old_H, old_W, C = np_image.shape

        # keypoints to list of tuples
        # need to clamp because some slightly exit from the image
        # abs_pixel_coord[:, 0] = np.clip(abs_pixel_coord[:, 0],
        #                                    a_min=0, a_max=old_W - 1e-3)
        # abs_pixel_coord[:, 1] = np.clip(abs_pixel_coord[:, 1],
        #                                    a_min=0, a_max=old_H - 1e-3)
        # Check whether some keypoints are outside the image
        x_coord_big = np.argwhere(abs_pixel_coord[:, :, 0] > old_W)
        y_coord_big = np.argwhere(abs_pixel_coord[:, :, 1] > old_H)
        x_coord_small = np.argwhere(abs_pixel_coord[:, :, 0] < 0)
        y_coord_small = np.argwhere(abs_pixel_coord[:, :, 1] < 0)

        assert x_coord_big.shape[0] == y_coord_big.shape[0] == x_coord_small.shape[0] == y_coord_small.shape[0] == 0, \
            f'Some traj points not within image, outside shapes: {x_coord_big.shape[0]}, {y_coord_big.shape[0]}, {x_coord_small.shape[0]} and {y_coord_small.shape[0]}'

        keypoints = list(map(tuple, abs_pixel_coord.reshape(-1, 2)))

        # Resize first to create Gaussian maps later
        transform = A.Compose([
            A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
        ],
            keypoint_params=A.KeypointParams(format='xy',
                                            remove_invisible=False))
        
        transformed = transform(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = np.array(transformed['keypoints'])
        
        input_traj_maps = create_CNN_inputs_loop(
            batch_abs_pixel_coords=torch.tensor(keypoints.reshape(self.args.seq_length, -1, 2)).float(),
            tensor_image=F.to_tensor(image))
        bs, T, old_H, old_W = input_traj_maps.shape
        input_traj_maps = input_traj_maps.view(bs * T, old_H, old_W).\
            permute(1, 2, 0).numpy().astype('float32')

        if augmentation:
            transform = A.Compose([
                # SAFE AUGS, flips and 90rots
                # A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
                A.augmentations.transforms.HorizontalFlip(p=0.5),
                A.augmentations.transforms.VerticalFlip(p=0.5),
                A.augmentations.transforms.Transpose(p=0.5),
                A.augmentations.geometric.rotate.RandomRotate90(p=1.0),
                A.augmentations.transforms.PadIfNeeded(min_height=self.final_resolution, min_width=self.final_resolution, border_mode=cv2.BORDER_CONSTANT, value=[0., 0., 0.]),
                # TODO implement continuous rotations from within [0;360] at some point, maybe change the transform order for that
                # in that case watch out for cases when image extends the image borders by a few pixels after rotation
                A.augmentations.geometric.rotate.Rotate(limit=45, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),

                # # HIGH RISKS - HIGH PROBABILITY OF KEYPOINTS GOING OUT
                # A.OneOf([  # perspective or shear
                #     A.augmentations.geometric.transforms.Perspective(
                #         scale=0.05, pad_mode=cv2.BORDER_CONSTANT, p=1.0),
                #     A.augmentations.geometric.transforms.Affine(
                #         shear=(-10, 10), mode=cv2.BORDER_CONSTANT, p=1.0),  # shear
                # ], p=0.2),

                # A.OneOf([  # translate
                #     A.augmentations.geometric.transforms.ShiftScaleRotate(
                #         shift_limit_x=0.01, shift_limit_y=0, scale_limit=0,
                #         rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
                #         p=1.0),  # x translations
                #     A.augmentations.geometric.transforms.ShiftScaleRotate(
                #         shift_limit_x=0, shift_limit_y=0.01, scale_limit=0,
                #         rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
                #         p=1.0),  # y translations
                #     A.augmentations.geometric.transforms.Affine(
                #         translate_percent=(0, 0.01),
                #         mode=cv2.BORDER_CONSTANT, p=1.0),  # random xy translate
                # ], p=0.2),
                # # random rotation
                # A.augmentations.geometric.rotate.Rotate(
                #     limit=10, border_mode=cv2.BORDER_CONSTANT,
                #     p=0.4),
            ],
                keypoint_params=A.KeypointParams(format='xy',
                                                remove_invisible=False),
                additional_targets={'traj_map': 'image'},
            )
        else:
            transform = A.Compose([
                # A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
                A.augmentations.transforms.PadIfNeeded(min_height=self.final_resolution, min_width=self.final_resolution, border_mode=cv2.BORDER_CONSTANT, value=[0., 0., 0.]),
            ],
                keypoint_params=A.KeypointParams(format='xy',
                                                remove_invisible=False),
                additional_targets={'traj_map': 'image'},
            )

        #################### TEST START ####################
        # plt.imshow(image/255.) # weg
        # img_np = image#/255.
        # traj = np.array(keypoints)
        # # traj = [t for t in traj]
        # trajs = [traj[x:x+20] for x in range(0, len(traj), 20)]
        # # for t_id in traj:
        # for traj in trajs:
        #     old_point = None
        #     r_val = random.uniform(0.4, 1.0)
        #     b_val = random.uniform(0.7, 1.0)
        #     for point in traj:
        #         x_n, y_n = round(point[0]), round(point[1])
        #         assert 0 <= x_n <= img_np.shape[1] and 0 <= y_n <= img_np.shape[0], f'{x_n} > {img_np.shape[1]} or {y_n} > {img_np.shape[0]}'
        #         if old_point != None:
        #             # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
        #             c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, y_n)))]
        #             for c in c_line:
        #                 img_np[c[1], c[0]] = np.array([r_val, 0., b_val])
        #             # plt.imshow(img_np)
        #         old_point = (x_n, y_n)

        # plt.imshow(img_np)
        # plt.close('all')
        #################### TEST END ####################

        transformed = transform(
            image=image, keypoints=keypoints, traj_map=input_traj_maps)
        # #################### TEST START ####################
        # plt.imshow(image/255.) # weg
        # img_np = transformed['image']#/255.

        traj = np.array(transformed['keypoints'])
        trajs = [traj[x:x+20] for x in range(0, len(traj), 20)]
        # for t_id in traj:
        # for traj in trajs:
        #     old_point = None
        #     r_val = random.uniform(0.4, 1.0)
        #     b_val = random.uniform(0.7, 1.0)
        #     for point in traj:
        #         x_n, y_n = round(point[0]), round(point[1])
        #         assert 0 <= x_n <= img_np.shape[1] and 0 <= y_n <= img_np.shape[0]
        #         if old_point != None:
        #             # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
        #             c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, y_n)))]
        #             for c in c_line:
        #                 img_np[c[1], c[0]] = np.array([r_val, 0., b_val])
        #             # plt.imshow(img_np)

        #     old_point = (x_n, y_n)
        # # cv2.imshow('namey', img_np)
        # plt.imshow(img_np)
        # plt.close('all')
        # #################### TEST END ####################
        # FROM NUMPY BACK TO TENSOR
        image = torch.tensor(transformed['image']).permute(2, 0, 1)
        C, new_H, new_W = image.shape
        abs_pixel_coord = torch.tensor(transformed['keypoints']).view(batch_data["abs_pixel_coord"].shape)
        input_traj_maps = torch.tensor(transformed['traj_map']).permute(2, 0, 1).view(bs, T, new_H, new_W)

        # Check whether some keypoints are outside the image
        x_coord_big = torch.argwhere(abs_pixel_coord[:, :, 0] > new_W)
        y_coord_big = torch.argwhere(abs_pixel_coord[:, :, 1] > new_H)
        x_coord_small = torch.argwhere(abs_pixel_coord[:, :, 0] < 0)
        y_coord_small = torch.argwhere(abs_pixel_coord[:, :, 1] < 0)

        if not (x_coord_big.size(0) == y_coord_big.size(0) == x_coord_small.size(0) == y_coord_small.size(0) == 0):
            print('After rotation, some trajectories dont lie within output image. Output shapes: ' + \
                f'{x_coord_big.size(0)}, {y_coord_big.size(0)}, {x_coord_small.size(0)} and { y_coord_small.size(0)}')

        # Clamping of trajectory values if they exceed the image boundaries (very rare)
        abs_pixel_coord[:, :, 0] = torch.clamp(abs_pixel_coord[:, :, 0], min=0, max=new_W)
        abs_pixel_coord[:, :, 1] = torch.clamp(abs_pixel_coord[:, :, 1], min=0, max=new_H)

        assert new_W == self.final_resolution and new_H == self.final_resolution

        # NEW AUGMENTATION: INVERT TIME
        # if random.random() > 0.5:
        #     abs_pixel_coord = abs_pixel_coord.flip(dims=(0,))
        #     input_traj_maps = input_traj_maps.flip(dims=(1,))

        # To torch.tensor.float32
        batch_data["tensor_image"] = image.float()
        batch_data["abs_pixel_coord"] = abs_pixel_coord.float()
        batch_data["input_traj_maps"] = input_traj_maps.float()
        batch_data["seq_list"] = torch.tensor(batch_data["seq_list"]).float()

        return batch_data


    def __getitem__(self, idx):
        
        # LOAD CSV DATA
        fragment = self.fragment_list[idx]

        abs_pixel_coord = np.moveaxis(np.stack([frag_coord['abs_pixel_coord'] for frag_coord in fragment['batch_of_fragments']], axis=0), 0, 1)

        seq_list =  np.ones((abs_pixel_coord.shape[0], abs_pixel_coord.shape[1]))

        # tensor_image, [scale_x, scale_y, rotation_angle, h_flip, v_flip] = self.image_and_traj_preprocessing(fragment['scene_path'], fragment['abs_pixel_coord'])
        
        # to get a clear image, convert to bool first to get rid of values betweet 1 and 254 (all stored as 1)
        np_image = np.array(h5py.File(fragment['scene_path'], 'r').get('img')).astype(bool).astype(np.float32)

        # input_traj_maps = create_CNN_inputs_loop(
        #     batch_abs_pixel_coords=torch.tensor(np.moveaxis(abs_pixel_coord, 0, 1)).float(),
        #     tensor_image=F.to_tensor(np_image))

        batch_data = {
            'abs_pixel_coord': abs_pixel_coord,
            'scene_id': fragment['scene_id'],
            'seq_list': seq_list
        }

        warnings.filterwarnings("ignore")
        batch_data = self.augment_traj_and_create_traj_maps(batch_data, np_image, self.data_augmentation)
        
        return batch_data


def get_dataloader_direct(args, set_name, full_dataset=True):
    """
    Create a data loader for a specific set/data split
    """
    assert set_name in ['train', 'valid', 'test']

    shuffle = args.shuffle_train_batches if set_name == 'train' else \
        args.shuffle_test_batches

    dataset = Dataset_LTCFP(args, split=set_name, full_dataset=full_dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=args.num_workers)

    return loader


def collate_fn(batch_list):

    scene_path_list = [el['scene_path'] for el in batch_list]
    # convert scene path list to set to check whether all paths are identical 
    assert len(set(scene_path_list)) == 1

    # load the scene
    np_image = np.array(h5py.File(batch_list[0]['scene_path'], 'r').get('img')).astype(np.float32)

    coord_list = [el['abs_pixel_coord'] for el in batch_list]
    abs_pixel_coord = np.concatenate(coord_list, axis=0)

    for batch_item in batch_list:
        abs_pixel_coord = batch_list['abs_pixel_coord']

        input_traj_maps = create_CNN_inputs_loop(
            batch_abs_pixel_coords=torch.tensor(abs_pixel_coord).float().unsqueeze(1),
            tensor_image=F.to_tensor(np_image))

    # use the collate to only load one augmented image for X trajectories

    batch_data = {
        'abs_pixel_coord': None,
        'seq_list': None,
        'scene': None,
        'scene_path': None

    }

    return batch_data