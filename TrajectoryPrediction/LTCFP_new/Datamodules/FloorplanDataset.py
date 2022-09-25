import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import h5py
from helper import SEP
import matplotlib.pyplot as plt
import albumentations as A
import random
import torch
import cv2
from Datamodules.SceneLoader import Scene_floorplan
import warnings
from Modules.goal.models.goal.utils import create_CNN_inputs_loop
import torchvision.transforms.functional as F

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
    """
    Dataset class to load iteratively pre-made batches saved as pickle files.
    Apply data augmentation when needed.
    """

    def __init__(self, img_list: list, csv_list: list, split: str, batch_size: int = 32):

        self.name = "floorplan"
        self.floorplan_root = 'C:\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS'
        self.dataset_folder_rgb = os.path.join(self.floorplan_root, "INPUT")
        self.dataset_folder_csv = os.path.join(self.floorplan_root, "CSV_GT_TRAJECTORIES")

        self.layout_types = ['corr_e2e', 'corr_cross']

        self.seq_length = 20
        self.data_augmentation = True if split == 'train' else False

        # self.dataset_folder = self.dataset_folder_rgb # os.path.join(self.dataset_folder_rgb, scene_name.split(SEP)[0], scene_name.split(SEP)[1])
        # self.scene_folder = os.path.join(self.dataset_folder, self.name)

        self.scenes = {i: Scene_floorplan(img_path, csv_path, verbose=False) for  i, (img_path, csv_path) in enumerate(zip(img_list, csv_list))}# i, scene_name in enumerate(img_list)}

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
                raw_agent_data = raw_agent_data.iloc[::1]

                for start_t in range(0, len(raw_agent_data)):
                    candidate_traj = raw_agent_data.iloc[start_t:start_t + self.seq_length]
                    if len(candidate_traj) == self.seq_length:
                        if is_legitimate_traj(candidate_traj, step=1):

                            fragments_per_scene.append({
                                'fragemet_id': fragment_id,
                                'abs_pixel_coord': np.array(candidate_traj[["x_coord", "y_coord"]].values).astype(np.float32)
                                # 'starting_frame': candidate_traj.frame_id.iloc[0],
                                # 'agent_id': candidate_traj.agent_id.iloc[0],
                            })

                            fragment_id += 1

            random.shuffle(fragments_per_scene)

            self.fragment_list += [{
                'batch_of_fragments': fragments_per_scene[x:x+batch_size],
                'scene_path': scene.RGB_image_path,
                'scene_data': {
                    'scene_id': sc_id,
                    'image_res_x': scene.image_res_x,
                    'image_res_y': scene.image_res_y,
                    'floorplan_min_x': scene.floorplan_min_x,
                    'floorplan_min_y': scene.floorplan_min_y,
                    'floorplan_max_x': scene.floorplan_max_x,
                    'floorplan_max_y': scene.floorplan_max_y
                    }
                } for x in range(0, len(fragments_per_scene), batch_size)]

        self.fragment_list = self.fragment_list[:2]

    def __len__(self):
        return len(self.fragment_list)

    def augment_traj_and_create_traj_maps(self, batch_data, np_image, augmentation):

        image = np_image
        abs_pixel_coord = batch_data["abs_pixel_coord"]
        # input_traj_maps = batch_data["input_traj_maps"]
        site_x = batch_data['scene_data']['floorplan_max_x']
        site_y = batch_data['scene_data']['floorplan_max_y']

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
            batch_abs_pixel_coords=torch.tensor(keypoints.reshape(self.seq_length, -1, 2)).float(),
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
            'scene_data': fragment['scene_data'],
            'seq_list': seq_list
        }

        warnings.filterwarnings("ignore")
        batch_data = self.augment_traj_and_create_traj_maps(batch_data, np_image, self.data_augmentation)
        
        return batch_data