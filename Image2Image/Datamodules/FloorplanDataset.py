from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import h5py
from helper import SEP
import matplotlib.pyplot as plt
from helper import get_color_from_array
import albumentations as A
import random
import torch
import cv2

class Dataset_Img2Img(Dataset):
    def __init__(
        self, 
        mode: str, 
        img_paths: list, 
        traj_paths: list, 
        transform = None, 
        vary_area_brightness: bool = True, 
        num_ts_per_floorplan: int = 1,
    ):
        # TODO maybe turn off transformations when eval/test
        self.transform = transform
        self.img_paths = img_paths
        self.traj_paths = traj_paths
        self.mode = mode
        self.num_ts_per_floorplan = num_ts_per_floorplan
        self.vary_area_brightness = vary_area_brightness
        self.pred_evac_times = True if self.mode == 'evac' else False
        self.mode = 'grayscale' if self.mode == 'evac' else self.mode
        self.max_floorplan_meters = 70
        self.final_resolution = 800

        assert mode in ['grayscale', 'grayscale_movie', 'evac'], 'Unknown mode setting!'
        assert len(self.traj_paths) == len(self.img_paths), 'Length of image paths and trajectory paths do not match, something went wrong!'

    def __len__(self):
        return(len(self.img_paths))
    
    def __getitem__(self, idx):
        img_path, traj_path = self.img_paths[idx], self.traj_paths[idx]

        self.floorplan_max_x = int(img_path.split(SEP)[-1].split('siteX_')[1].split('_siteY')[0])
        self.floorplan_max_y = int(img_path.split(SEP)[-1].split('siteY_')[1].split('_')[0])

        # quick check if paths are consistent
        assert '_'.join(img_path.split('_')[-2:]).replace('.h5', '') in traj_path
        
        evac_time = float(np.array(h5py.File(traj_path, 'r').get('max_time')))
        img = np.array(h5py.File(img_path, 'r').get('img'))
        img = img.astype(bool).astype(np.float32) # deal with compression errors: some values not shown in full color
        # plt.imshow(img)

        # Change origin and destination area brightnesses
        if self.vary_area_brightness:
            agent_variations = [15, 25, 50]
            # bright_limit = 100
            # dark_limit = 155
            # color_interval_length = (dark_limit+bright_limit)//len(agent_variations)
            agent_variations_index = agent_variations.index(int(traj_path.split(SEP)[-1].split('_')[-1].replace('.h5', '')))
            # reset origin color
            # or_color_range = [[255, 100, 100], [255, 50, 50], [255, 0, 0], [205, 0, 0], [155, 0, 0]]
            or_color_range = [[1., 0.392, 0.392], [1., 0, 0], [0.608, 0, 0]]
            or_area_color = np.array(or_color_range[agent_variations_index])
            or_coords = np.where(np.all(img == np.array([1.0, 0, 0]), axis=-1))
            img[or_coords[0], or_coords[1]] = or_area_color
            # reset destination color
            # dst_color_range = [[100, 255, 100], [50, 255, 50], [0, 255, 0], [0, 205, 0], [0, 155, 0]]
            dst_color_range = [[0.392, 1., 0.392], [0, 1., 0], [0, 0.608, 0]]
            dst_area_color = np.array(dst_color_range[agent_variations_index])
            dst_coords = np.where(np.all(img == np.array([0, 1.0, 0]), axis=-1))
            img[dst_coords[0], dst_coords[1]] = dst_area_color

            assert len(or_coords[0]) > 0 and len(or_coords[0])==len(or_coords[1])
            assert len(dst_coords[0]) > 0 and len(dst_coords[0])==len(dst_coords[1])
            # plt.imshow(img)

        if self.mode == 'grayscale':
            traj = np.array(h5py.File(traj_path, 'r').get('img')).astype('float32')

        elif self.mode == 'grayscale_movie':
            traj0 = np.array(h5py.File(traj_path, 'r').get('img')).astype('float32')
            # traj0 /= traj0.max() # Normalization
            # ts_limits = [(i+1)*traj0.max()/self.num_ts_per_floorplan for i in range(self.num_ts_per_floorplan)]
            ts_limits = [i * traj0.max()/self.num_ts_per_floorplan for i in range(self.num_ts_per_floorplan+1)]
            traj = [traj0] + [traj0.copy() for i in range(1,self.num_ts_per_floorplan)]
            
            for idx in range(len(ts_limits)-1):
                traj[idx][traj[idx] > ts_limits[idx+1]] = 0.
                traj[idx][traj[idx] <= ts_limits[idx]] = 0.
        
        # Visualize trajectory for checking correctness
        # non_zeros = np.argwhere(traj != 0.)
        # img_now = img.copy()
        # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 255])
        # plt.imshow(img_now, vmin=0, vmax=255)

        # if self.transform:
        #     img = self.transform(img)
        augmentation = True # change this later maybe for testing
        img, traj = self.augment_traj_and_images(img, traj, augmentation, evac_time)
        
        if self.pred_evac_times:
            img = (img, evac_time)
        
        if self.transform:
            img = self.transform(img)

        return img, traj


    def augment_traj_and_images(self, image, traj_image, augmentation, evac_time):

        scale_x = self.floorplan_max_x / self.max_floorplan_meters
        scale_y = self.floorplan_max_y / self.max_floorplan_meters

        assert 0.0 <= scale_x <= 1.0 and 0.0 <= scale_y <= 1.0

        scaled_resolution_x = int(self.final_resolution * scale_x)
        scaled_resolution_y = int(self.final_resolution * scale_y)

        # Resize first to create Gaussian maps later
        transform = A.Compose([
            A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
            ],
            additional_targets={'traj_map': 'image'})
        
        transformed = transform(image=image, traj_map=traj_image)

        #################### TEST START ####################
        # 2D plot pred
        # plt.imshow(image)
        # non_zeros = np.argwhere(traj_image != 0.)
        # img_now = image.copy()
        # pred_colors_from_timestamps = [get_color_from_array(traj_image[x, y], evac_time)/255. for x, y in non_zeros]
        # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array(pred_colors_from_timestamps)
        # # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 1.])
        # plt.imshow(img_now)
        
        # 3D plot pred
        # fig = plt.figure(figsize=(6,6))
        # ax = fig.add_subplot(111, projection='3d')
        # X,Y = np.meshgrid(np.arange(traj_image.shape[1]), np.arange(traj_image.shape[0]))
        # ax.plot_surface(X, Y, traj_image)
        # plt.show()
        # plt.close('all')
        #################### TEST END ####################

        image = transformed['image']
        traj_image = transformed['traj_map']

        #################### TEST START ####################
        # 2D plot pred
        # # plt.imshow(image)
        # non_zeros = np.argwhere(traj_image != 0.)
        # img_now = image.copy()
        # pred_colors_from_timestamps = [get_color_from_array(traj_image[x, y], evac_time)/255. for x, y in non_zeros]
        # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array(pred_colors_from_timestamps)
        # # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 1.])
        # plt.imshow(img_now)
        
        # # 3D plot pred
        # fig = plt.figure(figsize=(6,6))
        # ax = fig.add_subplot(111, projection='3d')
        # X,Y = np.meshgrid(np.arange(traj_image.shape[1]), np.arange(traj_image.shape[0]))
        # ax.plot_surface(X, Y, traj_image)
        # plt.show()
        # plt.close('all')
        #################### TEST END ####################

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
                additional_targets={'traj_map': 'image'},
            )
        else:
            transform = A.Compose([
                # A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
                A.augmentations.transforms.PadIfNeeded(min_height=self.final_resolution, min_width=self.final_resolution, border_mode=cv2.BORDER_CONSTANT, value=[0., 0., 0.]),
            ],
                additional_targets={'traj_map': 'image'},
            )


        transformed = transform(
            image=image, traj_map=traj_image)

        #################### TEST START ####################
         # 2D plot pred
        # plt.imshow(transformed['image'])
        # non_zeros = np.argwhere(transformed['traj_map'] != 0.)
        # img_now = transformed['image'].copy()
        # pred_colors_from_timestamps = [get_color_from_array(transformed['traj_map'][x, y], evac_time)/255. for x, y in non_zeros]
        # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array(pred_colors_from_timestamps)
        # # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 1.])
        # plt.imshow(img_now)
        
        # 3D plot pred
        # fig = plt.figure(figsize=(6,6))
        # ax = fig.add_subplot(111, projection='3d')
        # X,Y = np.meshgrid(np.arange(transformed['traj_map'].shape[1]), np.arange(transformed['traj_map'].shape[0]))
        # ax.plot_surface(X, Y, transformed['traj_map'])
        # plt.show()
        #################### TEST END ####################
        
        # FROM NUMPY BACK TO TENSOR
        image = torch.tensor(transformed['image']).permute(2, 0, 1)
        traj_image = torch.tensor(transformed['traj_map'])

        assert image.size(1) == self.final_resolution and image.size(2) == self.final_resolution
        assert traj_image.size(1) == self.final_resolution and traj_image.size(0) == self.final_resolution

        # NEW AUGMENTATION: INVERT TIME
        # if random.random() > 0.5:
        #     abs_pixel_coord = abs_pixel_coord.flip(dims=(0,))
        #     input_traj_maps = input_traj_maps.flip(dims=(1,))

        # Return torch.tensor.float32
        return image.float(), traj_image.float()
