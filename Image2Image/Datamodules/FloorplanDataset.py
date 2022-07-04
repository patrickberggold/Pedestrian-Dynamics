from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import random
import h5py
from helper import SEP
import matplotlib.pyplot as plt
from helper import get_color_from_array

class semantic_dataset(Dataset):
    def __init__(self, split = 'train', transform = None):
        # self.void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        # self.valid_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        # self.ignore_index = 250
        self.void_labels = []
        self.valid_labels = [0, 1, 2, 3, 4]
        self.ignore_index = 25
        self.class_map = dict(zip(self.valid_labels, range(5)))
        self.split = split
        # self.img_path = '/home/Datasets/Segmentation/KITTI/testing/image_2/'
        self.img_path = '/home/Datasets/Segmentation/Floorplans/HDF5_IMAGES_resolution_800_800'
        self.mask_path = None
        if self.split == 'train' or self.split == 'val':
            # self.img_path = '/home/Datasets/Segmentation/KITTI/training/image_2/'    
            # self.mask_path = '/home/Datasets/Segmentation/KITTI/training/semantic/'
            self.mask_path = '/home/Datasets/Segmentation/Floorplans/HDF5_GT_resolution_800_800'
        self.transform = transform
        
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = None
        if self.split == 'train' or self.split == 'val':
            self.mask_list = self.get_filenames(self.mask_path)
    	
        val_split_factor = 0.0
        test_split_factor = 0.2
        indices = list(range(len(self.img_list)))
        val_split_index = int(len(indices) * val_split_factor)
        test_split_index = int(len(indices) * test_split_factor)
        random.seed(0)
        random.shuffle(indices)
        if self.split == 'train':
            assert len(self.img_list) == len(self.mask_list)
            self.img_list = [self.img_list[idx] for idx in indices[(test_split_index + val_split_index):]]
            self.mask_list = [self.mask_list[idx] for idx in indices[(test_split_index + val_split_index):]]
        elif self.split == 'val':
            assert len(self.img_list) == len(self.mask_list)
            self.img_list = [self.img_list[idx] for idx in indices[test_split_index:(test_split_index + val_split_index)]]
            self.mask_list = [self.mask_list[idx] for idx in indices[test_split_index:(test_split_index + val_split_index)]]
        elif self.split == 'test':
            self.img_list = [self.img_list[idx] for idx in indices[:test_split_index]]
        else:
            raise ValueError()

    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        # img = cv2.imread(self.img_list[idx])
        # img = cv2.resize(img, (1242, 376))
        img = np.array(h5py.File(self.img_list[idx], 'r').get('img'))       
        mask = None
        if self.split == 'train':
            mask = np.array(h5py.File(self.mask_list[idx], 'r').get('img'))
            # mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
            # mask = cv2.resize(mask, (1242, 376))
            # mask = self.encode_segmap(mask)
            # assert(mask.shape == (376, 1242))
        
        # plot images to test correctness
        # plt.imsave(f'test_img_{idx}.jpeg', img, vmin=0, vmax=255)
        # if isinstance(mask, np.ndarray): 
        #     plt.imsave(f'test_mask_{idx}.jpeg', mask, cmap=get_customized_colormap())

        if self.transform:
            img = self.transform(img)
            # assert(img.shape == (3, 376, 1242))
        # else :
        #     assert(img.shape == (376, 1242, 3))

        if self.split == 'train':
            return img, mask
        else :
            return img
    
    def encode_segmap(self, mask):
        '''
        Sets void classes to zero so they won't be considered for training
        '''
        for voidc in self.void_labels :
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels :
            mask[mask == validc] = self.class_map[validc]
        return mask
    
    def get_filenames(self, path):
        files_list = list()
        for foldername in os.listdir(path):
            for filename in os.listdir(os.path.join(path, foldername)):
                if filename.endswith('.h5'):
                    files_list.append(os.path.join(path, foldername, filename))
        # return [os.path.join(path, filename) for filename in os.listdir(path)]
        # listy = [os.path.join(path, foldername, filename) for filename in os.listdir(os.path.join(path, foldername)) if filename.endswith('.h5') for foldername in os.listdir(path)]
        return files_list

class img2img_dataset_traj_1D(Dataset):
    def __init__(self, mode: str, img_paths: list, traj_paths: list, transform = None, non_traj_vals: float = 0., max_traj_factor: float = 1.0, vary_area_brightness: bool = False, num_ts_per_floorplan: int = 1):

        self.transform = transform
        self.img_paths = img_paths
        self.traj_paths = traj_paths
        self.mode = mode
        self.non_traj_vals = non_traj_vals
        self.max_traj_factor = max_traj_factor
        self.num_ts_per_floorplan = num_ts_per_floorplan
        self.vary_area_brightness = vary_area_brightness

        assert mode in ['grayscale', 'rgb', 'bool', 'timeAndId', 'grayscale_movie', 'counts'], 'Unknown mode setting!'
        assert len(self.traj_paths) == len(self.img_paths), 'Length of image paths and trajectory paths do not match, something went wrong!'

    def __len__(self):
        return(len(self.img_paths))
    
    def __getitem__(self, idx):
        img_path, traj_path = self.img_paths[idx], self.traj_paths[idx]
        # quick check if paths are consistent
        assert SEP.join(img_path.split(SEP)[-2:]) == SEP.join(traj_path.split(SEP)[-2:])
        
        img = np.array(h5py.File(img_path, 'r').get('img'))
        # plt.imsave(f'test_input_img_{idx}.jpeg', img, vmin=0, vmax=255)

        # Change origin and destination area brightnesses
        if self.vary_area_brightness:
            agent_variations = [10, 20, 30, 40, 50]
            # bright_limit = 100
            # dark_limit = 155
            # color_interval_length = (dark_limit+bright_limit)//len(agent_variations)
            agent_variations_index = agent_variations.index(int(traj_path.split(SEP)[-3].split('_')[-1]))
            # reset origin color
            or_color_range = [[255, 100, 100], [255, 50, 50], [255, 0, 0], [205, 0, 0], [155, 0, 0]]
            or_area_color = np.array(or_color_range[agent_variations_index])
            or_coords = np.where(np.all(img == np.array([255, 0, 0]), axis=-1))
            img[or_coords[0], or_coords[1]] = or_area_color
            # reset destination color
            dst_color_range = [[100, 255, 100], [50, 255, 50], [0, 255, 0], [0, 205, 0], [0, 155, 0]]
            dst_area_color = np.array(dst_color_range[agent_variations_index])
            dst_coords = np.where(np.all(img == np.array([0, 255, 0]), axis=-1))
            img[dst_coords[0], dst_coords[1]] = dst_area_color

            assert len(or_coords[0]) > 0 and len(or_coords[0])==len(or_coords[1])
            assert len(dst_coords[0]) > 0 and len(dst_coords[0])==len(dst_coords[1])
            # plt.imshow(img.astype('uint8'), vmin=0, vmax=255)

        if self.mode == 'grayscale':
            traj = np.array(h5py.File(traj_path, 'r').get('img')).astype('float32')/self.max_traj_factor # self.max_traj_factor=89.5 normalizes the traj values for 40 agents
            # SETTING 1
            # Scale timestamp values with respect to maximum timestamp value
            # traj /= 89.5
            # traj[traj == 0.] = -0.02
            # SETTING 2
            traj[traj == 0] = self.non_traj_vals

        elif self.mode == 'grayscale_movie':
            traj0 = np.array(h5py.File(traj_path, 'r').get('img')).astype('float32')
            # traj0 /= traj0.max() # Normalization
            # ts_limits = [(i+1)*traj0.max()/self.num_ts_per_floorplan for i in range(self.num_ts_per_floorplan)]
            ts_limits = [i * traj0.max()/self.num_ts_per_floorplan for i in range(self.num_ts_per_floorplan+1)]
            traj = [traj0] + [traj0.copy() for i in range(1,self.num_ts_per_floorplan)]
            
            # for idx, limit in enumerate(ts_limits):
            #     traj[idx][traj[idx] == 0] = self.non_traj_vals
            #     traj[idx][traj[idx] > limit] = self.non_traj_vals
            for idx in range(len(ts_limits)-1):
                traj[idx][traj[idx] == 0] = self.non_traj_vals
                # traj[idx][traj[idx] > limit] = self.non_traj_vals
                traj[idx][traj[idx] > ts_limits[idx+1]] = self.non_traj_vals
                traj[idx][traj[idx] <= ts_limits[idx]] = self.non_traj_vals
        
        if self.mode == 'counts':
            traj = np.array(h5py.File(traj_path, 'r').get('img')).astype('float32')[:,:,1] # self.max_traj_factor=89.5 normalizes the traj values for 40 agents
            # SETTING 1
            # Scale timestamp values with respect to maximum timestamp value
            # traj /= 89.5
            # traj[traj == 0.] = -0.02
            # SETTING 2
            traj[traj == 0] = self.non_traj_vals

        elif self.mode == 'rgb':
            traj = np.array(h5py.File(traj_path, 'r').get('img'))
            if self.transform:
                traj = self.transform(traj)
            traj = traj.astype('float32')
        elif self.mode == 'bool':
            traj = np.array(h5py.File(traj_path, 'r').get('img')).astype('bool').astype('float32')
        elif self.mode == 'timeAndId':
            traj = np.array(h5py.File(traj_path, 'r').get('img'))
            # SETTING 1
            # Scale timestamp values with respect to maximum timestamp value
            # traj[:,:,0] /= 89.5
            # traj[:,:,0][traj[:,:,0] == 0.] = -0.02
            # SETTING 2
            traj[:,:,0][traj[:,:,0] == 0] = self.non_traj_vals
            traj = traj.astype('float32')
            # traj = traj[:,:,1]

            # # Generate random arrayas for testing forward pass
            # traj = np.random.randint(0, high=41, size=(800,800))
            # # Append random int-array for testing
            # rand_int_array = np.random.randint(0, high=41, size=(800,800,1), dtype=int).astype('float32') # generates 800x800 random int array, min_val=0, max_val=40
            # # assign zeros where no trajectories are...
            # zeros_in_traj = np.argwhere(traj == -5.)
            # rand_int_array[zeros_in_traj[:,0], zeros_in_traj[:,1]] = 0.
            # traj = np.expand_dims(traj, 2)
            # traj_app = np.concatenate((traj, rand_int_array), axis=2)
            # traj = traj_app

            # non_zeros = np.argwhere(traj > 0.)
            # traj[non_zeros[:,0], non_zeros[:,1]] = 1

            # plt.imsave(f'test_GT_img_{idx}.jpeg', traj, vmin=0, vmax=255)
            # non_zeros = np.argwhere(traj != 0)
            # img[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 255])
            # plt.imsave(f'test_input_and_GT_{idx}.jpeg', img, vmin=0, vmax=255)
        
        # Visualize trajectory for checking correctness
        # for t in traj:
        #     non_zeros = np.argwhere(t != 0.)
        #     img_now = img.copy()
        #     img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 255])
        #     plt.imshow(img_now, vmin=0, vmax=255)
            
        # plot images to test correctness
        # plt.imsave(f'test_img_{idx}.jpeg', img, vmin=0, vmax=255)
        # if isinstance(mask, np.ndarray): 
        #     plt.imsave(f'test_mask_{idx}.jpeg', mask, cmap=get_customized_colormap())

        if self.transform:
            img = self.transform(img) 
            # TODO transforms for trajs too at some point (e.g. for rotation, resizing):
            # traj = self.transform(traj)

        return img, traj
