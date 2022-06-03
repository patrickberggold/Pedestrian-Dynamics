import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import random
import h5py
from helper import SEP

# TODO OPEN QUESTION
# What should be the input? RGB image with colored regions (so RGB format) or pixel annotations (so 'greyscale' values from 0 to num_classes)

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
    def __init__(self, mode: str, img_paths: list, traj_paths: list, transform = None):

        self.transform = transform
        self.img_paths = img_paths
        self.traj_paths = traj_paths
        self.mode = mode

        assert mode in ['grayscale', 'rgb', 'bool'], 'Unknown mode setting!'
        assert len(self.traj_paths) == len(self.img_paths), 'Length of image paths and trajectory paths do not match, something went wrong!'

        # if self.split == 'train':
        #     assert len(self.img_list) == len(self.traj_list)
        #     self.img_list = [self.img_list[idx] for idx in indices[(test_split_index + val_split_index):]]
        #     self.traj_list = [self.traj_list[idx] for idx in indices[(test_split_index + val_split_index):]]
        # elif self.split == 'val':
        #     assert len(self.img_list) == len(self.traj_list)
        #     self.img_list = [self.img_list[idx] for idx in indices[test_split_index:(test_split_index + val_split_index)]]
        #     self.traj_list = [self.traj_list[idx] for idx in indices[test_split_index:(test_split_index + val_split_index)]]
        # elif self.split == 'test':
        #     self.img_list = [self.img_list[idx] for idx in indices[:test_split_index]]
        #     self.traj_list = [self.traj_list[idx] for idx in indices[:test_split_index]]
        # else:
        #     raise ValueError()

    def __len__(self):
        return(len(self.img_paths))
    
    def __getitem__(self, idx):
        img_path, traj_path = self.img_paths[idx], self.traj_paths[idx]
        # quick check if paths are consistent
        assert SEP.join(img_path.split(SEP)[-2:]) == SEP.join(traj_path.split(SEP)[-2:])
        
        img = np.array(h5py.File(img_path, 'r').get('img'))
        # plt.imsave(f'test_input_img_{idx}.jpeg', img, vmin=0, vmax=255)

        traj = np.array(h5py.File(traj_path, 'r').get('img'))
        
        if self.mode == 'grayscale':
            # Scale timestamp values with respect to maximum timestamp value
            traj /= 89.5
            assert 0 <= traj.max() <= 1.
            # If passed as numpy, trajectory does not get normalized
        elif self.mode == 'rgb':
            # TODO Maybe change/remove this once transforms switch from Normalization to something else
            if self.transform:
                traj = self.transform(traj)
        elif self.mode == 'bool':
            traj = traj.astype('bool').astype('float32')
            
            # non_zeros = np.argwhere(traj > 0.) # (25647 (614353
            # traj[non_zeros[:,0], non_zeros[:,1]] = 1

            # zeros = np.argwhere(traj <= 0.)
            # traj[zeros[:,0], zeros[:,1]] = 0

            # plt.imsave(f'test_GT_img_{idx}.jpeg', traj, vmin=0, vmax=255)
            # non_zeros = np.argwhere(traj != 0)
            # img[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 255])
            # plt.imsave(f'test_input_and_GT_{idx}.jpeg', img, vmin=0, vmax=255)
            
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
            # TODO transforms for trajs too at some point (e.g. for rotation, resizing):
            # traj = self.transform(traj)

        return img, traj
