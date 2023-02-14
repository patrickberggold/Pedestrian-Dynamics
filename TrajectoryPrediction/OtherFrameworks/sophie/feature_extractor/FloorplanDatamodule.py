import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import os, random
from FloorplanDataset import semantic_dataset, img2img_dataset
from helper import SEP

class FloorplanDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        mode: str, 
        cuda_index: int, 
        batch_size: int = 4, 
        splits: list = [0.7, 0.15, 0.15],  
        num_workers: int = 0, 
        ):
        super().__init__()
        assert mode in ['img2img'], 'Unknown mode setting!'
        self.mode = mode
        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(), # THIS NORMALIZES ALREADY
            # transforms.Normalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
            # transforms.Normalize(mean=0, std=255)
        ])
        self.cuda_index = cuda_index
        self.num_workers = num_workers
        self.set_data_paths(splits)

    def setup(self, stage):
        if self.mode == 'segmentation':
            self.train_dataset = semantic_dataset(split='train', transform=self.transforms)
            self.val_dataset = semantic_dataset(split='val', transform=self.transforms)
            self.test_dataset = semantic_dataset(split='test', transform=self.transforms)
        else:
            self.train_dataset = img2img_dataset(self.mode, self.train_imgs_list, self.train_trajs_list, transform=self.transforms)
            self.val_dataset = img2img_dataset(self.mode, self.val_imgs_list, self.val_trajs_list, transform=self.transforms)
            self.test_dataset = img2img_dataset(self.mode, self.test_imgs_list, self.test_trajs_list, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def set_non_traj_vals(self, new_val: float = -5.):
        self.non_traj_vals = new_val

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.cuda_index != 'cpu':
            device = torch.device('cuda', self.cuda_index)
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            return batch

    def set_data_paths(self,  splits: list):
        
        assert sum(splits) == 1., 'Splits do not accumulate to 100%'
        assert len(splits) == 3, 'Splits are not transfered in correct format (which is [train_split, val_split, test_split])'
        self.splits = splits
        
        # self.current_split = current_split
        # assert self.current_split in ['train', 'val', 'test']

        self.img_path = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'HDF5_INPUT_IMAGES_resolution_800_800'])
        self.traj_path = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'HDF5_INPUT_IMAGES_resolution_800_800'])
        
        assert os.path.isdir(self.img_path)
        assert os.path.isdir(self.traj_path)
        self.img_list = self.get_filenames(self.img_path)
        self.traj_list = self.get_filenames(self.traj_path)

        assert len(self.img_list) == len(self.traj_list), 'Images list and trajectory list do not have same length, something went wrong!'
        # Randomly check if entries are the same
        index_check_list = random.sample(range(len(self.img_list)), 10)
        assert [SEP.join(self.img_list[i].split('\\')[-2:]) for i in index_check_list] == [SEP.join(self.traj_list[i].split('\\')[-2:]) for i in index_check_list], \
            'Images list and trajectory list do not have same entries, something went wrong!'
    	
        val_split_factor = self.splits[1]
        test_split_factor = self.splits[2]
        
        self.indices = list(range(len(self.img_list)))
        
        val_split_index = int(len(self.indices) * val_split_factor)
        test_split_index = int(len(self.indices) * test_split_factor)
        
        random.seed(42)
        random.shuffle(self.indices)

        self.train_imgs_list = [self.img_list[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        self.train_trajs_list = [self.traj_list[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        
        self.val_imgs_list = [self.img_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]
        self.val_trajs_list = [self.traj_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]

        self.test_imgs_list = [self.img_list[idx] for idx in self.indices[:test_split_index]]
        self.test_trajs_list = [self.traj_list[idx] for idx in self.indices[:test_split_index]]


    def get_filenames(self, path):
        files_list = list()
        folder_list = sorted(os.listdir(path), key=lambda x: int(x.split('__')[0]))
        for foldername in folder_list:
            files_in_folder = sorted(os.listdir(os.path.join(path, foldername)), key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            for filename in files_in_folder:
                if filename.endswith('.h5'):
                    files_list.append(os.path.join(path, foldername, filename))
        return files_list
