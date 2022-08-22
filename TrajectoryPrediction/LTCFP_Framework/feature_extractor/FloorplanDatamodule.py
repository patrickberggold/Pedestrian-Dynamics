import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import os, random
from .FloorplanDataset import img2img_dataset
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
            
        ])
        self.cuda_index = cuda_index
        self.num_workers = num_workers
        self.set_data_paths(splits)

    def setup(self, stage):
        self.train_dataset = img2img_dataset(self.mode, self.train_imgs_list, transform=self.transforms)
        self.val_dataset = img2img_dataset(self.mode, self.val_imgs_list, transform=self.transforms)
        self.test_dataset = img2img_dataset(self.mode, self.test_imgs_list, transform=self.transforms)

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

        self.img_path = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', 'INPUT'])
        
        assert os.path.isdir(self.img_path)
        self.img_list = self.get_filenames(self.img_path)
  	
        val_split_factor = self.splits[1]
        test_split_factor = self.splits[2]
        
        self.indices = list(range(len(self.img_list)))
        
        val_split_index = int(len(self.indices) * val_split_factor)
        test_split_index = int(len(self.indices) * test_split_factor)
        
        # random.seed(42) # seed already given
        random.shuffle(self.indices)

        self.train_imgs_list = [self.img_list[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        self.val_imgs_list = [self.img_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]
        self.test_imgs_list = [self.img_list[idx] for idx in self.indices[:test_split_index]]


    def get_filenames(self, path):

        files_list = list()
        for layout_type in os.listdir(path):
            folder_list = sorted(os.listdir(os.path.join(path, layout_type)), key=lambda x: int(x.split('__')[0]))
            for foldername in folder_list:
                files_in_folder = sorted(os.listdir(os.path.join(path, layout_type, foldername)), key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
                for filename in files_in_folder:
                    if filename.endswith('.h5'):
                        files_list.append(os.path.join(path, layout_type, foldername, filename))
        return files_list
