import torch
import pytorch_lightning as pl
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, Normalize, Resize, CenterCrop
from torch.utils.data import DataLoader
import os, random
from .FloorplanDataset import Dataset_Img2Img
from helper import SEP

class FloorplanDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        mode: str,
        arch: str, 
        cuda_index: int, 
        batch_size: int = 4, 
        splits: list = [0.7, 0.15, 0.15], 
        num_workers: int = 0, 
        num_ts_per_floorplan: int = 1,
        vary_area_brightness: bool = True,
        ):
        super().__init__()
        assert mode in ['grayscale', 'grayscale_movie', 'evac'], 'Unknown mode setting!'
        assert arch in ['DeepLab', 'BeIT', 'SegFormer']
        self.mode = mode
        self.batch_size = batch_size

        if arch in ['BeIT', 'SegFormer']:
            if arch == 'BeIT':
                from transformers import BeitFeatureExtractor
                feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
            elif arch == 'SegFormer':
                from transformers import SegformerFeatureExtractor
                feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-768-768')            
            self.train_transforms = Compose([
                Resize(feature_extractor.size),
                # RandomResizedCrop(feature_extractor.size),
                # RandomHorizontalFlip(),
                # ToTensor(),
                Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ])
            self.val_transforms = Compose([
                Resize(feature_extractor.size),
                # CenterCrop(feature_extractor.size),
                # ToTensor(),
                Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ])
        else:
            self.train_transforms = None
            self.val_transforms = None

        self.cuda_index = cuda_index
        self.num_workers = num_workers
        self.vary_area_brightness = vary_area_brightness
        if self.mode == 'grayscale_movie':
            assert num_ts_per_floorplan > 1 and isinstance(num_ts_per_floorplan, int)
            self.num_ts_per_floorplan = num_ts_per_floorplan
        else:
            self.num_ts_per_floorplan = 1
        self.set_data_paths(splits)

    def setup(self, stage):
        self.train_dataset = Dataset_Img2Img(self.mode, self.train_imgs_list, self.train_trajs_list, transform=self.train_transforms, num_ts_per_floorplan=self.num_ts_per_floorplan, vary_area_brightness=self.vary_area_brightness)
        self.val_dataset = Dataset_Img2Img(self.mode, self.val_imgs_list, self.val_trajs_list, transform=self.val_transforms, num_ts_per_floorplan=self.num_ts_per_floorplan, vary_area_brightness=self.vary_area_brightness)
        self.test_dataset = Dataset_Img2Img(self.mode, self.test_imgs_list, self.test_trajs_list, transform=self.val_transforms, num_ts_per_floorplan=self.num_ts_per_floorplan, vary_area_brightness=self.vary_area_brightness)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.cuda_index != 'cpu':
            device = torch.device('cuda', self.cuda_index)
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            return batch

    def set_data_paths(self,  splits: list):
        
        assert sum(splits) == 1., 'Splits do not accumulate to 100%'
        assert len(splits) == 3, 'Splits are not transfered in correct format (which is [train_split, val_split, test_split])'
        self.splits = splits

        if not self.vary_area_brightness:
            raise NotImplementedError('Dataset with overall same number of agents does not exist!')

        # self.img_path = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', 'HDF5_INPUT_IMAGES_resolution_800_800'])
        root_img_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', 'INPUT'])
        self.img_path = [os.path.join(root_img_dir, layout_dir, floorplan_dir) for layout_dir in os.listdir(root_img_dir) for floorplan_dir in os.listdir(os.path.join(root_img_dir, layout_dir))]
        
        # self.traj_path = [SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', f'HDF5_GT_TIMESTAMP_MASKS_resolution_800_800_numAgents_{var}_thickness_5']) for var in [10, 20, 30, 40, 50]]
        root_traj_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', f'HDF5_GT_TIMESTAMP_MASKS_thickness_5'])
        self.traj_path = [os.path.join(root_traj_dir, layout_dir, floorplan_dir) for layout_dir in os.listdir(root_traj_dir) if os.path.isdir(os.path.join(root_traj_dir, layout_dir)) for floorplan_dir in os.listdir(os.path.join(root_traj_dir, layout_dir))]

        self.set_filepaths()

        assert len(self.img_list) == len(self.traj_list), 'Images list and trajectory list do not have same length, something went wrong!'
        # Randomly check if entries are the same
        # index_check_list = random.sample(range(len(self.img_list)), 10)
        # assert [SEP.join(self.img_list[i].split('\\')[-2:]) for i in index_check_list] == [SEP.join(self.traj_list[i].split('\\')[-2:]) for i in index_check_list], \
        #     'Images list and trajectory list do not have same entries, something went wrong!'
    	
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

        # LIMIT THE DATASET
        # self.train_imgs_list = self.train_imgs_list[:1000]
        # self.train_trajs_list = self.train_trajs_list[:1000]

        # self.val_imgs_list = self.val_imgs_list[:200]
        # self.val_trajs_list = self.val_trajs_list[:200]

        # self.test_imgs_list = self.test_imgs_list[:200]
        # self.test_trajs_list = self.test_trajs_list[:200]

    def set_filepaths(self):

        self.img_list = []
        self.traj_list = []
        
        for img_path, traj_path in zip(self.img_path, self.traj_path):
            assert img_path.split(SEP)[-1] == traj_path.split(SEP)[-1]

            sorted_imgs = sorted(os.listdir(img_path), key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            sorted_traj = sorted(os.listdir(traj_path), key=lambda x: int(x.split('_')[-1]))

            for img_var, traj_var in zip(sorted_imgs, sorted_traj):
                
                assert traj_var in img_var # check if i.e. 'variation_9' in img_path
                
                for agent_var in os.listdir(os.path.join(traj_path, traj_var)):

                    self.img_list.append(os.path.join(img_path, img_var))
                    self.traj_list.append(os.path.join(traj_path, traj_var, agent_var))

                    assert os.path.isfile(self.img_list[-1])
                    assert os.path.isfile(self.traj_list[-1])

