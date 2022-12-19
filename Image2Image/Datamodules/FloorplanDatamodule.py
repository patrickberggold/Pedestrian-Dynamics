import torch
import pytorch_lightning as pl
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, Normalize, Resize, CenterCrop
from torch.utils.data import DataLoader
import os, random
from .FloorplanDataset import Dataset_Img2Img
from helper import SEP

class FloorplanDataModule(pl.LightningDataModule):
    def __init__(self, config: dict, num_workers: int = 2):
        super().__init__()
        self.mode = config['mode']
        self.batch_size = config['batch_size']
        self.additional_info = config['additional_info']
        arch = config['arch']
        self.cuda_device = config['cuda_device']
        self.vary_area_brightness = config['vary_area_brightness']
        self.limit_dataset = config['limit_dataset']
        self.num_workers = num_workers

        assert self.mode in ['grayscale', 'grayscale_movie', 'evac_only', 'class_movie', 'density_class', 'density_reg', 'denseClass_wEvac'], 'Unknown mode setting!'
        assert arch in ['DeepLab', 'BeIT', 'SegFormer']

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

        self.set_data_paths()

    def setup(self, stage):
        self.train_dataset = Dataset_Img2Img(self.mode, self.train_imgs_list, self.train_trajs_list, transform=self.train_transforms, additional_info = self.additional_info)
        self.val_dataset = Dataset_Img2Img(self.mode, self.val_imgs_list, self.val_trajs_list, transform=self.val_transforms, additional_info = self.additional_info)
        self.test_dataset = Dataset_Img2Img(self.mode, self.test_imgs_list, self.test_trajs_list, transform=self.val_transforms, additional_info = self.additional_info)

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
        if self.cuda_device != 'cpu':
            device = torch.device('cuda', self.cuda_device)
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            return batch

    def set_data_paths(self):
        
        self.splits = [0.7, 0.15, 0.15]

        if not self.vary_area_brightness:
            raise NotImplementedError('Dataset with overall same number of agents does not exist!')

        # self.img_path = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', 'HDF5_INPUT_IMAGES_resolution_800_800'])
        if self.mode=='class_movie':
            root_img_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'INPUT'])
            root_traj_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', f'SPARSE_GT_VELOCITY_MASKS_thickness_5_nframes_10'])
        elif self.mode in ['density_reg', 'density_class', 'denseClass_wEvac']:
            root_img_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'SPARSE_DENSITY_INPUT_640'])
            root_traj_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', f'SPARSE_DENSITY_BINS'])
            if self.mode == 'denseClass_wEvac': root_traj_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'SPARSE_DENSITY_BINS_wEVAC'])
        else:
            root_img_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', 'INPUT'])
            root_traj_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', f'HDF5_GT_TIMESTAMP_MASKS_thickness_5'])
        
        # self.traj_path = [SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', f'HDF5_GT_TIMESTAMP_MASKS_resolution_800_800_numAgents_{var}_thickness_5']) for var in [10, 20, 30, 40, 50]]
        self.traj_path = [os.path.join(root_traj_dir, layout_dir, floorplan_dir) for layout_dir in os.listdir(root_traj_dir) if os.path.isdir(os.path.join(root_traj_dir, layout_dir)) for floorplan_dir in os.listdir(os.path.join(root_traj_dir, layout_dir))]
        self.img_path = [os.path.join(root_img_dir, layout_dir, floorplan_dir) for layout_dir in os.listdir(root_img_dir) if os.path.isdir(os.path.join(root_img_dir, layout_dir)) for floorplan_dir in os.listdir(os.path.join(root_img_dir, layout_dir))]

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
        if self.limit_dataset:
            self.train_imgs_list = self.train_imgs_list[:self.limit_dataset]
            self.train_trajs_list = self.train_trajs_list[:self.limit_dataset]

            self.val_imgs_list = self.val_imgs_list[:self.limit_dataset//4]
            self.val_trajs_list = self.val_trajs_list[:self.limit_dataset//4]

            self.test_imgs_list = self.test_imgs_list[:self.limit_dataset//4]
            self.test_trajs_list = self.test_trajs_list[:self.limit_dataset//4]

    def set_filepaths(self):

        self.img_list = []
        self.traj_list = []
        
        for img_path, traj_path in zip(self.img_path, self.traj_path):
            assert img_path.split(SEP)[-1] == traj_path.split(SEP)[-1]

            if self.mode not in ['density_class', 'density_reg', 'denseClass_wEvac']:

                sorted_imgs = [i_path for i_path in os.listdir(img_path) if i_path.endswith('.h5')]
                sorted_imgs = sorted(sorted_imgs, key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
                sorted_traj = sorted(os.listdir(traj_path), key=lambda x: int(x.split('_')[-1]))

                for img_var, traj_var in zip(sorted_imgs, sorted_traj):
                    
                    assert traj_var in img_var # check if i.e. 'variation_9' in img_path
                    
                    for agent_var in os.listdir(os.path.join(traj_path, traj_var)):

                        self.img_list.append(os.path.join(img_path, img_var))
                        self.traj_list.append(os.path.join(traj_path, traj_var, agent_var))

                        assert os.path.isfile(self.img_list[-1])
                        assert os.path.isfile(self.traj_list[-1])
            else:
                sorted_imgs = sorted([path for path in os.listdir(img_path) if not path.endswith('.txt')], key=lambda x: int(x.split('_')[-1]))
                sorted_traj = sorted(os.listdir(traj_path), key=lambda x: int(x.split('_')[-1]))

                for img_var, traj_var in zip(sorted_imgs, sorted_traj):
                    
                    assert traj_var == img_var # check if i.e. 'variation_9' in img_path
                    for agent_var in os.listdir(os.path.join(traj_path, traj_var)):

                        self.img_list.append(os.path.join(img_path, img_var, agent_var))
                        self.traj_list.append(os.path.join(traj_path, traj_var, agent_var))

                        assert os.path.isfile(self.img_list[-1])
                        assert os.path.isfile(self.traj_list[-1])