import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import os, random
from .FloorplanDataset import Dataset_Seq2Seq
from helper import SEP

class FloorplanDatamodule(pl.LightningDataModule):
    def __init__(
        self, 
        mode: str, 
        cuda_index: int, 
        batch_size: int = 4, 
        num_workers: int = 2, 
        ):
        super().__init__()

        self.mode = mode
        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(), # THIS NORMALIZES ALREADY
            # transforms.Normalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
            # transforms.Normalize(mean=0, std=255)
        ])
        self.cuda_index = cuda_index
        self.num_workers = num_workers
        splits = [0.7, 0.15, 0.15]
        self.set_data_paths(splits)

    def setup(self, stage):
        # TODO
        self.train_dataset = Dataset_Seq2Seq(self.train_imgs_list, self.train_csv_list, split='train')
        self.val_dataset = Dataset_Seq2Seq(self.val_imgs_list, self.val_csv_list, split='val')
        self.test_dataset = Dataset_Seq2Seq(self.test_imgs_list, self.test_csv_list, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

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

        self.layout_types = ['corr_e2e', 'corr_cross', 'train_station']

        # self.img_path = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', 'HDF5_INPUT_IMAGES_resolution_800_800']) # '/home/Datasets/Segmentation/Floorplans/HDF5_INPUT_IMAGES_resolution_800_800'
        self.root_img_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', 'INPUT'])
        self.root_csv_dir = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', 'CSV_GT_TRAJECTORIES'])


        """ self.img_path = [os.path.join(root_img_dir, layout_dir, floorplan_dir) for layout_dir in os.listdir(root_img_dir) for floorplan_dir in os.listdir(os.path.join(root_img_dir, layout_dir))]



        self.img_path = [
            os.path.join(layout_type, flooplan_folder, variation_folder, csv_file) \
                for layout_type in self.layout_types \
                for flooplan_folder in os.listdir(os.path.join(root_img_dir, layout_type)) \
                for variation_folder in os.listdir(os.path.join(root_img_dir, layout_type, flooplan_folder)) \
                for csv_file in os.listdir(os.path.join(root_img_dir, layout_type, flooplan_folder, variation_folder))
        ]
        
        

        # self.traj_path = [SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', f'HDF5_GT_TIMESTAMP_MASKS_resolution_800_800_numAgents_{var}']) for var in [10, 20, 30, 40, 50]]
        # self.traj_path = [SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', f'HDF5_GT_TIMESTAMP_MASKS_resolution_800_800_numAgents_{var}_thickness_5']) for var in [10, 20, 30, 40, 50]]
        
        # TODO
        dataset_folder_csv = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', f'CSV_GT_TRAJECTORIES'])
        self.traj_path = [
            os.path.join(layout_type, flooplan_folder, variation_folder, csv_file) \
                for layout_type in self.layout_types \
                for flooplan_folder in os.listdir(os.path.join(dataset_folder_csv, layout_type)) \
                for variation_folder in os.listdir(os.path.join(dataset_folder_csv, layout_type, flooplan_folder)) \
                for csv_file in os.listdir(os.path.join(dataset_folder_csv, layout_type, flooplan_folder, variation_folder)) 
        ] """
        
        # for layout_dir in os.listdir(root_traj_dir):
        #     for floorplan_dir in os.listdir(os.path.join(root_traj_dir, layout_dir)):

        #         self.traj_path.append(os.path.join(root_traj_dir, layout_dir, floorplan_dir))
        # self.traj_path = [os.path.join(root_traj_dir, layout_dir, floorplan_dir) for layout_dir in os.listdir(root_traj_dir) for floorplan_dir in os.listdir(os.path.join(root_traj_dir, layout_dir))]

        self.set_filepaths()

        # self.img_list = []
        # self.traj_list = []
        # for path in self.traj_path:
        #     assert os.path.isdir(path)
        #     self.traj_list += self.get_filenames(path)
        #     self.img_list += self.get_filenames(self.img_path)

        # self.img_list = self.get_filenames(self.img_path)
        # self.traj_list = self.get_filenames(self.traj_path)

        assert len(self.img_list) == len(self.csv_list), 'Images list and trajectory list do not have same length, something went wrong!'
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
        self.train_csv_list = [self.csv_list[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        
        self.val_imgs_list = [self.img_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]
        self.val_csv_list = [self.csv_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]

        self.test_imgs_list = [self.img_list[idx] for idx in self.indices[:test_split_index]]
        self.test_csv_list = [self.csv_list[idx] for idx in self.indices[:test_split_index]]

        # LIMIT DATASETS
        self.train_imgs_list = self.train_imgs_list[:4]
        self.train_csv_list = self.train_csv_list[:4]
        
        self.val_imgs_list = self.val_imgs_list[:1]
        self.val_csv_list = self.val_csv_list[:1]

        self.test_imgs_list = self.test_imgs_list[:2]
        self.test_csv_list = self.test_csv_list[:2]

    def set_filepaths(self):

        self.img_list = []
        self.csv_list = []

        for layout_type in self.layout_types:
            layout_csv_path = os.path.join(self.root_csv_dir, layout_type)
            layout_img_path = os.path.join(self.root_img_dir, layout_type)

            for img_path, csv_path in zip(os.listdir(layout_csv_path), os.listdir(layout_img_path)):
                
                assert img_path == csv_path

                sorted_imgs = sorted(os.listdir(os.path.join(layout_img_path, img_path)), key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
                sorted_csv = sorted(os.listdir(os.path.join(layout_csv_path, csv_path)), key=lambda x: int(x.split('_')[-1]))

                for img_var, csv_var in zip(sorted_imgs, sorted_csv):
                    
                    assert csv_var in img_var # check if i.e. 'variation_9' in img_path
                    
                    for agent_var in os.listdir(os.path.join(layout_csv_path, csv_path, csv_var)):

                        self.img_list.append(os.path.join(layout_img_path, img_path, img_var))
                        self.csv_list.append(os.path.join(layout_csv_path, csv_path, csv_var, agent_var))

                        assert os.path.isfile(self.img_list[-1])
                    assert os.path.isfile(self.csv_list[-1])


    # def get_filenames(self, path):
    #     files_list = list()
    #     folder_list = sorted(os.listdir(path), key=lambda x: int(x.split('__')[0]))
    #     for foldername in folder_list:
    #         files_in_folder = sorted(os.listdir(os.path.join(path, foldername)), key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
    #         for filename in files_in_folder:
    #             if filename.endswith('.h5'):
    #                 files_list.append(os.path.join(path, foldername, filename))
    #     return files_list