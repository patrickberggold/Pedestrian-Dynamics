import torch
import pytorch_lightning as pl
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
import os, random
from .FloorplanDataset import Dataset_Seq2Seq
from helper import SEP, OPSYS, PREFIX
from tqdm import tqdm

class FloorplanDatamodule(pl.LightningDataModule):
    def __init__(
        self, 
        mode: str,
        arch: str,
        img_arch: str,
        cuda_index: int, 
        batch_size: int = 4, 
        num_workers: int = 2,
        data_format: str = 'by_frame', 
        traj_quantity: str = 'vel',
        normalize_dataset: bool = True,
        limit_dataset: bool = False,
        read_from_pickle: bool = False
        ):
        super().__init__()

        self.mode = mode
        self.arch = arch
        self.img_arch = img_arch
        self.batch_size = batch_size

        self.cuda_index = cuda_index
        self.num_workers = num_workers
        self.data_format = data_format
        self.traj_quantity = traj_quantity
        splits = [0.7, 0.15, 0.15]
        self.limit_dataset = limit_dataset
        self.set_data_paths(splits)
        self.normalize_dataset = normalize_dataset

        store_as_pickle = False
        self.read_from_pickle = read_from_pickle

        if img_arch in ['BeIT', 'SegFormer']:
            if img_arch == 'BeIT':
                from transformers import BeitFeatureExtractor
                feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
            elif img_arch == 'SegFormer':
                from transformers import SegformerFeatureExtractor
                # https://xieenze.github.io/segformer.pdf
                # feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
                feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b1-finetuned-cityscapes-1024-1024')
            else:
                raise NotImplementedError
            # img_size = feature_extractor.size['height'] if OPSYS=='Linux' else feature_extractor.size
            img_size = 640 # feature_extractor.size['height'] if OPSYS=='Linux' else feature_extractor.size
            self.train_transforms = {
                'feature_extractor_size': img_size,
                'transforms': Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
            }
            # Compose([
            #     # Resize(feature_extractor.size),
            #     # RandomResizedCrop(feature_extractor.size),
            #     # RandomHorizontalFlip(),
            #     # ToTensor(),
            #     Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            # ])
            self.val_transforms = {
                'feature_extractor_size': img_size,
                'transforms': Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
            }
            # self.val_transforms = Compose([
            #     # Resize(feature_extractor.size),
            #     # CenterCrop(feature_extractor.size),
            #     # ToTensor(),
            #     Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            # ])
        
        if store_as_pickle:
            import pickle
            self.setup('train')
            
            with open(SEP.join(['TrajectoryPrediction', 'LTCFP_new', 'Datamodules', 'dataset_train_whole_random_seq20.pickle']), 'wb') as handle:
                print('Storing train dataset as pickle...')
                pickle.dump(self.train_dataset.sequence_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(SEP.join(['TrajectoryPrediction', 'LTCFP_new', 'Datamodules', 'dataset_val_whole_random_seq20.pickle']), 'wb') as handle:
                print('Storing validation dataset as pickle...')
                pickle.dump(self.val_dataset.sequence_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(SEP.join(['TrajectoryPrediction', 'LTCFP_new', 'Datamodules', 'dataset_test_whole_random_seq20.pickle']), 'wb') as handle:
                print('Storing test dataset as pickle...')
                pickle.dump(self.test_dataset.sequence_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

            quit()

    def setup(self, stage):
        # TODO
        self.train_dataset = Dataset_Seq2Seq(self.arch, self.train_imgs_list, self.train_csv_list, data_format=self.data_format, traj_quantity=self.traj_quantity, split='train', normalize_dataset=self.normalize_dataset, transforms=self.train_transforms, read_from_pickle=self.read_from_pickle)
        self.val_dataset = Dataset_Seq2Seq(self.arch, self.val_imgs_list, self.val_csv_list, data_format=self.data_format, traj_quantity=self.traj_quantity, split='val', normalize_dataset=self.normalize_dataset, transforms=self.val_transforms, read_from_pickle=self.read_from_pickle)
        self.test_dataset = Dataset_Seq2Seq(self.arch, self.test_imgs_list, self.test_csv_list, data_format=self.data_format, traj_quantity=self.traj_quantity, split='test', normalize_dataset=self.normalize_dataset, transforms=self.val_transforms, read_from_pickle=self.read_from_pickle)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

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

        # self.img_path = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', 'HDF5_INPUT_IMAGES_resolution_800_800']) # '/home/Datasets/Segmentation/Floorplans/HDF5_INPUT_IMAGES_resolution_800_800'
        self.root_img_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'SPARSE_DENSITY_INPUT_640'])
        # self.root_csv_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'CSV_GT_TRAJECTORIES'])
        self.root_csv_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'CSV_GT_TRAJECTORIES_TRANSFORMED'])

        self.set_filepaths()

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
        if self.limit_dataset: 
            self.train_imgs_list = self.train_imgs_list[:self.limit_dataset]
            self.train_csv_list = self.train_csv_list[:self.limit_dataset]
        
            self.val_imgs_list = self.val_imgs_list[:self.limit_dataset//4]
            self.val_csv_list = self.val_csv_list[:self.limit_dataset//4]

            self.test_imgs_list = self.test_imgs_list[:1]#20]
            self.test_csv_list = self.test_csv_list[:1]#20]

        # import matplotlib.pyplot as plt
        # # pathy = 
        # # import h5py
        # # import numpy as np
        # # img = np.array(h5py.File(pathy, 'r').get('img'))
        # # plt.imshow(img)
        # self.train_imgs_list = [
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\INPUT\\train_station\\2__floorplan_siteX_65_siteY_25_numEscX_4_numEscY_2_SET_together\\HDF5_floorplan_siteX_65_siteY_25_numEscX_4_numEscY_2_SET_together_variation_497.h5',
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\INPUT\\corr_cross\\0__floorplan_siteX_35_siteY_20_CORRWIDTH_3_SIDECORRWIDTH_20_numCross_2\\HDF5_floorplan_siteX_35_siteY_20_CORRWIDTH_3_SIDECORRWIDTH_20_numCross_2_variation_56.h5',
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\INPUT\\train_station\\2__floorplan_siteX_65_siteY_25_numEscX_4_numEscY_2_SET_together\\HDF5_floorplan_siteX_65_siteY_25_numEscX_4_numEscY_2_SET_together_variation_573.h5',
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\INPUT\\train_station\\1__floorplan_siteX_55_siteY_30_numEscX_3_numEscY_2_SET_apart\\HDF5_floorplan_siteX_55_siteY_30_numEscX_3_numEscY_2_SET_apart_variation_233.h5'
        # ]
        # self.train_csv_list = [
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\CSV_GT_TRAJECTORIES\\train_station\\2__floorplan_siteX_65_siteY_25_numEscX_4_numEscY_2_SET_together\\variation_497\\variation_497_num_agents_15.txt',
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\CSV_GT_TRAJECTORIES\\corr_cross\\0__floorplan_siteX_35_siteY_20_CORRWIDTH_3_SIDECORRWIDTH_20_numCross_2\\variation_56\\variation_56_num_agents_15.txt',
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\CSV_GT_TRAJECTORIES\\train_station\\2__floorplan_siteX_65_siteY_25_numEscX_4_numEscY_2_SET_together\\variation_573\\variation_573_num_agents_15.txt',
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\CSV_GT_TRAJECTORIES\\train_station\\1__floorplan_siteX_55_siteY_30_numEscX_3_numEscY_2_SET_apart\\variation_233\\variation_233_num_agents_15.txt'
        # ]

        # self.val_imgs_list = [
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\INPUT\\corr_e2e\\2__floorplan_siteX_40_siteY_15_CORRWIDTH_3_SET_rl_NUMROOMS_4_4\\HDF5_floorplan_siteX_40_siteY_15_CORRWIDTH_3_SET_rl_NUMROOMS_4_4_variation_31.h5'
        # ]
        # self.val_csv_list = [
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\CSV_GT_TRAJECTORIES\\corr_e2e\\2__floorplan_siteX_40_siteY_15_CORRWIDTH_3_SET_rl_NUMROOMS_4_4\\variation_31\\variation_31_num_agents_15.txt'
        # ]

        # self.test_imgs_list = [
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\INPUT\\train_station\\1__floorplan_siteX_55_siteY_30_numEscX_3_numEscY_2_SET_apart\\HDF5_floorplan_siteX_55_siteY_30_numEscX_3_numEscY_2_SET_apart_variation_288.h5'
        # ]
        # self.test_csv_list = [
        #     f'{PREFIX}\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\CSV_GT_TRAJECTORIES\\train_station\\1__floorplan_siteX_55_siteY_30_numEscX_3_numEscY_2_SET_apart\\variation_288\\variation_288_num_agents_15.txt'
        # ]

    def set_filepaths(self):

        self.img_list = []
        self.csv_list = []

        """ for layout_type in self.layout_types:
            layout_csv_path = os.path.join(self.root_csv_dir, layout_type)
            layout_img_path = os.path.join(self.root_img_dir, layout_type)

            for img_path, csv_path in zip(os.listdir(layout_csv_path), os.listdir(layout_img_path)):
                
                assert img_path == csv_path

                sorted_imgs = sorted(os.listdir(os.path.join(layout_img_path, img_path)), key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
                sorted_csv = sorted(os.listdir(os.path.join(layout_csv_path, csv_path)), key=lambda x: int(x.split('_')[-1]))

                for img_var, csv_var in zip(sorted_imgs, sorted_csv):
                    
                    assert csv_var in img_var # check if i.e. 'variation_9' in img_path
                    
                    for agent_var in os.listdir(os.path.join(layout_csv_path, csv_path, csv_var)):
                        if not agent_var.replace('.txt', '').endswith('15'):
                            continue # TODO change this back

                        self.img_list.append(os.path.join(layout_img_path, img_path, img_var))
                        self.csv_list.append(os.path.join(layout_csv_path, csv_path, csv_var, agent_var))

                        assert os.path.isfile(self.img_list[-1])
                    assert os.path.isfile(self.csv_list[-1])
 """

        for layout_type in self.layout_types:
            layout_csv_path = os.path.join(self.root_csv_dir, layout_type)
            layout_img_path = os.path.join(self.root_img_dir, layout_type)

            # for img_path, csv_path in zip(os.listdir(layout_csv_path), os.listdir(layout_img_path)):
            for img_path, csv_path in tqdm(zip(os.listdir(layout_csv_path), os.listdir(layout_img_path)), desc=f'[Loading files, layout_type=={layout_type}...]', total=len(os.listdir(layout_csv_path))):

                # assert img_path == csv_path

                sorted_imgs = sorted([path for path in os.listdir(os.path.join(layout_img_path, img_path)) if not path.endswith('.txt')], key=lambda x: int(x.split('_')[-1]))
                sorted_traj = sorted(os.listdir(os.path.join(layout_csv_path, csv_path)), key=lambda x: int(x.split('_')[-1]))

                for img_var, csv_var in zip(sorted_imgs, sorted_traj):
                    
                    # assert csv_var == img_var # check if i.e. 'variation_9' in img_path
                    for agent_var in os.listdir(os.path.join(layout_csv_path, csv_path, csv_var)):

                        self.img_list.append(os.path.join(layout_img_path, img_path, img_var, agent_var.replace('.txt', '.npz')))
                        self.csv_list.append(os.path.join(layout_csv_path, csv_path, csv_var, agent_var))

                        # assert os.path.isfile(self.img_list[-1])
                        # assert os.path.isfile(self.csv_list[-1])

