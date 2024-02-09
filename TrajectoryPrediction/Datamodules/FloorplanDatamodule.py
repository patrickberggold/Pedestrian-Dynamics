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
        config: dict,
        num_workers: int = 0,
        ):
        super().__init__()

        self.config = config
        self.arch = config['arch']
        self.img_arch = config['img_arch']
        self.cuda_index = config['cuda_device']
        self.batch_size = config['imgs_per_batch']

        self.num_workers = num_workers
        self.data_format = config['data_format']
        self.traj_quantity = config['traj_quantity']
        splits = [0.7, 0.15, 0.15]
        self.limit_dataset = config['limit_dataset']
        self.set_data_paths(splits)
        self.normalize_dataset = config['normalize_dataset']

        store_as_pickle = False
        self.train_loader_shuffle = True
        self.val_loader_shuffle = True
        self.read_from_pickle = config['read_from_pickle']

        self.collate_fn = None if True else self.custom_collate # self.config['data_format'] != 'full_tokenized_by_frame'

        if self.img_arch in ['BeIT', 'SegFormer']:
            if self.img_arch == 'BeIT':
                from transformers import BeitFeatureExtractor
                feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
            elif self.img_arch == 'SegFormer':
                from transformers import SegformerFeatureExtractor
                # https://xieenze.github.io/segformer.pdf
                # feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
                feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b1-finetuned-cityscapes-1024-1024')
            else:
                raise NotImplementedError
            # img_size = feature_extractor.size['height'] if OPSYS=='Linux' else feature_extractor.size
            img_size = 640 # feature_extractor.size['height'] if OPSYS=='Linux' else feature_extractor.size
            self.train_transforms = None # {
            #     'feature_extractor_size': img_size,
            #     'transforms': Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
            # }
            self.val_transforms = None # {
            #     'feature_extractor_size': img_size,
            #     'transforms': Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
            # }
        
        if store_as_pickle:
            import pickle
            self.setup('train')
            
            with open(SEP.join(['TrajectoryPrediction', 'Datamodules', 'dataset_train_whole_random_seq20.pickle']), 'wb') as handle:
                print('Storing train dataset as pickle...')
                pickle.dump(self.train_dataset.sequence_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(SEP.join(['TrajectoryPrediction', 'Datamodules', 'dataset_val_whole_random_seq20.pickle']), 'wb') as handle:
                print('Storing validation dataset as pickle...')
                pickle.dump(self.val_dataset.sequence_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(SEP.join(['TrajectoryPrediction', 'Datamodules', 'dataset_test_whole_random_seq20.pickle']), 'wb') as handle:
                print('Storing test dataset as pickle...')
                pickle.dump(self.test_dataset.sequence_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

            quit()

    def setup(self, stage):

        self.train_dataset = Dataset_Seq2Seq(self.config, self.train_imgs_list, self.train_csv_list, split='train', semanticMap_list=self.train_semanticMap_list, globalSemMap_list=self.train_global_semanticMap_list, transforms=self.train_transforms)
        self.val_dataset = Dataset_Seq2Seq(self.config, self.val_imgs_list, self.val_csv_list, split='val', semanticMap_list=self.val_semanticMap_list, globalSemMap_list=self.val_global_semanticMap_list, transforms=self.val_transforms)
        self.test_dataset = Dataset_Seq2Seq(self.config, self.test_imgs_list, self.test_csv_list, split='test', semanticMap_list=self.test_semanticMap_list, globalSemMap_list=self.test_global_semanticMap_list, transforms=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.train_loader_shuffle, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.val_loader_shuffle, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def custom_collate(self, batch):
        if self.data_format == 'random':
            images, coords = zip(*[(d.values()) for d in batch])
            return {'images': torch.stack(images, dim=0), 'coords': coords}
        elif self.data_format == 'full_tokenized_by_frame':
            images, coords, tokens, is_traj_mask = zip(*[(d.values()) for d in batch])
            return {'images': torch.stack(images, dim=0),  'coords': coords, 'tokens': tokens, 'is_traj_mask': is_traj_mask}
        batch_unzipped = zip(*batch)
        images, labels, bboxes, numAgentsIds = zip(*batch)
        # return torch.stack(images, dim=0), labels, bboxes, torch.stack(numAgentsIds, dim=0)
        return torch.stack(images, dim=0), labels, bboxes, torch.stack(numAgentsIds, dim=0)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.cuda_index != 'cpu':
            device = torch.device('cuda', self.cuda_index)
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            return batch

    def set_data_paths(self,  splits: list):
        
        assert sum(splits) == 1., 'Splits do not accumulate to 100%'
        assert len(splits) == 3, 'Splits are not transfered in correct format (which is [train_split, val_split, test_split])'
        self.splits = splits

        self.layout_types = ['corr_e2e', 'corr_cross'] #, 'train_station']

        # self.img_path = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'SIMPLE_FLOORPLANS', 'HDF5_INPUT_IMAGES_resolution_800_800']) # '/home/Datasets/Segmentation/Floorplans/HDF5_INPUT_IMAGES_resolution_800_800'
        self.root_img_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'SPARSE_DENSITY_INPUT_640'])
        # self.root_csv_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'CSV_GT_TRAJECTORIES'])
        self.root_csv_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'CSV_GT_TRAJECTORIES_TRANSFORMED'])
        self.semantic_maps_dir = None
        
        # new dataset
        self.root_img_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'SPARSE_WITH_DESTINATIONS', 'SPARSE_IMAGES'])
        self.root_csv_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'SPARSE_WITH_DESTINATIONS', 'TRAJECTORIES_PIX'])
        self.semantic_maps_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'SPARSE_WITH_DESTINATIONS', 'SEMANTIC_MAPS'])
        self.global_semantic_maps_dir = SEP.join([PREFIX, 'Users', 'Remotey', 'Documents', 'Datasets', 'SPARSE_WITH_DESTINATIONS', 'GLOBAL_SEMANTIC_MAPS'])

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
        self.train_semanticMap_list = [self.semantic_maps_list[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        self.train_global_semanticMap_list = [self.global_semantic_maps_list[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        
        self.val_imgs_list = [self.img_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]
        self.val_csv_list = [self.csv_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]
        self.val_semanticMap_list = [self.semantic_maps_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]
        self.val_global_semanticMap_list = [self.global_semantic_maps_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]

        self.test_imgs_list = [self.img_list[idx] for idx in self.indices[:test_split_index]]
        self.test_csv_list = [self.csv_list[idx] for idx in self.indices[:test_split_index]]
        self.test_semanticMap_list = [self.semantic_maps_list[idx] for idx in self.indices[:test_split_index]]
        self.test_global_semanticMap_list = [self.global_semantic_maps_list[idx] for idx in self.indices[:test_split_index]]

    
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
        self.semantic_maps_list = []
        self.global_semantic_maps_list = []
        if self.semantic_maps_dir is not None:
            self.layout_types = ['corr_e2e']

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
            layout_semantic_maps_path = os.path.join(self.semantic_maps_dir, layout_type) if self.semantic_maps_dir is not None else None
            layout_semantic_maps_paths = os.listdir(layout_semantic_maps_path) if layout_semantic_maps_path is not None else [None]*len(os.listdir(layout_img_path))
            layout_global_semantic_maps_path = os.path.join(self.global_semantic_maps_dir, layout_type) if self.global_semantic_maps_dir is not None else None
            layout_global_semantic_maps_paths = os.listdir(layout_global_semantic_maps_path) if layout_global_semantic_maps_path is not None else [None]*len(os.listdir(layout_img_path))

            # for img_path, csv_path in zip(os.listdir(layout_csv_path), os.listdir(layout_img_path)):
            for img_path, csv_path, semMap_path, globSemMap_path in tqdm(zip(os.listdir(layout_csv_path), os.listdir(layout_img_path), layout_semantic_maps_paths, layout_global_semantic_maps_paths), desc=f'[Loading files, layout_type=={layout_type}...]', total=len(os.listdir(layout_csv_path))):

                # assert img_path == csv_path

                sorted_imgs = sorted([path for path in os.listdir(os.path.join(layout_img_path, img_path)) if not path.endswith('.txt')], key=lambda x: int(x.split('_')[-1]))
                sorted_traj = sorted(os.listdir(os.path.join(layout_csv_path, csv_path)), key=lambda x: int(x.split('_')[-1]))
                sorted_semMaps = sorted(os.listdir(os.path.join(layout_semantic_maps_path, semMap_path)), key=lambda x: int(x.split('_')[-1])) if semMap_path is not None else [None]*len(sorted_traj)
                sorted_globSemMaps = sorted(os.listdir(os.path.join(layout_global_semantic_maps_path, globSemMap_path)), key=lambda x: int(x.split('_')[-1])) if globSemMap_path is not None else [None]*len(sorted_traj)

                for img_var, csv_var, sem_var, glSem_var in zip(sorted_imgs, sorted_traj, sorted_semMaps, sorted_globSemMaps):

                    if self.semantic_maps_dir is None:
                        # assert csv_var == img_var # check if i.e. 'variation_9' in img_path
                        for agent_var in os.listdir(os.path.join(layout_csv_path, csv_path, csv_var)):

                            self.img_list.append(os.path.join(layout_img_path, img_path, img_var, agent_var.replace('.txt', '.npz')))
                            self.csv_list.append(os.path.join(layout_csv_path, csv_path, csv_var, agent_var))
                            self.semantic_maps_list.append(None)
                            self.global_semantic_maps_list.append(None)

                            # assert os.path.isfile(self.img_list[-1])
                            # assert os.path.isfile(self.csv_list[-1])
                    else:
                        assert img_var == csv_var == sem_var
                        img_files = sorted(os.listdir(os.path.join(layout_img_path, img_path, img_var)))
                        assert len(img_files) == 2
                        csv_files = sorted(os.listdir(os.path.join(layout_csv_path, csv_path, csv_var)))
                        assert len(csv_files) == 6
                        sem_files = sorted(os.listdir(os.path.join(layout_semantic_maps_path, semMap_path, sem_var)))
                        assert len(sem_files) == 6
                        glSem_files = sorted([d for d in os.listdir(os.path.join(layout_global_semantic_maps_path, globSemMap_path, glSem_var)) if d.endswith('.npz')])
                        assert len(glSem_files) == 2
                        assert (img_files[0] == glSem_files[0]) and (img_files[1] == glSem_files[1])
                        for _ in range(3):
                            self.img_list.append(os.path.join(layout_img_path, img_path, img_var, img_files[0]))
                            self.img_list.append(os.path.join(layout_img_path, img_path, img_var, img_files[1]))
                            self.global_semantic_maps_list.append(os.path.join(layout_global_semantic_maps_path, globSemMap_path, glSem_var, glSem_files[0]))
                            self.global_semantic_maps_list.append(os.path.join(layout_global_semantic_maps_path, globSemMap_path, glSem_var, glSem_files[1]))
                        for csv_config, semMap_config in zip(csv_files, sem_files):
                            assert csv_config.replace('.txt', '') == semMap_config.replace('.npz', '')
                            self.csv_list.append(os.path.join(layout_csv_path, csv_path, csv_var, csv_config))
                            self.semantic_maps_list.append(os.path.join(layout_semantic_maps_path, semMap_path, sem_var, semMap_config))

                        # some checks
                        for f1, f2 in zip(self.img_list[-6:][0::2], self.img_list[-6:][1::2]):
                            assert not f1.endswith('_wObs.npz') and f2.endswith('_wObs.npz')
                        for i1, i2 in zip(self.global_semantic_maps_list[-6:][0::2], self.global_semantic_maps_list[-6:][1::2]):
                            assert not i1.endswith('_wObs.npz') and i2.endswith('_wObs.npz')
                        for g1, g2 in zip(self.csv_list[-6:], self.semantic_maps_list[-6:]):
                            assert g1.split(os.sep)[-1].replace('.txt', '') == g2.split(os.sep)[-1].replace('.npz', '')
                        for h1, h2 in zip(self.img_list[-6:], self.csv_list[-6:]):
                            assert (('wObs' in h1) == ('wObs' in h2)) or (('wObs' not in h1) == ('wObs' not in h2))
