import os

from tqdm import tqdm

from src.data_src.dataset_src.dataset_floorplan import Dataset_floorplan
from src.data_src.experiment_src.experiment_base import Experiment_base


class Experiment_floorplan(Experiment_base):
    def __init__(self, dataset, test_set, args, full_dataset=True):
        super().__init__(args)
        self.dataset = dataset
        self.test_set = test_set
        self.dataset_folder = self.dataset.dataset_folder_csv

        self.protocol = "26_7"
        self.train_valid_strategy = "validate_on_test"
        self.downsample_frame_rate = 1

        self.set_name_to_scenes = {
            "train": self.dataset.train_scenes,
            "valid": self.dataset.test_scenes,
            "test": self.dataset.test_scenes}

        self._load_all()

    def _load_data_files(self, set_name):
        scene_names = self.set_name_to_scenes[set_name]
        set_name_data = []
        for scene_name in tqdm(scene_names):
            scene = self.dataset.scenes[scene_name]
            file_path = os.path.join(self.dataset_folder, scene_name)

            # load raw data table
            raw_data = scene._load_raw_data_table(file_path)

            ####################### EQUISPACE CHECK #######################
            raw_data_np = raw_data.to_numpy()
            import numpy as np
            agent_ids = np.unique(raw_data_np[:,1])
            len_agents = 0
            for id in agent_ids:
                ids = raw_data_np[:,1]
                id_args = np.argwhere(ids == id)
                len_agents += len(id_args)
                agent_data = raw_data_np[id_args]
                # check equispace
                frames = agent_data.squeeze()[:,0]
                equi_frames = np.arange(frames[0], frames[-1]+1.0)
                if not np.array_equal(frames, equi_frames):
                    raise ValueError(f'frames are not equi-spaced for agent {id} in scene {scene_name}') 
            assert len_agents == len(raw_data_np)
            ####################### EQUISPACE CHECK #######################

            # world to pixel
            raw_pixel_data = scene._make_pixel_coord_pandas(
                raw_data.copy())

            set_name_data.append({
                "file_path": file_path,
                "scene_name": scene_name,
                "downsample_frame_rate": self.downsample_frame_rate,
                "set_name": set_name,
                "raw_pixel_data": raw_pixel_data,
                # "RGB_image": scene.RGB_image,
            })
        return set_name_data

    def _load_train_val_test(self):
        for set_name in ['train', 'valid', 'test']:
            self.data[set_name] = self._load_data_files(set_name)
