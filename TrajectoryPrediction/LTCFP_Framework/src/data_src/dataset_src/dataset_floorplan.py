import os
import random
from src.data_src.dataset_src.dataset_base import Dataset_base
from src.data_src.scene_src.scene_floorplan import Scene_floorplan


class Dataset_floorplan(Dataset_base):
    """
    The data of this Intersection Drone Dataset is the original one,
    in meter coordinates, where non-pedestrian agents have already been
    filtered out.
    There is a total of 33 different recordings recorded at 4 unique locations.
    The standard protocol is to train on 3 locations and test on the
    remaining one. The test location contains recordings 00 to 06 and
    corresponds to scene ID4 in the original inD paper.
    AT the end there are 26 train scenes and 7 test scenes.
    """
    def __init__(self, verbose=False, full_dataset=True):
        super().__init__()
        self.name = "floorplan"
        self.floorplan_root = 'C:\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS'
        self.dataset_folder_rgb = os.path.join(self.floorplan_root, "INPUT")
        self.dataset_folder_csv = os.path.join(self.floorplan_root, "CSV_GT_TRAJECTORIES")

        self.layout_types = ['corr_e2e', 'corr_cross']

        # self.dataset_folder = self.dataset_folder_rgb # os.path.join(self.dataset_folder_rgb, scene_name.split(SEP)[0], scene_name.split(SEP)[1])
        # self.scene_folder = os.path.join(self.dataset_folder, self.name)

        csv_path_list = [
            os.path.join(layout_type, flooplan_folder, variation_folder, csv_file) \
                for layout_type in self.layout_types \
                for flooplan_folder in os.listdir(os.path.join(self.dataset_folder_csv, layout_type)) \
                for variation_folder in os.listdir(os.path.join(self.dataset_folder_csv, layout_type, flooplan_folder)) \
                for csv_file in os.listdir(os.path.join(self.dataset_folder_csv, layout_type, flooplan_folder, variation_folder))
        ]

        len_rgb_dataset = len(csv_path_list)
        random.shuffle(csv_path_list)
        test_split_index = int(0.2 * len_rgb_dataset)

        if full_dataset:
            self.test_scenes = [csv_path_list[i] for i in range(0, test_split_index)]
            self.train_scenes = [csv_path_list[i] for i in range(test_split_index, len_rgb_dataset)]

        else:
            self.test_scenes = [csv_path_list[i] for i in range(0, 1)]
            self.train_scenes = [csv_path_list[i] for i in range(1, 3)]
        
        self.scenes = {key: Scene_floorplan(key, verbose) for
                       key in self.train_scenes+self.test_scenes}