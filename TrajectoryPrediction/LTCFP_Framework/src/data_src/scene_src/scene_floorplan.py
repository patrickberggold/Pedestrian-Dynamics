import os
import pandas as pd
from src.data_src.scene_src.scene_base import Scene_base
from helper import SEP
from collections import OrderedDict

class Scene_floorplan(Scene_base):
    def __init__(self, scene_name, verbose=False):
        super().__init__()
        self.name = os.path.join(scene_name.split(SEP)[0], scene_name.split(SEP)[1])
        self.dataset_name = "floorplan"
        self.dataset_folder_rgb = os.path.join(self.floorplan_root, "INPUT")
        self.dataset_folder_csv = os.path.join(self.floorplan_root, "CSV_GT_TRAJECTORIES")

        self.raw_scene_data_path = os.path.join(self.dataset_folder_csv, scene_name)
        self.RGB_image_name = os.path.join(f'HDF5_{scene_name.split(SEP)[1].split("__")[-1]}_{scene_name.split(SEP)[2]}.h5')
        assert os.path.isfile(self.raw_scene_data_path)

        self.dataset_folder = self.dataset_folder_rgb # os.path.join(self.dataset_folder_rgb, scene_name.split(SEP)[0], scene_name.split(SEP)[1])
        self.scene_folder = os.path.join(self.dataset_folder, self.name)
        # self.raw_scene_data_path = os.path.join(self.scene_folder,
        #                                         f"{scene_name}.csv")
        # self.RGB_image_name = f"{scene_name}_background.jpg"
        # self.semantic_map_gt_name = "scene_mask.png"

        self.column_names = [
            'frame_id',
            'agent_id',
            'x_coord',
            'y_coord',
        ]
        self.column_dtype = {
            'frame_id': float,
            'agent_id': int,
            'x_coord': float,
            'y_coord': float,
        }

        # semantic classes
        self.semantic_classes = OrderedDict([
            ('walkable', 'white'),
            ('wall', 'black'),
            ('origin', 'red'),
            ('destination', 'green'),
        ])

        self.frames_per_second = 2
        self.delta_frame = 1
        self.unit_of_measure = 'meter'

        self.has_H = False
        self.has_semantic_map_gt = False
        self.has_semantic_map_pred = True
        # used for meter <--> pixel conversion
        self.scale_down_factor = 1

        # self.load_scene_all(verbose)
        self.delta_time = 1 / self.frames_per_second
        RGB_image = self._load_RGB_image()
        # self.semantic_map_pred = self._load_rec_img()
        # import matplotlib.pyplot as plt
        # plt.imshow(self.RGB_image)

        layout_type = scene_name.split(SEP)[0]
        self.floorplan_min_x = 0
        self.floorplan_max_x = float(scene_name.split('siteX_')[-1].split('_')[0])
        self.floorplan_min_y = 0
        self.floorplan_max_y = float(scene_name.split('siteY_')[-1].split('_')[0])
        # For now, this is hardcoded, but in the future, the true floorplan sizes will be included in the dataset
        if layout_type != 'train_station':
            # including the wall thicknesses
            self.floorplan_min_x -= 0.15
            self.floorplan_min_y -= 0.15
            self.floorplan_max_x += 0.15
            self.floorplan_max_y += 0.15

        self.image_res_x = RGB_image.shape[1]
        self.image_res_y = RGB_image.shape[0]

        self.raw_pixel_data = self._make_pixel_coord_pandas(self._load_raw_data_table(self.raw_scene_data_path))


    def _load_raw_data_table(self, path):
        # load .csv raw data table with header
        raw_data = pd.read_csv(path,
                               engine='python',
                               header=None,
                               skiprows=1,
                               names=self.column_names,
                               dtype=self.column_dtype)

        columns_to_drop = []
        raw_data = raw_data.drop(columns=columns_to_drop)
        # convert timestamps to frame ids
        raw_data.frame_id= raw_data.frame_id*2-1
        raw_data.frame_id = raw_data.frame_id.astype(int)
        return raw_data

    def _linear_interpolation(self, curr_points_or, lim_min_or, lim_max_or, lim_min_proj, lim_max_proj):
        return lim_min_proj + (curr_points_or - lim_min_or) * (lim_max_proj - lim_min_proj) / (lim_max_or - lim_min_or)
    
    def _make_pixel_coord_pandas(self, raw_world_data):
        raw_pixel_data = raw_world_data.copy()
        # raw_pixel_data_or = raw_world_data.copy()

        raw_pixel_data['x_coord'] = self._linear_interpolation(raw_pixel_data['x_coord'], self.floorplan_min_x, self.floorplan_max_x, 0, self.image_res_x)
        raw_pixel_data['y_coord'] = self._linear_interpolation(raw_pixel_data['y_coord'], self.floorplan_min_y, self.floorplan_max_y, 0, self.image_res_y)
        
        # # test trafo back
        # x_back = self._linear_interpolation(raw_pixel_data['x_coord'], 0, self.image_res_x, self.floorplan_min_x, self.floorplan_max_x).values
        # y_back = self._linear_interpolation(raw_pixel_data['y_coord'], 0, self.image_res_y, self.floorplan_min_y, self.floorplan_max_y).values

        # t = raw_pixel_data_or['x_coord'].values == x_back
        # d = raw_pixel_data_or['y_coord'].values == y_back

        # x_diffs = 0
        # y_diffs = 0
        # for idx, x_i in enumerate(x_back):
        #     x_or = round(raw_pixel_data_or['x_coord'].values[idx], 5)
        #     x_proj = round(x_i, 5)
        #     x_diffs += abs(x_or - x_proj)
        #     if abs(x_or - x_proj) > 0.01:
        #         he = 2

        # for idy, y_i in enumerate(y_back):
        #     y_or = round(raw_pixel_data_or['y_coord'].values[idy], 5)
        #     y_proj = round(y_i, 5)
        #     y_diffs += abs(y_or - y_proj)
        #     if abs(y_or - y_proj) > 0.01:
        #         he = 2

        return raw_pixel_data

    def _make_world_coord_pandas(self, raw_pixel_data):
        raise NotImplementedError
        world_batch_coord = pixel_batch_coord.clone()
        world_batch_coord['x_coord'] = self._linear_interpolation(world_batch_coord['x_coord'], 0, self.image_res_x, self.floorplan_min_x, self.floorplan_max_x)
        world_batch_coord['y_coord'] = self._linear_interpolation(world_batch_coord['y_coord'], 0, self.image_res_y, self.floorplan_min_y, self.floorplan_max_y)
        return world_batch_coord

    def make_pixel_coord_torch(self, world_batch_coord):
        raise NotImplementedError
        pixel_batch_coord = world_batch_coord.clone()
        pixel_batch_coord[:, :, 1] *= -1
        pixel_batch_coord /= (self.ortho_px_to_meter * self.scale_down_factor)
        return pixel_batch_coord

    def make_world_coord_torch(self, pixel_batch_coord):
        world_batch_coord = pixel_batch_coord.clone()
        world_batch_coord[:, :, 0] = self._linear_interpolation(world_batch_coord[:, :, 0], 0, self.image_res_x, self.floorplan_min_x, self.floorplan_max_x)
        world_batch_coord[:, :, 1] = self._linear_interpolation(world_batch_coord[:, :, 1], 0, self.image_res_y, self.floorplan_min_y, self.floorplan_max_y)
        return world_batch_coord