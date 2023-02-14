import os
import pandas as pd
# from src.data_src.scene_src.scene_base import Scene_base
from helper import SEP
from collections import OrderedDict
import numpy as np
from PIL import Image
import h5py
import sparse

class Scene_floorplan():
    def __init__(self, img_path, csv_path, verbose=False):
        super().__init__()
        self.RGB_image_path = img_path
        self.raw_scene_data_path = csv_path

        assert os.path.isfile(self.RGB_image_path)
        assert os.path.isfile(self.raw_scene_data_path)

        self.column_names = [
            'frame_id',
            'agent_id',
            'x_coord',
            'y_coord',
        ]
        self.column_dtype = {
            'frame_id': np.float32,
            'agent_id': int,
            'x_coord': np.float32,
            'y_coord': np.float32,
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

        # used for meter <--> pixel conversion
        self.scale_down_factor = 1

        # self.load_scene_all(verbose)
        self.delta_time = 1 / self.frames_per_second
        # self.semantic_map_pred = self._load_rec_img()
        # import matplotlib.pyplot as plt
        # plt.imshow(self.RGB_image)

        layout_type = img_path.split(SEP)[-2]
        self.floorplan_min_x = 0
        self.floorplan_max_x = float(img_path.split('siteX_')[-1].split('_')[0])
        self.floorplan_min_y = 0
        self.floorplan_max_y = float(img_path.split('siteY_')[-1].split('_')[0])
        # For now, this is hardcoded, but in the future, the true floorplan sizes will be included in the dataset
        if layout_type != 'train_station':
            # including the wall thicknesses
            self.floorplan_min_x -= 0.15
            self.floorplan_min_y -= 0.15
            self.floorplan_max_x += 0.15
            self.floorplan_max_y += 0.15

        # RGB_image = self._load_RGB_image()
        # self.image_res_x = 640 # RGB_image.shape[1]
        # self.image_res_y = 640 # RGB_image.shape[0]

        # self.raw_pixel_data = self._make_pixel_coord_pandas(self._load_raw_data_table(self.raw_scene_data_path))
        self.raw_pixel_data = self._load_raw_data_table(self.raw_scene_data_path)

    def _load_RGB_image(self):
        if self.RGB_image_path.endswith('.jpeg') or self.RGB_image_path.endswith('.jpg'):
            with Image.open(self.RGB_image_path) as f:
                image = np.asarray(f)
        elif self.RGB_image_path.endswith('.h5'):
            image = np.array(h5py.File(self.RGB_image_path, 'r').get('img'))
        elif self.RGB_image_path.endswith('.npz'):
            image = sparse.load_npz(self.RGB_image_path).todense()
            image = image.astype(np.float32) #/ 255.
            image = np.clip(image, 0.0, 1.0)
        else:
            raise NotImplementedError
        return image


    def _load_raw_data_table(self, path):
        # load .csv raw data table with header
        raw_data = pd.read_csv(path,
                               engine='python',
                               header=None,
                               skiprows=None,
                               names=self.column_names,
                               dtype=self.column_dtype)

        columns_to_drop = []
        raw_data = raw_data.drop(columns=columns_to_drop)
        # convert timestamps to frame ids
        raw_data.frame_id= raw_data.frame_id*2-1
        raw_data.frame_id = raw_data.frame_id.astype(int)
        return raw_data

    @staticmethod
    def _linear_interpolation(curr_points_or, lim_min_or, lim_max_or, lim_min_proj, lim_max_proj):
        return lim_min_proj + (curr_points_or - lim_min_or) * (lim_max_proj - lim_min_proj) / (lim_max_or - lim_min_or)
    
    def _make_pixel_coord_pandas(self, raw_world_data):
        raw_pixel_data = raw_world_data.copy()
        # raw_pixel_data_or = raw_world_data.copy()

        raw_pixel_data['x_coord'] = Scene_floorplan._linear_interpolation(raw_pixel_data['x_coord'], self.floorplan_min_x, self.floorplan_max_x, 0, self.image_res_x)
        raw_pixel_data['y_coord'] = Scene_floorplan._linear_interpolation(raw_pixel_data['y_coord'], self.floorplan_min_y, self.floorplan_max_y, 0, self.image_res_y)
        
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
        world_batch_coord['x_coord'] = Scene_floorplan._linear_interpolation(world_batch_coord['x_coord'], 0, self.image_res_x, self.floorplan_min_x, self.floorplan_max_x)
        world_batch_coord['y_coord'] = Scene_floorplan._linear_interpolation(world_batch_coord['y_coord'], 0, self.image_res_y, self.floorplan_min_y, self.floorplan_max_y)
        return world_batch_coord

    def make_pixel_coord_torch(self, world_batch_coord):
        raise NotImplementedError
        pixel_batch_coord = world_batch_coord.clone()
        pixel_batch_coord[:, :, 1] *= -1
        pixel_batch_coord /= (self.ortho_px_to_meter * self.scale_down_factor)
        return pixel_batch_coord

    def make_world_coord_torch(self, pixel_batch_coord):
        world_batch_coord = pixel_batch_coord.clone()
        world_batch_coord[:, :, 0] = Scene_floorplan._linear_interpolation(world_batch_coord[:, :, 0], 0, self.image_res_x, self.floorplan_min_x, self.floorplan_max_x)
        world_batch_coord[:, :, 1] = Scene_floorplan._linear_interpolation(world_batch_coord[:, :, 1], 0, self.image_res_y, self.floorplan_min_y, self.floorplan_max_y)
        return world_batch_coord
    
    @staticmethod
    def make_world_coord_torch_static(pixel_batch_coord, image_res_x, image_res_y, floorplan_min_x, floorplan_min_y, floorplan_max_x, floorplan_max_y):
        world_batch_coord = pixel_batch_coord.clone()
        world_batch_coord[:, :, 0] = Scene_floorplan._linear_interpolation(world_batch_coord[:, :, 0], 0, image_res_x, floorplan_min_x, floorplan_max_x)
        world_batch_coord[:, :, 1] = Scene_floorplan._linear_interpolation(world_batch_coord[:, :, 1], 0, image_res_y, floorplan_min_y, floorplan_max_y)
        return world_batch_coord