import os
import pandas as pd
# from src.data_src.scene_src.scene_base import Scene_base
from helper import SEP
from collections import OrderedDict
import numpy as np
import cv2
from PIL import Image
import h5py
import sparse

class Scene_floorplan():
    def __init__(self, img_path, csv_path, semanticMap_path=None, global_semanticMap_path=None, find_dsts=False, verbose=False):
        super().__init__()
        self.RGB_image_path = img_path
        self.raw_scene_data_path = csv_path
        self.semanticMap_path = semanticMap_path
        self.global_semanticMap_path = global_semanticMap_path

        assert os.path.isfile(self.RGB_image_path)
        assert os.path.isfile(self.raw_scene_data_path)
        if semanticMap_path is not None: assert os.path.isfile(self.semanticMap_path)
        if global_semanticMap_path is not None: assert os.path.isfile(self.global_semanticMap_path)

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
        self.new_ds = True # includes destinations in the first row of the csv file, and also includes semantic maps for each agent

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
        layout_type = img_path.split(SEP)[-4]
        self.destinations = []
        # assuming that all floorplans have a padded real-world length/width of 64m

        assert layout_type in ['corr_e2e', 'corr_cross'], 'train stations do not have the same floorplan size'
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

        if find_dsts:
            destinations = []
            # find all green rectangles in the image
            import matplotlib.pyplot as plt
            np_image = sparse.load_npz(self.RGB_image_path).todense()
            bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            lower_green = np.array([0, 100, 0], dtype=np.uint8)
            upper_green = np.array([100, 255, 100], dtype=np.uint8)
            mask = cv2.inRange(bgr_image, lower_green, upper_green)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 10 or h < 10:
                    continue
                destinations.append((x, y, x+w, y+h))
                cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (150, 0, 150), 3)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # plt.imshow(rgb_image)
            assert 0 < len(destinations) < 3, 'destinations must be between 1 and 2'
            # plt.close('all')
            if self.all_scene_destinations == []: self.all_scene_destinations = destinations
            else:
                np_destinations = np.array(destinations)
                np_self_destinations = np.array(self.all_scene_destinations)
                # sort arrays
                np_destinations = np_destinations[np.argsort(np_destinations[:, 0])]
                np_self_destinations = np_self_destinations[np.argsort(np_self_destinations[:, 0])]
                diffs = np.abs(np_destinations - np_self_destinations)
                assert np.all(diffs < 2), 'destinations must be the same for all scenes'

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
        if self.new_ds:
            skiprows = [0]
            f = open(path, 'r')
            destinations_str = f.readlines()[0]
            f.close()
            self.destinations_per_agent = [[int(dc) for dc in dest] for dest in [d.split('[')[-1].split(']')[0].split(',') for d in destinations_str.split(';')]]
            destinations_set = set()
            # remove duplicates
            for dst in self.destinations_per_agent:
                destinations_set.add(tuple(dst))
            self.all_scene_destinations = [list(entry_tuple) for entry_tuple in destinations_set]    
        else:
            skiprows = None
        raw_data = pd.read_csv(path,
                               engine='python',
                               header=None,
                               skiprows=skiprows,
                               names=self.column_names,
                               dtype=self.column_dtype)

        columns_to_drop = []
        raw_data = raw_data.drop(columns=columns_to_drop)
        # convert timestamps to frame ids
        if raw_data.frame_id.values[0] == 1.0:
            raw_data.frame_id = raw_data.frame_id*2-1
        elif raw_data.frame_id.values[0] == 0.0:
            raw_data.frame_id = raw_data.frame_id*2+1
        else:
            raise NotImplementedError
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