import os
import pickle
import random

import numpy as np
import torch

from src.data_src.dataset_src.dataset_create import create_dataset
from src.data_src.experiment_src.experiment_create import create_experiment
from src.models.model_utils.cnn_big_images_utils import create_tensor_image, \
    create_CNN_inputs_loop
from src.utils import maybe_makedir
from helper import SEP
from collections import OrderedDict
from feature_extractor.image2image_module import Image2ImageModule

import h5py
from torchvision import transforms
import torchvision.transforms.functional as F
from src.m_transforms import m_RandomHorizontalFlip, m_RandomRotation, m_RandomVerticalFlip
import random
import numpy as np
from skimage.draw import line
from tqdm import tqdm

def is_legitimate_traj(traj_df, step):
    agent_id = traj_df.agent_id.values
    # check if I only have 1 agent (always the same)
    if not (agent_id[0] == agent_id).all():
        print("not same agent")
        return False
    frame_ids = traj_df.frame_id.values
    equi_spaced = np.arange(frame_ids[0], frame_ids[-1] + 1, step, dtype=int)
    if len(frame_ids) != len(equi_spaced):
        raise ValueError
    # check that frame rate is evenly-spaced
    if not (frame_ids == equi_spaced).all():
        print("not equi_spaced")
        return False
    # if checks are passed
    return True

def is_legitimate_traj_np(traj_df_np, step):
    agent_id = traj_df_np[:,1]
    # check if I only have 1 agent (always the same)
    if not (agent_id[0] == agent_id).all():
        print("not same agent")
        return False
    frame_ids = traj_df_np[:,0]
    equi_spaced = np.arange(frame_ids[0], frame_ids[-1] + 1, step, dtype=int)
    # check that frame rate is evenly-spaced
    if len(frame_ids) != len(equi_spaced):
        print(f"agent {agent_id[0]} not equi_spaced")
        return False
    if not (frame_ids == equi_spaced).all() :
        print("not equi_spaced")
        return False
    # if checks are passed
    return True


class Trajectory_Data_Pre_Process(object):
    def __init__(self, args):
        self.args = args

        # Segmentation network
        if args.dataset == 'floorplan':
            self.final_resolution = 1000
            self.max_floorplan_meters = 70
            self._load_seg_network(args.device)
            print('Loaded segmentation network successfully')

        # Trajectories and data_batches folder
        self.data_batches_path = os.path.join(
            self.args.save_dir, 'data_batches')
        maybe_makedir(self.data_batches_path)

        # Creating batches folders and files
        self.batches_folders = {}
        self.batches_confirmation_files = {}
        for set_name in ['train', 'valid', 'test']:
            # batches folders
            self.batches_folders[set_name] = os.path.join(
                self.data_batches_path, f"{set_name}_batches")
            maybe_makedir(self.batches_folders[set_name])
            # batches confirmation file paths
            self.batches_confirmation_files[set_name] = os.path.join(
                self.data_batches_path, f"finished_{set_name}_batches.txt")

        # exit pre-processing early
        if os.path.exists(self.batches_confirmation_files["test"]):
            print('Data batches already created!\n')
            return

        print("Loading dataset and experiment ...")
        # Define the train, val, test files
        self.dataset = create_dataset(self.args.dataset, full_dataset=False)
        # Load the data into dataframes
        self.experiment = create_experiment(self.args.dataset)(
            self.dataset,self.args.test_set, self.args, full_dataset=False)
        print("Done.\n")

        print("Preparing data batches ...")
        self.num_batches = {}
        for set_name in ['train', 'valid', 'test']:
            if not os.path.exists(self.batches_confirmation_files[set_name]):
                self.num_batches[set_name] = 0
                print(f"\nPreparing {set_name} batches ...")
                self.create_data_batches(set_name)

        print('Data batches created!\n')

    def create_data_batches(self, set_name):
        """
        Create data batches for the DataLoader object
        """
        for scene_data in self.experiment.data[set_name]:
            # break if fast_debug
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break
            self.make_batches(scene_data, set_name)
            print(f"Saved a total of {self.num_batches[set_name]} {set_name} "
                  f"batches ...")

        with open(self.batches_confirmation_files[set_name], "w") as f:
            f.write(f"Number of {set_name} batches: "
                    f"{self.num_batches[set_name]}")

    def make_batches(self, scene_data, set_name):
        """
        Query the trajectories fragments and make data batches.
        Notes: Divide the fragment if there are too many people; accumulate some
        fragments if there are few people.
        """
        scene_name = scene_data["scene_name"]
        scene = self.dataset.scenes[scene_name]
        delta_frame = scene.delta_frame
        downsample_frame_rate = scene_data["downsample_frame_rate"]

        df = scene_data['raw_pixel_data']

        if set_name == 'train':
            shuffle = self.args.shuffle_train_batches
        elif set_name == 'test':
            shuffle = self.args.shuffle_test_batches
        else:
            shuffle = self.args.shuffle_test_batches
        assert scene_data["set_name"] == set_name

        fragment_list = []  # list of fragments per scene
        
        ####################### NUMPY VERSION #######################
        # fragment_list_np = []
        # df_np = df.to_numpy()
        # agent_ids = np.unique(df_np[:, 1])
        # # agent_data = [df_np[df_np[:,0] == i] for i in agent_ids]
        # for agent_id in agent_ids:
        #     agent_id_data = df_np[df_np[:,1] == agent_id]
        #     agent_id_data = agent_id_data[::downsample_frame_rate]
        #     for start_t in range(0, len(agent_id_data), self.args.skip_ts_window):
        #         candidate_traj = agent_id_data[start_t:start_t + self.args.seq_length]
        #         if len(candidate_traj) == self.args.seq_length:
        #             if is_legitimate_traj_np(candidate_traj,
        #                                   step=downsample_frame_rate * delta_frame):
        #                 fragment_list_np.append(candidate_traj)
        ####################### NUMPY VERSION #######################

        for agent_i in set(df.agent_id):
            hist = df[df.agent_id == agent_i]
            # downsample frame rate happens here, at the single agent level
            hist = hist.iloc[::downsample_frame_rate]

            for start_t in range(0, len(hist), self.args.skip_ts_window):
                candidate_traj = hist.iloc[start_t:start_t + self.args.seq_length]
                if len(candidate_traj) == self.args.seq_length:
                    if is_legitimate_traj(candidate_traj,
                                          step=downsample_frame_rate * delta_frame):
                        fragment_list.append(candidate_traj)
        # TODO trajectories are read in single fashion right now, only one agent trajectory will be loaded ==> load all agents within one frame (up to max agent, like AgentFormer)
        if shuffle:
            random.shuffle(fragment_list)

        batch_acculumator = []
        batch_ids = {
            "scene_name": scene_name, # 'corr_cross\\1__floorplan_siteX_45_siteY_20_CORRWIDTH_3_SIDECORRWIDTH_25_numCross_3\\variation_184\\variation_184_num_agents_15.txt'
            "starting_frames": [],
            "agent_ids": [],
            "data_file_path": scene_data["file_path"]} # 'C:\\Users\\Remotey\\Documents\\Datasets\\ADVANCED_FLOORPLANS\\CSV_GT_TRAJECTORIES\\corr_cross\\1__floorplan_siteX_45_siteY_20_CORRWIDTH_3_SIDECORRWIDTH_25_numCross_3\\variation_184\\variation_184_num_agents_15.txt'

        pbar = tqdm(total=len(fragment_list)//self.args.batch_size+1)
        pbar.set_description(f"Processing {scene_name}")

        for fragment_df in fragment_list:
            # break if fast_debug
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break

            batch_ids["starting_frames"].append(fragment_df.frame_id.iloc[0])
            batch_ids["agent_ids"].append(fragment_df.agent_id.iloc[0])
            # TODO right now, batch_accumulator consists of only random trajectories stacked together, from different agents at different times ==> change that
            batch_acculumator.append(fragment_df[["x_coord", "y_coord"]].values)

            # save batch if big enough
            if len(batch_acculumator) == self.args.batch_size:
                # create and save batch
                self.massup_batch_and_save(batch_acculumator,
                                           batch_ids, set_name)

                # reset batch_acculumator and ids for new batch
                batch_acculumator = []
                batch_ids = {
                    "scene_name": scene_name,
                    "starting_frames": [],
                    "agent_ids": [],
                    "data_file_path": scene_data["file_path"]}

                pbar.update(1)

        # save last (incomplete) batch if there is some fragment left
        if batch_acculumator:
            # create and save batch
            self.massup_batch_and_save(batch_acculumator,
                                       batch_ids, set_name)
            
            pbar.update(1)

    def massup_batch_and_save(self, batch_acculumator, batch_ids, set_name):
        """
        Mass up data fragments to form a batch and then save it to disk.
        From list of dataframe fragments to saved batch.
        """
        abs_pixel_coord = np.stack(batch_acculumator).transpose(1, 0, 2) # shape: (32, 20, 2)
        seq_list = np.ones((abs_pixel_coord.shape[0],
                            abs_pixel_coord.shape[1]))

        data_dict = {
            "abs_pixel_coord": abs_pixel_coord,
            "seq_list": seq_list,
        }

        # add cnn maps and inputs
        data_dict = self.add_pre_computed_cnn_maps(data_dict, batch_ids)

        # increase batch number count
        self.num_batches[set_name] += 1
        batch_name = os.path.join(
            self.batches_folders[set_name],
            f"{set_name}_batch" + "_" + str(
                self.num_batches[set_name]).zfill(4) + ".pkl")
        # save batch
        with open(batch_name, "wb") as f:
            pickle.dump((data_dict, batch_ids), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def add_pre_computed_cnn_maps(self, data_dict, batch_ids):
        """
        Pre-compute CNN maps used by the goal modules and add them to data_dict
        """
        abs_pixel_coord = data_dict["abs_pixel_coord"]
        scene_name = batch_ids["scene_name"]
        scene = self.dataset.scenes[scene_name]

        # numpy semantic map from 0 to 1
        if self.dataset.name == 'floorplan':
            # img = self._load_recon_image(rgb_path=os.path.join(scene.scene_folder, scene.RGB_image_name))
            img = np.array(h5py.File(os.path.join(scene.scene_folder, scene.RGB_image_name), 'r').get('img'))
            # # self.final_resolution = 500 # scale_down = 2
            # tensor_image, [scale_x, scale_y, rotation_angle, h_flip, v_flip] = self._image_preprocessing(os.path.join(scene.scene_folder, scene.RGB_image_name))

            # # rescale the trajectories
            # new_x = abs_pixel_coord[:,:,0] * scale_x
            # new_y = abs_pixel_coord[:,:,1] * scale_y

            # C, H, W = tensor_image.shape
            # new_heigth = int(H / self.args.down_factor)
            # new_width = int(W / self.args.down_factor)
            # tensor_image = F.resize(tensor_image, (new_heigth, new_width),
            #                                     interpolation=transforms.InterpolationMode.BILINEAR)
            # data_dict["augmentation"] = [rotation_angle, h_flip, v_flip]

        else:
            img = scene.semantic_map_pred

        tensor_image = create_tensor_image(
            big_numpy_image=img,
            down_factor=self.args.down_factor) # resize W & H by down_factor

        # import cv2
        # import matplotlib.pyplot as plt
        # img_np = tensor_image.permute(1, 2, 0).detach().cpu().numpy()
        # traj = np.moveaxis(abs_pixel_coord/2., 1, 0)
        # traj = [t for t in traj]
        # for t_id in traj:
        #     old_point = None
        #     r_val = random.uniform(0.4, 1.0)
        #     b_val = random.uniform(0.7, 1.0)
        #     for point in t_id:
        #         x_n, y_n = round(point[0]), round(point[1])
        #         assert 0 <= x_n <= img_np.shape[1] and 0 <= y_n <= img_np.shape[0]
        #         if old_point != None:
        #             # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
        #             c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, img_np.shape[0]-y_n)))]
        #             for c in c_line:
        #                 img_np[c[1], c[0]] = np.array([r_val, 0., b_val])
        #             # plt.imshow(img_np)

        #         old_point = (x_n, img_np.shape[0]-y_n)
        # plt.imshow(img_np)

        # input_traj_maps = create_CNN_inputs_loop(
        #     batch_abs_pixel_coords=torch.tensor(abs_pixel_coord).float() /
        #                            self.args.down_factor,
        #     tensor_image=tensor_image)
        torchy = torch.randn(20, 2)

        input_traj_maps = create_CNN_inputs_loop(
            batch_abs_pixel_coords=torchy.float().unsqueeze(1) /
                                   self.args.down_factor,
            tensor_image=tensor_image)


        data_dict["tensor_image"] = tensor_image
        data_dict["input_traj_maps"] = input_traj_maps

        return data_dict

    def _load_seg_network(self, device):
        ckpt_seg_net = SEP.join(['TrajectoryPrediction','LTCFP_Framework','feature_extractor','checkpoints', 'model_deepLab_img2img_epoch=10-step=11660.ckpt'])
        state_dict = OrderedDict([(key, tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(ckpt_seg_net)['state_dict'].items()])
        self.model = Image2ImageModule(mode='img2img', relu_at_end=True)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.net.eval()

    def _load_recon_image(self, rgb_path):
        img, [scale_x, scale_y, rotation_angle, h_flip, v_flip] = self._image_preprocessing(rgb_path)

        pred_img = self.model.forward(img.unsqueeze(0).to(self.args.device))['out'].squeeze()

        import matplotlib.pyplot as plt
        img_np = img.transpose(0,1).transpose(1, 2).cpu().detach().numpy()
        img_pred_np = pred_img.transpose(0,1).transpose(1, 2).detach().cpu().numpy()
        img_total = np.concatenate((img_np, img_pred_np), axis=1)
        # plt.imshow(img_total)

        return pred_img

    def _image_preprocessing(self, img_path):

        import matplotlib.pyplot as plt

        floorplan_size_x = float(img_path.split('siteX_')[-1].split('_')[0])
        floorplan_size_y = float(img_path.split('siteY_')[-1].split('_')[0])

        img = transforms.ToTensor()(np.array(h5py.File(img_path, 'r').get('img')))

        scale_x = floorplan_size_x / self.max_floorplan_meters
        scale_y = floorplan_size_y / self.max_floorplan_meters

        assert 0.0 <= scale_x <= 1.0 and 0.0 <= scale_y <= 1.0

        scaled_resolution_x = int(self.final_resolution * scale_x)
        scaled_resolution_y = int(self.final_resolution * scale_y)

        # Scale image down
        img = transforms.Resize(size=(scaled_resolution_y, scaled_resolution_x))(img)

        # Rotate the image by an angle
        transform_rot = m_RandomRotation((0, 360), F.InterpolationMode.BILINEAR, expand=True, fill=0)
        rotated_img = transform_rot(img)
        rotation_angle = transform_rot.angle

        # Make sure that the rotated image doesnt exceed the final resolution
        if rotated_img.size()[-2] > self.final_resolution or rotated_img.size()[-1] > self.final_resolution:
            rotation_angle = random.choice([0,90,180,270])
            rotated_img = transforms.RandomRotation((rotation_angle, rotation_angle), expand=False, fill=0)(img)

        # pad the rest of the image with zeros
        width = rotated_img.size()[-1]
        height = rotated_img.size()[-2]

        padd_width_x0 = 3 #(self.final_resolution - width)//2
        padd_width_x1 = self.final_resolution - width - padd_width_x0

        padd_width_y0 = 3# (self.final_resolution - height)//2
        padd_width_y1 = self.final_resolution - height - padd_width_y0

        assert width+padd_width_x0+padd_width_x1 == self.final_resolution
        assert height+padd_width_y0+padd_width_y1 == self.final_resolution

        # left, top, right and bottom borders respectively
        rotated_img = transforms.Pad([padd_width_x0, padd_width_y0, padd_width_x1, padd_width_y1], fill=0., padding_mode='constant')(rotated_img)
        # plt.imshow(rotated_img.permute(1, 2, 0))

        # include random flipping
        # transform_hf = m_RandomHorizontalFlip(p=0.5)
        # rotated_img = transform_hf(rotated_img)
        # h_flip = transform_hf.h_flip
        # transform_vf = m_RandomVerticalFlip(p=0.5)
        # rotated_img = transform_vf(rotated_img)
        # v_flip = transform_vf.v_flip

        # plt.imshow(rotated_img.permute(1, 2, 0))
        assert rotated_img.size()[-2] == self.final_resolution and rotated_img.size()[-1] == self.final_resolution
        
        return rotated_img, [padd_width_x0, padd_width_y0, scale_x, scale_y, rotation_angle, True, True]

