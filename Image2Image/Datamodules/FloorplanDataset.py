from torch.utils.data import Dataset
import numpy as np
import h5py
from helper import SEP
import matplotlib.pyplot as plt
import albumentations as A
import torch
import cv2
import sparse
from itertools import product

class Dataset_Img2Img(Dataset):
    def __init__(
        self, 
        mode: str, 
        img_paths: list, 
        traj_paths: list, 
        transform = None, 
        additional_info: bool = False,
        vary_area_brightness: bool = True, 
    ):
        # TODO maybe turn off transformations when eval/test
        self.transform = transform
        self.img_paths = img_paths
        self.traj_paths = traj_paths
        self.mode = mode
        self.vary_area_brightness = vary_area_brightness
        self.max_floorplan_meters = 70
        self.final_resolution = 800
        self.use_addit_info = additional_info

        assert mode in ['grayscale', 'evac_only', 'class_movie', 'density_class', 'density_reg', 'denseClass_wEvac'], 'Unknown mode setting!'
        assert len(self.traj_paths) == len(self.img_paths), 'Length of image paths and trajectory paths do not match, something went wrong!'

        if self.use_addit_info:
            self.data_dict = {}
            add_data_path = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS', 'INPUT'])
            if self.mode == 'denseClass_wEvac':
                add_data_path = SEP.join(['C:', 'Users', 'Remotey', 'Documents', 'Datasets', 'ADVANCED_FLOORPLANS_SPARSE', 'SPARSE_DENSITY_INPUT_640'])
            self.max_ors = 0
            self.max_dsts = 0
            self.load_additional_info(add_data_path)
        if self.mode in ['density_class', 'density_reg', 'denseClass_wEvac']:
            # delete the resize operation for the new dataset, has already been done...
            if len(self.transform.transforms)==2:
                del self.transform.transforms[0]

    def __len__(self):
        return(len(self.img_paths))

    def __getitem__(self, idx):
        img_path, traj_path = self.img_paths[idx], self.traj_paths[idx]

        if self.mode in ['density_reg', 'density_class', 'denseClass_wEvac']:

            traj_aug_mode = 0 if self.mode=='density_reg' else 1
            
            img = sparse.load_npz(img_path).todense()
            img = img.astype(np.float32) / 255.
            if self.mode != 'denseClass_wEvac':
                traj = sparse.load_npz(traj_path).todense().transpose(1,2,0)
            else:
                with np.load(traj_path) as fp:
                    coords = fp["coords"]
                    data = fp["data"]
                    evac_time = float(data[-1])/2.
                    data = np.delete(data, -1)
                    shape = tuple(fp["shape"])
                    fill_value = fp["fill_value"][()]
                    traj = sparse.COO(
                        coords=coords,
                        data=data,
                        shape=shape,
                        sorted=True,
                        has_duplicates=False,
                        fill_value=fill_value,
                    )
                assert evac_time > 10.
                traj = traj.todense().transpose(1,2,0)
                traj = np.clip(traj, 0, 4)
            
            """ # for frame in traj.numpy().astype(np.int32).transpose(2,0,1):
            for frame in traj.transpose(2,0,1):
                nnz_coords = np.argwhere(frame > 0).squeeze()
                bin_count = frame[nnz_coords[:, 0], nnz_coords[:, 1]]
                cell_size = 4

                bin4image = np.repeat(nnz_coords, cell_size**2, axis=0) * cell_size
                binCount4image = np.repeat(bin_count, cell_size**2, axis=0)

                add_vector = np.zeros((16,2), dtype=np.int32)
                for x in range(4):
                    for y in range(4):
                        add_vector[4*x + y] = np.array([x, y], dtype=np.int32)
                add_vector = np.tile(add_vector.transpose(1,0), nnz_coords.shape[0]).transpose(1,0)
                bin4image += add_vector

                counts1 = np.argwhere(binCount4image==1).squeeze()
                count1_coords = bin4image[counts1]

                counts2 = np.argwhere(binCount4image==2).squeeze()
                count2_coords = bin4image[counts2]

                counts3 = np.argwhere(binCount4image==3).squeeze()
                count3_coords = bin4image[counts3]

                counts4 = np.argwhere(binCount4image==4).squeeze()
                count4_coords = bin4image[counts4]

                counts5 = np.argwhere(binCount4image>=5).squeeze()
                count5_coords = bin4image[counts5]

                # binned_image = img.numpy().copy().transpose(1,2,0)
                binned_image = img.copy()
                binned_image = (binned_image+1)/2. * 255.
                binned_image = binned_image.astype(np.int32)

                binned_image[count1_coords[:, 0], count1_coords[:, 1]] = np.array([0, 150, 255])
                binned_image[count2_coords[:, 0], count2_coords[:, 1]] = np.array([0, 0, 255])
                binned_image[count3_coords[:, 0], count3_coords[:, 1]] = np.array([0, 0, 155])
                binned_image[count4_coords[:, 0], count4_coords[:, 1]] = np.array([0, 0, 100])
                binned_image[count5_coords[:, 0], count5_coords[:, 1]] = np.array([100, 0, 100])

                # plt.imshow(binned_image)
                plt.imsave('testImg.png', binned_image.astype(np.uint8))
                plt.close('all') """

            if self.mode == 'density_class': 
                traj = np.clip(traj, 0, 4)

            img, traj = self.augment_traj_and_images4density(img, traj, traj_aug_mode)

            if self.transform:
                img = self.transform(img)

            if self.use_addit_info:
                info_dict = self.data_dict[SEP.join([img_path.split(SEP)[-4], img_path.split(SEP)[-3]])]['data'][img_path.split(SEP)[-2]]
                distances = self.extract_distances(info_dict)
            
            if self.mode == 'denseClass_wEvac':
                if self.use_addit_info:
                    return img, (traj, evac_time), [distances.astype(np.float32)] 
                else:
                    return img, (traj, evac_time)

            if self.use_addit_info:
                return img, traj, [distances.astype(np.float32)] 
            else:
                return img, traj

        self.floorplan_max_x = float(img_path.split(SEP)[-1].split('siteX_')[1].split('_siteY')[0])
        self.floorplan_max_y = float(img_path.split(SEP)[-1].split('siteY_')[1].split('_')[0])

        # quick check if paths are consistent
        assert '_'.join(img_path.split('_')[-2:]).replace('.h5', '') in traj_path
        
        img = np.array(h5py.File(img_path, 'r').get('img'))
        img = img.astype(bool).astype(np.float32) # deal with compression errors: some values not shown in full color
        # plt.imshow(img)

        # Change origin and destination area brightnesses
        if self.vary_area_brightness:
            agent_variations = [15, 25, 50]
            # bright_limit = 100
            # dark_limit = 155
            # color_interval_length = (dark_limit+bright_limit)//len(agent_variations)
            agent_variations_index = agent_variations.index(int(traj_path.split(SEP)[-1].split('_')[-1].split('.')[0]))
            # reset origin color
            # or_color_range = [[255, 100, 100], [255, 50, 50], [255, 0, 0], [205, 0, 0], [155, 0, 0]]
            or_color_range = [[1., 0.392, 0.392], [1., 0, 0], [0.608, 0, 0]]
            or_area_color = np.array(or_color_range[agent_variations_index])
            or_coords = np.where(np.all(img == np.array([1.0, 0, 0]), axis=-1))
            img[or_coords[0], or_coords[1]] = or_area_color
            # reset destination color
            # dst_color_range = [[100, 255, 100], [50, 255, 50], [0, 255, 0], [0, 205, 0], [0, 155, 0]]
            dst_color_range = [[0.392, 1., 0.392], [0, 1., 0], [0, 0.608, 0]]
            dst_area_color = np.array(dst_color_range[agent_variations_index])
            dst_coords = np.where(np.all(img == np.array([0, 1.0, 0]), axis=-1))
            img[dst_coords[0], dst_coords[1]] = dst_area_color

            assert len(or_coords[0]) > 0 and len(or_coords[0])==len(or_coords[1])
            assert len(dst_coords[0]) > 0 and len(dst_coords[0])==len(dst_coords[1])
            # plt.imshow(img)

        if self.mode in ['grayscale']:
            traj = np.array(h5py.File(traj_path, 'r').get('img')).astype('float32')
            normalize_traj = False
            if normalize_traj:
                traj /= np.max(traj)

        elif self.mode == 'class_movie':
            sparse_matrix = sparse.load_npz(traj_path)
            slow_class_coords = np.argwhere(sparse_matrix.data <= 0.9).squeeze()
            fast_class_coords = np.argwhere(sparse_matrix.data > 0.9).squeeze()
            sparse_matrix.data[slow_class_coords] = 1
            sparse_matrix.data[fast_class_coords] = 2
            sparse_matrix.data = sparse_matrix.data.astype(int)
            traj = sparse_matrix.todense().transpose(2,1,0)
        elif self.mode == 'evac_only':
            traj = None
        
        # Visualize trajectory for checking correctness
        # non_zeros = np.argwhere(traj != 0.)
        # img_now = img.copy()
        # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 255])
        # plt.imshow(img_now, vmin=0, vmax=255)

        augmentation = True # change this later maybe for testing
        if self.mode in ['grayscale']:
            traj_aug_mode = 0 # regression
        elif self.mode == 'evac_only':
            traj_aug_mode = None
        elif self.mode == 'class_movie':
            traj_aug_mode = 1 # classification
        else:
            raise NotImplementedError

        img, traj = self.augment_traj_and_images(img, traj, augmentation, traj_aug_mode)

        if self.transform:
            img = self.transform(img)
        
        if self.mode in ['evac_only']:
            evac_time = float(np.array(h5py.File(traj_path, 'r').get('max_time')))
            img = (img, evac_time)

            if self.mode == 'evac_only':
                traj = 0.

        if self.use_addit_info:
            info_dict = self.data_dict[SEP.join([img_path.split(SEP)[-3], img_path.split(SEP)[-2]])]['data'][f'variation_{img_path.split("_")[-1].replace(".h5", "")}']
            distances = self.extract_distances(info_dict)
            return img, traj, [distances.astype(np.float32)] 

        return img, traj


    def augment_traj_and_images(self, image, traj_image, augmentation, traj_aug_mode):

        scale_x = self.floorplan_max_x / self.max_floorplan_meters
        scale_y = self.floorplan_max_y / self.max_floorplan_meters

        assert 0.0 <= scale_x <= 1.0 and 0.0 <= scale_y <= 1.0

        scaled_resolution_x = int(self.final_resolution * scale_x)
        scaled_resolution_y = int(self.final_resolution * scale_y)

        # Resize first to create Gaussian maps later
        transform = A.Compose([
            A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_NEAREST),
            ],
            # additional_targets={'traj_map': 'image'}
            )

        if traj_aug_mode == 0:
            # transform.add_targets({'traj_map': 'image'})
            transformed = transform(image=image) #, traj_map=traj_image)
            # traj_image = transformed['traj_map']
            traj_image = torch.from_numpy(traj_image).unsqueeze(0)
            traj_image = torch.nn.AdaptiveMaxPool2d((scaled_resolution_y, scaled_resolution_x))(traj_image).squeeze().numpy()
        elif traj_aug_mode == 1:
            transformed = transform(image=image, mask=traj_image)
            traj_image = transformed['mask']
        else:
            transformed = transform(image=image)

        image = transformed['image']

        #################### TEST START ####################
        # # 2D plot pred
        # plt.imshow(image)
        # max_time = np.max(traj_image)
        # non_zeros = np.argwhere(traj_image != 0.)
        # img_now = image.copy()
        # pred_colors_from_timestamps = [get_color_from_array(traj_image[x, y], max_time)/255. for x, y in non_zeros]
        # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array(pred_colors_from_timestamps)
        # # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 1.])
        # plt.imshow(img_now)
        
        # # 3D plot pred
        # fig = plt.figure(figsize=(6,6))
        # ax = fig.add_subplot(111, projection='3d')
        # X,Y = np.meshgrid(np.arange(traj_image.shape[1]), np.arange(traj_image.shape[0]))
        # ax.plot_surface(X, Y, traj_image)
        # plt.show()
        # plt.close('all')
        #################### TEST END ####################

        #################### TEST START ####################
        # 2D plot pred
        # # plt.imshow(image)
        # non_zeros = np.argwhere(traj_image != 0.)
        # img_now = image.copy()
        # pred_colors_from_timestamps = [get_color_from_array(traj_image[x, y], evac_time)/255. for x, y in non_zeros]
        # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array(pred_colors_from_timestamps)
        # # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 1.])
        # plt.imshow(img_now)
        
        # # 3D plot pred
        # fig = plt.figure(figsize=(6,6))
        # ax = fig.add_subplot(111, projection='3d')
        # X,Y = np.meshgrid(np.arange(traj_image.shape[1]), np.arange(traj_image.shape[0]))
        # ax.plot_surface(X, Y, traj_image)
        # plt.show()
        # plt.close('all')
        #################### TEST END ####################

        if augmentation:
            transform = A.Compose([
                # SAFE AUGS, flips and 90rots
                # A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
                A.augmentations.transforms.HorizontalFlip(p=0.5),
                A.augmentations.transforms.VerticalFlip(p=0.5),
                A.augmentations.transforms.Transpose(p=0.5),
                A.augmentations.geometric.rotate.RandomRotate90(p=1.0),
                A.augmentations.transforms.PadIfNeeded(min_height=self.final_resolution, min_width=self.final_resolution, border_mode=cv2.BORDER_CONSTANT, value=[0., 0., 0.]),
                # TODO implement continuous rotations from within [0;360] at some point, maybe change the transform order for that
                # in that case watch out for cases when image extends the image borders by a few pixels after rotation
                # A.augmentations.geometric.rotate.Rotate(limit=45, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),

                # # HIGH RISKS - HIGH PROBABILITY OF KEYPOINTS GOING OUT
                # A.OneOf([  # perspective or shear
                #     A.augmentations.geometric.transforms.Perspective(
                #         scale=0.05, pad_mode=cv2.BORDER_CONSTANT, p=1.0),
                #     A.augmentations.geometric.transforms.Affine(
                #         shear=(-10, 10), mode=cv2.BORDER_CONSTANT, p=1.0),  # shear
                # ], p=0.2),

                # A.OneOf([  # translate
                #     A.augmentations.geometric.transforms.ShiftScaleRotate(
                #         shift_limit_x=0.01, shift_limit_y=0, scale_limit=0,
                #         rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
                #         p=1.0),  # x translations
                #     A.augmentations.geometric.transforms.ShiftScaleRotate(
                #         shift_limit_x=0, shift_limit_y=0.01, scale_limit=0,
                #         rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
                #         p=1.0),  # y translations
                #     A.augmentations.geometric.transforms.Affine(
                #         translate_percent=(0, 0.01),
                #         mode=cv2.BORDER_CONSTANT, p=1.0),  # random xy translate
                # ], p=0.2),
                # # random rotation
                # A.augmentations.geometric.rotate.Rotate(
                #     limit=10, border_mode=cv2.BORDER_CONSTANT,
                #     p=0.4),
            ],
                # additional_targets={'traj_map': 'image'},
            )
        else:
            transform = A.Compose([
                # A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
                A.augmentations.transforms.PadIfNeeded(min_height=self.final_resolution, min_width=self.final_resolution, border_mode=cv2.BORDER_CONSTANT, value=[0., 0., 0.]),
            ],
                # additional_targets={'traj_map': 'image'},
            )

        if traj_aug_mode == 0:
            transform.add_targets({'traj_map': 'image'})
            transformed = transform(image=image, traj_map=traj_image)
            traj_image = transformed['traj_map']
        elif traj_aug_mode == 1:
            transformed = transform(image=image, mask=traj_image)
            traj_image = transformed['mask']
        else:
            transformed = transform(image=image)

        #################### TEST START ####################
         # 2D plot pred
        # plt.imshow(transformed['image'])
        # non_zeros = np.argwhere(transformed['traj_map'] != 0.)
        # img_now = transformed['image'].copy()
        # maxy = np.max(transformed['traj_map'])
        # pred_colors_from_timestamps = [get_color_from_array(transformed['traj_map'][x, y], np.max(transformed['traj_map']))/255. for x, y in non_zeros]
        # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array(pred_colors_from_timestamps)
        # # img_now[non_zeros[:,0], non_zeros[:,1]] = np.array([0, 0, 1.])
        # plt.imshow(img_now)
        
        # 3D plot pred
        # fig = plt.figure(figsize=(6,6))
        # ax = fig.add_subplot(111, projection='3d')
        # X,Y = np.meshgrid(np.arange(transformed['traj_map'].shape[1]), np.arange(transformed['traj_map'].shape[0]))
        # ax.plot_surface(X, Y, transformed['traj_map'])
        # plt.show()
        #################### TEST END ####################
        
        # FROM NUMPY BACK TO TENSOR
        image = torch.tensor(transformed['image']).permute(2, 0, 1)
        traj_image = torch.tensor(transformed['mask']).permute(2, 0, 1).long() if 'mask' in transformed else None
        traj_image = torch.tensor(transformed['traj_map']).float() if 'traj_map' in transformed else traj_image

        # assert image.size(1) == self.final_resolution and image.size(2) == self.final_resolution
        # assert traj_image.size(1) == self.final_resolution and traj_image.size(0) == self.final_resolution

        # NEW AUGMENTATION: INVERT TIME
        # if random.random() > 0.5:
        #     abs_pixel_coord = abs_pixel_coord.flip(dims=(0,))
        #     input_traj_maps = input_traj_maps.flip(dims=(1,))

        # Return torch.tensor.float32
        return image, traj_image


    def augment_traj_and_images4density(self, image, traj_image, traj_aug_mode):

        # flipping, transposing, random 90 deg rotations
        transform = A.Compose([
            A.augmentations.transforms.HorizontalFlip(p=0.5),
            A.augmentations.transforms.VerticalFlip(p=0.5),
            A.augmentations.transforms.Transpose(p=0.5),
            A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        ])
        if traj_aug_mode == 0:
            transform.add_targets({'traj_map': 'image'})
            transformed = transform(image=image, traj_map=traj_image)
            traj_image = transformed['traj_map']
        elif traj_aug_mode == 1:
            transformed = transform(image=image, mask=traj_image)
            traj_image = transformed['mask']
        else:
            transformed = transform(image=image)

        image = torch.tensor(transformed['image']).permute(2, 0, 1).float()
        traj_image = torch.tensor(transformed['mask']).permute(2, 0, 1).long() if 'mask' in transformed else None
        traj_image = torch.tensor(transformed['traj_map']).float() if 'traj_map' in transformed else traj_image

        return image, traj_image


    def extract_distances(self, info_dict):

        origin_list = [[area['x_start'], area['x_end'], area['y_start'], area['y_end']] for area in list(info_dict['origins'].values())]
        dst_list = [[area['x_start'], area['x_end'], area['y_start'], area['y_end']] for area in list(info_dict['destinations'].values())]

        # distances
        origin_list = [((area[0]+area[1])/2., (area[2]+area[3])/2.) for area in origin_list] # calc center points
        dst_list = [((area[0]+area[1])/2., (area[2]+area[3])/2.) for area in dst_list] # calc center points
        # calc center points
        or_dst_pairs = np.array(list(product(origin_list, dst_list)))
        distances = (or_dst_pairs[:, 0, 0]-or_dst_pairs[:, 1, 0])**2 + (or_dst_pairs[:, 0, 1]-or_dst_pairs[:, 1, 1])**2
        distances = -np.sort(-np.sqrt(distances)) # sort in descending order

        distances = distances[:16] if len(distances) >= 16 else np.append(distances, (16-len(distances)) * [0])

        return distances


    def load_additional_info(self, add_data_path):
        
        import os
        import re
        assert os.path.isdir(add_data_path)
        
        for layout_dir in os.listdir(add_data_path):
            for floorplan_dir in os.listdir(os.path.join(add_data_path, layout_dir)):
                temp_dir ={}
                self.data_dict[SEP.join([layout_dir, floorplan_dir])] = {}
                floorplan_data_path = os.path.join(add_data_path, layout_dir, floorplan_dir, 'data.txt')
                assert os.path.isfile(floorplan_data_path)
                lines = open(floorplan_data_path, 'r').readlines()
                for idx, line in enumerate(lines):
                    if line.startswith('site'):
                        self.data_dict[SEP.join([layout_dir, floorplan_dir])][line.split(":")[0]] = float(line.split(":")[1])
                        """.update({
                            SEP.join([layout_dir, floorplan_dir]): {
                                line.split(":")[0]: float(line.split(":")[1])
                                }
                            })"""
                    elif line.startswith('variation'):
                        data_line = lines[idx+1]
                        or_data_line = data_line.split(':')[1].split('#')[0].strip()
                        dst_data_line = data_line.split(':')[2].strip()

                        origins = re.findall(r'\[.*?\]', or_data_line)
                        destinations = re.findall(r'\[.*?\]', dst_data_line)

                        if len(origins) > self.max_ors:
                            self.max_ors = len(origins)
                        if len(destinations) > self.max_dsts:
                            self.max_dsts = len(destinations) 


                        def extract_area_points(areas, a_type):
                            variation_dict = {}
                            for id_o, area in enumerate(areas):
                                xs = area.split('=(')[1]
                                x_start = float(xs.split(',')[0])
                                x_end = float(xs.split(',')[1].split(')')[0])
                                ys = area.split('=(')[2]
                                y_start = float(ys.split(',')[0])
                                y_end = float(ys.split(',')[1].split(')')[0])
                                
                                variation_dict.update({
                                    f'{a_type}_{id_o}': {
                                        'x_start': x_start,
                                        'x_end': x_end,
                                        'y_start': y_start,
                                        'y_end': y_end 
                                    }
                                })
                            return variation_dict
                        
                        extracted_origins = extract_area_points(origins, 'origin')
                        extracted_destinations = extract_area_points(destinations, 'dst')

                        temp_dir.update(
                            {
                                line.split(':')[0]: {
                                'origins': extracted_origins,
                                'destinations': extracted_destinations
                                }
                            })

                    else:
                        continue

                self.data_dict[SEP.join([layout_dir, floorplan_dir])].update({'data': temp_dir})