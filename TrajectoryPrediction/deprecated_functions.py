""" in file FloorplanDataset.py """
# from Modules.goal.models.goal.utils import create_CNN_inputs_loop

# class Dataset_Seq2Seq(Dataset):

#     def augment_traj_and_create_traj_maps(self, batch_data, np_image, augmentation):

#         image = np_image
#         abs_pixel_coord = batch_data["abs_pixel_coord"]
#         # input_traj_maps = batch_data["input_traj_maps"]
#         site_x = batch_data['scene_data']['floorplan_max_x']
#         site_y = batch_data['scene_data']['floorplan_max_y']

#         scale_x = site_x / self.max_floorplan_meters
#         scale_y = site_y / self.max_floorplan_meters

#         assert 0.0 <= scale_x <= 1.0 and 0.0 <= scale_y <= 1.0

#         scaled_resolution_x = int(self.final_resolution * scale_x)
#         scaled_resolution_y = int(self.final_resolution * scale_y)

#         # get old channels for safety checking
#         old_H, old_W, C = np_image.shape

#         # keypoints to list of tuples
#         # need to clamp because some slightly exit from the image
#         # abs_pixel_coord[:, 0] = np.clip(abs_pixel_coord[:, 0],
#         #                                    a_min=0, a_max=old_W - 1e-3)
#         # abs_pixel_coord[:, 1] = np.clip(abs_pixel_coord[:, 1],
#         #                                    a_min=0, a_max=old_H - 1e-3)
#         if batch_data['type'] == 'pos':
#             # Check whether some keypoints are outside the image
#             x_coord_big = np.argwhere(abs_pixel_coord[:, :, 0] > old_W)
#             y_coord_big = np.argwhere(abs_pixel_coord[:, :, 1] > old_H)
#             x_coord_small = np.argwhere(abs_pixel_coord[:, :, 0] < 0)
#             y_coord_small = np.argwhere(abs_pixel_coord[:, :, 1] < 0)

#             assert x_coord_big.shape[0] == y_coord_big.shape[0] == x_coord_small.shape[0] == y_coord_small.shape[0] == 0, \
#                 f'Some traj points not within image, outside shapes: {x_coord_big.shape[0]}, {y_coord_big.shape[0]}, {x_coord_small.shape[0]} and {y_coord_small.shape[0]}'

#         keypoints = list(map(tuple, abs_pixel_coord.reshape(-1, 2)))

#         # Resize first to create Gaussian maps later
#         transform = A.Compose([
#             A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
#         ],
#             keypoint_params=A.KeypointParams(format='xy',
#                                             remove_invisible=False))
        
#         transformed = transform(image=image, keypoints=keypoints)
#         image = transformed['image']
#         keypoints = np.array(transformed['keypoints'])
        
#         if self.arch == 'goal':
#             additional_targets = {'traj_map': 'image'}
#             input_traj_maps = create_CNN_inputs_loop(
#                 batch_abs_pixel_coords=torch.tensor(keypoints.reshape(self.seq_length, -1, 2)).float(),
#                 tensor_image=F.to_tensor(image))
#             bs, T, old_H, old_W = input_traj_maps.shape
#             input_traj_maps = input_traj_maps.view(bs * T, old_H, old_W).\
#                 permute(1, 2, 0).numpy().astype('float32')
#         else:
#             additional_targets = {}

#         if augmentation:
            

#             transform = A.Compose([
#                 # SAFE AUGS, flips and 90rots
#                 # A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
#                 A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
#                 A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
#                 A.augmentations.geometric.transforms.Transpose(p=0.5),
#                 A.augmentations.geometric.rotate.RandomRotate90(p=1.0),
#                 A.augmentations.geometric.transforms.PadIfNeeded(min_height=self.final_resolution, min_width=self.final_resolution, border_mode=cv2.BORDER_CONSTANT, value=[0., 0., 0.]),
#                 # TODO implement continuous rotations from within [0;360] at some point, maybe change the transform order for that
#                 # in that case watch out for cases when image extends the image borders by a few pixels after rotation
#                 A.augmentations.geometric.rotate.Rotate(limit=45, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),

#                 # # HIGH RISKS - HIGH PROBABILITY OF KEYPOINTS GOING OUT
#                 # A.OneOf([  # perspective or shear
#                 #     A.augmentations.geometric.transforms.Perspective(
#                 #         scale=0.05, pad_mode=cv2.BORDER_CONSTANT, p=1.0),
#                 #     A.augmentations.geometric.transforms.Affine(
#                 #         shear=(-10, 10), mode=cv2.BORDER_CONSTANT, p=1.0),  # shear
#                 # ], p=0.2),

#                 # A.OneOf([  # translate
#                 #     A.augmentations.geometric.transforms.ShiftScaleRotate(
#                 #         shift_limit_x=0.01, shift_limit_y=0, scale_limit=0,
#                 #         rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
#                 #         p=1.0),  # x translations
#                 #     A.augmentations.geometric.transforms.ShiftScaleRotate(
#                 #         shift_limit_x=0, shift_limit_y=0.01, scale_limit=0,
#                 #         rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
#                 #         p=1.0),  # y translations
#                 #     A.augmentations.geometric.transforms.Affine(
#                 #         translate_percent=(0, 0.01),
#                 #         mode=cv2.BORDER_CONSTANT, p=1.0),  # random xy translate
#                 # ], p=0.2),
#                 # # random rotation
#                 # A.augmentations.geometric.rotate.Rotate(
#                 #     limit=10, border_mode=cv2.BORDER_CONSTANT,
#                 #     p=0.4),
#             ],
#                 keypoint_params=A.KeypointParams(format='xy',
#                                                 remove_invisible=False),
#                 additional_targets=additional_targets,
#             )
#         else:
#             transform = A.Compose([
#                 # A.augmentations.geometric.resize.Resize(scaled_resolution_y, scaled_resolution_x, interpolation=cv2.INTER_AREA),
#                 A.augmentations.transforms.PadIfNeeded(min_height=self.final_resolution, min_width=self.final_resolution, border_mode=cv2.BORDER_CONSTANT, value=[0., 0., 0.]),
#             ],
#                 keypoint_params=A.KeypointParams(format='xy',
#                                                 remove_invisible=False),
#                 additional_targets=additional_targets,
#             )

#         #################### TEST START ####################
#         # plt.imshow(image/255.) # weg
#         # img_np = image#/255.
#         # traj = np.array(keypoints)
#         # # traj = [t for t in traj]
#         # trajs = [traj[x:x+20] for x in range(0, len(traj), 20)]
#         # # for t_id in traj:
#         # for traj in trajs:
#         #     old_point = None
#         #     r_val = random.uniform(0.4, 1.0)
#         #     b_val = random.uniform(0.7, 1.0)
#         #     for point in traj:
#         #         x_n, y_n = round(point[0]), round(point[1])
#         #         assert 0 <= x_n <= img_np.shape[1] and 0 <= y_n <= img_np.shape[0], f'{x_n} > {img_np.shape[1]} or {y_n} > {img_np.shape[0]}'
#         #         if old_point != None:
#         #             # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
#         #             c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, y_n)))]
#         #             for c in c_line:
#         #                 img_np[c[1], c[0]] = np.array([r_val, 0., b_val])
#         #             # plt.imshow(img_np)
#         #         old_point = (x_n, y_n)

#         # plt.imshow(img_np)
#         # plt.close('all')
#         #################### TEST END ####################

#         if self.arch == 'goal':
#             transformed = transform(
#                 image=image, keypoints=keypoints, traj_map=input_traj_maps)
#         else:
#             transformed = transform(
#                 image=image, keypoints=keypoints)
#         # #################### TEST START ####################
#         # plt.imshow(image/255.) # weg
#         # img_np = transformed['image']#/255.

#         # traj = np.array(transformed['keypoints'])
#         # trajs = [traj[x:x+20] for x in range(0, len(traj), 20)]
#         # for t_id in traj:
#         # for traj in trajs:
#         #     old_point = None
#         #     r_val = random.uniform(0.4, 1.0)
#         #     b_val = random.uniform(0.7, 1.0)
#         #     for point in traj:
#         #         x_n, y_n = round(point[0]), round(point[1])
#         #         assert 0 <= x_n <= img_np.shape[1] and 0 <= y_n <= img_np.shape[0]
#         #         if old_point != None:
#         #             # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
#         #             c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, y_n)))]
#         #             for c in c_line:
#         #                 img_np[c[1], c[0]] = np.array([r_val, 0., b_val])
#         #             # plt.imshow(img_np)

#         #     old_point = (x_n, y_n)
#         # # cv2.imshow('namey', img_np)
#         # plt.imshow(img_np)
#         # plt.close('all')
#         # #################### TEST END ####################
#         # FROM NUMPY BACK TO TENSOR
#         image = torch.tensor(transformed['image']).permute(2, 0, 1)
#         C, new_H, new_W = image.shape
#         abs_pixel_coord = torch.tensor(transformed['keypoints']).view(batch_data["abs_pixel_coord"].shape)
#         if self.arch == 'goal':
#             input_traj_maps = torch.tensor(transformed['traj_map']).permute(2, 0, 1).view(bs, T, new_H, new_W)
#             batch_data["input_traj_maps"] = input_traj_maps.float()

#         if batch_data['type'] == 'pos':
#             # Check whether some keypoints are outside the image
#             x_coord_big = torch.argwhere(abs_pixel_coord[:, :, 0] > new_W)
#             y_coord_big = torch.argwhere(abs_pixel_coord[:, :, 1] > new_H)
#             x_coord_small = torch.argwhere(abs_pixel_coord[:, :, 0] < 0)
#             y_coord_small = torch.argwhere(abs_pixel_coord[:, :, 1] < 0)

#             if not (x_coord_big.size(0) == y_coord_big.size(0) == x_coord_small.size(0) == y_coord_small.size(0) == 0):
#                 print('After rotation, some trajectories dont lie within output image. Output shapes: ' + \
#                     f'{x_coord_big.size(0)}, {y_coord_big.size(0)}, {x_coord_small.size(0)} and { y_coord_small.size(0)}')

#             # Clamping of trajectory values if they exceed the image boundaries (very rare)
#             abs_pixel_coord[:, :, 0] = torch.clamp(abs_pixel_coord[:, :, 0], min=0, max=new_W)
#             abs_pixel_coord[:, :, 1] = torch.clamp(abs_pixel_coord[:, :, 1], min=0, max=new_H)

#         assert new_W == self.final_resolution and new_H == self.final_resolution

#         # NEW AUGMENTATION: INVERT TIME
#         # if random.random() > 0.5:
#         #     abs_pixel_coord = abs_pixel_coord.flip(dims=(0,))
#         #     input_traj_maps = input_traj_maps.flip(dims=(1,))

#         # To torch.tensor.float32
#         batch_data["tensor_image"] = image.float()
#         batch_data["abs_pixel_coord"] = abs_pixel_coord.float()
#         batch_data["seq_list"] = torch.tensor(batch_data["seq_list"]).float()

#         return batch_data