from typing import Optional, Sequence, Union
from torch.utils.data import Dataset
import os
import numpy as np
import random
from helper import SEP
import matplotlib.pyplot as plt
import albumentations as A
import torch
import cv2
from PIL import Image
from helper import xywh2xyxy, xyxy2xywh, xyxy2xywhn, xyxy2xyxyn
from albumentations.augmentations.crops import functional as F
from albumentations.core.bbox_utils import union_of_bboxes
from itertools import combinations

class ObjDetDataset(Dataset):
    def __init__(
        self, 
        config: dict,
        img_paths: list, 
        bboxes: list, 
        batch_size: int,
        transform = None,
    ):
        # TODO maybe turn off transformations when eval/test
        self.transform = transform
        self.img_paths = img_paths
        # self.target_paths = target_paths
        self.bboxes = bboxes
        # self.max_floorplan_meters = 64
        # self.final_resolution = 640
        self.p_dim_tf = [0.0, 0.0, 0.0, 0.0]
        self.arch = config['arch']
        self.img_max_width, self.img_max_height = config['img_max_size']
        self.augment_batch_level = config['augment_batch_level']
        self.augment_brightness = True if config['additional_queries'][0] == 'vanilla_imgAugm' else False
        self.color_mappings = {
            0: {(255, 255, 0): (100, 100, 0), (255, 0, 0): (100, 0, 0)},
            1: {(255, 255, 0): (175, 175, 0), (255, 0, 0): (175, 0, 0)}
        } if self.augment_brightness else None

        self.batch_size = batch_size
        self.agentIds = {'10': 0, '30': 1, '50': 2}
        self.ascentIds = {'En_Sn': 0, 'E1_S0': 1, 'E2_S0': 2, 'E3_S0': 3, 'E1_S2.4': 4, 'E2_S2.4': 5, 'E3_S2.4': 6} # (E=1/2/3 + S=0/2.4 + None) _E2_S2.4
        self.obstaclesIds = {'C0': 0, 'C2': 1}

        assert len(self.bboxes) == len(self.img_paths), 'Length of image paths and trajectory paths do not match, something went wrong!'

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        bbox_format = 'pascal_voc' # 'coco', 'pascal_voc'
        img_path, boxes_per_image = self.img_paths[idx], self.bboxes[idx]
        ## DELETE FROM HERE ...
        # figures_list = ['T2_E1_S0_O0_A20_C0', 'T4_E2_S1.8_O0_A20_C2', 'T6_E1_S0_O1_A20_C2']
        # lll = [l.split(SEP)[-1].replace('.png', '') for l in self.img_paths]
        # index = None
        # for fig in figures_list:
        #     try: 
        #         index = lll.index(fig)
        #     except ValueError: 
        #         continue 
        # if index is not None:
        #     img_path, boxes_per_image = self.img_paths[index], self.bboxes[index]
        ## ... TO HERE
        index = None
        a, e, s, es, ss, c = boxes_per_image[-1]
        agentId = self.agentIds[a]
        # info = img_path.split(os.sep)[-1].replace('.png', '').split('_')
        # e, s, es, ss, c = info[4][1:], info[5][1:], info[8][2:], info[9][2:], info[2]
        central_id, sides_id, obstacles_id = self.ascentIds[f'E{e}_S{s}'], self.ascentIds[f'E{es}_S{ss}'], self.obstaclesIds[f'C{c}']
        boxes_per_image = boxes_per_image[:-1]

        labels = np.zeros((len(boxes_per_image)), dtype=np.int64) if self.arch not in ['FasterRCNN', 'FasterRCNN_custom'] else np.ones((len(boxes_per_image)), dtype=np.int64)

        img = np.array(Image.open(img_path))[...,:3]
        if self.augment_brightness and agentId != 2:
            # Create masks for the colors to be replaced
            masks = [np.all(img == color, axis=-1) for color in self.color_mappings[agentId]]
            # Apply the color mappings using masks
            for color, mask in zip(self.color_mappings[agentId].values(), masks):
                img[mask] = color
            # plt.imshow(img)

        img = img.astype(np.float32) / 255.
       
        if self.augment_batch_level:
            if self.arch in ['EfficientDet', 'YoloV5']: raise NotImplementedError
            return img, labels, boxes_per_image, torch.tensor([agentId, central_id, sides_id, obstacles_id], dtype=torch.long)
        # implement reading the bbox
        # bboxes = np.clip(bboxes, 0, 1e4)

        # apply (4) image transformation once per batch
        # if idx % self.batch_size == 0:
        #     self.p_dim_tf = [random.randint(0, 1) for _ in range(4)]
        #     print(f'Batch id: {idx}', self.p_dim_tf)

        temp_img_save = img_path.split(os.sep)[-1] # .replace('_A10', f'_A{num_agents}') if '_A10' in img_path else img_path.replace('.jpg', f'_A{num_agents}.jpg')
        img, bboxes, labels = self.augmentations(img, boxes_per_image, labels, p_dim_tf=self.p_dim_tf, format=bbox_format, file=temp_img_save, index=index) #None) #
        if len(bboxes) > 0:
            bboxes = np.array([(round(box[0], 3), round(box[1], 3), round(box[2], 3), round(box[3], 3)) for box in bboxes])
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0,4), dtype=np.float32)
            labels = np.zeros((0), dtype=np.int64)

        # bboxes = [(box[0]/self.img_max_height, box[1]/self.img_max_width, box[2]/self.img_max_height, box[3]/self.img_max_width) for box in bboxes]

        # augment_hsv() in the yoloV5 project?
        # xyxy 2 xywh
        # bboxes = xyxy2xywhn(bboxes, self.img_max_height, self.img_max_width)

        # include non-empty tensors in case of bbox absincenc
        # if bboxes.nelement()==0:
        #     bboxes = torch.zeros((0,4),dtype=torch.float32)
        #     # bboxes = torch.FloatTensor([[0.,0.,1.,1.]])
        #     labels = torch.zeros((0),dtype=torch.int64) # torch.LongTensor([0])
        #     raise NotImplementedError('check zero-box option for EfficientDet and Yolo')
        # TODO: YoloV5 uses array([], shape=(0, 5), dtype=float32), first entry is the class
        if self.arch in ['Detr', 'Detr_custom']:
            if bboxes.shape[0] != 0:
                assert (bboxes[:, 2:] >= bboxes[:, :2]).all()
            bboxes = xyxy2xywhn(bboxes, self.img_max_width, self.img_max_height)
            if bboxes.shape[0] > 0: assert 1.0 >= np.max(bboxes)
        
        elif self.arch == 'EfficientDet':
            img = img.transpose((2, 0, 1)).astype(np.float32) / 255.
            # bboxes = xyxy2xyxyn(bboxes, self.img_max_height, self.img_max_width)
            target = dict(bbox=bboxes, cls=np.ones_like(labels, dtype=np.int64))

            return img, target, torch.tensor([agentId, central_id, sides_id, obstacles_id], dtype=torch.long)
        elif self.arch == 'YoloV5':
            bboxes = xyxy2xywhn(bboxes, self.img_max_width, self.img_max_height)
            assert 1.0 >= np.max(bboxes)

        img = torch.tensor(img).permute(2, 0, 1).float()
        bboxes = torch.tensor(bboxes).float()#.unsqueeze(0)
        labels = torch.tensor(labels).long()

        # if format == 'pascal_voc' and bboxes.nelement() != 0 and self.arch=='Detr':
        #     # from pascal to coco (required by DETR)
        #     assert (bboxes[:, 0] <= bboxes[:, 2]).all(), 'x_s dont fit'
        #     assert (bboxes[:, 1] <= bboxes[:, 3]).all(), 'y_s dont fit'
        #     x_c = (bboxes[:, 0] + bboxes[:, 2]) / 2.
        #     y_c = (bboxes[:, 1] + bboxes[:, 3]) / 2.
        #     w = bboxes[:, 2] - bboxes[:, 0]
        #     h = bboxes[:, 3] - bboxes[:, 1]

        #     bboxes = torch.stack([x_c, y_c, w, h], dim=-1)
        #     bboxes = xyxy2xywhn(bboxes, self.img_max_height, self.img_max_width)

        if self.transform:
            img = self.transform(img)
        
        return img, labels, bboxes, torch.tensor([agentId, central_id, sides_id, obstacles_id], dtype=torch.long)


    def augmentations(self, image, bboxes, labels, p_dim_tf, format='pascal_voc', file=None, index=None):
        tf_probs = [0.0, 1.]
        p_trans = 1.0
        assert image.shape[1] <= 4470 and image.shape[0] <= 1840
        # pad_square_size = 4470 # 4224 # goes for LENGTH_IN_METER = 140.2
        pad_square_size_width = 4470 + 200 # 4224 # goes for LENGTH_IN_METER = 140.2
        pad_square_size_height = 1840 + 100 # 4224 # goes for LENGTH_IN_METER = 140.2
        # pad_square_size_big = pad_square_size * 212.48 / 150.2 # 140.2
        resized_img = 2000 * 212.48 / 150.2 # 140.2
        max_long_size = 1500
        if '2SBSS' in file or 'U9_colored' in file or 'saved_from_xml' in file:
            pad_square_size = 2000 + 90 # keeping the padding ration to normal input...
            tf_probs = [1., 0.] # [0., 1.]
            p_trans = 0.0
            max_long_size = 1536 if 'saved_from_xml' in file else 1500
            # if 'old': # img_size: (1486, 408)
            #     pad_square_size_width = 1486
            # elif 'new': # img_size: (1604, 403)

        # if '2SBSS' in file: pad_square_size = pad_square_size * 212.48 / 140.2
        # elif 'U9_colored' in file: pad_square_size = pad_square_size * 199.206 / 140.2
        # width_p = image.shape[1] 

        # meter_per_pixel_para = 0.033607137707905174
        # meter_per_pixel_u9mo = 199.206 / 2000 = 0.099603
        # meter_per_pixel_sbss = 212.48 / 2000 = 0.10624

        # flipping, transposing, random 90 deg rotations
        # transform = A.Compose([
        #     # A.PadIfNeeded(pad_square_size, pad_square_size, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True), # (4224, 4224) (561, 4174) # 1344
        #     A.PadIfNeeded(1024, 1024, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True), # (4224, 4224) (561, 4174) # 1344
        #     # A.SmallestMaxSize(max_size=self.img_max_width),
        #     # A.Resize(self.img_max_width, self.img_max_height),
        #     A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
        #     A.augmentations.geometric.transforms.VerticalFlip(p=0.5),

        #     # A.RandomResizedCrop(height=1024, width=1024, p=1.0), #, scale=(0.8, 1.0), ratio=(0.9, 1.11)) # from YoloV5
        #     # RandomResizePad(target_size=img_size, interpolation=interpolation, fill_color=fill_color) from EfficientDet
        #     # A.augmentations.geometric.transforms.Transpose(p=p_dim_tf[2]),
        #     A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        #     FixedSizedBBoxSafeCrop(height=self.img_max_height, width=self.img_max_width, p=1.0),
        # ], bbox_params=A.BboxParams(format=format, label_fields=['category_ids']))
        
        # transform_pad_square = A.Compose([
        #     A.PadIfNeeded(pad_square_size, pad_square_size, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True), # (4224, 4224) (561, 4174) # 1344
        #     A.Resize(self.img_max_width, self.img_max_height),
        #     # A.augmentations.geometric.transforms.Affine(translate_px=(0, self.img_max_width//2), p=p_trans),
        #     A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
        #     A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
        #     A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        # ], bbox_params=A.BboxParams(format=format, label_fields=['category_ids']))
       
        # transform_cropping = A.Compose([
        #     A.PadIfNeeded(self.img_max_height, self.img_max_width, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True), # (4224, 4224) (561, 4174) # 1344
        #     A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
        #     A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
        #     A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        #     FixedSizedBBoxSafeCrop(height=self.img_max_height, width=self.img_max_width, p=1.0),
        # ], bbox_params=A.BboxParams(format=format, label_fields=['category_ids']))

        transform_rectangular = A.Compose([
            A.LongestMaxSize(max_size=max_long_size, always_apply=True),
            A.PadIfNeeded(self.img_max_height, self.img_max_width, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True), # (4224, 4224) (561, 4174) # 1344
            A.augmentations.geometric.transforms.Affine(translate_px={'x': (0, 50), 'y': (0, 50)}, p=p_trans),
            A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
            A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
            # A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        ], bbox_params=A.BboxParams(format=format, label_fields=['category_ids']))

        # transform = A.OneOf([
        #     transform_pad_square,
        #     transform_cropping,
        # ], p=1.0)
        # transform.transforms_ps = tf_probs

        transformed = transform_rectangular(image=image, bboxes=bboxes, category_ids=labels)        

        if index is not None: visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], {0: 'Critical area'}, file=file)

        return transformed['image'], transformed['bboxes'], transformed['category_ids']
    
    
def visualize(image, bboxes, category_ids, category_id_to_name, format='pascal_voc',file=None):

    def visualize_bbox(img, bbox, class_name, color=(0, 175, 0), thickness=2, format=format):
        
        img = np.ascontiguousarray(img) 
        if format=='coco':
            x_min, y_min, w, h = bbox
            x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        elif format=='pascal_voc':
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (0, 175, 0), -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35, 
            color=(255, 255, 255), 
            lineType=cv2.LINE_AA,
        )
        # img_cp = img_cp.astype(np.float32) / 255.
        return img

    img = (image.copy()*255).astype(np.uint8)
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)

    # plt.figure(figsize=(12, 12))
    if file is not None:
        plt.imsave(f'test_imgs\\{file.split(SEP)[-1]}', img)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.close('all')


def augment_on_batch_level(batch, config, img_transform):
    arch = config['arch']
    images, labels, bboxes, numAgentsIds = zip(*batch)
    max_height, max_width = np.max([img.shape[:2] for img in images], axis=0)

    # padding transformation
    images_pad, bboxes_pad, labels_pad = [], [], []
    transform_pad = A.Compose([
        A.PadIfNeeded(max_height, max_width, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    
    for img, bbox, label in zip(images, bboxes, labels):
        transformed_pad = transform_pad(image=img, bboxes=bbox, category_ids=label)
        images_pad.append(transformed_pad['image'])
        bboxes_pad.append(transformed_pad['bboxes'])
        labels_pad.append(transformed_pad['category_ids'])
    
    # visualize the paddings
    # for idx, (img_v, bbox_v, label_v) in enumerate(zip(images_pad, bboxes_pad, labels_pad)):
    #     visualize((img_v*255).astype(np.uint8), bbox_v, label_v,  {0: 'Critical area', 1: 'Critical area'}, file=SEP+f'testy_pad_{idx}.png')
    
    # geometric transformations
    images_tf, bboxes_tf, labels_tf = [], [], []
    tf_probs = [random.randint(0, 1) for _ in range(4)]
    rot_dir = random.choice([1, 3]) # +90==left, -90==right
    transform = A.Compose([
        A.augmentations.geometric.transforms.HorizontalFlip(p=tf_probs[0]),
        A.augmentations.geometric.transforms.VerticalFlip(p=tf_probs[1]),
        A.augmentations.geometric.transforms.Transpose(p=tf_probs[2]),
        # A.augmentations.geometric.rotate.Rotate(limit=(rot_dir * 90, rot_dir * 90), p=1),
        # A.augmentations.geometric.rotate.RandomRotate90(p=1),
        CustomRotate90(p=tf_probs[3], factor=rot_dir),
        # A.Rotate(limit=(rot_dir * 90, rot_dir * 90), p=1), # Rotate left
        A.LongestMaxSize(max_size=1024, p=1)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    for img_tf, bbox_tf, label_tf in zip(images_pad, bboxes_pad, labels_pad):    
        transformed = transform(image=img_tf, bboxes=bbox_tf, category_ids=label_tf) 
        images_tf.append(transformed['image'])
        bboxes_tf.append(transformed['bboxes'])
        labels_tf.append(transformed['category_ids'])
    
    # visualizes all transformations
    # for idx, (img_v, bbox_v, label_v) in enumerate(zip(images_tf, bboxes_tf, labels_tf)):
    #     visualize((img_v*255).astype(np.uint8), bbox_v, label_v,  {0: 'Critical area', 1: 'Critical area'}, file=SEP+f'testy_tf_{idx}.png')

    # formatting to list of arrays
    boxes_list = []
    labels_list = []
    for boxes_per_img, labels_per_img in zip(bboxes_tf, labels_tf):
        if len(boxes_per_img) == 0:
            boxes_list.append(torch.zeros((0,4)).float())
            labels_list.append(torch.zeros((0)).long())
        else:
            if arch in ['Detr', 'Detr_custom']:
                max_2 = np.max(np.array(boxes_per_img)[:,2]) 
                max_3 = np.max(np.array(boxes_per_img)[:,3])
                assert (images_tf[0].shape[1] > max_2) and (images_tf[0].shape[0] > max_3)
                boxes_list.append(xyxy2xywhn(torch.tensor(boxes_per_img).float(), images_tf[0].shape[1], images_tf[0].shape[0]))
            else:
                boxes_list.append(torch.tensor(boxes_per_img).float())
                # concat = []
                # for box, label in zip(boxes_per_img, labels_per_img):
                #     concat.append(np.array([round(box[0], 3), round(box[1], 3), round(box[2], 3), round(box[3], 3)]))
                # boxes_list.append(np.stack(concat, axis=0))
            labels_list.append(torch.tensor(labels_per_img).long())
    
    images_tf = torch.tensor(np.stack(images_tf, axis=0)).permute(0, 3, 1, 2)
    # print(f'\nImage shape: {images_tf.size()}')

    if img_transform:
        images_tf = img_transform(images_tf)

    return images_tf, labels_list, boxes_list, torch.LongTensor(numAgentsIds)


class CustomRotate90(A.augmentations.geometric.rotate.RandomRotate90):
    def __init__(self, always_apply: bool = False, p: float = 0.5, factor: int = 0):
        super().__init__(always_apply, p)
        self.factor = factor
    
    def apply(self, img, factor=0, **params):
        return super().apply(img, self.factor, **params)


class FixedSizedBBoxSafeCrop(A.BBoxSafeRandomCrop):
    def __init__(self, height, width, erosion_rate=0, always_apply=False, p=1):
        super().__init__(erosion_rate, always_apply, p)

        self.height = height
        self.width = width

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]

        crop_height = self.height
        crop_width = self.width
        
        valid_crop_found = False
        attempts = 0
        if len(params["bboxes"]) > 0:
            bboxes = np.array([box[:4] for box in params["bboxes"]])
            bboxes[:, 0] *= img_w
            bboxes[:, 1] *= img_h
            bboxes[:, 2] *= img_w
            bboxes[:, 3] *= img_h
        else:

            h_start = 0. if img_h==crop_height else random.random()
            w_start = 0. if img_w==crop_width else random.random()
            assert h_start < 1.0 and w_start < 1.0
            return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}
        
        while not valid_crop_found:
            attempts += 1

            bx, by = random.random(), random.random()
            bx2, by2 = bx + crop_width / img_w, by + crop_height / img_h
            bw, bh = bx2 - bx, by2 - by

            h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0 - 1e-5)
            w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0 - 1e-5)

            y1 = int((img_h - crop_height + 1) * h_start)
            y2 = y1 + crop_height
            x1 = int((img_w - crop_width + 1) * w_start)
            x2 = x1 + crop_width

            crop_pix = [x1, y1, x2, y2]

            valid_crop_found, num_within_boxes = is_crop_valid(crop_pix, bboxes)

            if attempts > 99: print(f'Cropping attempt nr. {attempts} attempts, {num_within_boxes} included boxes inside crop')
        
        #### ideal code
        # bboxes = [box[:4] for box in params["bboxes"]]
        # bboxes_np = np.array(bboxes)
        # all_box_combinations = []

        # for i in range(1, len(params["bboxes"])+1):
        #     box_combinations_given_num = list(combinations(bboxes, i))
        #     box_combinations_given_num = np.array(box_combinations_given_num)
        #     unions_from_combinations = compute_union_boxes(bboxes=box_combinations_given_num)
        #     union_boxes_validation_mask = valid_boxes(unions_from_combinations, bboxes_np, image_width=img_w, image_height=img_h, crop_size=self.width)
        #     valid_union_boxes = unions_from_combinations[union_boxes_validation_mask]
        #     all_box_combinations += valid_union_boxes.tolist()

        # # pick some box
        # selected_crop = random.choice(all_box_combinations)
        # offset_x1 = selected_crop[0]-bboxes_np[:,0]
        assert h_start < 1.0 and w_start < 1.0

        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}


def is_crop_valid(crop, bboxes):
    # if no bboxes in the image, just return...    
    crop = np.array(crop)

    intersections_x1 = np.maximum(crop[0], bboxes[:, 0])
    intersections_y1 = np.maximum(crop[1], bboxes[:, 1])
    intersections_x2 = np.minimum(crop[2], bboxes[:, 2])
    intersections_y2 = np.minimum(crop[3], bboxes[:, 3])

    # Calculate the width and height of the intersection rectangles
    intersections_widths = np.maximum(0, intersections_x2 - intersections_x1)
    intersections_heights = np.maximum(0, intersections_y2 - intersections_y1)

     # Calculate the area of the intersection rectangles for all smaller boxes
    intersections_areas = intersections_widths * intersections_heights

    # Calculate the area of each smaller box
    smaller_box_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Create a boolean array indicating whether each smaller box meets the criteria
    valid_per_boxes = np.logical_or(
        (intersections_areas == 0),  # Small box is completely outside large box
        (intersections_areas == smaller_box_areas)  # Small box is completely within large box
    )
    num_within_boxes = np.count_nonzero(intersections_areas == smaller_box_areas)
    
    is_valid = np.all(valid_per_boxes) and (num_within_boxes > 0)

    return is_valid, num_within_boxes





def valid_boxes(union_boxes, smaller_boxes, image_width, image_height, crop_size):
    
    valid_boxes = []
    # Find the coordinates of the intersection rectangle
    for union_box in union_boxes:
        union_width = round((union_box[2]-union_box[0])*image_width)
        union_height = round((union_box[3]-union_box[1])*image_height)
        if (union_width > crop_size) or (union_height > crop_size):
            valid_boxes.append(False)
            continue
        intersections_x1 = np.maximum(union_box[0], smaller_boxes[:, 0])
        intersections_y1 = np.maximum(union_box[1], smaller_boxes[:, 1])
        intersections_x2 = np.minimum(union_box[2], smaller_boxes[:, 2])
        intersections_y2 = np.minimum(union_box[3], smaller_boxes[:, 3])

        # Calculate the width and height of the intersection rectangles
        intersections_widths = np.maximum(0, intersections_x2 - intersections_x1)
        intersections_heights = np.maximum(0, intersections_y2 - intersections_y1)

        # Calculate the area of the intersection rectangles for all smaller boxes
        intersections_areas = intersections_widths * intersections_heights

        # Calculate the area of each smaller box
        smaller_box_areas = (smaller_boxes[:, 2] - smaller_boxes[:, 0]) * (smaller_boxes[:, 3] - smaller_boxes[:, 1])

        # Create a boolean array indicating whether each smaller box meets the criteria
        # valid_per_boxes = np.logical_or(
        #     (intersections_areas == 0),  # Small box is completely outside large box
        #     (intersections_areas == smaller_box_areas)  # Small box is completely within large box
        # )
        # valid_all_boxes = np.all(valid_per_boxes)

        outside_condition = intersections_areas == 0
        within_condition = intersections_areas == smaller_box_areas
        valid_all_boxes = outside_condition | within_condition

        if valid_all_boxes:
            conditioned_boxes = smaller_boxes[outside_condition]
            offset_x1 = np.abs(union_box[0] - conditioned_boxes[:, 2])
        valid_boxes.append(valid_all_boxes)
            
    return valid_boxes


# def valid_boxes_2(union_box, small_boxes):
#     large_x1, large_y1, large_x2, large_y2 = union_box
#     small_boxes = np.array(small_boxes)
    
#     # Check if the small boxes are completely outside the large box
#     outside_condition = (small_boxes[:, 2] <= large_x1) | (small_boxes[:, 0] >= large_x2) | \
#                         (small_boxes[:, 3] <= large_y1) | (small_boxes[:, 1] >= large_y2)
    
#     # Check if the small boxes are completely within the large box
#     within_condition = (large_x1 <= small_boxes[:, 0]) & (small_boxes[:, 2] <= large_x2) & \
#                        (large_y1 <= small_boxes[:, 1]) & (small_boxes[:, 3] <= large_y2)
    
#     # Combine the conditions to determine the relationship
#     relationships = outside_condition | within_condition
    
#     return relationships



def compute_union_boxes(bboxes):
    """Calculate union of bounding boxes.

    Args:
        height (float): Height of image or space.
        width (float): Width of image or space.
        bboxes (List[tuple]): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.
        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    # Compute the minimum and maximum coordinates along each axis
    mins = bboxes[:, :, :2]
    maxes = bboxes[:, :, 2:]
    min_coords = np.min(mins, axis=1)
    max_coords = np.max(maxes, axis=1)

    # Construct the union box using the minimum and maximum coordinates
    union_boxes = np.concatenate([min_coords, max_coords], axis=1)

    return union_boxes
