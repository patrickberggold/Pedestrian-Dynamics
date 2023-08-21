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
from helper import xywh2xyxy, xyxy2xywh, xyxy2xywhn

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

        self.batch_size = batch_size
        self.agentIds = {'005': 0, '020': 1, '050': 2}

        assert len(self.bboxes) == len(self.img_paths), 'Length of image paths and trajectory paths do not match, something went wrong!'

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        format = 'pascal_voc' # 'coco', 'pascal_voc'
        # try:
        img_path, boxes_per_image = self.img_paths[idx], self.bboxes[idx]
        agentId = self.agentIds[boxes_per_image[-1]]
        boxes_per_image = boxes_per_image[:-1]

        # assert os.path.isfile(img_path)
        # except AssertionError:
        # width, height = Image.open(img_path).size
        # boxes_per_image = []
        # for i in range(5):
        #     if format == 'coco':
        #         box = [width//6 * i + width//20, height//5 * i + height//20, width//10, height//10] # x_c, y_c, w, y_max
        #     elif format == 'pascal_voc':
        #         box = [width//6 * i, height//5 * i, width//6 * i + width//10, height//5 * i + height//10] # x_min, y_min, x_max, y_max
        #     boxes_per_image.append(box)
        # labels = np.ones_like([1 for i in range(len(boxes_per_image))])
        labels = np.zeros_like([1 for i in range(len(boxes_per_image))])

        img = np.array(Image.open(img_path))[...,:3]
        # implement reading the bbox

        # apply (4) image transformation once per batch
        # if idx % self.batch_size == 0:
        #     self.p_dim_tf = [random.randint(0, 1) for _ in range(4)]
        #     print(f'Batch id: {idx}', self.p_dim_tf)

        img, bboxes, labels = self.augmentations(img, boxes_per_image, labels, p_dim_tf=self.p_dim_tf, format=format, file=img_path) #None) #
        bboxes = np.array([(round(box[0], 3), round(box[1], 3), round(box[2], 3), round(box[3], 3)) for box in bboxes])

        # normalize input pixels + bboxes
        img = img.astype(np.float32) / 255.
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
        if self.arch == 'EfficientDet':
            img = img.transpose((2, 0, 1)).astype(np.float32) / 255.
            bboxes = np.array(bboxes, dtype=np.float32)
            target = dict(bbox=bboxes, cls=np.array(labels, dtype=np.int64))

            return img, target, agentId
        elif self.arch == 'YoloV5':
            bboxes = xyxy2xywhn(bboxes, self.img_max_height, self.img_max_width)
            assert 1.0 >= np.max(bboxes)
        
        elif self.arch=='Detr':
            bboxes = xyxy2xywhn(bboxes, self.img_max_height, self.img_max_width)
            assert 1.0 >= np.max(bboxes)
        
        img = torch.tensor(img).permute(2, 0, 1).float()
        bboxes = torch.tensor(bboxes).float()#.unsqueeze(0)
        labels = torch.tensor(labels).long()


        # self.visualize(img, bboxes, labels, {1: 'Detection'})

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
        
        return img, labels, bboxes, agentId


    def augmentations(self, image, bboxes, labels, p_dim_tf, format='pascal_voc', file=None):

        # flipping, transposing, random 90 deg rotations
        transform = A.Compose([
            A.PadIfNeeded(3072, 3072, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True), #1.344
            # A.SmallestMaxSize(max_size=self.img_max_width),
            A.Resize(self.img_max_width, self.img_max_height),
            A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
            A.augmentations.geometric.transforms.VerticalFlip(p=0.5),

            # A.RandomResizedCrop(height=800, width=800, p=0.0) #, scale=(0.8, 1.0), ratio=(0.9, 1.11)) # from YoloV5
            # RandomResizePad(target_size=img_size, interpolation=interpolation, fill_color=fill_color) from EfficientDet
            # A.augmentations.geometric.transforms.Transpose(p=p_dim_tf[2]),
            A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        ], bbox_params=A.BboxParams(format=format, label_fields=['category_ids']))
        transformed = transform(image=image, bboxes=bboxes, category_ids=labels)        

        # self.visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], {0: 'Detection'}, file=file)

        return transformed['image'], transformed['bboxes'], transformed['category_ids']
    
    
    def visualize(self, image, bboxes, category_ids, category_id_to_name, format='pascal_voc',file=None):

        def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2, format=format):
            if format=='coco':
                x_min, y_min, w, h = bbox
                x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
            elif format=='pascal_voc':
                x_min, y_min, x_max, y_max = bbox
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
            
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
            cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (255, 0, 0), -1)
            cv2.putText(
                img,
                text=class_name,
                org=(x_min, y_min - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.35, 
                color=(255, 255, 255), 
                lineType=cv2.LINE_AA,
            )
            return img

        img = image.copy()
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, bbox, class_name)

        # plt.figure(figsize=(12, 12))
        if file is not None:
            plt.imsave(f'test_imgs\\{file.split(SEP)[-1]}', img)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.close('all')
