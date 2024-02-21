import torch
import pytorch_lightning as pl
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torch.utils.data import DataLoader
import os, random
from ObjDetDataset import ObjDetDataset, augment_on_batch_level
from helper import SEP, PREFIX

# line extraction
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

# extra Object Detection modules
# EfficientDet
from models.EfficientDet.data.loader import DetectionFastCollate


class ObjDetDatamodule(pl.LightningDataModule):
    def __init__(self, config: dict, num_workers: int = 0):
        super().__init__()
        # self.mode = config['mode']
        self.config = config
        self.batch_size = config['batch_size']
        # self.additional_info = config['additional_info']
        # self.dataset_type = config['dataset']
        self.arch = config['arch']
        self.cuda_device = config['cuda_device']
        # self.vary_area_brightness = config['vary_area_brightness']
        self.limit_dataset = config['limit_dataset']
        self.num_workers = num_workers
        self.transforms = None
        self.augment_batch_level = config['augment_batch_level']

        assert self.arch in ['Detr', 'Detr_custom', 'FasterRCNN', 'FasterRCNN_custom', 'YoloV5', 'EfficientDet']

        if self.arch == 'Detr':
            from transformers import DetrFeatureExtractor
            feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
            self.img_size = self.config['img_max_size'] # (960, 3200)
            self.transforms = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        elif self.arch == 'EfficientDet':
            pass
            # import albumentations as A
            # self.transforms = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            # EfficientDet image values in data/transforms: IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD (if needed)
        else:
            self.transforms = None # ToTensor()

        self.set_data_paths()

    def setup(self, stage):
        self.train_dataset = ObjDetDataset(self.config, self.train_imgs_list, self.train_targets_list, transform=self.transforms, batch_size = self.batch_size)
        self.val_dataset = ObjDetDataset(self.config, self.val_imgs_list, self.val_targets_list, transform=self.transforms, batch_size = self.batch_size)
        self.test_dataset = ObjDetDataset(self.config, self.test_imgs_list, self.test_targets_list, transform=self.transforms, batch_size = self.batch_size)
        self.inference_dataset = ObjDetDataset(self.config, self.inference_imgs_list, self.inference_targets_list, transform=self.transforms, batch_size = self.batch_size)

    def train_dataloader(self):
        if self.arch=='EfficientDet':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=DetectionFastCollate(self.config)) # maybe implement PrefetchLoader at some point ...
        elif self.arch=='YoloV5':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.yolo_collate)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def val_dataloader(self):
        if self.arch=='EfficientDet':
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=DetectionFastCollate(self.config, mode=1)) # maybe implement PrefetchLoader at some point ...
        elif self.arch=='YoloV5':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.yolo_collate)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)
        

    def test_dataloader(self):
        if self.arch=='EfficientDet':
            return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=DetectionFastCollate(self.config, mode=1)) # maybe implement PrefetchLoader at some point ...
        elif self.arch=='YoloV5':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.yolo_collate)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_collate)
        

    def predict_dataloader(self):
        if self.arch in ['EfficientDet', 'YoloV5']: return NotImplementedError
        return DataLoader(self.inference_dataset, batch_size=1, collate_fn=self.custom_collate)
    

    def custom_collate(self, batch):
        if self.augment_batch_level:
            return augment_on_batch_level(batch, self.config, self.transforms)
        else:
            images, labels, bboxes, numAgentsIds = zip(*batch)
        return torch.stack(images, dim=0), labels, bboxes, torch.stack(numAgentsIds, dim=0)
    

    def yolo_collate(self, batch):
        images, labels, numAgentsIds = [], [], []
        for i, batchItem in enumerate(batch):
            images.append(batchItem[0])
            id = (torch.ones((batchItem[1].size(0), 1), dtype=torch.int64) * i).float()
            labels.append(torch.cat([
                id, batchItem[1].unsqueeze(1), batchItem[2]], 
                dim=1))
            numAgentsIds.append(batchItem[3])
        return torch.stack(images, dim=0), torch.cat(labels, dim=0).float(), torch.LongTensor(numAgentsIds)


    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.cuda_device != 'cpu':
            device = torch.device('cuda', self.cuda_device)
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            return batch

    def set_data_paths(self):
        
        # self.splits = [0.7, 0.15, 0.15]
        self.splits = [0.75, 0.25, 0.0]

        self.img_path = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\images'
        # self.img_path = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\images_transformed'
        self.crowdit_path = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\sims\\crowdit'
        self.boxes_temp = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\boxes_temp_corrected'

        # for now use yellow rectangles:
        self.img_path = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\images_rectangles_black_mode' #_larger'
        self.boxes_temp = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\boxes_rectangles_black_mode' #_larger'

        self.img_path = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Trackline_V3_Complete-Data'
        self.boxes_temp = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Trackline_V3_boxes'
        self.img_path = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Trackline_V3_Input_Images'
        self.boxes_temp = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\crowdit-V3\\konvertiert'

        # new dataset
        self.img_path = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Trackline_V3_Input_Images-V4'
        self.boxes_temp = 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\crowdit-V4'

        self.set_filepaths()
        self.set_inference_paths()

        assert len(self.img_list) == len(self.bboxes), 'Images list and trajectory list do not have same length, something went wrong!'
        # Randomly check if entries are the same
        # index_check_list = random.sample(range(len(self.img_list)), 10)
        # assert [SEP.join(self.img_list[i].split('\\')[-2:]) for i in index_check_list] == [SEP.join(self.traj_list[i].split('\\')[-2:]) for i in index_check_list], \
        #     'Images list and trajectory list do not have same entries, something went wrong!'
    	
        val_split_factor = self.splits[1]
        test_split_factor = self.splits[2]
        
        self.indices = list(range(len(self.img_list)))
        
        val_split_index = int(len(self.indices) * val_split_factor)
        test_split_index = int(len(self.indices) * test_split_factor)
        # to the next batch size divisible
        # val_split_index = ((len(self.indices) * val_split_factor + self.batch_size - 1) // self.batch_size) * self.batch_size
        # test_split_index = ((len(self.indices) * test_split_factor + self.batch_size - 1) // self.batch_size) * self.batch_size
        
        random.seed(42)
        random.shuffle(self.indices)

        self.train_imgs_list = [self.img_list[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        self.train_targets_list = [self.bboxes[idx] for idx in self.indices[(test_split_index + val_split_index):]]
        
        self.val_imgs_list = [self.img_list[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]
        self.val_targets_list = [self.bboxes[idx] for idx in self.indices[test_split_index:(test_split_index + val_split_index)]]

        self.test_imgs_list = self.img_list_inference # [self.img_list[idx] for idx in self.indices[:test_split_index]]
        self.test_targets_list = self.bboxes_inference # [self.bboxes[idx] for idx in self.indices[:test_split_index]]

        self.inference_imgs_list = self.img_list_inference
        self.inference_targets_list = self.bboxes_inference

        # LIMIT THE DATASET
        if self.limit_dataset:
            # raise ValueError('Check first if limitations set correctly')
            self.train_imgs_list = self.train_imgs_list[:self.limit_dataset]
            self.train_targets_list = self.train_targets_list[:self.limit_dataset]

            self.val_imgs_list = self.val_imgs_list[:self.limit_dataset//4]
            self.val_targets_list = self.val_targets_list[:self.limit_dataset//4]

            self.test_imgs_list = self.test_imgs_list[:self.limit_dataset//4]
            self.test_targets_list = self.test_targets_list[:self.limit_dataset//4]


    def set_filepaths(self):

        self.img_list = []
        self.crowdit_list = []
        self.bboxes = []

        if 'crowdit-V' in self.boxes_temp:
            from tqdm import tqdm
            box_dir = [d for d in os.listdir(self.boxes_temp) if d.endswith('_res') and ('_A10' in d or '_A30' in d or '_A50' in d)]
            img_dir = [d for d in os.listdir(self.img_path) if d.endswith('.png')] # if '_A10' in d]
            assert len(box_dir) == len(img_dir)#*3
            for b in tqdm(box_dir):
                img_file = b.replace('_res', '.png')
                # if '_A30' in img_file: 
                #     img_file = img_file.replace('_A30', '_A10')
                # elif '_A50' in img_file:
                #     img_file = img_file.replace('_A50', '_A10')
                img_file = os.path.join(self.img_path, img_file)
                box_file = os.path.join(self.boxes_temp, b, 'criticalAreas.txt')
                assert os.path.isfile(img_file) and os.path.isfile(box_file)

                boxes_per_file = []
                f = open(box_file, "r")
                file_lines = f.readlines()
                for line in file_lines:
                    x1, y1, x2, y2 = line.strip().split(',')
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if y1==y2 or x1==x2:
                        continue
                    assert y2 > y1 and x2 > x1
                    boxes_per_file.append([x1, y1, x2, y2])
                f.close()
                additional_info = b.replace('_res', '').split('_')
                a, e, s, es, ss, c = additional_info[1][1:], additional_info[4][1:], additional_info[5][1:], additional_info[8][2:], additional_info[9][2:], additional_info[2][1]
                # boxes_per_file.append([b.split('_A')[-1][:2]])
                assert a in ['10', '30', '50'] and e in ['n', '1', '2', '3'] and es in ['n', '1', '2', '3'] and s in ['n', '0', '2.4'] and ss in ['n', '0', '2.4'] and c in ['0', '2']
                boxes_per_file.append([a, e, s, es, ss, c])
                # if b.split('_')[4][1:] == '10':
                #     continue
                self.bboxes.append(boxes_per_file)
                self.img_list.append(img_file)

            return

            # for img_file, box_file in tqdm(zip(img_dir, box_dir), total=len(box_dir)):


        elif self.img_path.endswith('larger'):
            from tqdm import tqdm
            assert len(os.listdir(self.img_path)) == len(os.listdir(self.boxes_temp))
            for img_file, box_file in tqdm(zip(os.listdir(self.img_path), os.listdir(self.boxes_temp)), total=len(os.listdir(self.img_path))):
                f = open(os.path.join(self.boxes_temp, box_file), 'r')
                lines = f.readlines()
                boxes_per_file = []
                for line in lines:
                    box_str = line.strip().split(',')
                    boxes_per_file += [[int(b) for b in box_str]]
                boxes_per_file.append('005')
                self.bboxes.append(boxes_per_file)
                self.img_list.append(os.path.join(self.img_path, img_file))

            return

        elif self.img_path.endswith('Input_Images'):
            from tqdm import tqdm
            assert len(os.listdir(self.img_path)) == len(os.listdir(self.boxes_temp))
            for img_file, box_file in tqdm(zip(os.listdir(self.img_path), os.listdir(self.boxes_temp)), total=len(os.listdir(self.img_path))):
                assert img_file.replace('.png','') == box_file.replace('.txt','')
                if 'A30' in box_file: continue
                f = open(os.path.join(self.boxes_temp, box_file), 'r')
                lines = f.readlines()
                boxes_per_file = []
                for line in lines:
                    box_str = line.strip().split(',')
                    boxes_per_file += [[int(b) if int(b)>0 else 0 for b in box_str]]
                boxes_per_file.append(img_file.split('_')[-2][1:].zfill(3))
                self.bboxes.append(boxes_per_file)
                self.img_list.append(os.path.join(self.img_path, img_file))
            return

        # x_dict = {}
        # y_dict = {}
        # from PIL import Image
        for train_test_img, train_test_cro, train_test_boxes in zip(os.listdir(self.img_path), os.listdir(self.crowdit_path), os.listdir(self.boxes_temp)):
            assert train_test_img == train_test_cro == train_test_boxes

            sorted_imgs = [i_path for i_path in os.listdir(os.path.join(self.img_path, train_test_img)) if i_path.endswith('.png')]
            sorted_imgs = sorted(sorted_imgs, key=lambda x: x.replace('_floorplan.png', ''))
            sorted_cros = [c_path for c_path in os.listdir(os.path.join(self.crowdit_path, train_test_cro)) if c_path.endswith('res')]
            sorted_cros = sorted(sorted_cros, key=lambda x: x.split('accurate_')[-1].replace('_res', ''))
            sorted_temp_boxes = [i_box for i_box in os.listdir(os.path.join(self.boxes_temp, train_test_boxes)) if i_box.endswith('.txt')]
            sorted_temp_boxes = sorted(sorted_temp_boxes, key=lambda x: x.replace('.txt', ''))

            assert len(sorted_imgs)==len(sorted_cros)==len(sorted_temp_boxes)
            
            self.img_list += [os.path.join(self.img_path, train_test_img, img_file) for img_file in sorted_imgs]
            # CORRECTION CODE
            """ for imgFile in self.img_list:
                tttt = imgFile.split(SEP)[-1]
                tttt = tttt.replace('_floorplan.png', '')
                if not imgFile.split(SEP)[-1].replace('_floorplan.png', '').startswith('F2_T25_W3_L200_H15_E2'):
                    continue
                img = np.array(Image.open(imgFile))
                plt.imshow(img)
                obs_zones_c1 = img[:,:,0] < 210
                obs_zones_c2 = img[:,:,1] < 210
                obs_zones_c3 = img[:,:,2] < 210
                obs_zones = np.argwhere(obs_zones_c1 & obs_zones_c2 & obs_zones_c3).squeeze()
                # 236,207,205
                or_zones_c1 = (220 < img[:,:,0]) & (img[:,:,0] < 250)
                or_zones_c2 = (190 < img[:,:,1]) & (img[:,:,1] < 220)
                or_zones_c3 = (190 < img[:,:,2]) & (img[:,:,2] < 220)
                or_zones = np.argwhere(or_zones_c1 & or_zones_c2 & or_zones_c3).squeeze()
                corrected_img = np.copy(img)
                corrected_img[obs_zones[:,0], obs_zones[:,1]] = np.array([[0, 0, 0]]*len(obs_zones))
                corrected_img[or_zones[:,0], or_zones[:,1]] = np.array([[180, 0, 0]]*len(or_zones))
                # pink areas
                pink_zones_c1 = (50 < corrected_img[:,:,0]) & (corrected_img[:,:,0] < 253)
                pink_zones_c2 = (50 < corrected_img[:,:,1]) & (corrected_img[:,:,1] < 253)
                pink_zones_c3 = (50 < corrected_img[:,:,2]) & (corrected_img[:,:,2] < 253)
                pink_zones = np.argwhere(pink_zones_c1 & pink_zones_c2 & pink_zones_c3).squeeze()
                #filter coords
                pink_zones_in = []
                # for coord in pink_zones:
                #     if 110 <= coord[1] <= 460 or 1440 <= coord[1] <= 1810:
                #         if 130 <= coord[0] <= 350 or 570 <= coord[0] <= 630:
                #             pink_zones_in.append(coord)
                # pink_zones = np.array(pink_zones_in)
                mask_area1 = (pink_zones[:,1] >= 110) & (pink_zones[:,1] <= 460)
                mask_area2 = (pink_zones[:,1] >= 1440) & (pink_zones[:,1] <= 1810)
                mask_area3 = (pink_zones[:,0] >= 130) & (pink_zones[:,0] <= 350)
                mask_area4 = (pink_zones[:,0] >= 570) & (pink_zones[:,0] <= 630)
                pink_zones = pink_zones[mask_area1 | mask_area2 | mask_area3 | mask_area4]
                corrected_img[pink_zones[:,0], pink_zones[:,1]] = np.array([[255, 0, 255]]*len(pink_zones))
                # # bounding boxes
                boxes = [
                    [387, 240,485, 370],
                    [436, 573, 507, 636],
                    [1413, 280,1523, 367],
                    [1422, 557, 1533, 636]
                ]
                confidences = ['98', '91', '95', '95']
                colors = [(230,0,0), (0,176,80), (255, 217,102), (46,117,182)] # (0, 180, 0)
                # labels = [1,1,1,1]

                # import albumentations as A
                # transform = A.Compose([A.PadIfNeeded(1360, 3200, mask_value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
                # transformed = transform(image=corrected_img, bboxes=boxes, category_ids=labels)
                # corrected_img_tf = transformed['image']
                # boxes = transformed['bboxes']

                for box, conf, color in zip(boxes, confidences, colors):
                    fontScale = 0.35
                    text = f'Critical area: {conf}%'
                    x_min, y_min, x_max, y_max = box
                    cv2.rectangle(corrected_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=2)
                    ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)    
                    cv2.rectangle(corrected_img, (int(x_min), int(y_min) - int(1.3 * text_height)), (int(x_min) + text_width, int(y_min)), color, -1)
                    cv2.putText(
                        corrected_img,
                        text=text,
                        org=(int(x_min), int(y_min) - int(0.3 * text_height)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=fontScale, 
                        color=(255, 255, 255), 
                        lineType=cv2.LINE_AA,
                    )
               
                plt.close('all')
                plt.imshow(corrected_img)
                plt.close('all')
                plt.imsave('floorplan_input.png', corrected_img) """
            # CORRECTION CODE

        """ 
            self.crowdit_list += [os.path.join(self.crowdit_path, train_test_cro, c_project, 'out-1', 'criticalAreas.csv') for c_project in sorted_cros]
            for boxFile in sorted_temp_boxes:
                boxFileGlobal = os.path.join(self.boxes_temp, train_test_boxes, boxFile)
                f = open(boxFileGlobal, 'r')
                box_lines = f.readlines()
                boxes_per_file = []
                for row in box_lines:
                    b_strings = row.strip().split(',')
                    x1 = int(b_strings[0])
                    y1 = int(b_strings[1])
                    x2 = int(b_strings[2])
                    y2 = int(b_strings[3])
                    assert x1 < x2
                    assert y1 < y2
                    boxes_per_file += [[x1, y1, x2, y2]]
                num_agents = boxFile.split('_P')[-1][:3]
                assert num_agents in ['005', '020', '050']
                boxes_per_file.append(num_agents)
                f.close()
                boxes_per_file = [ [200,150,300,250], [700,300,850,400], '005']
                self.bboxes.append(boxes_per_file)
            
        # delete all samples without box input:
        box_indices = [i for i, box in enumerate(self.bboxes) if len(box)>1]
        self.bboxes = [self.bboxes[i] for i in box_indices]
        self.img_list = [self.img_list[i] for i in box_indices] """

        # detection of origin areas
        from tqdm import tqdm
        or_box_folder = "C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND"+SEP+"oArea_boxes"
        for img_file in self.img_list:
            f = open(os.path.join(self.boxes_temp, img_file.split(SEP)[-2], img_file.split(SEP)[-1].replace('.png', '.txt')), 'r')
            lines = f.readlines()
            boxes_per_file = []
            for line in lines:
                box_str = line.strip().split(',')
                boxes_per_file += [[int(b) for b in box_str]]
            boxes_per_file.append('005')
            self.bboxes.append(boxes_per_file)

        return
        
        if not os.path.isdir(or_box_folder): os.mkdir(or_box_folder)
        for img_file in tqdm(self.img_list):
            image = cv2.imread(img_file)
            # RGB: (236,207,205)
            lower_red = np.array([190, 190, 220])
            upper_red = np.array([220, 220, 240])
            mask = cv2.inRange(image, lower_red, upper_red)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes_per_file = []
            f = open(os.path.join(or_box_folder, img_file.split(SEP)[-1].replace('.png', '.txt')), 'w')

            # Iterate through the contours
            for contour in contours:
                # Approximate the contour to a polygon
                epsilon = 0.03 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                approx = approx.squeeze()
                x_min, y_min = approx[0]
                x_max, y_max = approx[1]
                boxes_per_file += [[int(x_min), int(y_min), int(x_max), int(y_max)]]
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                f.write(f'{int(x_min)}, {int(y_min)}, {int(x_max)}, {int(y_max)}\n')

                # If the polygon has four vertices, it could be a rectangle
                # if len(approx) == 4:
                #     # Draw a bounding box around the detected rectangle
                #     rect = cv2.boundingRect(approx)
                #     x, y, w, h = rect
                #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green bounding box
            # rgb = image[...,::-1].copy()
            # plt.imsave(f'C:\\Users\\Remotey\\Documents\\Pedestrian-Dynamics\\test_imgs\\gt_{img_file.split(SEP)[-1]}', rgb)
            # plt.close('all')
            boxes_per_file.append('005')
            f.write('005')
            self.bboxes.append(boxes_per_file)
            f.close()



        """
        # check once
        # for i, c in zip(self.img_list, self.crowdit_list):
        #     p1 = i.split(SEP)[8].replace('_floorplan.png', '')
        #     p2 = c.split(SEP)[8].split('accurate_')[-1].replace('_res', '')
        #     assert i.split(SEP)[8].replace('_floorplan.png', '') == c.split(SEP)[9].split('accurate_')[-1].replace('_res', '')
        
        # read bounding boxes from files + pixel coordinate conversion
        for csvFile, imgFile in tqdm(zip(self.crowdit_list, self.img_list), desc="Progress bboxes", total=len(self.crowdit_list)):

            file = open(csvFile)
            info = csv.reader(file) # id,floorId,minX,minY,maxX,maxY
            bboxes = [[float(row[2]), float(row[3]), float(row[4]), float(row[5])] for i, row in enumerate(info) if i > 0]
            # for i, row in enumerate(info):
            #     if i > 0:
            #         bboxes.append([float(row[2]), float(row[3]), float(row[4]), float(row[5])])
            bboxes_real = []
            save_images = False
            if len(bboxes)==0:
                self.bboxes.append([])
                # continue
                if save_images:
                    img = cv2.imread(imgFile)
                    img_bb = np.copy(img)
                    fileName = os.path.join('C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\boxed_images_corrected\\no_boxes', csvFile.split(SEP)[9].replace('_res', '_boxed.png'))
                    plt.imsave(fileName, img_bb[...,::-1])
            else:
                img = cv2.imread(imgFile)
                height, width, _ = img.shape
                plot = False
                # plt.imshow(img)
                line, x_start, x_end, y_start, y_end = self.extract_lines(img)

                line_length_pix = float(line[2] - line[0])

                # calculate dim
                site_x = width / line_length_pix * 40
                site_y = height / line_length_pix * 40

                x_start_real = x_start / width * site_x
                x_end_real = x_end / width * site_x
                y_start_real = y_start / height * site_y
                y_end_real = y_end / height * site_y

                if site_x not in x_dict:
                    x_dict.update({site_x: 1})
                else:
                    x_dict[site_x] += 1
                if site_y not in y_dict:
                    y_dict.update({site_y: 1})
                else:
                    y_dict[site_y] += 1

                for box in bboxes:
                    # add_y = 0
                    # add_x = 0
                    # if imgFile.split(SEP)[-1].split('_floor')[0] == 'F2_T15_W4_L200_H25_E3_P005':
                    #     hhh = box[0]
                    #     qqq = box[2]
                    if box[0] <= 25 or box[2] >= x_end_real-25:
                        continue 
                    if box[0]==box[2]:
                        print(f'\nsame x-values in file {csvFile}')
                        continue
                        # add_x = 5
                    if box[1]==box[3]:
                        print(f'\nsame y-values in file {csvFile}')
                        continue
                        # add_y = 5
                    x_min_real = box[0] / (x_end_real-x_start_real) * width
                    y_min_real = height - box[3] / (y_end_real-y_start_real) * height
                    x_max_real = box[2] / (x_end_real-x_start_real) * width
                    y_max_real = height - box[1] / (y_end_real-y_start_real) * height
                    x_min_real_b, x_max_real_b = round(x_min_real), round(x_max_real)
                    y_min_real_b, y_max_real_b = round(y_min_real), round(y_max_real)

                    # x_max_real_b += add_x
                    # y_max_real_b += add_y

                    if x_min_real_b == x_max_real_b:
                        x_max_real_b += 1
                        plot = True
                    if y_min_real_b == y_max_real_b:
                        y_max_real_b += 1
                        plot = True
                    
                    bboxes_real.append([x_min_real_b, y_min_real_b, x_max_real_b, y_max_real_b])

                    # x_c = (x_min_real + x_max_real)/2.
                    # y_c = (y_min_real + y_max_real)/2.
                    # w = x_max_real - x_min_real
                    # h = y_max_real - y_min_real
                    # bboxes_real.append([round(x_c), round(y_c), round(w), round(h)])

                if plot:
                    img_bb = np.copy(img)
                    for box in bboxes_real:
                        x, y, x2, y2 = box
                        cv2.rectangle(img_bb, (x, y), (x2, y2), (0, 255, 0), 2)
                        # x_c, y_c, w, h = box
                        # cv2.rectangle(img_bb, (x_c-w//2, y_c-h//2), (x_c+w//2, y_c+h//2), (0, 255, 0), 2)
                    plt.imshow(img_bb)
                    plt.close('all')
                    plot = False
                if save_images:
                    img_bb = np.copy(img)
                    small_box = False
                    for box in bboxes_real:
                        x, y, x2, y2 = box
                        area = (x2-x)*(y2-y)
                        if area <= 1e3:
                            small_box = True
                        cv2.rectangle(img_bb, (x, y), (x2, y2), (0, 255, 0), 2)
                        # x_c, y_c, w, h = box
                        # cv2.rectangle(img_bb, (x_c-w//2, y_c-h//2), (x_c+w//2, y_c+h//2), (0, 255, 0), 2)
                    if len(bboxes_real)==0:
                        fileName = os.path.join('C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\boxed_images_corrected\\no_boxes', csvFile.split(SEP)[9].replace('_res', '_boxed.png'))
                    elif small_box:
                        fileName = os.path.join('C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\boxed_images_corrected\\small_box', csvFile.split(SEP)[9].replace('_res', '_boxed.png'))
                    else:
                        fileName = os.path.join('C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\boxed_images_corrected\\normal_boxes', csvFile.split(SEP)[9].replace('_res', '_boxed.png'))
                    plt.imsave(fileName, img_bb[...,::-1])
                    
                self.bboxes.append(bboxes_real)
                
            save_boxes = True
            if save_boxes:
                temp_box_folder = os.path.join(self.boxes_temp, csvFile.split(SEP)[8])
                if not os.path.isdir(temp_box_folder): os.mkdir(temp_box_folder)
                temp_box_file = os.path.join(temp_box_folder, csvFile.split(SEP)[9]).replace('_res', '_boxes.txt')
                f = open(temp_box_file, 'w')
                for ib, box in enumerate(bboxes_real):
                    f.write(f'{box[0]},{box[1]},{box[2]},{box[3]}')
                    if ib != len(bboxes_real)-1: 
                        f.write('\n')
                f.close()
        quit()
        stop = 3
        """


    def set_inference_paths(self):
        self.img_list_inference = []
        self.bboxes_inference = []
        self.img_inference_paths = [
            # 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Trackline-V3-U9\\Modell_U9_colored_blacked.jpg',
            'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Use cases\\old_stage\\Modell_U9_colored_blacked_processed_final.png',
            # 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Trackline-Vnew\\saved_from_xml_2_altered_final_padded.png',
            # 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Use cases\\new_stage\\saved_from_xml_2_altered_final_padded_processed.png',
            'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Use cases\\new_stage\\saved_from_xml_2_altered_final_padded__new_processed.png'
            # 'C:\\Users\\Remotey\\Documents\\Datasets\\BEYOND\\U9-Trackline-V3-2SBSS\\U9-Trackline-V3-2SBSS_refined_blacked.jpg'
        ]
        
        for img_path in self.img_inference_paths:
            all_result_runs_folder = img_path.replace(img_path.split(SEP)[-1], 'results_old')
            assert os.path.isdir(all_result_runs_folder)
            dir_list = [d for d in os.listdir(all_result_runs_folder) if d.endswith('_res')]

            for result_folder in dir_list:
                self.img_list_inference.append(img_path)
                boxes_per_file = []

                f = open(os.path.join(all_result_runs_folder, result_folder, 'criticalAreas.txt'), 'r')
                bboxes_lines = f.readlines()
                for line in bboxes_lines:
                    x1, y1, x2, y2 = line.strip().split(',')
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    boxes_per_file.append([x1, y1, x2, y2])

                # boxes_per_file.append(result_folder.split('_A')[-1][:2])
                if 'old_stage' in img_path:
                    boxes_per_file.append([result_folder.split('_A')[-1][:2], 'n', 'n', '3', '0', '2'])
                elif 'new_stage' in img_path:
                    boxes_per_file.append([result_folder.split('_A')[-1][:2], '2', '0', '3', '0', '2'])
                self.bboxes_inference.append(boxes_per_file)
                f.close()


    def extract_lines(self, img):
        h,w,ch = img.shape
        # cv2.imshow(imgFile)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        lines = lines.squeeze()

        # get the limits
        row_indeces_x_ymin = np.argsort(lines[:, 0])

        x_start = 200
        y_start = 200
        x_end = 200
        y_end = 200
        # plt.imshow(img)

        for x1_ind in row_indeces_x_ymin[:10]:
            line = lines[x1_ind]
            if (line[2] - line[0]) > 0.9*w:
                cv2.line(line_image,(line[0],line[1]),(line[2],line[3]),(255,0,0),5)
                lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
                x_start = min(x_start, int(line[0]))
                x_end = max(x_end, int(line[2]))
            if line[2]-line[0]<2:
                cv2.line(line_image,(line[0],line[1]),(line[2],line[3]),(255,0,0),5)
                lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
                y_start = min(y_start, int(line[3])) # cv2 switches y-axis
                y_end = max(y_end, int(line[1])) # cv2 switches y-axis
        
        if x_start == 200 or y_start == 200 or x_end == 200 or y_end == 200:
            print('WARNING: a value is still 200')
            plt.imshow(lines_edges)


        # get the marker
        row_indices_sorted_ymin = np.argsort(lines[:, 3])[::-1]
        for y1_ind in row_indices_sorted_ymin[:5]:
            marker = lines[y1_ind]
            marker_lenght = marker[2] - marker[0]
            if marker[1]==marker[3] and (100 <= marker_lenght <= 600):
                # cv2.line(line_image,(marker[0],marker[1]),(marker[2],marker[3]),(255,0,0),5)
                # lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
                # plt.imshow(lines_edges)
                break
            marker = None
        if marker is None:
            print('Warning: Marker is None')
        # visualize all lines
        # for line in lines:
        #     for x1,y1,x2,y2 in line:
        #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

        # # Draw the lines on the  image
        # lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        # plt.imshow(lines_edges)

        return marker, x_start, x_end, y_start, y_end
