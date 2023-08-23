import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau, LambdaLR
from torch.optim import Adam, AdamW, SGD
import pytorch_lightning as pl
# from torchsummary import summary
# import numpy as np
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
from helper import SEP, xywhn2xyxy, xywh2xyxy, xyxyn2xyxy
from torchvision.models.detection.faster_rcnn import FasterRCNN
from collections import OrderedDict
from typing import Tuple, List
import warnings
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss, keypointrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from decimal import Decimal
from models.DETR.loss import compute_loss

class ObjDetModule(pl.LightningModule):
    def __init__(
        self, 
        config: dict,
        train_config: dict,
        ):
        super(ObjDetModule, self).__init__()
        # self.mode = config['mode']
        self.arch = config['arch']
        self.config = config
        
        # assert self.mode in ['grayscale', 'evac_only', 'class_movie', 'density_reg', 'density_class', 'denseClass_wEvac'], 'Unknown mode setting!'
        assert self.arch in ['Detr', 'FasterRCNN', 'FasterRCNN_custom', 'EfficientDet', 'YoloV5'], 'Unknown arch setting!'

        self.learning_rate = train_config['learning_rate']
        self.lr_scheduler = train_config['lr_scheduler']
        self.lr_sch_step_size4lr_step = train_config['lr_sch_step_size4lr_step']
        self.lr_sch_step_size4cosAnneal = train_config['lr_sch_step_size4cosAnneal']
        self.lr_sch_gamma4redOnPlat_and_stepLR = train_config['lr_sch_gamma4redOnPlat_and_stepLR']
        self.lr_sch_gamma4expLR = train_config['lr_sch_gamma4expLR']
        self.lr_sch_patience4redOnPlat = train_config['lr_sch_patience4redOnPlat']
        # self.lr_sch_step_size = lr_sch_step_size if lr_scheduler=='StepLR' else 50
        # self.lr_sch_gamma = lr_sch_gamma
        self.opt = train_config['opt']
        self.weight_decay = train_config['weight_decay']
        # self.additional_info = config['additional_info']
        self.model_ema_decay = train_config['model_ema_decay']
        self.model_ema = None
        self.customized_optim = train_config['customized_optim']

        self.num_heads = 1
        assert self.lr_scheduler in [CosineAnnealingLR.__name__, StepLR.__name__, ExponentialLR.__name__, ReduceLROnPlateau.__name__], 'Unknown LR Scheduler!'

        self.train_losses = {}
        self.train_losses_per_epoch = {}
        self.val_losses = {}
        self.val_losses_per_epoch = {}
        
        self.log_result = {'validation': [], 'training': []}
        self.backbone_frozen = False
        self.confusion_matrix_train_epoch = None
        self.confusion_matrix_val_epoch = None
        self.tversky_weights = None


        self.img_max_width, self.img_max_height = config['img_max_size']

        if self.arch == 'Detr':
            from transformers import DetrForObjectDetection, DetrConfig
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            # self.model = DetrForObjectDetection(DetrConfig.from_pretrained("facebook/detr-resnet-50"))
            # output_dim = 1 for regression, output_dim > 1 for classification (adjust model.config.num_labels accordingly)
            # output_dim = max_num_objects in the dataset (for facebook 91), num_queries = max_num_objects per image (detections padded with 'no object' entries)
            self.model.class_labels_classifier = torch.nn.Linear(self.model.config.d_model, self.config['num_classes'] + 1) # detection + no_detection
            if self.config['top_k'] != self.model.model.query_position_embeddings.num_embeddings:
                self.model.model.query_position_embeddings = torch.nn.Embedding(self.config['top_k'], self.model.config.d_model)
            self.model.config.num_labels = self.config['num_classes']
            def init_fct(m):
                try:
                    if hasattr(m, 'weight'):
                        torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                        # torch.nn.init.xavier_normal_(m.weight)
                        # torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                    elif hasattr(m, 'bias'):
                        m.bias.data.zero_()
                except ValueError:
                    m.weight.data.zero_()
            # self.model.apply(init_fct)
        
        elif self.arch == 'FasterRCNN':

            from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
            from helper import validate_faster_rcnn_with_loss

            pretrained_backbone = True
            self.model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=self.config['num_classes'] + 1, pretrained_backbone=pretrained_backbone, trainable_backbone_layers=None)

            validate_faster_rcnn_with_loss(self.model)


        elif self.arch == 'FasterRCNN_custom':

            from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
            from torchvision.ops import misc as misc_nn_ops
            from torchvision.models.resnet import resnet50

            trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
            backbone = resnet50(pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
            backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
            self.model = FasterRCNN_custom(backbone, self.config['num_classes'] + 1)

        elif self.arch == 'YoloV5':
            """ from ultralytics_yolov5 """
            from models.Yolo.models.yolo import DetectionModel
            from models.Yolo.utils.loss import ComputeLoss
            import yaml
            # model_config = 'ObjectDetection\\models\\Yolo\\models\\configs\\yolov5s.yaml'
            hyper_para_config = 'ObjectDetection\\models\\Yolo\\data\\hyps\\hyp.scratch-high.yaml'
            hyp = yaml.load(open(hyper_para_config), Loader=yaml.FullLoader)
            
            self.model = DetectionModel(self.config['model_config'], ch=3, nc=self.config['num_classes'], anchors=hyp.get('anchors'))
            
            from timm.utils import ModelEmaV2
            self.model_ema = ModelEmaV2(self.model, decay=self.model_ema_decay) if self.model_ema_decay else None
            
            # TODO: NMS for all models already there?, pretrained?
            # TODO: which add info models exist?
            # TODO look up multi-gpu training for yolo
            # TUNING possibilities: loss weights, image augmentations, box normalization, 
            nl = self.model.model[-1].nl  # number of detection layers (to scale hyps)
            # loss scaling parameters:
            hyp['box'] *= 3 / nl  # scale to layers
            hyp['cls'] *= self.config['num_classes'] / 80 * 3 / nl  # cls loss gain, scale to classes and layers
            hyp['obj'] *= (max(self.config['img_max_size']) / 640) ** 2 * 3 / nl  # obj loss gain, scale to image size and layers
            # hyp['anchor_t'] = 8.

            self.model.nc = self.config['num_classes']  # attach number of classes to model
            self.model.hyp = hyp  # attach hyperparameters to model
            
            pretrained = False
            if pretrained:
                print('Using pre-trained model...')
                # download models at: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
                from models.Yolo.utils.general import intersect_dicts
                resume = False
                weights = 'ObjectDetection\\checkpoints\\yolov5'+self.config['version']+'.pt' # 'YOLOv5l.pt'
                ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
                model = DetectionModel(self.config['model_config'], ch=3, nc=self.config['num_classes'], anchors=hyp.get('anchors'))
                exclude = ['anchor'] if (self.config['model_config']) and not resume else []  # exclude keys
                csd = intersect_dicts(ckpt['model'].float().state_dict(), model.state_dict(), exclude=exclude)  # intersect
                model.load_state_dict(csd, strict=False)  # load

            self.loss_fn = ComputeLoss(self.model, device=self.config["cuda_device"])



        elif self.arch == "EfficientDet":
            """ https://github.com/rwightman/efficientdet-pytorch """
            from models.EfficientDet.efficientdet import EfficientDet
            from models.EfficientDet.helpers import load_pretrained, load_checkpoint
            from models.EfficientDet.loss import DetectionLoss

            self.model_config = self.config['model_config']
            self.model_config.max_detection_points = 1000
            self.model_config.max_det_per_image = 20
            pretrained_backbone = True
            pretrained = False

            self.model = EfficientDet(self.model_config, pretrained_backbone=pretrained_backbone)

            from timm.utils import ModelEmaV2
            self.model_ema = ModelEmaV2(self.model, decay=self.model_ema_decay) if self.model_ema_decay else None

            # reset model head if num_classes doesn't match configs
            if pretrained:
                load_pretrained(self.model, self.model_config.url)            

            if self.config['num_classes'] is not None and self.config['num_classes'] != self.model_config.num_classes:
                self.model.reset_head(num_classes=self.config['num_classes'])

            # load an argument specified training checkpoint
            checkpoint_path = None
            if checkpoint_path:
                load_checkpoint(self.model, checkpoint_path, use_ema=False)

            self.loss_fn = DetectionLoss(self.model_config)

        else:
            raise NotImplementedError
        
        """ img = torch.randn((2,3,640,640)).to('cuda:0')
        self.model = self.model.to('cuda:0')
        # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        # [num_target_boxes, 4], center_x, center_y, width, height --> x0, y0, x1, y1
        box_low = torch.randn((1,2)).to('cuda:0')
        distances = torch.tensor([[1,1]]).to('cuda:0')
        box = torch.cat([box_low, distances], dim=1)
        boxes = torch.cat(4*[box], dim=0)
        labels = [{"class_labels": torch.tensor([1,1,1,1]).to('cuda:0'), "boxes": boxes}]*2
        pred = self.model(img, labels=labels)
        hihi = 3 """


    def training_step(self, batch, batch_idx: int):

        if self.arch in ['FasterRCNN', 'FasterRCNN_custom']:
            img, labels_b, bboxes_b, numAgentsIds_b = batch

            images = list(image for image in img)
            targets = []
            for i in range(len(img)):
                d = {}
                d['boxes'] = bboxes_b[i]
                d['labels'] = torch.ones_like(labels_b[i])
                targets.append(d)

            losses = self.model(images, targets) if self.arch == 'FasterRCNN' else self.model(images, numAgentsIds_b, targets)
            train_loss = sum([losses[k] for k in losses])
        
        elif self.arch == 'Detr':
            img, labels_b, bboxes_b, numAgentsIds_b = batch

            target = [{"class_labels": labels, "boxes": bboxes} for labels, bboxes in zip(labels_b, bboxes_b)]
            prediction = self.model(img, labels=target)
            train_loss = prediction.loss

            self.internal_log({'train_loss': train_loss, 'loss_ce': prediction.loss_dict['loss_ce'], 'loss_bbox': prediction.loss_dict['loss_bbox'], 'loss_giou': prediction.loss_dict['loss_giou']}, stage='train')

            # prediction = self.model(img, labels=None)
            # train_loss = compute_loss(self.model, target, prediction.logits, prediction.pred_boxes, prediction.auxiliary_outputs)

        elif self.arch == 'EfficientDet':
            img, target, numAgentsIds_b = batch

            class_out, box_out = self.model(img)
            ###########################
            train_loss, class_loss, box_loss = self.loss_fn(
                class_out,
                box_out,
                target
            )
            ###########################

        elif self.arch == 'YoloV5':
            img, target, numAgentsIds_b = batch

            train_out = self.model(img)
            train_loss, loss_items = self.loss_fn(train_out, target)

        if self.arch != 'Detr': self.internal_log({'train_loss': train_loss}, stage='train')
        self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
        
        return {'loss' : train_loss}


    def validation_step(self, batch, batch_idx: int) -> None:

        if self.arch in ['FasterRCNN', 'FasterRCNN_custom']:
            img, labels_b, bboxes_b, numAgentsIds_b = batch

            images = list(image for image in img)
            targets = []
            for i in range(len(img)):
                d = {}
                d['boxes'] = bboxes_b[i]
                d['labels'] = labels_b[i]
                targets.append(d)
            
            losses, prediction = self.model(images, targets) if self.arch == 'FasterRCNN' else self.model(images, numAgentsIds_b, targets)
            val_loss = sum([losses[k] for k in losses])
            
            # prediction = self.model(images, numAgentsIds_b, targets)
            pred_boxes, confidences, pred_labels = [],[],[]
            for pred in prediction:
                # selected_ids = torch.argwhere(pred["labels"] == 1).squeeze()
                b = pred["boxes"]#[selected_ids]
                s = pred["scores"]#[selected_ids]
                l = pred["labels"]#[selected_ids]
                if s.ndim == 0:
                    b, s, l = b.unsqueeze(0), s.unsqueeze(0), l.unsqueeze(0)
                pred_boxes.append(b)
                confidences.append(s)
                pred_labels.append(l)
            true_labels = labels_b
            true_boxes = bboxes_b
            map = metrics(true_boxes, true_labels, pred_boxes, pred_labels, confidences)

        elif self.arch == 'Detr':
            img, labels, bboxes, numAgentsIds_b = batch

            target = [{"class_labels": labels, "boxes": bboxes} for labels, bboxes in zip(labels, bboxes)]
            prediction = self.model(img, labels=target)
            val_loss = prediction.loss

            # prediction = self.model(img, labels=None)
            # val_loss = compute_loss(self.model, target, prediction.logits, prediction.pred_boxes, prediction.auxiliary_outputs)

            # from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=BiwSmd2i-Wkf
            # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            # keep = probas.max(-1).values > 0.7

            pred_boxes, pred_labels, confidences = [], [], []
            for i in range(prediction.logits.size(0)):
                scores = torch.max(prediction.logits[i], dim=-1).values
                argmaxes = torch.argmax(prediction.logits[i], dim=-1)
                # select only boxes with class 0 (class 1 == 'no-object')
                # argmaxes = torch.ones_like(argmaxes)
                box_detection_indices = torch.argwhere(argmaxes == 0).squeeze()
                
                selected_boxes = prediction.pred_boxes[i][box_detection_indices]
                selected_labels = argmaxes[box_detection_indices]
                selected_scores = scores[box_detection_indices]
                if selected_scores.ndim==0:
                    selected_boxes, selected_labels, selected_scores = selected_boxes.unsqueeze(0), selected_labels.unsqueeze(0), selected_scores.unsqueeze(0)
                # print(f'selection length: {selected_scores.size(0)}')

                pred_boxes.append(xywhn2xyxy(selected_boxes, self.img_max_height, self.img_max_width))
                confidences.append(torch.softmax(selected_scores, 0))
                pred_labels.append(selected_labels.long())
            
            true_labels = labels
            true_boxes = [xywhn2xyxy(box, self.img_max_height, self.img_max_width) for box in bboxes]

            map = metrics(true_boxes, true_labels, pred_boxes, pred_labels, confidences)

            self.internal_log({'val_loss': val_loss, 'loss_ce': prediction.loss_dict['loss_ce'], 'loss_bbox': prediction.loss_dict['loss_bbox'], 'loss_giou': prediction.loss_dict['loss_giou'], 'AP': map}, stage='val')

        elif self.arch == 'EfficientDet':
            img, target, numAgentsIds_b = batch
            labels_b, bboxes_b = target['gt_targets']

            ###########################
            class_out, box_out = self.model(img)

            val_loss, class_loss, box_loss = self.loss_fn(
                class_out,
                box_out,
                target
            )
            
            predictions = self.model.predict(class_out, box_out) 
           
            # trick for now: only take top K=target length prediction
            pred_boxes, pred_labels, confidences = [], [], []
            for i in range(predictions.size(0)):
                p = predictions[i]
                # pred_boxes.append(xyxyn2xyxy(p[:, :4], self.img_max_height, self.img_max_width))
                selected_ids = torch.argwhere(p[:, 5] == 1).squeeze()
                pred_boxes.append(p[:, :4][selected_ids])
                confidences.append(p[:, 4:5][selected_ids].squeeze())
                pred_labels.append(p[:, 5][selected_ids].long()) # in EffDet/PyTorch 0 is background

            true_labels, true_boxes = target['gt_targets']

            map = metrics(true_boxes, true_labels, pred_boxes, pred_labels, confidences) # confidences of 0.1 or 0.95 doesnt make a difference, 
            h = 2
            # map = torch.FloatTensor([0.5])
            ###########################

        elif self.arch == 'YoloV5':
            img, target, numAgentsIds_b = batch

            predictions, val_out = self.model(img)
            val_loss, val_items = self.loss_fn(val_out, target)

            # NMS
            single_cls = True
            # raise ValueError('metrics calculation throws error when in torchmetrics.detection.mean_ap._input_validator for one value targets that dont have dims (line 182).')

            predictions = self.model.nms(predictions,
                conf_thres=0.0,
                iou_thres=0.5,
                labels=(),
                multi_label=False,
                agnostic=single_cls,
                max_det=20)
            
            pred_boxes, pred_labels, confidences = [], [], []
            for p in predictions:
                p_box = xywhn2xyxy(p[:,:4], self.img_max_height, self.img_max_width)
                pred_boxes.append(p_box)
                confidences.append(p[:, 4:5].squeeze())
                pred_labels.append(p[:, 5:6].long().squeeze())

            true_labels, true_boxes = [], []
            unique_ids = target[:, 0].unique()
            for id_ in unique_ids:
                mask = target[:, 0] == id_
                labels_i = target[mask, 1].long()
                boxes_i = target[mask, 2:6]
                boxes_i = xywhn2xyxy(boxes_i, self.img_max_height, self.img_max_width)
                true_boxes.append(boxes_i)
                true_labels.append(labels_i)

            map = metrics(true_boxes, true_labels, pred_boxes, pred_labels, confidences)

        if self.arch != 'Detr': self.internal_log({'val_loss': val_loss, 'AP': map}, stage='val')
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {'val_loss': val_loss}


    def internal_log(self, losses_it, stage):
        if self.trainer.state.stage == 'sanity_check': return

        losses_logger = self.train_losses if stage=='train' else self.val_losses

        for key, val in losses_it.items():
            if key not in losses_logger:
                losses_logger.update({key: [val]})
            else:
                losses_logger[key].append(val)


    def configure_optimizers(self):
        
        if self.customized_optim:
            
            if self.arch == 'EfficientDet':
                from timm.scheduler import CosineLRScheduler
                decay_params, no_decay_params = [], []
                weight_decay = 4e-5
                new_lr = 0.0042
                if self.learning_rate != new_lr:
                    self.learning_rate = new_lr
                    print(f'WARNING: Learning rate is different from custom. Changing it to {self.learning_rate}')
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
                opt = Adam([{'params': no_decay_params, 'lr': self.learning_rate}, {'params': decay_params, 'lr': self.learning_rate, 'weight_decay': weight_decay}])
                
                t_initial = 100

                warmup_args = dict(
                    warmup_lr_init = 0.0001,
                    warmup_t = 1,
                    warmup_prefix = False,
                )

                noise_range = [n * t_initial for n in [0.4, 0.9]]
                noise_args = dict(
                    noise_range_t=noise_range,
                    noise_pct=0.67,
                    noise_std=1.0,
                    noise_seed=42,
                )

                cycle_args = dict(
                    cycle_mul = 1.0,
                    cycle_decay = 0.1,
                    cycle_limit = 1,
                )

                sch = CosineLRScheduler(
                    opt,
                    t_initial=t_initial,
                    lr_min=1e-05,
                    t_in_epochs=True,
                    **cycle_args,
                    **warmup_args,
                    **noise_args,
                    k_decay=1.0,
                )

                self.lr_scheduler_step = self.custom_step_EffDet

                return [opt], [{"scheduler": sch, "interval": "epoch"}] 

            elif self.arch == 'YoloV5':
                weight_decay = 4e-5
                new_lr = 0.01
                momentum = 0.937
                warmup_momentum = 0.8

                g = [], [], []  # optimizer parameter groups
                if self.learning_rate != new_lr:
                    self.learning_rate = new_lr
                    print(f'WARNING: Learning rate is different from custom. Changing it to {self.learning_rate}')
                
                bn = tuple(v for k, v in torch.nn.__dict__.items() if 'Norm' in k)
                for v in self.model.modules():
                    for p_name, p in v.named_parameters(recurse=0):
                        if p_name == 'bias':  # bias (no decay)
                            g[2].append(p)
                        elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                            g[1].append(p)
                        else:
                            g[0].append(p)  # weight (with decay)
                
                opt = torch.optim.SGD(g[2], lr=self.learning_rate, momentum=momentum, nesterov=True)
                opt.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
                opt.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)

                # lf = lambda x: (1 - x / last_epoch) * (1.0 - lrf) + lrf
                def lambdaLR_with_warmup(epoch):
                    warmup_epochs = 1
                    lrf = 0.01
                    last_epoch = 100

                    if epoch < warmup_epochs:
                        warmup_bias_lr = 0.1
                        lr_scale = [warmup_bias_lr, 0.0, 0.0]
                    else:
                        lr_scale = (1 - epoch / last_epoch) * (1.0 - lrf) + lrf
                    return lr_scale

                sch = LambdaLR(opt, lambdaLR_with_warmup, verbose=True)

                return [opt], [sch]

            elif self.arch == 'Detr':
                param_dicts = [
                    {"params": [p for n, p in self.named_parameters() if "backbone" not in n]},
                    {"params": [p for n, p in self.named_parameters() if "backbone" in n], "lr": 1e-5}
                ]
                opt = Adam(param_dicts, lr=self.learning_rate)
                sch = ReduceLROnPlateau(opt, factor=self.lr_sch_gamma4redOnPlat_and_stepLR, patience=self.lr_sch_patience4redOnPlat, verbose=True)
                # Because of a weird issue with ReduceLROnPlateau, the monitored value needs to be returned... See https://github.com/PyTorchLightning/pytorch-lightning/issues/4454
                # if self.mode == 'evac': raise NotImplementedError('how to implement here for two optimizers')
                return {
                    'optimizer': opt,
                    'lr_scheduler': sch,
                    'monitor': 'val_loss'
                }
            else:
                raise NotImplementedError()
                
        if self.weight_decay not in [None, 0.0]:
            # dont apply weight decay for layer norms https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348 
            # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/7
            decay_params = []
            no_decay_params = []
            for name, param in self.named_parameters():
                if 'weight' in name and 'evac' in name:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
                
            # dec_k = list(decay.keys())
            # no_dec_k = list(no_decay.keys())
            if self.opt == 'Adam':
                opt = Adam([{'params': no_decay_params, 'lr': self.learning_rate}, {'params': decay_params, 'lr': self.learning_rate, 'weight_decay': self.weight_decay}])
            elif self.opt == 'AdamW':
                opt = AdamW([{'params': no_decay_params, 'lr': self.learning_rate}, {'params': decay_params, 'lr': self.learning_rate, 'weight_decay': self.weight_decay}])
        else:
            if self.opt == 'Adam':
                opt = Adam(self.model.parameters(), lr = self.learning_rate)
            elif self.opt == 'AdamW':
                opt = AdamW(self.model.parameters(), lr = self.learning_rate)
            
        # opt = Adam(self.parameters(), lr = self.learning_rate)
        
        # opt = Adam(list(self.image_head.parameters()) + list(self.backbone.parameters()) + list(self.evac_head.parameters()), lr = self.learning_rate)
        # if self.mode == 'evac':
            # opt_evac = Adam(self.evac_head.parameters(), lr = self.learning_rate)
        
        if self.lr_scheduler == CosineAnnealingLR.__name__:
            sch = CosineAnnealingLR(opt, T_max = self.lr_sch_step_size4cosAnneal, verbose=True)
            # if self.mode == 'evac': sch_evac = CosineAnnealingLR(opt_evac, T_max = self.lr_sch_step_size)
        elif self.lr_scheduler == StepLR.__name__:
            sch = StepLR(opt, step_size=self.lr_sch_step_size4lr_step, gamma=self.lr_sch_gamma4redOnPlat_and_stepLR, verbose=True)
            # if self.mode == 'evac': sch_evac = StepLR(opt_evac, step_size=self.lr_sch_step_size, gamma=self.lr_sch_gamma)
        elif self.lr_scheduler == ExponentialLR.__name__:
            sch = ExponentialLR(opt, gamma=self.lr_sch_gamma4expLR, verbose=True)
            # if self.mode == 'evac': sch_evac = ExponentialLR(opt_evac, gamma=0.9)
        elif self.lr_scheduler == ReduceLROnPlateau.__name__:
            sch = ReduceLROnPlateau(opt, factor=self.lr_sch_gamma4redOnPlat_and_stepLR, patience=self.lr_sch_patience4redOnPlat, verbose=True)
            # Because of a weird issue with ReduceLROnPlateau, the monitored value needs to be returned... See https://github.com/PyTorchLightning/pytorch-lightning/issues/4454
            # if self.mode == 'evac': raise NotImplementedError('how to implement here for two optimizers')
            return {
            'optimizer': opt,
            'lr_scheduler': sch,
            'monitor': 'val_loss'
            }
        else:
            raise NotImplementedError('Scheduler has not been implemented yet!')

        optimizers = [opt]
        schedulers = [sch]
        # if self.mode == 'evac':
            # optimizers.append(opt_evac)
            # schedulers.append(sch_evac)
        return optimizers, schedulers


    def on_before_zero_grad(self, *args, **kwargs):
        if self.model_ema:
            self.model_ema.update(self.model)
        # total_norm = 0
        # for p in self.model.parameters():
        #     if p.grad is None:
        #         continue
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5 if total_norm != 0 else 0.0
        # print(f'grad norm: {total_norm}')
    
    # For customized schedulers
    def custom_step_EffDet(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value


    def on_fit_start(self) -> None:

        # return super().on_fit_start()
        print(f"\nFREEZE STRATEGY INITIALIZED IN EPOCH {self.current_epoch}\n")
        module_list = list(self.model._modules.keys())
        for param in self.parameters():
            param.requires_grad = True
        # for key, module in self.model._modules.items():
        #     if key == 'auxiliary_head': continue
        #     if key == 'model':
        #         for p in module.parameters():
        #             p.requires_grad = False
        #     else:
        #         for p in module.parameters():
        #             p.requires_grad = True
        return super().on_fit_start()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 3:
            print(f'\nFreezing model at epoch {self.current_epoch}\n')
            for key, module in self.model._modules.items():
                if key == 'model':
                    for p in module.parameters():
                        p.requires_grad = False
            # for param in self.parameters():
            #     param.requires_grad = True

        # if self.current_epoch == 8:
        #     print(f'\n\nUnfreezing all parameters in epoch {self.current_epoch}...')
        #     for param in self.parameters():
        #         param.requires_grad = True

        if self.trainer.state.stage in ['sanity_check']: return super().on_epoch_end()
        
        if self.current_epoch > 0: 
            self.print_logs()
    

    def print_logs(self):
        # Training Logs
        for key, val in self.train_losses.items():
            if key not in self.train_losses_per_epoch:
                mean = torch.as_tensor(val).nanmean()
                self.train_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.train_losses_per_epoch[key].append(torch.as_tensor(val).nanmean().item())

        # Validation logs
        for key, val in self.val_losses.items():
            if key not in self.val_losses_per_epoch:
                mean = torch.as_tensor(val).nanmean()
                self.val_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.val_losses_per_epoch[key].append(torch.as_tensor(val).nanmean().item())

        # Reset
        self.train_losses = {}
        self.val_losses = {}
        
        print('\nTRAINING RESULT:')
        train_string = f'Epoch\t'
        train_vals = [val for val in self.train_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.train_losses_per_epoch.keys())):
            if id_k == 0:
                train_string += key+':'
            else:
                train_string += '\t\t' + key+':'
        for i_epoch in range(len(train_vals[0])):
            for i_loss in range(len(train_vals)):
                if i_loss == 0:
                    train_string += f'\n{i_epoch}:\t{Decimal(train_vals[i_loss][i_epoch]):.5e}'
                    # print(f"{Decimal('0.0000000201452342000'):.8e}")
                else:
                    train_string += f'\t\t{Decimal(train_vals[i_loss][i_epoch]):.5e}'
        print(train_string) 


        print('\nVALIDATION RESULT:')
        val_string = f'Epoch\t'
        val_vals = [val for val in self.val_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.val_losses_per_epoch.keys())):
            if id_k == 0:
                val_string += key+':'
            else:
                val_string += '\t\t' + key+':'
        for i_epoch in range(len(val_vals[0])):
            for i_loss in range(len(val_vals)):
                if i_loss == 0:
                    # val_string += f'\n{i_epoch}:\t{val_vals[i_loss][i_epoch]:.5f}'
                    val_string += f'\n{i_epoch}:\t{Decimal(val_vals[i_loss][i_epoch]):.5e}'
                else:
                    # val_string += f'\t\t\t{val_vals[i_loss][i_epoch]:.5f}'
                    val_string += f'\t\t{Decimal(val_vals[i_loss][i_epoch]):.5e}'
        print(val_string) 
        
    
def metrics(true_boxes, true_labels, pred_boxes, pred_labels, confidences):
    preds, targets = [], []
    # should iterate over each image
    for pboxes, plabels, confs in zip(pred_boxes, pred_labels, confidences):
        preds.append({
            "boxes": pboxes, # boxes / image
            "labels": plabels,
            "scores": confs
        })
    for tboxes, tlabels in zip(true_boxes, true_labels):
        targets.append({
            "boxes": tboxes, # boxes / image
            "labels": tlabels
        })

    mapInstance = MeanAveragePrecision()
    results = mapInstance.forward(preds, targets)
    return results.map






class FasterRCNN_custom(FasterRCNN):
    
    def __init__(self, backbone, num_classes):
        super().__init__(backbone, num_classes)

        self.rpn.head = RPNHead_custom()
        self.roi_heads.box_head = TwoMLPHead_custom(256 * 7 ** 2, 1024)
        self.forward = self.forward
        self.rpn.forward = self.forward_rpn
        self.roi_heads.forward = self.forward_roi
        self.metrics_mode = False
        
        self.simulator_embeddings = torch.nn.Embedding(3, 5*512)


    def forward(self, images, numAgentIds, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if targets is None:
            raise ValueError("In training mode, targets should be passed")
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
            else:
                raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # embed simulator settings
        ag_embeddings = self.simulator_embeddings(numAgentIds)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, ag_embeddings, targets) # features: [0]=256, 144, 336 // [1]=256, 72, 168 // [2]= 256, 36, 84 // [3]=256, 18, 42 from [XXXX]=3,1360, 3200
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, ag_embeddings, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            # return self.eager_outputs(losses, detections)
            # return detections if self.metrics_mode else losses
            if self.training:
                return losses
            return losses, detections


    def forward_roi(
        self,
        features, 
        proposals,
        image_shapes, 
        ag_embeddings,
        targets=None, 
    ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        # if targets is not None and not self.metrics_mode:
        if targets is not None and self.roi_heads.training:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                if self.roi_heads.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, "target keypoints must of float type"


        # get the losses
        proposals_w_gt, matched_idxs, labels, regression_targets = self.roi_heads.select_training_samples(proposals, targets)
        box_features = self.roi_heads.box_roi_pool(features, proposals_w_gt, image_shapes)
        box_features = self.roi_heads.box_head(box_features, ag_embeddings)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        result = []
        losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}


        # get the detections
        if not self.training:
            labels = None
            regression_targets = None
            matched_idxs = None
            box_features = self.roi_heads.box_roi_pool(features, proposals, image_shapes)
            box_features = self.roi_heads.box_head(box_features, ag_embeddings)
            class_logits, box_regression = self.roi_heads.box_predictor(box_features)

            boxes, scores, labels = self.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.roi_heads.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            # if self.roi_heads.training:
            assert matched_idxs is not None
            # during training, only focus on positive boxes
            num_images = len(proposals)
            mask_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                mask_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
            # else:
            #     pos_matched_idxs = None

            if self.roi_heads.mask_roi_pool is not None:
                mask_features = self.roi_heads.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.roi_heads.mask_head(mask_features)
                mask_logits = self.roi_heads.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            # if self.roi_heads.training:
            assert targets is not None
            assert pos_matched_idxs is not None
            assert mask_logits is not None

            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
            loss_mask = {"loss_mask": rcnn_loss_mask}
            # else:
                # labels = [r["labels"] for r in result]
                # masks_probs = maskrcnn_inference(mask_logits, labels)
                # for mask_prob, r in zip(masks_probs, result):
                #     r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.roi_heads.keypoint_roi_pool is not None
            and self.roi_heads.keypoint_head is not None
            and self.roi_heads.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            # if self.roi_heads.training:
                # during training, only focus on positive boxes
            num_images = len(proposals)
            keypoint_proposals = []
            pos_matched_idxs = []
            assert matched_idxs is not None
            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                keypoint_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
            # else:
            #     pos_matched_idxs = None

            keypoint_features = self.roi_heads.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.roi_heads.keypoint_head(keypoint_features)
            keypoint_logits = self.roi_heads.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            # if self.roi_heads.training:
            assert targets is not None
            assert pos_matched_idxs is not None

            gt_keypoints = [t["keypoints"] for t in targets]
            rcnn_loss_keypoint = keypointrcnn_loss(
                keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
            )
            loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            # else:
            #     assert keypoint_logits is not None
            #     assert keypoint_proposals is not None

            #     keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
            #     for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
            #         r["keypoints"] = keypoint_prob
            #         r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses
    

    def forward_rpn(
        self,
        images,
        features,
        ag_embeddings,
        targets = None,
    ):
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        # objectness, pred_bbox_deltas = self.rpn.head(features)
        objectness, pred_bbox_deltas = self.rpn.head(features, ag_embeddings)
        anchors = self.rpn.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
      
        assert targets is not None
        labels, matched_gt_boxes = self.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses
    

class RPNHead_custom(torch.nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels: int = 256, num_anchors: int = 3) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = torch.nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = torch.nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
            torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

        # customized
        self.sim_fc = torch.nn.ModuleList([torch.nn.Linear(512, 21*9) for i in range(5)])
        self.dim_red = torch.nn.ModuleList([torch.nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1) for i in range(5)])
        self.dim_inc = torch.nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.size_up = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=1, stride=2, padding=0, output_padding=1)


    def forward(self, x: List[Tensor], ag_embeddings) -> Tuple[List[Tensor], List[Tensor]]:
        ag_embeddings = torch.split(ag_embeddings, 512, dim=-1)

        logits = []
        bbox_reg = []
        for i_t, feature in enumerate(x):
            # new:
            emb = self.sim_fc[i_t](ag_embeddings[i_t]).view(-1, 1, 9, 21)
            feat_red = self.dim_red[i_t](feature)
            assert feat_red.size()[2] // 9 == feat_red.size()[3] // 21
            size_factor = feat_red.size()[2] // 9
            feat_red = torch.nn.MaxPool2d(size_factor)(feat_red)
            feat_out = self.dim_inc(emb + feat_red)
            for _r in range(len(x) - i_t - 1):
                feat_out = self.size_up(feat_out)
            feat_out = F.relu(feat_out)

            # old:
            t = F.relu(self.conv(feature))
            t = t + feat_out # also new
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
    

class TwoMLPHead_custom(torch.nn.Module):

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = torch.nn.Linear(in_channels, representation_size)
        self.fc7 = torch.nn.Linear(representation_size, representation_size)

    def forward(self, x, ag_embeddings):
        ag_embeddings = torch.split(ag_embeddings, 512, dim=-1)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x