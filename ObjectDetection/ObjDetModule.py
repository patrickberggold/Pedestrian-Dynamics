import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau, LambdaLR
from torch.optim import Adam, AdamW, SGD
import pytorch_lightning as pl
from helper import SEP, xywhn2xyxy, xywh2xyxy, xyxyn2xyxy
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import BinaryPrecisionRecallCurve
from decimal import Decimal
import numpy as np
import warnings
from sklearn.metrics import average_precision_score, auc


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
        self.batch_size = config['batch_size']
        
        # assert self.mode in ['grayscale', 'evac_only', 'class_movie', 'density_reg', 'density_class', 'denseClass_wEvac'], 'Unknown mode setting!'
        assert self.arch in ['Detr', 'Detr_custom', 'FasterRCNN', 'FasterRCNN_custom', 'EfficientDet', 'YoloV5'], 'Unknown arch setting!'

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
        self.tversky_weights = None
        self.post_inference_call = True
        self.num_use_cases = 2
        self.pred_boxes, self.pred_labels, self.confidences = [], [], []
        self.true_boxes, self.true_labels = [], []

        self.img_max_width, self.img_max_height = config['img_max_size']
        self.save_results = config['save_results']
        self.txt_path = config['store_path'] if self.save_results and 'store_path' in config else None

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

        elif self.arch == "Detr_custom":
            from models.Detr_custom import Detr_custom
            self.model = Detr_custom.from_pretrained("facebook/detr-resnet-50")
            self.model.update_model(self.config)
        
        elif self.arch == 'FasterRCNN':

            # from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
            # self.model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=self.config['num_classes'] + 1, pretrained_backbone=pretrained_backbone, trainable_backbone_layers=None)
            from helper import validate_faster_rcnn_with_loss
            from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
            from torchvision.models.resnet import resnet50
            from torchvision.models.detection import FasterRCNN
            from torchvision.ops import misc as misc_nn_ops

            pretrained_backbone = True
            pretrained=False
            trainable_backbone_layers=None
            num_classes=self.config['num_classes'] + 1

            trainable_backbone_layers = _validate_trainable_layers(
                pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
            )

            backbone = resnet50(pretrained=pretrained_backbone, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
            backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
            self.model = FasterRCNN(backbone, num_classes, box_score_thresh=0.5, box_nms_thresh=0.0)

            validate_faster_rcnn_with_loss(self.model)

        elif self.arch == 'FasterRCNN_custom':
            from models.FasterRCNN_custom import FasterRCNN_custom
            self.model = FasterRCNN_custom(backbone=None, num_classes=self.config['num_classes'] + 1, config=self.config)

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
        
        elif self.arch in ['Detr', 'Detr_custom']:
            img, labels, bboxes, numAgentsIds = batch

            target = [{"class_labels": l, "boxes": b} for l, b in zip(labels, bboxes)]
            prediction = self.model(img, labels=target) if self.arch == 'Detr' else self.model(img, numAgentsIds, labels=target)
            train_loss = prediction.loss

            # self.internal_log({'train_loss': train_loss, 'loss_ce': prediction.loss_dict['loss_ce'], 'loss_bbox': prediction.loss_dict['loss_bbox'], 'loss_giou': prediction.loss_dict['loss_giou']}, stage='train')
            # self.internal_log({'train_loss': train_loss}, stage='train')

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

        self.internal_log({'train_loss': train_loss}, stage='train')
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
            for pred in prediction:
                # selected_ids = torch.argwhere(pred["labels"] == 1).squeeze()
                b = pred["boxes"]#[selected_ids]
                s = pred["scores"]#[selected_ids]
                l = pred["labels"]#[selected_ids]
                if s.ndim == 0:
                    b, s, l = b.unsqueeze(0), s.unsqueeze(0), l.unsqueeze(0)
                self.pred_boxes += [b]
                self.confidences += [s]
                self.pred_labels += [l]
            self.true_labels += labels_b
            self.true_boxes += bboxes_b
            # map, f1_score _ = metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences)

        elif self.arch in ['Detr', 'Detr_custom']:
            img, labels, bboxes, numAgentsIds = batch

            target = [{"class_labels": l, "boxes": b} for l, b in zip(labels, bboxes)]
            prediction = self.model(img, labels=target) if self.arch == 'Detr' else self.model(img, numAgentsIds, labels=target)
            val_loss = prediction.loss

            # prediction = self.model(img, labels=None)
            # val_loss = compute_loss(self.model, target, prediction.logits, prediction.pred_boxes, prediction.auxiliary_outputs)

            # from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=BiwSmd2i-Wkf
            # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            # keep = probas.max(-1).values > 0.7

            for i in range(prediction.logits.size(0)):
                scores = torch.softmax(prediction.logits[i].detach(), dim=-1)
                argmaxes = torch.argmax(scores, dim=-1)
                # box_detection_indices_o = torch.argwhere(argmaxes == 0).squeeze() # select only boxes with class 0 (class 1 == 'no-object')
                box_detection_indices = torch.argwhere((argmaxes == 0) & (scores[:, 0] > 0.7)).squeeze() # select only boxes with class 0 (class 1 == 'no-object')
                selected_scores = scores[box_detection_indices, 0]
                assert torch.all(selected_scores > scores[box_detection_indices,1])
                box_proposals_per_batch = prediction.pred_boxes[i].detach()
                selected_boxes = box_proposals_per_batch[box_detection_indices]
                selected_labels = argmaxes[box_detection_indices].detach()

                # for score predictions consisting of a single number
                if selected_scores.ndim==0:
                    selected_boxes, selected_labels, selected_scores = selected_boxes.unsqueeze(0), selected_labels.unsqueeze(0), selected_scores.unsqueeze(0)
                # print(f'selection length: {selected_scores.size(0)}')

                self.pred_boxes += [xywhn2xyxy(selected_boxes, img.size(3), img.size(2))] # 2048, 789
                self.confidences += [selected_scores]
                self.pred_labels += [selected_labels.long()]
            
            self.true_labels += labels
            self.true_boxes += [xywhn2xyxy(box, img.size(3), img.size(2)) for box in bboxes]

            # map, f1_score, _ = metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences)

            # self.internal_log({'val_loss': val_loss, 'AP': map}, stage='val')

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

            map, f1_score, _ = metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences) # confidences of 0.1 or 0.95 doesnt make a difference, 
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
                p_box = xywhn2xyxy(p[:,:4], self.img_max_width, self.img_max_height)
                pred_boxes.append(p_box)
                confidences.append(p[:, 4:5].squeeze())
                pred_labels.append(p[:, 5:6].long().squeeze())

            true_labels, true_boxes = [], []
            unique_ids = target[:, 0].unique()
            for id_ in unique_ids:
                mask = target[:, 0] == id_
                labels_i = target[mask, 1].long()
                boxes_i = target[mask, 2:6]
                boxes_i = xywhn2xyxy(boxes_i, self.img_max_width, self.img_max_height)
                true_boxes.append(boxes_i)
                true_labels.append(labels_i)

            map, f1_score, _ = metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences)

        self.internal_log({'val_loss': val_loss}, stage='val')
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {'val_loss': val_loss}


    def test_step(self, batch):
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
            
            # prediction = self.model(images, numAgentsIds_b, targets)
            for pred in prediction:
                # selected_ids = torch.argwhere(pred["labels"] == 1).squeeze()
                b = pred["boxes"]#[selected_ids]
                s = pred["scores"]#[selected_ids]
                l = pred["labels"]#[selected_ids]
                if s.ndim == 0:
                    b, s, l = b.unsqueeze(0), s.unsqueeze(0), l.unsqueeze(0)
                self.pred_boxes_test += [b]
                self.confidences_test += [s]
                self.pred_labels_test += [l]
            self.true_labels_test += labels_b
            self.true_boxes_test += bboxes_b
            # map, f1_score, _ = metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences)

        elif self.arch in ['Detr', 'Detr_custom']:
            img, labels, bboxes, numAgentsIds = batch

            target = [{"class_labels": l, "boxes": b} for l, b in zip(labels, bboxes)]
            prediction = self.model(img, labels=target) if self.arch == 'Detr' else self.model(img, numAgentsIds, labels=target)

            for i in range(prediction.logits.size(0)):
                scores = torch.softmax(prediction.logits[i].detach(), dim=-1)
                argmaxes = torch.argmax(scores, dim=-1)
                box_detection_indices = torch.argwhere(argmaxes == 0).squeeze() # select only boxes with class 0 (class 1 == 'no-object')
                selected_scores = scores[box_detection_indices, 0]
                assert torch.all(selected_scores > scores[box_detection_indices,1])
                box_proposals_per_batch = prediction.pred_boxes[i].detach()
                selected_boxes = box_proposals_per_batch[box_detection_indices]
                selected_labels = argmaxes[box_detection_indices].detach()

                if selected_scores.ndim==0:
                    selected_boxes, selected_labels, selected_scores = selected_boxes.unsqueeze(0), selected_labels.unsqueeze(0), selected_scores.unsqueeze(0)

                self.pred_boxes_test += [xywhn2xyxy(selected_boxes, img.size(3), img.size(2))] # 2048, 789
                self.confidences_test += [selected_scores]
                self.pred_labels_test += [selected_labels.long()]
            
            self.true_labels_test += labels
            self.true_boxes_test += [xywhn2xyxy(box, img.size(3), img.size(2)) for box in bboxes]

    
    def internal_log(self, losses_it, stage):
        if self.trainer.state.stage == 'sanity_check': return

        losses_logger = self.train_losses if stage=='train' else self.val_losses

        for key, val in losses_it.items():
            if isinstance(val, torch.Tensor): val = val.detach()
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

            elif self.arch in ['Detr', 'Detr_custom']:
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
        # if self.current_epoch == 0:
        #     print(f'\nFreezing model at epoch {self.current_epoch}\n')
        #     for key, module in self.model._modules.items():
        #         if key == 'model':
        #             for p in module.parameters():
        #                 p.requires_grad = False
            # for param in self.parameters():
            #     param.requires_grad = True

        # if self.current_epoch == 8:
        #     print(f'\n\nUnfreezing all parameters in epoch {self.current_epoch}...')
        #     for param in self.parameters():
        #         param.requires_grad = True

        if self.trainer.state.stage in ['sanity_check']: return super().on_train_epoch_end()
        
        if self.current_epoch > 0: 
            self.print_logs()
    

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.state.stage == 'sanity_check':
            map, f1_score, _ = metrics_sklearn(self.true_boxes, self.true_labels, self.pred_boxes, self.pred_labels, self.confidences)
            self.internal_log({'AP': map}, stage='val')
            self.true_boxes, self.true_labels, self.pred_boxes, self.pred_labels, self.confidences = [], [], [], [], []
            self.true_boxes_test, self.true_labels_test, self.pred_boxes_test, self.pred_labels_test, self.confidences_test = [], [], [], [], []
            if self.post_inference_call:
                # assert all parameters are in eval mode right now
                assert not self.model.training
                # call the test dataloader
                test_dataloader = self.trainer.datamodule.test_dataloader()
                for batch in test_dataloader:
                    batch_d = (batch[0].to(self.device), tuple([l.to(self.device) for l in batch[1]]), tuple([b.to(self.device) for b in batch[2]]), batch[3].to(self.device))
                    self.test_step(batch_d) 
                map_u9_old, f1_score_sbahn, stats_u9_old = metrics_sklearn(self.true_boxes_test[:3], self.true_labels_test[:3], self.pred_boxes_test[:3], self.pred_labels_test[:3], self.confidences_test[:3], verbose=True)
                map_u9_new, f1_score_u9_new, stats_u9_new = metrics_sklearn(self.true_boxes_test[3:6], self.true_labels_test[3:6], self.pred_boxes_test[3:6], self.pred_labels_test[3:6], self.confidences_test[3:6], verbose=True)
                self.internal_log({'AP U9 old': map_u9_old}, stage='val')
                self.internal_log({'AP U9 new': map_u9_new}, stage='val')
                if True: # map_u9_old > 0.9:
                    s_o_tps, s_o_fps, s_o_fns = stats_u9_old['tps'], stats_u9_old['fps'], stats_u9_old['fns']
                    print(f'[map_u9_old = {map_u9_old}]: tps={s_o_tps}, fps={s_o_fps}, fns={s_o_fns}')
                if True: #map_u9_new > 0.9:
                    s_n_tps, s_n_fps, s_n_fns = stats_u9_new['tps'], stats_u9_new['fps'], stats_u9_new['fns']
                    print(f'[map_u9_new = {map_u9_new}]: tps={s_n_tps}, fps={s_n_fps}, fns={s_n_fns}')
        return super().on_validation_epoch_end()


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
        
        # print('\nTRAINING RESULT:')
        train_string = f'TRAINING RESULT:\nEpoch\t'
        train_vals = [val for val in self.train_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.train_losses_per_epoch.keys())):
            if id_k == 0:
                train_string += key+':'
            else:
                train_string += '\t\t' + key+':'
        for i_epoch in range(len(train_vals[0])):
            for i_loss in range(len(train_vals)):
                if i_loss == 0:
                    train_string += f'\n{i_epoch}:\t{Decimal(train_vals[i_loss][i_epoch]):.3e}'
                    # print(f"{Decimal('0.0000000201452342000'):.8e}")
                else:
                    train_string += f'\t\t{Decimal(train_vals[i_loss][i_epoch]):.3e}'
        print('\n\n'+train_string) 


        # print('\nVALIDATION RESULT:')
        val_string = f'\nVALIDATION RESULT:\nEpoch\t'
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
                    val_string += f'\n{i_epoch}:\t{Decimal(val_vals[i_loss][i_epoch]):.3e}'
                else:
                    # val_string += f'\t\t\t{val_vals[i_loss][i_epoch]:.5f}'
                    val_string += f'\t\t{Decimal(val_vals[i_loss][i_epoch]):.3e}'
        print(val_string+'\n')

        if self.save_results and self.txt_path is not None:
            save_string = train_string+'\n\n'+val_string
            f = open(self.txt_path, 'w')
            f.write(f'Latest learning rate:{self.learning_rate}\n\n')
            f.write(save_string)
            f.close()


def metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences, return_curve = False, verbose = False):
    # for precision-recall-curve
    scores_list, gt_classes_list, statistics = [], [], {'tps': 0, 'tp_ious': [], 'fps': 0, 'fns': 0}
    iou_thresholds = [0.9] # [0.75] 0.9
    
    # iterate over each image
    for tboxes, tlabels, pboxes, plabels, confs in zip(true_boxes, true_labels, pred_boxes, pred_labels, confidences):    
        new_scores, new_gt_classes, new_statistics = get_scores_and_classes(tboxes, pboxes, confs, iou_thresholds)
        scores_list += new_scores
        gt_classes_list += new_gt_classes
        for key in statistics.keys():
            statistics[key] += new_statistics[key]

        assert len(scores_list) == len(gt_classes_list)
    
    if len(scores_list) > 0:
        precision, recall, thresholds = precision_recall_curve(gt_classes_list, scores_list, fns=statistics['fns'])
        average_precision = average_precision_score(gt_classes_list, scores_list) # -np.sum(np.diff(recall) * np.array(precision)[:-1]) 
        area_under_curve = auc(recall, precision) # based on np.trapz
    else:
        average_precision = 0.

    try:
        recall_n = statistics['tps']/(statistics['tps']+statistics['fns'])
        precision_n = statistics['tps']/(statistics['tps']+statistics['fps'])
        f1_score = 2 * (precision_n * recall_n) / (precision_n + recall_n)
    except ZeroDivisionError:
        f1_score = 0.

    if verbose:
        if average_precision > 0.5:
            print(f"[verbose call] average_precision={average_precision:.3e}, TPs: {statistics['tps']}, FPs: {statistics['tps']}, FNs: {statistics['fns']}")

    # f = open("curve_values_DETR_vanilla_imgAugm_94.11.txt", "w")
    # for idx, list_item in enumerate([precision, recall, thresholds]):
    #     f.write(f"__{idx}\n")
    #     for item in list_item:
    #         f.write(f"{item}\n")
    # f.close()

    if return_curve:
        # plt.plot(recall, precision)
        # plt.show()
        """ from torch import Tensor
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        fig, ax = plt.subplots() if ax is None else (None, ax)

        if isinstance(recall, Tensor) and isinstance(precision, Tensor) and recall.ndim == 1 and precision.ndim == 1:
            # label = f"AUC={score.item():0.3f}" if score is not None else None
            ax.plot(recall.detach().cpu(), precision.detach().cpu(), linestyle="-", linewidth=2, label=labels[0], color='blue')
            if label_names is not None:
                ax.set_xlabel(label_names[0])
                ax.set_ylabel(label_names[1])
            if label is not None or labels is not None:
                ax.legend()
        elif (isinstance(recall, list) and isinstance(precision, list)) or (
            isinstance(recall, Tensor) and isinstance(precision, Tensor) and recall.ndim == 2 and precision.ndim == 2
        ):
            for i, (x_, y_) in tqdm(enumerate(zip(recall, precision)), total=len(recall)):
                label = f"{legend_name}_{i}" if legend_name is not None else str(i)
                # label += f" AUC={score[i].item():0.3f}" if score is not None else ""
                ax.plot(x_, y_, label=label)
                ax.legend()
        else:
            raise ValueError(
                f"Unknown format for argument `x` and `y`. Expected either list or tensors but got {type(recall)} and {type(precision)}."
            )
        ax.grid(True)
        ax.set_title(plot_name)
        ax.set_ylim(0.0, 1.05) """
        from torchmetrics.utilities.plot import plot_curve
        fig, ax = plot_curve(
            (torch.tensor(recall), torch.tensor(precision)), score=average_precision, ax=None, label_names=("Recall", "Precision"), name='Precision-Recall-Curve'
        )
        return average_precision, f1_score, statistics, (fig, ax)
    
    return average_precision, f1_score, statistics


    
def metrics_torch(true_boxes, true_labels, pred_boxes, pred_labels, confidences, calc_stats = True, return_curve = False):
    preds, targets = [], []
    # for precision-recall-curve
    scores_list, gt_classes_list, statistics = [], [], {'tps': 0, 'tp_ious': [], 'fps': 0, 'fns': 0}
    iou_thresholds = [0.5]
    if return_curve: assert calc_stats, 'Statistics calculation must be turned on for PR curve!'

    # iterate over each image
    for tboxes, tlabels, pboxes, plabels, confs in zip(true_boxes, true_labels, pred_boxes, pred_labels, confidences):

        preds.append({
            "boxes": pboxes, # boxes / image
            "labels": plabels,
            "scores": confs
        })
        targets.append({
            "boxes": tboxes, # boxes / image
            "labels": tlabels
        })

        if calc_stats:
            new_scores, new_gt_classes, new_statistics = get_scores_and_classes(tboxes, pboxes, confs, iou_thresholds)
            scores_list += new_scores
            gt_classes_list += new_gt_classes
            for key in statistics.keys():
                statistics[key] += new_statistics[key]

            assert len(scores_list) == len(gt_classes_list)

    mapInstance = MeanAveragePrecision()
    results = mapInstance.forward(preds, targets)
    m_ap = results['map'] # this is likely over all APs from .5 to .95 (and all classes) --> 0.43 is actually quite good then...

    if return_curve:
        import matplotlib.pyplot as plt
        from torchmetrics.utilities.compute import _auc_compute_without_check
        from torchmetrics.utilities.plot import plot_curve
        bprc = BinaryPrecisionRecallCurve(thresholds=None)
        curve_computed = [item for item in bprc(torch.tensor(scores_list), torch.tensor(gt_classes_list))]
        score = _auc_compute_without_check(curve_computed[0], curve_computed[1], 1.0)
        curve_computed[1], curve_computed[0] = curve_computed[0], curve_computed[1] # torchmetrics plots recall-precision vice versa
        fig, ax = plot_curve(
            curve_computed, score=score, ax=None, label_names=("Recall", "Precision"), name='Precision-Recall-Curve'
        )
        # f = open("curve_values_FasterRCNN.txt", "w")
        # for idx, list_item in enumerate(curve_computed):
        #     f.write(f"__{idx}\n")
        #     for item in list_item:
        #         f.write(f"{item}\n")
        # f.close()
        return m_ap, statistics, fig

    return m_ap, statistics



def calc_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    box1 (numpy array): [x1, y1, x2, y2] coordinates of the first bounding box.
    box2 (numpy array): [x1, y1, x2, y2] coordinates of the second bounding box.

    Returns:
    float: IoU value.
    """
    # Calculate the coordinates of the intersection rectangle
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2_intersection - x1_intersection) * max(0, y2_intersection - y1_intersection)

    # Calculate the area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the Union area by adding the areas of both boxes and subtracting the intersection area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou_value = intersection_area / union_area

    return iou_value


def precision_recall_curve(
    y_true, probas_pred, fns=0, *, pos_label=1, sample_weight=None, drop_intermediate=False
):
    """
    source implementation from sklearn
    """
    classes = np.unique(y_true)
    assert (np.array_equal(classes, [0, 1]) or np.array_equal(classes, [0]) or np.array_equal(classes, [1])), f'Failure with given classes: {classes.tolist()}'
    
    from sklearn.metrics._ranking import _binary_clf_curve
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight
    )

    if drop_intermediate and len(fps) > 2:
        # Drop thresholds corresponding to points where true positives (tps)
        # do not change from the previous or subsequent point. This will keep
        # only the first and last point for each tps value. All points
        # with the same tps value have the same recall and thus x coordinate.
        # They appear as a vertical line on the plot.
        optimal_idxs = np.where(
            np.concatenate(
                [[True], np.logical_or(np.diff(tps[:-1]), np.diff(tps[1:])), [True]]
            )
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    ps = tps + fps
    # Initialize the result array with zeros to make sure that precision[ps == 0]
    # does not contain uninitialized values.
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / (tps[-1] + fns)

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]


def get_scores_and_classes(tboxes, pboxes, confs, iou_thresholds):

    # maskUtils.iou(dt,gt), dt/gt = List[List[4xfloat]]
    # import pycocotools.mask as maskUtils

    new_scores, new_gt_classes = [], []
    tps, fps, fns = 0, 0, 0
    ious = np.zeros((len(pboxes), len(tboxes)), dtype=np.float32)
    scores = np.zeros((len(pboxes), len(tboxes)), dtype=np.float32)

    if tboxes.shape[0]==0:
        app_list = confs.tolist()
        new_scores += app_list
        new_gt_classes += [0]*len(app_list)
        assert len(new_scores) == len(new_gt_classes)
        return new_scores, new_gt_classes, {'tps': 0, 'tp_ious': [], 'fps': len(app_list), 'fns': 0}
    if pboxes.shape[0]==0:
        # new_gt_classes += [1]*tboxes.shape[0]
        # new_scores += [0.0]*tboxes.shape[0]
        assert len(new_scores) == len(new_gt_classes)
        return new_scores, new_gt_classes, {'tps': 0, 'tp_ious': [], 'fps': 0, 'fns': tboxes.shape[0]}
    
    for ip, p_box in enumerate(pboxes):
        for it, gt_box in enumerate(tboxes):
            iou_ = calc_iou(p_box, gt_box)
            # iou_c = maskUtils.iou([p_box.numpy().tolist()], [gt_box.numpy().tolist()])
            if iou_ > 0.:
                assert confs[ip] > 0.
                ious[ip, it] = iou_
                scores[ip, it] = confs[ip]
    
    # for testing
    # ious = np.hstack((np.zeros((len(ious), 3)), ious))
    # scores = np.hstack((2.5*np.ones((len(ious), 3)), scores))
    
    sum_over_ious = np.sum(ious, axis=0)
    no_match_gts = np.argwhere(sum_over_ious==0)
    if no_match_gts.shape[0] > 0:
        hi = 43
    neg_fn_mask = (sum_over_ious != 0)
    fns = no_match_gts.shape[0]

    # append the false negatives
    # new_scores += [0.0]*no_match_gts.shape[0]
    # new_gt_classes += [1]*no_match_gts.shape[0]
        
    # get the coordinates of the biggest IOUs for their corresponding scores
    # pbox_indices = np.argmax(ious, axis=0)
    # pbox_indices = np.column_stack((pbox_indices, np.arange(len(pbox_indices))))
    # pbox_indices = pbox_indices[neg_fn_mask]
    max_ious = np.max(ious, axis=0)

    valid_iou_coordinates = np.argwhere((max_ious == ious) & (max_ious > iou_thresholds[0]))
    
    if max_ious.shape[0] > valid_iou_coordinates.shape[0]:
        lll = 3
    
    scores_tps = scores[valid_iou_coordinates[:, 0], valid_iou_coordinates[:, 1]]
    ious_tps = ious[valid_iou_coordinates[:, 0], valid_iou_coordinates[:, 1]].tolist()
    # assign zeros after extraction
    scores[valid_iou_coordinates[:, 0], valid_iou_coordinates[:, 1]] = 0

    new_scores += [float(val) for val in scores_tps] # true positives
    new_gt_classes += [1]*scores_tps.shape[0]
    tps = scores_tps.shape[0]

    remaining_score_indizes = np.nonzero(scores)
    remaining_scores = scores[remaining_score_indizes].tolist()

    new_scores += remaining_scores
    new_gt_classes += [0]*len(remaining_scores)
    fps = len(remaining_scores)

    return new_scores, new_gt_classes, {'tps': tps, 'tp_ious': ious_tps, 'fps': fps, 'fns': fns}