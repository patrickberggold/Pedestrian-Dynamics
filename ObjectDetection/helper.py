import os
import torch
import platform
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import keypointrcnn_inference, keypointrcnn_loss, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference
import numpy as np

OPSYS = platform.system()
SEP = os.sep
PREFIX = '/mnt/c' if OPSYS == 'Linux' else 'C:'


def dir_maker(store_folder_path, description_log, config, train_config):
    if os.path.isdir(store_folder_path):
        print('Path already exists!')
        quit()
    else:
        os.mkdir(store_folder_path)
        with open(os.path.join(store_folder_path, 'description.txt'), 'w') as f:
            f.write(description_log)
            f.write("\n\nCONFIG: {\n")
            for k in config.keys():
                f.write("'{}':'{}'\n".format(k, str(config[k])))
            f.write("}")
            f.write("\n\nTRAIN_CONFIG: {\n")
            for k in config.keys():
                f.write("'{}':'{}'\n".format(k, str(config[k])))
            f.write("}\n\n")
        f.close()


# replace the roi and rpn forward functions to return the loss when validating (not the predictions)
def validate_faster_rcnn_with_loss(faster_rcnn_model: GeneralizedRCNN):

    # turn off metrics return for rpn
    def rpn_forward(images, features, targets):
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
        objectness, pred_bbox_deltas = faster_rcnn_model.rpn.head(features)
        anchors = faster_rcnn_model.rpn.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas) # after here
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = faster_rcnn_model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = faster_rcnn_model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}

        assert targets is not None
        labels, matched_gt_boxes = faster_rcnn_model.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = faster_rcnn_model.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = faster_rcnn_model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    
    # turn off metrics return for roi
    def roi_forward(features, proposals,  image_shapes, targets=None,):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                if faster_rcnn_model.roi_heads.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, "target keypoints must of float type"

        # get the losses
        proposals_w_gt, matched_idxs, labels, regression_targets = faster_rcnn_model.roi_heads.select_training_samples(proposals, targets)
        box_features = faster_rcnn_model.roi_heads.box_roi_pool(features, proposals_w_gt, image_shapes)
        box_features = faster_rcnn_model.roi_heads.box_head(box_features)
        class_logits, box_regression = faster_rcnn_model.roi_heads.box_predictor(box_features)

        result = []
        losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}


        # get the detections
        if not faster_rcnn_model.training:
            labels = None
            regression_targets = None
            matched_idxs = None
            box_features = faster_rcnn_model.roi_heads.box_roi_pool(features, proposals, image_shapes)
            box_features = faster_rcnn_model.roi_heads.box_head(box_features)
            class_logits, box_regression = faster_rcnn_model.roi_heads.box_predictor(box_features)

            boxes, scores, labels = faster_rcnn_model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        """ if True: # faster_rcnn_model.roi_heads.training:
            proposals, matched_idxs, labels, regression_targets = faster_rcnn_model.roi_heads.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = faster_rcnn_model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = faster_rcnn_model.roi_heads.box_head(box_features)
        class_logits, box_regression = faster_rcnn_model.roi_heads.box_predictor(box_features)

        result = []
        losses = {}
        if True: # faster_rcnn_model.roi_heads.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = faster_rcnn_model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if faster_rcnn_model.roi_heads.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if True: # faster_rcnn_model.roi_heads.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if faster_rcnn_model.roi_heads.mask_roi_pool is not None:
                mask_features = faster_rcnn_model.roi_heads.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = faster_rcnn_model.roi_heads.mask_head(mask_features)
                mask_logits = faster_rcnn_model.roi_heads.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if True: # faster_rcnn_model.roi_heads.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            faster_rcnn_model.roi_heads.keypoint_roi_pool is not None
            and faster_rcnn_model.roi_heads.keypoint_head is not None
            and faster_rcnn_model.roi_heads.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if True: # faster_rcnn_model.roi_heads.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = faster_rcnn_model.roi_heads.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = faster_rcnn_model.roi_heads.keypoint_head(keypoint_features)
            keypoint_logits = faster_rcnn_model.roi_heads.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if True: # faster_rcnn_model.roi_heads.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint) """

        return result, losses


    # replace eager_outputs
    def eager_outputs(losses, detections):
        if faster_rcnn_model.training:
            return losses
        return losses, detections

    faster_rcnn_model.eager_outputs = eager_outputs

    faster_rcnn_model.rpn.forward = rpn_forward

    faster_rcnn_model.roi_heads.forward = roi_forward


""" From Yolo.utils.general """
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xyxyn(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] / w 
    y[..., 1] = x[..., 1] / h 
    y[..., 2] = x[..., 2] / w  
    y[..., 3] = x[..., 3] / h
    return y


def xyxyn2xyxy(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] * w 
    y[..., 1] = x[..., 1] * h 
    y[..., 2] = x[..., 2] * w  
    y[..., 3] = x[..., 3] * h
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    def clip_boxes(boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def update_config(config: dict):

    if config['arch'] == 'EfficientDet':
    
        from models.EfficientDet.config import get_efficientdet_config
        if config['version'] == '0':
            model_name = 'efficientdet_d0'
        elif config['version'] == '1':
            model_name = 'tf_efficientdet_d1'
        elif config['version'] == '4':
            model_name = 'tf_efficientdet_d4'
        elif config['version'] == '5':
            model_name = 'tf_efficientdet_d5'
        else:
            raise ValueError()

        model_config = get_efficientdet_config(model_name=model_name)

        model_config.name = 'efficientdet_d0'
        model_config.max_det_per_image = 20
        model_config.num_classes = config['num_classes']
        model_config.image_size = config['img_max_size']

        config.update({
            'model_config': model_config
        })
    
    elif config['arch'] == 'YoloV5':

        import yaml
        assert config['version'] in ['s', 'm']
        cfg_path = 'C:\\Users\\Remotey\\Documents\\Pedestrian-Dynamics\\ObjectDetection\\models\\yolov5\\models\\yolov5'+config['version']+'.yaml'

        with open(cfg_path, encoding='ascii', errors='ignore') as f:
            model_config = yaml.safe_load(f)
        
        model_config['nc'] = config['num_classes']
        
        config.update({
            'model_config': model_config
        })
    

