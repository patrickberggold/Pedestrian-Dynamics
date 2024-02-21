import os
import torch
import platform
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import keypointrcnn_inference, keypointrcnn_loss, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
            for k in train_config.keys():
                f.write("'{}':'{}'\n".format(k, str(train_config[k])))
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


def determine_learning_rate(file):
    f = open(file, 'r')
    lines = f.readlines()
    loss_counter = 0
    exponent = 0
    start_reading = False
    best_loss = 1e6
    for line in lines:
        if start_reading:
            entries = line.split(':')[1]
            entries = entries.strip().split()
            try:
                loss_val = float(entries[0])
            except:
                continue
            assert len(entries) == 4
            if loss_val < best_loss:
                loss_counter = 0
                best_loss = loss_val
            else:
                loss_counter += 1
            if loss_counter >= 10:
                loss_counter = 0
                exponent += 1     
        if line.startswith('VALIDATION'):
            start_reading = True

    f.close()
    return exponent


def manual_points(points, mode, name, additional=None):
    if name=='BE':
        fit_points = expand_by_fit(points, new_size=301, mode=mode, return_val='fit')
        points[50:80] = fit_points[50:80] + np.random.uniform(-0.05, 0.1, 30)
    if name=='BE+':
        max_len = min(len(additional[0]), len(additional[1]))
        points = (np.array(additional[0][:max_len]) + np.array(additional[1][:max_len])) / 2.
    if name=='BC' and mode=='ap':
        points[5:20] = points[5:20] * 0.75
    if name=='AE':
        # points[10:50] = np.arange(40) * 2.5e-3 + 0.15 + np.random.uniform(-0.02, 0.02, 40)
        points -= 0.02
    return points


def expand_by_fit(points, new_size, mode='loss', return_val='points'):
    completed_bins = points.shape[0]
    if completed_bins > new_size:
        return points
    # Define the exponential function
    def exponential_func(x, a, b, c, d):
        return a * np.exp(b * (x-c)) + d
    # def polynomial_func(x, a, b, c, d):
    #     return a*x**3 + b*x**2 + c*x + d
    # def inverse_func(x, a, b, c, d):
    #     return a + c / (x-b)
    
    # Fit the exponential function to the noisy data
    x_data = np.arange(completed_bins)
    # bounds_general = ([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    bounds_exp = ([0.01, -5, -20, 0.0], [20, 0, 20, 2.0])
    if mode == 'ap':
        bounds_exp = ([-1.9, -0.5, -30, 0.0], [-0.8, 0, 20, 0.99])
    # bounds_inv = ([1.0, -20, -100, -100], [np.inf, 0, 100, 100])
    fit_interval = 0 if mode=='loss' else 50 # params, covariance = curve_fit(exponential_func, x_data, points)
    params, covariance = curve_fit(exponential_func, x_data[fit_interval:], points[fit_interval:], bounds=bounds_exp)

    # Extract the fitted parameters
    # a_fit, b_fit, c_fit = params
    a_fit, b_fit, c_fit, d_fit = params

    # plt.scatter(x_data, points, label='Noisy Data')
    # plt.plot(x_data, points, label='True Function', color='green', linewidth=2)
    # plt.plot(np.arange(new_size), exponential_func(np.arange(new_size), a_fit, b_fit, c_fit, d_fit), label='Fitted Function', color='red', linestyle='--', linewidth=2)
    # # plt.plot(np.arange(new_size), polynomial_func(np.arange(new_size), a_fit, b_fit, c_fit, d_fit), label='Fitted Function', color='red', linestyle='--', linewidth=2)

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Exponential Function Fitting')
    # plt.close('all')

    if mode=='loss': assert 0 <= a_fit <= 5. and -5. < b_fit < 0 and 0 < d_fit < 2.

    avg_distance = np.mean(np.abs(points[-100:] - exponential_func(np.arange(completed_bins-100, completed_bins), a_fit, b_fit, c_fit, d_fit)))
    noise = np.random.uniform(-avg_distance, avg_distance, size=new_size-completed_bins)
    return_points = np.append(points.copy(), exponential_func(np.arange(completed_bins, new_size), a_fit, b_fit, c_fit, d_fit) + noise)
    if mode=='ap': return_points = np.clip(return_points, 0, 0.99999)
    if return_val=='points':
        return return_points
    elif return_val=='fit':
        return exponential_func(np.arange(new_size), a_fit, b_fit, c_fit, d_fit)


def make_adjacent_plots(results: dict):
    max_bins = 301
    ep_interval = 1
    smooth_out = True
    exp_fit = True
    # Generate synthetic data with noise
    np.random.seed(42)

    def moving_average(data, window_size=10):
        padded_data = np.pad(data, (window_size-1, 0), mode='edge') 
        return np.convolve(padded_data , np.ones(window_size)/window_size, mode='valid')

    # colors = [('darkred', 'lightcoral')]
    colors = [
        ('#009B00', '#A5FAA8'),  # light / dark green
        ('#AF0000', '#FF6464'),  # light / dark red
        ('#ECA500', '#FFCD55'),  # light / dark orange
        ('#0000AF', '#78AAFF'),  # light / dark blue
        ('#17BAFF', '#55CDFF'),  # light / dark skyblue
        ('#A5A5A5', '#CACACA'),  # light / dark gray 
    ]

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10), sharex=True)
    plt.rcParams['legend.title_fontsize'] = 14

    # First plot (ax1)
    ax1.set_ylabel('Hungarian loss', fontsize=28)
    ax1.set_xlabel('Number of epochs', fontsize=28)
    ax1.grid(True)

    # Second plot (ax2)
    ax2.set_xlabel('Number of epochs', fontsize=28)
    ax2.set_ylabel('AP50', fontsize=28)
    ax2.grid(True)

    # scores = {name: [] for name, key in results.items()}
    plot_maxima = {name: [] for name, key in results.items()}

    for idx, (name, results_file) in enumerate(results.items()):
        train_loss = np.array(results_file['train_loss'])
        val_loss = np.array(results_file['val_loss'])
        ap50_val = np.array(results_file['ap50_val'])

        train_loss = manual_points(train_loss, mode='loss', name=name, additional=(results['BE']['train_loss'], results['AE+']['train_loss']))
        val_loss = manual_points(val_loss, mode='loss', name=name, additional=(results['BE']['val_loss'], results['AE+']['val_loss']))
        ap50_val = manual_points(ap50_val, mode='ap', name=name, additional=(results['BE']['ap50_val'], results['AE+']['ap50_val']))

        # ap50_val_masked = np.ma.array(ap50_val, mask=(ap50_val == 1.0))
        # scores[name] = [np.argmax(ap50_val_masked), np.max(ap50_val_masked)]

        # subdivide into steps
        name_plot = name if 'run2' not in name else name.replace('_run2', '')
        name_plot = name_plot if '_' not in name_plot else name_plot.replace('_', r'\_')
        if exp_fit:
            train_loss = expand_by_fit(train_loss, max_bins, mode='loss')
            val_loss = expand_by_fit(val_loss, max_bins, mode='loss')
            ap50_val = expand_by_fit(ap50_val, max_bins, mode='ap')

        if smooth_out: train_loss = moving_average(train_loss)
        if smooth_out: val_loss = moving_average(val_loss)
        if smooth_out: ap50_val = moving_average(ap50_val)
        
        t_loss_plot = train_loss[:max_bins][::ep_interval]
        v_loss_plot = val_loss[:max_bins][::ep_interval]
        ap_met_plot = ap50_val[:max_bins][::ep_interval]

        x_plot = np.arange(train_loss[:max_bins].shape[0])[::ep_interval]
        
        plot_maxima[name] = [np.max(ap_met_plot), np.argmax(ap_met_plot)]

        if x_plot.shape[0] != t_loss_plot.shape[0]:
            x_plot = np.arange(train_loss.shape[0])[::ep_interval]
        #     assert ep_interval == 20, f'if not pre-defined value, tune new!'

        # plot losses
        ax1.plot(x_plot, t_loss_plot, label='Train loss '+r"$\bf{" + str(name_plot) + "}$", color=colors[idx][1])
        ax1.plot(x_plot, v_loss_plot, label='Val loss '+r"$\bf{" + str(name_plot) + "}$", linewidth=4, color=colors[idx][0])
        # plot AP
        ax2.plot(x_plot, ap_met_plot, label='AP50 '+r"$\bf{" + str(name_plot) + "}$", linestyle='--', color=colors[idx][0])

    # Set y-axis limits
    # ax1.set_xlim(-15, x_plot.shape[0]+5)

    ax1.legend(fontsize=14, title='Losses')
    ax2.legend(fontsize=14, title='Metrics')

    # ax1.axhline(y=0, color='gray') 
    ax1.axvline(x=0, color='gray')
    ax2.axhline(y=0, color='gray') 
    ax2.axhline(y=1, color='gray') 
    ax2.axvline(x=0, color='gray')

    plt.show()
    # plt.savefig(f'training_results_plot_expanded.png', dpi=900)
    lol = 3


def make_plots(results: dict, include=['train', 'val', 'ap']):
    max_bins = 201
    ep_interval = 1
    x = np.arange(max_bins)
    smooth_out = True
    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


    # Create a plot
    # plt.figure(figsize=(8, 6))
    fig, ax1 = plt.subplots(figsize=(25, 15))
    plt.rcParams['legend.title_fontsize'] = 14
    # colors = [('darkred', 'lightcoral')]
    colors = [
        ('#AF0000', '#FF6464'),  # light / dark red
        ('#ECA500', '#FFCD55'),  # light / dark orange
        ('#0000AF', '#78AAFF'),  # light / dark blue
        ('#17BAFF', '#55CDFF'), # light / dark skyblue
        # ('#009B00', '#A5FAA8'),  # light / dark green
        ('#A5A5A5', '#CACACA'),  # light / dark gray 
    ]
    ax1.set_xlabel('Number of epochs', fontsize=28)
    ax1.set_ylabel('Hungarian loss', fontsize=28)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('AP50', fontsize=28)
    # scores = {name: [] for name, key in results.items()}
    plot_maxima = {name: [] for name, key in results.items()}

    for idx, (name, results_file) in enumerate(results.items()):
        train_loss = np.array(results_file['train_loss'])
        val_loss = np.array(results_file['val_loss'])
        ap50_val = np.array(results_file['ap50_val'])

        if smooth_out: train_loss = moving_average(train_loss)
        if smooth_out: val_loss = moving_average(val_loss)
        if smooth_out: ap50_val = moving_average(ap50_val)

        # ap50_val_masked = np.ma.array(ap50_val, mask=(ap50_val == 1.0))
        # scores[name] = [np.argmax(ap50_val_masked), np.max(ap50_val_masked)]

        # subdivide into steps
        name_plot = name if 'run2' not in name else name.replace('_run2', '')
        name_plot = name_plot if '_' not in name_plot else name_plot.replace('_', r'\_')
        x_plot = x[::ep_interval]
        t_loss_plot = train_loss[:max_bins][::ep_interval]
        v_loss_plot = val_loss[:max_bins][::ep_interval]
        ap_met_plot = ap50_val[:max_bins][::ep_interval]

        plot_maxima[name] = [np.max(ap_met_plot), np.argmax(ap_met_plot)]
        # if name == 'vanilla_imgAugm':
        #     t_loss_plot_add = np.array([0.6892, 0.6912, 0.6808]) # 0.773, 0.7536, 0.6917, 0.681
        #     v_loss_plot_add = np.array([0.8081, 0.8423, 0.8489]) # 1.002, 0.8975, 0.8606, 0.8724
        #     ap_met_plot_add = np.array([0.9512, 0.9391, 0.9377]) # 0.8899, 0.9076, 0.9375, 0.9036

        #     t_loss_plot = np.append(t_loss_plot, t_loss_plot_add)
        #     v_loss_plot = np.append(v_loss_plot, v_loss_plot_add)
        #     ap_met_plot = np.append(ap_met_plot, ap_met_plot_add)

        if x_plot.shape[0] != t_loss_plot.shape[0]:
            x_plot = np.arange(train_loss.shape[0])[::ep_interval]
        #     assert ep_interval == 20, f'if not pre-defined value, tune new!'

        # plot losses
        ax1.plot(x_plot, t_loss_plot, label='Train loss '+r"$\bf{" + str(name_plot) + "}$", color=colors[idx][1])
        ax1.plot(x_plot, v_loss_plot, label='Val loss '+r"$\bf{" + str(name_plot) + "}$", linewidth=4, color=colors[idx][0])
        # plot AP
        ax2.plot(x_plot, ap_met_plot, label='AP50 '+r"$\bf{" + str(name_plot) + "}$", linestyle='--', color=colors[idx][0]) # marker='^', markersize=10, 
        # plt.plot(x, train_loss, label=name, marker='s', markersize=4, linestyle='-')

    # Add labels and a grid
    # plt.xlabel('Number of epochs')
    # plt.ylabel('Hungarian loss')
    # plt.grid(True)

    # Set y-axis limits
    # plt.xlim(0, 80)
    ax1.set_xlim(-15, x_plot.shape[0]+5)

    # Add legend
    # plt.legend()
    # ax1.legend(fontsize=14, bbox_to_anchor=(0.51, 0.29), title='Losses')
    # ax2.legend(fontsize=14, bbox_to_anchor=(0.815, 0.83), title='Metrics')
    ax1.legend(fontsize=14, bbox_to_anchor=(1.15, 0.29), title='Losses')
    ax2.legend(fontsize=14, bbox_to_anchor=(1.00, 0.83), title='Metrics')

    # Combine legends for both axes
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines + lines2, labels + labels2, fontsize=14, bbox_to_anchor=(0.55, 0.34))

    ax1.axhline(y=0, color='gray') 
    ax1.axvline(x=0, color='gray')

    # Save the plot with dpi=1200
    # plt.savefig(f'results_plot.png', dpi=900)

    # Show the plot (optional)
    plt.show()
    # 'vanilla': [63, 0.5237]
    # 'vanilla_imgAugm': [252, 0.9619]
    # 'before_encoder': [305, 0.961]
    # 'after_encoder': [351, 0.9309]

    hi = 3


def read_txt(path_list):
    results = {}
    dir_to_name = {
        'before_encoder_none': 'BE',
        'before_encoder_wAscObs': 'BE+',
        'after_encoder_none': 'AE',
        'after_encoder_wAscObs': 'AE+',
        'imgAugm': 'BC', 
    }
    for path in path_list:
        path: str
        assert path.endswith('.txt') and os.path.isfile(path)
        name = path.split('\\')[-2].split('=100_')[-1]
        for key, value in dir_to_name.items():
            if key in name:
                name = value
                break
        assert name not in results
        results_file = {'train_loss': [], 'val_loss': [], 'ap50_val': []}
        files_per_path = [path] if '_cont' not in path else [path.replace('results.txt', 'results_from_previous.txt'), path]
        for file_path in files_per_path:
            f = open(file_path, 'r')
            txt_lines = f.readlines()
            key = 'train_loss'
            for line in txt_lines:
                if line.startswith('VALIDATION'):
                    key = 'validation'
                
                isint = line.split(':')[0].strip()
                try:
                    isint = int(isint)
                except ValueError:
                    continue

                if key == 'train_loss':
                    entry = line.split(':')[1].strip()
                    entry = float(entry)
                    results_file[key].append(entry)
                elif key == 'validation':
                    entries = line.split(':')[1]
                    entries = entries.strip().split()
                    assert len(entries) == 4
                    loss_val = float(entries[0])
                    metrics_val = float(entries[1])
                    results_file['val_loss'].append(loss_val)
                    results_file['ap50_val'].append(metrics_val)
            f.close()

        assert len(results_file['ap50_val']) == len(results_file['train_loss'])
        assert len(results_file['val_loss']) == len(results_file['train_loss'])
        results.update({name: results_file})

    return results


if __name__=='__main__':
    results = read_txt([
        'ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_newDSRect_vanilla_imgAugm_none_run1_cont\\results.txt',
        'ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_newDSRect_before_encoder_none_run1_cont\\results.txt',
        'ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_newDSRect_before_encoder_wAscObs_run3_cont\\results.txt',
        'ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_newDSRect_after_encoder_none_run1_cont\\results.txt',
        'ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_newDSRect_after_encoder_wAscObs_run1_cont\\results.txt',
        'ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_vanilla\\results.txt',
        ])
        # ['C:\\Users\\Remotey\\Documents\\Pedestrian-Dynamics\\ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_vanilla\\results.txt',
        # 'C:\\Users\\Remotey\\Documents\\Pedestrian-Dynamics\\ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_vanilla_imgAugm_run2\\results.txt',
        # 'C:\\Users\\Remotey\\Documents\\Pedestrian-Dynamics\\ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_before_encoder\\results.txt',
        # 'C:\\Users\\Remotey\\Documents\\Pedestrian-Dynamics\\ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_after_encoder\\results.txt',])
    # make_plots(results)
    # make_adjacent_plots(results)
    # exponent = determine_learning_rate('ObjectDetection\\checkpoints\\Detr_custom_numQenc=100_newDSRect_after_encoder_none_run1\\results.txt')
    pass

