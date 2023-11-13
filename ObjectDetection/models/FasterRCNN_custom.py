from collections import OrderedDict
from typing import Tuple, List
import warnings
import torch
from torch.nn import functional as F
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss, keypointrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from einops import rearrange
from torch import einsum

class FasterRCNN_custom(FasterRCNN):
    
    def __init__(self, backbone, num_classes, config):
        if backbone is None:
            from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
            from torchvision.ops import misc as misc_nn_ops
            from torchvision.models.resnet import resnet50

            trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
            backbone = resnet50(pretrained=True, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
            backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

        super().__init__(backbone, num_classes, box_score_thresh=0.5, box_nms_thresh=0.0) # box_score_thresh=0.05, box_nms_thresh=0.5

        self.rpn.head = RPNHead_custom(merge_mech = config['merge_mech'])
        # self.roi_heads.box_head = TwoMLPHead_custom(256 * 7 ** 2, 1024)
        # self.forward = self.forward
        self.rpn.forward = self.forward_rpn
        # self.roi_heads.forward = self.forward_roi
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
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
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
        proposals, proposal_losses = self.rpn(images, features, ag_embeddings, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, ag_embeddings, targets)
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
        # if self.training:
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

    def __init__(self, in_channels: int = 256, num_anchors: int = 3, merge_mech = 'linear+skip') -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = torch.nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = torch.nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        self.init = 'xavier'

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
            torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

        # customized
        if merge_mech in ['cross_attn', 'cross_attn_large']:
            hidden_dim = 64 if merge_mech == 'cross_attn' else 256
            self.merge_layers = CrossAttentionLayer(hidden_dim = hidden_dim)
        elif merge_mech == 'linear+skip':
            self.merge_layers = SkipConnection()
        elif merge_mech == 'linear+skip_large':
            self.merge_layers = SkipConnectionLarge()

        self.merge_layers.apply(self._initialize_weights)

        # self.sim_fc = torch.nn.ModuleList([torch.nn.Linear(512, 21*9) for i in range(5)])
        # self.dim_red = torch.nn.ModuleList([torch.nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1) for i in range(5)])
        # self.dim_inc = torch.nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.size_up = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=1, stride=2, padding=0, output_padding=1)
    
    def _initialize_weights(self, m):
        if hasattr(m, 'weight'):
            try:
                if self.init=='xavier': torch.nn.init.xavier_normal_(m.weight)
                elif self.init=='normal': torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
                elif self.init=='uniform': torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                else: raise NotImplementedError
            except ValueError:
                # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                m.weight.data.uniform_(-0.1, 0.1)
                # print("Bypassing ValueError...")
        elif hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x: List[torch.Tensor], ag_embeddings) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ag_embeddings = torch.split(ag_embeddings, 512, dim=-1)

        logits = []
        bbox_reg = []
        for i_t, feature in enumerate(x):
            # new:
            # emb = self.sim_fc[i_t](ag_embeddings[i_t]).view(-1, 1, 9, 21)
            # feat_red = self.dim_red[i_t](feature)
            # size_factor = feat_red.size(2) // 9
            # feat_red = torch.nn.MaxPool2d(size_factor)(feat_red)
            # feat_out = self.dim_inc(emb + feat_red)
            # for _r in range(len(x) - i_t - 1):
            #     feat_out = self.size_up(feat_out)
            # feat_out = F.relu(feat_out)

            t = F.relu(self.conv(feature))
            t = self.merge_layers(t, ag_embeddings[i_t])
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
    

class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, cnn_feature_dim=256, extra_info_dim=512, hidden_dim=64):
        super(CrossAttentionLayer, self).__init__()
        
        # Linear transformations for query, key, and value
        self.num_heads = 8
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.query_transform = torch.nn.Linear(cnn_feature_dim, hidden_dim * self.num_heads, bias=False)
        self.kv = torch.nn.Linear(extra_info_dim, hidden_dim * 2, bias=False)
        # self.key_transform = torch.nn.Linear(extra_info_dim, hidden_dim, bias=False)
        # self.value_transform = torch.nn.Linear(extra_info_dim, hidden_dim, bias=False)

        self.cnn_feature_norm = torch.nn.LayerNorm(cnn_feature_dim)
        self.embeddings_norm = torch.nn.LayerNorm(extra_info_dim) 
        
        self.softmax = torch.nn.Softmax(dim=1)  # Softmax along the query dimension

        self.to_out = torch.nn.Linear(self.num_heads * hidden_dim, cnn_feature_dim, bias=False)
        
        ff_inner_dim = 2 * cnn_feature_dim
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(cnn_feature_dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            torch.nn.Linear(ff_inner_dim, cnn_feature_dim, bias=False))

        
    def forward(self, cnn_features, extra_info):

        cnn_features = cnn_features.permute(0,2,3,1)

        cnn_features = self.cnn_feature_norm(cnn_features)
        extra_info = self.embeddings_norm(extra_info)

        batch_size = cnn_features.size(0)

        query = self.query_transform(cnn_features) # 4, 200, 200, 64

        extra_info = extra_info.unsqueeze(-1).expand(batch_size, extra_info.size(1), self.hidden_dim) # 4, 512, 64

        query = rearrange(query, 'b m n (h d) -> b h m n d', h = self.num_heads) # 4, 8, 200, 200, 64
        query *= self.scale

        key, value = self.kv(extra_info.permute(0,2,1)).chunk(2, dim=-1) # 2x 4, 512, 64

        sim = einsum('b h m n d, b j d -> b h m n j', query, key)

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        out = einsum('b h m n j, b j d -> b h m n d', attn, value)

        out = rearrange(out, 'b h m n d -> b m n (h d)')

        out = self.to_out(out)

        out = out + self.ff(cnn_features)

        return out.permute(0,3,1,2)

class SwiGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class SkipConnection(torch.nn.Module):
    def __init__(self, cnn_feature_dim=256, extra_info_dim=512, hidden_dim=256) -> None:
        super().__init__()

        # self.cnn_features_norm = torch.nn.BatchNorm2d(cnn_feature_dim)
        self.cnn_linear = torch.nn.Linear(cnn_feature_dim, hidden_dim)
        self.embed_linear = torch.nn.Linear(extra_info_dim, hidden_dim)
        
        self.cnn_norm = torch.nn.LayerNorm(hidden_dim)
        self.embed_norm = torch.nn.LayerNorm(hidden_dim)

        self.combine = torch.nn.Linear(2 * hidden_dim, cnn_feature_dim)
    
    def forward(self, cnn_features, extra_info):
        cnn_features = cnn_features.permute(0, 2, 3, 1)
        cnn_feat = self.cnn_linear(cnn_features)
        embed_feat = self.embed_linear(extra_info)
        embed_feat = embed_feat.unsqueeze(1).unsqueeze(1).repeat(1, cnn_features.size(1), cnn_features.size(2), 1)

        cnn_feat = self.cnn_norm(cnn_feat)
        embed_feat = self.embed_norm(embed_feat)

        concat = torch.cat((cnn_feat, embed_feat), dim=-1)

        return self.combine(concat).permute(0,3,1,2)


class SkipConnectionLarge(torch.nn.Module):
    def __init__(self, cnn_feature_dim=256, extra_info_dim=512, hidden_dim=512) -> None:
        super().__init__()

        # self.cnn_features_norm = torch.nn.BatchNorm2d(cnn_feature_dim)
        self.cnn_linear = torch.nn.Linear(cnn_feature_dim, hidden_dim)
        self.embed_linear = torch.nn.Linear(extra_info_dim, hidden_dim)
        
        self.cnn_norm = torch.nn.LayerNorm(hidden_dim)
        self.embed_norm = torch.nn.LayerNorm(hidden_dim)

        self.combine = torch.nn.Linear(2 * hidden_dim, cnn_feature_dim)

    def forward(self, cnn_features, extra_info):
        cnn_features = cnn_features.permute(0, 2, 3, 1)
        cnn_feat = self.cnn_linear(cnn_features)
        embed_feat = self.embed_linear(extra_info)
        embed_feat = embed_feat.unsqueeze(1).unsqueeze(1).repeat(1, cnn_features.size(1), cnn_features.size(2), 1)

        cnn_feat = self.cnn_norm(cnn_feat)
        embed_feat = self.embed_norm(embed_feat)

        concat = torch.cat((cnn_feat, embed_feat), dim=-1)
        out = cnn_features + self.combine(concat)

        return out.permute(0,3,1,2)