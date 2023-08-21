import torch
from transformers.models.detr.modeling_detr import DetrHungarianMatcher, DetrLoss, DetrForObjectDetection

def compute_loss(model: DetrForObjectDetection, labels, logits, pred_boxes, auxiliary_outputs):
    matcher = DetrHungarianMatcher(
        class_cost=model.config.class_cost, bbox_cost=model.config.bbox_cost, giou_cost=model.config.giou_cost
    )
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality"]
    criterion = DetrLoss_custom(
        matcher=matcher,
        num_classes=model.config.num_labels,
        eos_coef=model.config.eos_coefficient,
        losses=losses,
    )
    criterion.to(model.device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    # if model.config.auxiliary_loss:
    #     intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
    #     outputs_class = model.class_labels_classifier(intermediate)
    #     outputs_coord = model.bbox_predictor(intermediate).sigmoid()
    #     auxiliary_outputs = model._set_aux_loss(outputs_class, outputs_coord)
    #     outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_bbox": model.config.bbox_loss_coefficient}
    weight_dict["loss_giou"] = model.config.giou_loss_coefficient
    # if model.config.auxiliary_loss:
    #     aux_weight_dict = {}
    #     for i in range(model.config.decoder_layers - 1):
    #         aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    return loss


class DetrLoss_custom(DetrLoss):
    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__(matcher, num_classes, eos_coef, losses)
        self.empty_weight = None

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        src_logits = outputs["logits"]
        assert src_logits.size(2) == 1, 'Assuming one class prediction for now, for multiple classes fix this loss!'

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(
        #     src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        # )
        # target_classes[idx] = target_classes_o

        # one class prediction
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )

        loss_ce = torch.nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses 