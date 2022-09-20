from turtle import forward
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.models import mobilenetv3
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation._utils import  _SimpleSegmentationModel, _load_weights
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.utils import _log_api_usage_once

__all__ = [
    "DeepLabV3",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
]


model_urls = {
    "deeplabv3_resnet50_coco": "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
    "deeplabv3_resnet101_coco": "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
    "deeplabv3_mobilenet_v3_large_coco": "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
}


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class DeepLabV3MulitHead(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifiers: nn.ModuleList, aux_classifier: Optional[nn.Module] = None, pred_evac_time: bool = False) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifiers = classifiers
        self.aux_classifier = aux_classifier
        self.pred_evac_time = pred_evac_time

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        result["out"] = []
        
        for idx, classifier in enumerate(self.classifiers):
            x_class = classifier(x)
            if self.pred_evac_time:
                if idx != (len(self.classifiers)-1):
                    x_class = F.interpolate(x_class, size=input_shape, mode="bilinear", align_corners=False)
            else:
                x_class = F.interpolate(x_class, size=input_shape, mode="bilinear", align_corners=False)
            result["out"].append(x_class)

        # if self.aux_classifier is not None:
        #     x = features["aux"]
        #     x = self.aux_classifier(x)
        #     x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        #     result["aux"] = x

        return result

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, relu_at_end: bool = False, dropout = None) -> None:
        if not relu_at_end:
            if not dropout:
                super().__init__(
                    ASPP(in_channels, [12, 24, 36]),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, num_classes, 1),
                )
            else:
                super().__init__(
                    ASPP(in_channels, [12, 24, 36]),
                    nn.Dropout(p=dropout),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Conv2d(256, num_classes, 1),
                )
        else:
            if not dropout:
                super().__init__(
                    ASPP(in_channels, [12, 24, 36]),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, num_classes, 1),
                    nn.ReLU()
                )
            else:
                super().__init__(
                    ASPP(in_channels, [12, 24, 36]),
                    nn.Dropout(p=dropout),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Conv2d(256, num_classes, 1),
                    nn.ReLU()
                )

class EvacTimeHead(nn.Module):
    def __init__(self, in_channels: int, dropout: bool = False):
        super(EvacTimeHead, self).__init__()
        self.evac_time_predictor = nn.Sequential(
            DeepLabHead(in_channels, 1, relu_at_end=False, dropout=dropout),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(1, 1, 3),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(1, 1, 5),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(100,1)
            # nn.ReLU(),
        )
    def forward(self, x):
        return self.evac_time_predictor(x)


class IntermediateLayerGetter_custom(IntermediateLayerGetter):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str], dropout=None) -> None:
        self.dropout = dropout
        super().__init__(model, return_layers)
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        if self.dropout:
            x = nn.Dropout(p=self.dropout)(x)
        return out


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet(
    backbone: resnet.ResNet,
    num_classes: int,
    aux: Optional[bool],
    relu_at_end: bool = False,
    num_heads: int = 1,
    pred_evac_time: bool = False
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    if num_heads == 1 and not pred_evac_time:
        classifier = DeepLabHead(2048, num_classes, relu_at_end=relu_at_end)
        return DeepLabV3(backbone, classifier, aux_classifier)
    elif num_heads == 1 and pred_evac_time:
        classifiers = nn.ModuleList([DeepLabHead(2048, num_classes, relu_at_end=relu_at_end), EvacTimeHead(2048)])
        return DeepLabV3MulitHead(backbone, classifiers, aux_classifier, pred_evac_time=pred_evac_time)
    elif num_heads > 1 and not pred_evac_time:
        classifiers = nn.ModuleList([DeepLabHead(2048, num_classes, relu_at_end=relu_at_end) for i in range(num_heads)])
        return DeepLabV3MulitHead(backbone, classifiers, aux_classifier)
    elif num_heads > 1 and pred_evac_time:
        classifiers = nn.ModuleList([DeepLabHead(2048, num_classes, relu_at_end=relu_at_end) for i in range(num_heads)])
        classifiers.append(EvacTimeHead(2048))
        return DeepLabV3MulitHead(backbone, classifiers, aux_classifier, pred_evac_time=pred_evac_time)


def _deeplabv3_mobilenetv3(
    backbone: mobilenetv3.MobileNetV3,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux:
        return_layers[str(aux_pos)] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


def deeplabv3_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    output_channels: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    relu_at_end: bool = False,
    num_heads: int = 1,
    pred_evac_time: bool = False
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, output_channels, aux_loss, relu_at_end=relu_at_end, num_heads=num_heads, pred_evac_time=pred_evac_time)

    if pretrained:
        arch = "deeplabv3_resnet50_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def deeplabv3_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool, optional): If True, include an auxiliary classifier
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet101(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if pretrained:
        arch = "deeplabv3_resnet101_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def deeplabv3_mobilenet_v3_large(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = mobilenetv3.mobilenet_v3_large(pretrained=pretrained_backbone, dilated=True)
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

    if pretrained:
        arch = "deeplabv3_mobilenet_v3_large_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model
