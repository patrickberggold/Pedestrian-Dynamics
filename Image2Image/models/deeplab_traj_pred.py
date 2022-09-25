import torch
from models.deeplabv3_resnet50_custom import DeepLabHead, DeepLabV3MulitHead, DeepLabV3, EvacTimeHead, IntermediateLayerGetter_custom
from torchvision.models import resnet
from torch.nn import functional as F

class DeepLabTraj(torch.nn.Module):
    def __init__(self, mode, output_channels, relu_at_end, p_dropout, num_heads) -> None:
        super().__init__()

        self.mode = mode
        self.output_channels = output_channels
        self.relu_at_end = relu_at_end
        self.p_dropout = p_dropout
        self.num_heads = num_heads

        self.backbone = resnet.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.backbone = IntermediateLayerGetter_custom(self.backbone, return_layers={"layer4": "out"}, dropout=self.p_dropout)
        if self.mode == 'grayscale':
            self.image_head = DeepLabHead(2048, self.output_channels, relu_at_end=self.relu_at_end)
        elif self.mode == 'grayscale_movie':
            self.image_head = torch.nn.ModuleList([DeepLabHead(2048, self.output_channels, relu_at_end=self.relu_at_end) for i in range(self.num_heads)])
        elif self.mode == 'evac':
            self.image_head = DeepLabHead(2048, self.output_channels, relu_at_end=self.relu_at_end, dropout=None)
            self.evac_head = EvacTimeHead(2048, dropout=self.p_dropout)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)['out']
        if self.mode == 'grayscale':
            # result = OrderedDict()
            # x = features["out"]
            x = self.image_head(features)
            
            result = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        
        elif self.mode == 'grayscale_movie':
            result = []
            for idx, classifier in enumerate(self.image_head):
                x_class = classifier(features)
                x_class = F.interpolate(x_class, size=input_shape, mode="bilinear", align_corners=False)
                result.append(x_class)
        
        elif self.mode == 'evac':
            x = self.image_head(features)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            ev = self.evac_head(features)
            result = (x, ev)
        return result

    def append_conv_layers(self):
        len_dict = len(self.image_head._modules)
        self.image_head.add_module(f'{len_dict}', torch.nn.Conv2d(1, 3, kernel_size=3, padding=1))
        self.image_head.add_module(f'{len_dict+1}', torch.nn.BatchNorm2d(3))
        self.image_head.add_module(f'{len_dict+2}', torch.nn.ReLU())
        self.image_head.add_module(f'{len_dict+3}', torch.nn.Conv2d(3, 1, kernel_size=3, padding=1))
        self.image_head.add_module(f'{len_dict+4}', torch.nn.ReLU())

        