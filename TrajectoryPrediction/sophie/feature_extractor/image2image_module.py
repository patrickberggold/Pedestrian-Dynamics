import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau
from torch.optim import Adam
import pytorch_lightning as pl
from torchsummary import summary
from model import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
# from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.vgg import vgg19_bn, vgg19
from model import DeepLabHead, Interpolate

class Image2ImageModule(pl.LightningModule):
    def __init__(
        self, 
        mode: str, 
        learning_rate: float = 1e-3, 
        lr_scheduler: str = StepLR.__name__, 
        lr_sch_step_size: int = 8, 
        lr_sch_gamma: float = 0.3, 
        unfreeze_backbone_at_epoch: int = 8,
        relu_at_end: bool = False,
        loss_fct: str = 'mse'
        ):
        super(Image2ImageModule, self).__init__()
        
        assert mode in ['img2img'], 'Unknown mode setting!'
        
        self.mode = mode
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_sch_step_size = lr_sch_step_size
        self.lr_sch_gamma = lr_sch_gamma
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.relu_at_end = relu_at_end
        assert self.lr_scheduler in [CosineAnnealingLR.__name__, StepLR.__name__, ExponentialLR.__name__, ReduceLROnPlateau.__name__], 'Unknown LR Scheduler!'
        assert loss_fct in ['mse', 'l1_loss', 'crafted'], 'Unknown loss function'
        if loss_fct == 'mse':
            self.loss_fct = F.mse_loss
        elif loss_fct == 'l1_loss':
            self.loss_fct = F.l1_loss
        elif loss_fct == 'crafted':
            self.loss_fct = self.crafted_loss
        else:
            raise NotImplementedError('Unknown loss function')
        
        if self.mode == 'img2img':
            self.output_channels = 3
        else:
            raise NotImplementedError

        # self.net = UNet(num_classes = self.num_classes, bilinear = False)
        # self.net = ENet(num_classes = self.num_classes)
        # self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = self.num_classes)

        # self.net = deeplabv3_resnet50(pretrained = False, progress = True, output_channels = self.output_channels, relu_at_end = self.relu_at_end)
        self.backbone = vgg19_bn(pretrained=True, progress=True).features
        self.avg_pool = nn.AdaptiveAvgPool2d((15, 15))
        self.reconstruction = DeepLabHead(in_channels=512, num_classes=self.output_channels)
        self.net = nn.Sequential(
            vgg19_bn(pretrained=True, progress=True).features,
            nn.AdaptiveAvgPool2d((15, 15)),
            Interpolate((100,100), 'bilinear'),
            DeepLabHead(in_channels=512, num_classes=self.output_channels)
        )

        # Check intermediate layers and sizes
        # summary(self.net.to('cuda:0'), (3, 800, 800), device='cuda') # IMPORTANT: INCLUDE non-Instance of torch.Tensor exclusion, otherwise exception
        # print('\n\n##########################################\n\n')
        # print(self.net)
        # output_tensor = self.net(torch.rand((2,3,800,800)).to('cuda:0'))   #   ([2, 256, 800, 800])

        # children_list = list(self.net.children())
        # children_backbone = list(children_list[0].children())
        # children_head = list(children_list[1].children())
        
        # first_part = list(self.net.children())[0]
        # first_result = first_part(torch.ones((1,3,800,800)).to('cuda:1'))['out']
        # summary(first_part, (3, 800, 800), device='cuda')
    
    def crafted_loss(self, prediction, target):
        return torch.mean(torch.abs((prediction - target)**3))
    
    def forward(self, x):
        # x = self.backbone(x)
        # x = self.avg_pool(x)
        # x = F.interpolate(x, size=(100,100), mode="bilinear", align_corners=False)
        # x = self.reconstruction(x)
        x = self.net(x)
        return F.interpolate(x, size=(800,800), mode="bilinear", align_corners=False)

    def training_step(self, batch, batch_idx: int):
        # TODO maybe implement scheduler change once backbone is unfreezed
        img, traj = batch
        img = img.float()
        traj_pred = self.forward(img)#['out']

        if self.mode == 'img2img':
            train_loss = self.loss_fct(traj_pred.squeeze(), traj.float())
        else:
            raise NotImplementedError('Mode not implemented!')
        
        self.log('loss', train_loss, on_step=False, on_epoch=True)
        return {'loss' : train_loss}

    def validation_step(self, batch, batch_idx: int) -> None:

        img, traj = batch
        img = img.float()
        traj_pred = self.forward(img)#['out']

        if self.mode == 'img2img':
            val_loss = self.loss_fct(traj_pred.squeeze(), traj.float())
        else:
            raise NotImplementedError('Mode not implemented!')
        
        self.log('val_loss', val_loss)
        return {'val_loss' : val_loss}

        # data, target = batch
        # output = self(data)
        # pred = output.argmax(dim=1, keepdim=True)
        # accuracy = pred.eq(target.view_as(pred)).float().mean()
        # self.log("val_acc", accuracy)
        # self.log("hp_metric", accuracy, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        opt = Adam(self.net.parameters(), lr = self.learning_rate)
        if self.lr_scheduler == CosineAnnealingLR.__name__:
            sch = CosineAnnealingLR(opt, T_max = self.lr_sch_step_size)
        elif self.lr_scheduler == StepLR.__name__:
            sch = StepLR(opt, step_size=self.lr_sch_step_size, gamma=self.lr_sch_gamma)
        elif self.lr_scheduler == ExponentialLR.__name__:
            sch = ExponentialLR(opt, gamma=0.9)
        elif self.lr_scheduler == ReduceLROnPlateau.__name__:
            sch = ReduceLROnPlateau(opt, factor=self.lr_sch_gamma)
            # Because of a weird issue with ReduceLROnPlateau, the monitored value needs to be returned... See https://github.com/PyTorchLightning/pytorch-lightning/issues/4454
            return {
            'optimizer': opt,
            'lr_scheduler': sch,
            'monitor': 'val_loss'
        }
        else:
            raise NotImplementedError('Scheduler has not been implemented yet!')
        return [opt], [sch]

    def on_epoch_start(self) -> None:
        # return super().on_epoch_start()
        if self.unfreeze_backbone_at_epoch:
            if self.current_epoch == 0:
                # freeze backbone weights
                for idx, child in enumerate(self.net.children()):
                    if idx == 0:
                        for param in child.parameters():
                            param.requires_grad = False
            elif self.current_epoch == self.unfreeze_backbone_at_epoch:
                print(f"\nUNFREEZING BACKBONE AFTER {self.unfreeze_backbone_at_epoch} EPOCHS\n")
                # unfreeze backbone weights
                for idx, child in enumerate(self.net.children()):
                    if idx == 0:
                        for param in child.parameters():
                            param.requires_grad = True
