import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchsummary import summary
from models import deeplabv3_resnet50
# from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50

class Image2ImageModule(pl.LightningModule):
    def __init__(self, mode: str, learning_rate: float = 1e-3):
        super(Image2ImageModule, self).__init__()
        
        assert mode in ['grayscale', 'rgb', 'bool', 'segmentation'], 'Unknown mode setting!'
        
        self.mode = mode
        self.learning_rate = learning_rate
        
        if self.mode == 'grayscale':
            # Regression task
            self.output_channels = 1
        elif self.mode == 'bool':
            # Classification task
            self.output_channels = 2 # Classes: True, False
        elif self.mode == 'rgb':
            # Regression task
            self.output_channels = 3
        elif self.mode == 'segmentation':
            # Classification
            self.output_channels = 5 # Classes: Bkg, 'unpassable', 'walkable area', 'spawn_zone', 'destination'
        else:
            raise ValueError

        # self.net = UNet(num_classes = self.num_classes, bilinear = False)
        # self.net = ENet(num_classes = self.num_classes)
        # self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = self.num_classes)

        self.net = deeplabv3_resnet50(pretrained = False, progress = True, output_channels = self.output_channels)

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
    
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch) :

        img, traj = batch
        img = img.float()
        traj_pred = self.forward(img)['out']

        if self.mode == 'bool' or self.mode == 'segmentation':
            loss_val = F.cross_entropy(traj_pred, traj.long(), ignore_index = 250)
        else:
            loss_val = F.mse_loss(traj_pred.squeeze(), traj.float())
        
        self.log('loss', loss_val, on_step=False, on_epoch=True)
        return {'loss' : loss_val}
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]

