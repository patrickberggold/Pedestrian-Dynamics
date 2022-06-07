from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from torch.optim import Adam
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

    def training_step(self, batch, batch_idx: int):

        img, traj = batch
        img = img.float()
        traj_pred = self.forward(img)['out']

        if self.mode == 'bool' or self.mode == 'segmentation':
            train_loss = F.cross_entropy(traj_pred, traj.long(), ignore_index = 250)
        else:
            train_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
        
        self.log('loss', train_loss, on_step=False, on_epoch=True)
        return {'loss' : train_loss}

    def validation_step(self, batch, batch_idx: int) -> None:

        img, traj = batch
        img = img.float()
        traj_pred = self.forward(img)['out']

        if self.mode == 'bool' or self.mode == 'segmentation':
            val_loss = F.cross_entropy(traj_pred, traj.long(), ignore_index = 250)
        else:
            val_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
        
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
        # TODO implement other schedulers
        sch = CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]

    # def configure_callbacks(self):
    #     early_stop = EarlyStopping(monitor="val_acc", mode="max")
    #     checkpoint = ModelCheckpoint(monitor="val_loss")
    #     return [early_stop, checkpoint]