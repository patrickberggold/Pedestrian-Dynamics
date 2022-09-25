import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau
from torch.optim import Adam, AdamW
import pytorch_lightning as pl
from torchsummary import summary
from models import deeplabv3_resnet50
# from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
from .deeplab_traj_pred import DeepLabTraj
from .beit_traj_pred import BeITTraj
from .segformer_traj_pred import SegFormerTraj

class Image2ImageModule(pl.LightningModule):
    def __init__(
        self, 
        mode: str,
        arch: str, 
        learning_rate: float = 1e-3, 
        lr_scheduler: str = CosineAnnealingLR.__name__, 
        lr_sch_step_size: int = 8,
        lr_sch_gamma: float = 0.3, 
        opt: str = 'Adam',
        weight_decay = None, # 1e-6,
        unfreeze_backbone_at_epoch: int = 8,
        alternate_unfreezing: int = False,
        relu_at_end: bool = True,
        p_dropout = None,
        num_heads: int = 1
        ):
        super(Image2ImageModule, self).__init__()
        
        assert mode in ['grayscale', 'grayscale_movie', 'evac'], 'Unknown mode setting!'
        assert arch in ['DeepLab', 'BeIT', 'SegFormer'], 'Unknown arch setting!'
        
        self.mode = mode
        self.arch = arch
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_sch_step_size = lr_sch_step_size if lr_scheduler=='StepLR' else 50
        self.lr_sch_gamma = lr_sch_gamma
        self.weight_decay = weight_decay
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.alternate_unfreezing = alternate_unfreezing
        self.relu_at_end = relu_at_end
        self.opt = opt
        self.num_heads = 1
        self.pred_evac_time = True if self.mode=='evac' else False
        assert self.lr_scheduler in [CosineAnnealingLR.__name__, StepLR.__name__, ExponentialLR.__name__, ReduceLROnPlateau.__name__], 'Unknown LR Scheduler!'      
        
        self.train_losses = {}
        self.train_losses_per_epoch = {}
        self.val_losses = {}
        self.val_losses_per_epoch = {}

        if self.mode in ['grayscale', 'evac']:
            # Regression task
            self.output_channels = 1
        elif self.mode == 'grayscale_movie':
            self.output_channels = 1
            assert num_heads > 1 and isinstance(num_heads, int)
            self.num_heads = num_heads
        else:
            raise ValueError
        
        self.log_result = {'validation': [], 'training': []}
        self.backbone_frozen = False
        self.p_dropout = p_dropout

        if self.pred_evac_time:
            self.automatic_optimization = False

        if arch == 'DeepLab':
            self.model = DeepLabTraj(self.mode, self.output_channels, self.relu_at_end, self.p_dropout, self.num_heads)
        elif arch == 'BeIT':
            self.model = BeITTraj(self.mode, self.output_channels, self.relu_at_end, self.p_dropout, self.num_heads)
        elif arch == 'SegFormer':
            self.model = SegFormerTraj(self.mode, self.output_channels, self.relu_at_end, self.p_dropout, self.num_heads)

        # self.net = deeplabv3_resnet50(pretrained = False, progress = True, output_channels = self.output_channels, relu_at_end = self.relu_at_end, num_heads=self.num_heads, pred_evac_time=self.pred_evac_time)

        # for idx, child in enumerate(self.net.children()):
        #     if idx == 1:
        #         for param in child[1].parameters():
        #             param.requires_grad = False

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
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        # TODO maybe implement scheduler change once backbone is unfreezed
        img, traj = batch

        if self.pred_evac_time:
            self.backbone.train()
            self.image_head.train()
            self.evac_head.train()

            # for several optimizers: https://pytorch-lightning.readthedocs.io/en/stable/model/manual_optimization.html
            opt = self.optimizers()
            # opt, opt_evac = self.optimizers()
            opt.zero_grad()
            # opt_evac.zero_grad()

            img, evac_time = img
            traj_pred, evac_time_pred = self.forward(img.float())
            """ # train evac
            self.backbone.requires_grad_(False)
            self.image_head.requires_grad_(False)
            self.evac_head.requires_grad_(True)

            evac_loss = F.mse_loss(evac_time_pred.squeeze(), evac_time.float())
            self.manual_backward(evac_loss, retain_graph=True)
            opt_evac.step()

            # train img
            self.backbone.requires_grad_(not self.backbone_frozen)
            self.image_head.requires_grad_(True)
            self.evac_head.requires_grad_(False)

            img_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
            self.manual_backward(img_loss)
            opt.step()

            # if .data is neglected, autograd will be stored in the returned loss throughout each epoch, which accumulates memory and leads to OOM errors
            train_loss = img_loss.data + evac_loss.data/1000. """

            """ Test this """
            img_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
            evac_loss = F.mse_loss(evac_time_pred.squeeze(), evac_time.float())
            train_loss = img_loss + evac_loss/1000.

            self.manual_backward(train_loss)
            opt.step()
            # opt_evac.step()

            self.internal_log({'train_loss': train_loss}, stage='train')
            self.log("train_loss", train_loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
            # self.log("loss", {'loss': train_loss, 'img_loss': img_loss, 'evac_loss': evac_loss/1000.}, on_step=True, on_epoch=True, prog_bar=True, logger=False)

            # loss_dict = {'img_loss': img_loss, 'evac_loss': evac_loss}
            # losses = sum(loss for loss in loss_dict.values())
            # return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}


            # self.log('loss', train_loss, on_step=False, on_epoch=True)
            # # self.log('loss', {"img_loss": img_loss, "evac_loss": evac_loss}, img_loss, on_step=False, on_epoch=True)
            return {'train_loss' : train_loss, 'img_loss': img_loss.data, 'evac_loss': evac_loss.data}
        
        else:
            traj_pred = self.forward(img.float())
            if self.mode in ['grayscale']:
                train_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
            elif self.mode == 'grayscale_movie':
                train_loss = sum([F.mse_loss(traj_pred_ts.squeeze(), traj[idx].float()) for idx, traj_pred_ts in enumerate(traj_pred)])
            
            self.internal_log({'train_loss': train_loss}, stage='train')
            self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
            return {'loss' : train_loss}

    def validation_step(self, batch, batch_idx: int) -> None:

        img, traj = batch

        if self.pred_evac_time:
            self.backbone.eval()
            self.image_head.eval()
            self.evac_head.eval()

            img, evac_time = img
            traj_pred, evac_time_pred = self.forward(img.float())
            img_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
            evac_loss = F.mse_loss(evac_time_pred.squeeze(), evac_time.float())

            val_loss = img_loss + evac_loss/1000.

            self.internal_log({'val_loss': val_loss}, stage='val')
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
            return {'val_loss' : val_loss, 'img_loss': img_loss, 'evac_loss': evac_loss}

        else:
            traj_pred = self.forward(img.float())
            if self.mode == 'grayscale':
                val_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
            elif self.mode == 'grayscale_movie':
                val_loss = sum([F.mse_loss(traj_pred_ts.squeeze(), traj[idx].float()) for idx, traj_pred_ts in enumerate(traj_pred)])

            self.internal_log({'val_loss': val_loss}, stage='val')
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            return {'val_loss': val_loss}


    def internal_log(self, losses_it, stage):
        if self.trainer.state.stage == 'sanity_check': return

        losses_logger = self.train_losses if stage=='train' else self.val_losses

        for key, val in losses_it.items():
            if key not in losses_logger:
                losses_logger.update({key: [val]})
            else:
                losses_logger[key].append(val)


    def configure_optimizers(self):
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
            sch = CosineAnnealingLR(opt, T_max = self.lr_sch_step_size)
            # if self.mode == 'evac': sch_evac = CosineAnnealingLR(opt_evac, T_max = self.lr_sch_step_size)
        elif self.lr_scheduler == StepLR.__name__:
            sch = StepLR(opt, step_size=self.lr_sch_step_size, gamma=self.lr_sch_gamma)
            # if self.mode == 'evac': sch_evac = StepLR(opt_evac, step_size=self.lr_sch_step_size, gamma=self.lr_sch_gamma)
        elif self.lr_scheduler == ExponentialLR.__name__:
            sch = ExponentialLR(opt, gamma=1.-self.lr_sch_gamma)
            # if self.mode == 'evac': sch_evac = ExponentialLR(opt_evac, gamma=0.9)
        elif self.lr_scheduler == ReduceLROnPlateau.__name__:
            sch = ReduceLROnPlateau(opt, factor=self.lr_sch_gamma, patience=3)
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

    def on_fit_start(self) -> None:

        return super().on_epoch_start()
        if self.arch == 'DeepLab':
            print('Freezing all layers except new convolutions...')
            for idx, child in enumerate(self.model.backbone.named_children()):
                for param in child[1].parameters():
                    param.requires_grad = False
            for idx, child in enumerate(self.model.image_head.named_children()):
                if int(child[0]) > 5:
                    for param in child[1].parameters():
                        param.requires_grad = True
                else:
                    for param in child[1].parameters():
                        param.requires_grad = False
        return super().on_fit_start()

        # determine whether backbone freezed at all
        if self.unfreeze_backbone_at_epoch:
            if self.current_epoch == 0:
                print(f"\nFREEZE BACKBONE INITIALLY IN EPOCH {self.current_epoch}\n")
                # freeze backbone weights
                # for idx, child in enumerate(self.net.children()):
                #     if idx == 0:
                #         for param in child.parameters():
                #             param.requires_grad = False
                self.backbone_frozen = True
            else:
                # if alternate freeze
                if self.alternate_unfreezing:
                    if self.current_epoch % self.unfreeze_backbone_at_epoch == 0:
                        if self.backbone_frozen:
                            print(f"\nUNFREEZING BACKBONE IN EPOCH {self.current_epoch}\n")
                            # unfreeze backbone
                            # for idx, child in enumerate(self.net.children()):
                            #     if idx == 0:
                            #         for param in child.parameters():
                            #             param.requires_grad = True
                            self.backbone_frozen = False
                        else:
                            # freeze backbone
                            print(f"\nFREEZING BACKBONE IN EPOCH {self.current_epoch}\n")
                            # for idx, child in enumerate(self.net.children()):
                            #     if idx == 0:
                            #         for param in child.parameters():
                            #             param.requires_grad = False
                            self.backbone_frozen = True
                
                # if unfreeze after number of epochs
                else:
                    if self.current_epoch == self.unfreeze_backbone_at_epoch:
                        print(f"\nUNFREEZING BACKBONE AFTER {self.unfreeze_backbone_at_epoch} EPOCHS\n")
                        # unfreeze backbone weights
                        for idx, child in enumerate(self.net.children()):
                            if idx == 0:
                                for param in child.parameters():
                                    param.requires_grad = True


    # def validation_epoch_end(self, outputs) -> None:
    #     if self.trainer.state.stage == 'sanity_check' or isinstance(outputs[0], torch.Tensor): 
    #         return super().validation_epoch_end(outputs)

    #     feedback_string = ''
    #     keys = list(outputs[0].keys())
    #     feedback_dict = {key: 0. for key in keys}
    #     for output in outputs:
    #         for key in keys:
    #             feedback_dict[key] += output[key]
    #         # loss = sum(output[key] for output in outputs) / len(outputs)
    #     for key, loss in feedback_dict.items():
    #         feedback_string += f'{key}: {loss/len(outputs):.3f}, '
        
    #     # print(f'\Valid result {self.current_epoch}: ', feedback_string[:-1]+'\n')
    #     self.log_result['validation'].append(feedback_string[:-1])


    # def training_epoch_end(self, outputs) -> None:
    #     if isinstance(outputs[0], torch.Tensor) or isinstance(outputs[0], torch.Tensor):
    #         return super().training_epoch_end(outputs)

    #     feedback_string = ''
    #     keys = list(outputs[0].keys())
    #     feedback_dict = {key: 0. for key in keys}
    #     for output in outputs:
    #         for key in keys:
    #             feedback_dict[key] += output[key]
    #         # loss = sum(output[key] for output in outputs) / len(outputs)
    #     for key, loss in feedback_dict.items():
    #         feedback_string += f'{key}: {loss/len(outputs):.3f}, '
    #     # print(f'\nTrain result epoch {self.current_epoch}: ', feedback_string[:-1]+'\n')
    #     self.log_result['training'].append(feedback_string[:-1])

    # def on_fit_end(self):
    #     print('VALIDATION RESULT:')
    #     for result in self.log_result['validation']:
    #         print(result) 
    #     print('\nTRAINING RESULT:')
    #     for result in self.log_result['training']:
    #         print(result) 

    def on_epoch_end(self) -> None:
        if self.trainer.state.stage in ['sanity_check', 'train']: return super().on_epoch_end()

        # Training Logs
        for key, val in self.train_losses.items():
            if key not in self.train_losses_per_epoch:
                mean = sum(val)/len(val)
                self.train_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.train_losses_per_epoch[key].append(sum(val).item()/len(val))

        # Validation logs
        for key, val in self.val_losses.items():
            if key not in self.val_losses_per_epoch:
                mean = sum(val)/len(val)
                self.val_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.val_losses_per_epoch[key].append(sum(val).item()/len(val))

        # Reset
        self.train_losses = {}
        self.val_losses = {}
        
        print('\nTRAINING RESULT:')
        for result in self.train_losses_per_epoch['train_loss']:
            print(result) 

        print('\nVALIDATION RESULT:')
        for result in self.val_losses_per_epoch['val_loss']:
            print(result)
        print('')
        