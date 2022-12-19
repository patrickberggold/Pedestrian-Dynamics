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
        config: dict,
        train_config: dict,
        ):
        super(Image2ImageModule, self).__init__()
        self.mode = config['mode']
        self.arch = config['arch']
        
        assert self.mode in ['grayscale', 'evac_only', 'class_movie', 'density_reg', 'density_class', 'denseClass_wEvac'], 'Unknown mode setting!'
        assert self.arch in ['DeepLab', 'BeIT', 'SegFormer'], 'Unknown arch setting!'

        self.learning_rate = train_config['learning_rate']
        self.lr_scheduler = train_config['lr_scheduler']
        self.lr_sch_step_size4lr_step = train_config['lr_sch_step_size4lr_step']
        self.lr_sch_step_size4cosAnneal = train_config['lr_sch_step_size4cosAnneal']
        self.lr_sch_gamma4redOnPlat_and_stepLR = train_config['lr_sch_gamma4redOnPlat_and_stepLR']
        self.lr_sch_gamma4expLR = train_config['lr_sch_gamma4expLR']
        self.lr_sch_patience4redOnPlat = train_config['lr_sch_patience4redOnPlat']
        # self.lr_sch_step_size = lr_sch_step_size if lr_scheduler=='StepLR' else 50
        # self.lr_sch_gamma = lr_sch_gamma
        self.opt = train_config['opt']
        self.weight_decay = train_config['weight_decay']
        self.additional_info = config['additional_info']

        self.num_heads = 1
        assert self.lr_scheduler in [CosineAnnealingLR.__name__, StepLR.__name__, ExponentialLR.__name__, ReduceLROnPlateau.__name__], 'Unknown LR Scheduler!'

        self.train_losses = {}
        self.train_losses_per_epoch = {}
        self.val_losses = {}
        self.val_losses_per_epoch = {}
        self.loss_func_params = train_config['loss_dict']

        if self.mode in ['grayscale', 'evac_only', 'grayscale_norm_evac']:
            # Regression task
            self.output_channels = 1
        elif self.mode == 'class_movie':
            self.output_channels = 3 # number of classes
            self.num_heads = 10
            # Class weights: class imbalance in file C:\Users\Remotey\Documents\Datasets\ADVANCED_FLOORPLANS_SPARSE\SPARSE_GT_VELOCITY_MASKS_thickness_5_nframes_10\class_statistics.txt
            self.velocity_class_weights = torch.tensor([1., 20., 20.])
        elif self.mode in ['density_reg', 'density_class']:
            self.num_heads = 8
            if self.mode=='density_reg':
                self.output_channels = 1
            else:
                self.output_channels = 5
        elif self.mode == 'denseClass_wEvac':
            self.output_channels = 5
            self.num_heads = 8
        else:
            raise NotImplementedError
        
        self.log_result = {'validation': [], 'training': []}
        self.backbone_frozen = False

        # if self.mode=='evac':
            # self.automatic_optimization = False

        if self.arch == 'DeepLab':
            self.model = DeepLabTraj(self.mode, self.output_channels, self.num_heads)
        elif self.arch == 'BeIT':
            self.model = BeITTraj(self.mode, self.output_channels, self.num_heads, additional_info=self.additional_info)
        elif self.arch == 'SegFormer':
            self.model = SegFormerTraj(self.mode, self.output_channels)

        # self.net = deeplabv3_resnet50(pretrained = False, progress = True, output_channels = self.output_channels, relu_at_end = True, num_heads=self.num_heads, pred_evac_time=self.pred_evac_time)


    def dice_loss(self, gt, logits, eps=1e-7):
        # gt_img = gt[:, 0].unsqueeze(1)
        # lgt_img = logits[:, :, 0]

        # true_1_hot = torch.eye(3)[gt_img.squeeze(1)]
        # true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        true_1_hot = torch.eye(self.output_channels)[gt.unsqueeze(1).squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        intersection = torch.sum(probas * true_1_hot, (0, 2, 3, 4))
        cardinality = torch.sum(probas + true_1_hot, (0, 2, 3, 4))
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        
        return (1 - dice_loss)

    
    def tversky_loss(self, gt, logits, eps=1e-7):
        alpha = self.loss_func_params['alpha']
        beta = self.loss_func_params['beta']
        # alpha false positive penalty, lower?
        # beta false negative penalty, higher?
        # focal tversky loss: https://arxiv.org/pdf/1810.07842.pdf

        true_1_hot = torch.eye(self.output_channels)[gt.unsqueeze(1).squeeze(1)] # GT: 4, 8, 160, 160
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        intersection = torch.sum(probas * true_1_hot, (0, 2, 3, 4))
        fps = torch.sum(probas * (1 - true_1_hot), (0, 2, 3, 4))
        fns = torch.sum((1 - probas) * true_1_hot, (0, 2, 3, 4))
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_loss = (num / (denom + eps)).mean()
        return (1 - tversky_loss)


    def focal_tversky_loss(self, gt, logits, gamma, eps=1e-7):
        return self.tversky_loss(gt, logits, eps)**(1./gamma)


    def forward(self, x, *args):
        return self.model(x, *args)

    def training_step(self, batch, batch_idx: int):
        # TODO maybe implement scheduler change once backbone is unfreezed
        if not self.additional_info:
            img, traj = batch
        else:
            img, traj, add_info = batch

        if self.mode=='evac':
            # for several optimizers: https://pytorch-lightning.readthedocs.io/en/stable/model/manual_optimization.html
            opt = self.optimizers()
            # opt, opt_evac = self.optimizers()
            if not self.automatic_optimization:
                self.backbone.train()
                self.image_head.train()
                self.evac_head.train()
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
            train_loss = img_loss + evac_loss/100.

            if not self.automatic_optimization:
                self.manual_backward(train_loss)
                opt.step()
                # opt_evac.step()

            self.internal_log({'img_loss': img_loss, 'evac_loss': evac_loss}, stage='train')
            self.log("loss", train_loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
            # self.log("loss", {'loss': train_loss, 'img_loss': img_loss, 'evac_loss': evac_loss/1000.}, on_step=True, on_epoch=True, prog_bar=True, logger=False)

            return {'loss' : train_loss, 'img_loss': img_loss.data, 'evac_loss': evac_loss.data}
        
        elif self.mode=='evac_only':
            img, evac_time = img
            if not self.additional_info:
                evac_time_pred = self.forward(img.float())
            else:
                evac_time_pred = self.forward(img.float(), add_info)
            train_loss = F.mse_loss(evac_time_pred.squeeze(), evac_time.float())

            evac_l1_loss = F.l1_loss(evac_time_pred.clone().detach(), evac_time.clone().detach())

            self.internal_log({'evac_loss': train_loss, 'L1_loss': evac_l1_loss}, stage='train')
            self.log("loss", train_loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
            return {'loss' : train_loss}

        elif self.mode == 'class_movie':
            traj_pred = self.forward(img.float())

            # train_loss = torch.nn.CrossEntropyLoss(self.velocity_class_weights.to(traj_pred.device))(traj_pred, traj)
            train_loss = self.tversky_loss(traj, traj_pred)
            self.internal_log({'train_loss': train_loss}, stage='train')
            self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
            return {'loss' : train_loss}
        elif self.mode in ['density_reg', 'density_class']:
            traj_pred = self.forward(img.float())
            if self.mode == 'density_reg':
                train_loss = F.mse_loss(traj_pred, traj)
            else:
                train_loss = self.tversky_loss(traj, traj_pred.permute(0,1,4,2,3))
                # train_loss = torch.nn.CrossEntropyLoss()(traj_pred, traj)
            self.internal_log({'train_loss': train_loss}, stage='train')
            self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
            return {'loss' : train_loss}
        
        elif self.mode == 'denseClass_wEvac':
            traj, evac_time = traj
            traj_pred, evac_time_pred = traj_pred, evac_time_pred = self.forward(img, add_info) if self.additional_info else self.forward(img)

            img_loss = self.tversky_loss(traj, traj_pred)
            evac_loss = F.mse_loss(evac_time_pred.squeeze(), evac_time.float())

            train_loss = img_loss*100 + evac_loss
            evac_l1_loss = F.l1_loss(evac_time_pred.clone().detach().squeeze(), evac_time.clone().detach())
            evac_perc_loss = torch.abs(evac_time_pred.clone().detach().squeeze() - evac_time.clone().detach()) / evac_time.clone().detach()
            evac_perc_loss = evac_perc_loss.mean()

            self.internal_log({'img_loss': img_loss, 'evac_loss': evac_loss, 'evac_L1': evac_l1_loss, 'evac_perc': evac_perc_loss}, stage='train')
            # self.internal_log({'img_loss': img_loss, 'evac_loss': evac_loss}, stage='train')
            self.log('loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            return {'loss' : train_loss}

        else:
            traj_pred = self.forward(img)
            train_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
            
            self.internal_log({'train_loss': train_loss}, stage='train')
            self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
            return {'loss' : train_loss}

    def validation_step(self, batch, batch_idx: int) -> None:

        if not self.additional_info:
            img, traj = batch
        else:
            img, traj, add_info = batch

        if self.mode=='evac':
            if not self.automatic_optimization:
                self.backbone.eval()
                self.image_head.eval()
                self.evac_head.eval()

            img, evac_time = img
            traj_pred, evac_time_pred = self.forward(img.float())
            img_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
            evac_loss = F.mse_loss(evac_time_pred.squeeze(), evac_time.float())

            val_loss = img_loss + evac_loss/100.

            self.internal_log({'img_loss': img_loss, 'evac_loss': evac_loss}, stage='val')
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
            return {'val_loss' : val_loss, 'img_loss': img_loss, 'evac_loss': evac_loss}

        elif self.mode=='evac_only':
            img, evac_time = img
            if not self.additional_info:
                evac_time_pred = self.forward(img.float())
            else:
                evac_time_pred = self.forward(img.float(), add_info).squeeze()
            val_loss = F.mse_loss(evac_time_pred, evac_time.float())

            evac_l1_loss = F.l1_loss(evac_time_pred.clone().detach().squeeze(), evac_time.clone().detach())

            self.internal_log({'evac_loss': val_loss, 'L1_loss': evac_l1_loss}, stage='val')
            self.log("val_loss", val_loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
            return {'val_loss': val_loss}

        elif self.mode == 'class_movie':
            traj_pred = self.forward(img.float())
            # val_loss = torch.nn.CrossEntropyLoss(self.velocity_class_weights.to(traj_pred.device))(traj_pred, traj)
            val_loss = self.tversky_loss(traj, traj_pred)
            self.internal_log({'val_loss': val_loss}, stage='val')
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            return {'val_loss': val_loss}

        elif self.mode in ['density_reg', 'density_class']:
            traj_pred = self.forward(img.float())
            if self.mode == 'density_reg':
                val_loss = F.mse_loss(traj_pred, traj)
            else:
                val_loss = self.tversky_loss(traj, traj_pred.permute(0,1,4,2,3))
                # val_loss = torch.nn.CrossEntropyLoss()(traj_pred, traj)
            self.internal_log({'val_loss': val_loss}, stage='val')
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            return {'val_loss' : val_loss}
        
        elif self.mode == 'denseClass_wEvac':
            traj, evac_time = traj
            traj_pred, evac_time_pred = self.forward(img, add_info) if self.additional_info else self.forward(img)

            img_loss = self.tversky_loss(traj, traj_pred)
            evac_loss = F.mse_loss(evac_time_pred.squeeze(), evac_time.float())

            val_loss = img_loss*100 + evac_loss
            evac_l1_loss = F.l1_loss(evac_time_pred.clone().detach().squeeze(), evac_time.clone().detach())
            evac_perc_loss = torch.abs(evac_time_pred.clone().detach().squeeze() - evac_time.clone().detach()) / evac_time.clone().detach()
            evac_perc_loss = evac_perc_loss.mean()

            self.internal_log({'img_loss': img_loss, 'evac_loss': evac_loss, 'evac_L1': evac_l1_loss, 'evac_perc': evac_perc_loss}, stage='val')
            # self.internal_log({'img_loss': img_loss, 'evac_loss': evac_loss}, stage='val')
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            return {'val_loss' : val_loss}

        else:
            traj_pred = self.forward(img)
            val_loss = F.mse_loss(traj_pred.squeeze(), traj.float())
            
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
            sch = CosineAnnealingLR(opt, T_max = self.lr_sch_step_size4cosAnneal)
            # if self.mode == 'evac': sch_evac = CosineAnnealingLR(opt_evac, T_max = self.lr_sch_step_size)
        elif self.lr_scheduler == StepLR.__name__:
            sch = StepLR(opt, step_size=self.lr_sch_step_size4lr_step, gamma=self.lr_sch_gamma4redOnPlat_and_stepLR)
            # if self.mode == 'evac': sch_evac = StepLR(opt_evac, step_size=self.lr_sch_step_size, gamma=self.lr_sch_gamma)
        elif self.lr_scheduler == ExponentialLR.__name__:
            sch = ExponentialLR(opt, gamma=self.lr_sch_gamma4expLR)
            # if self.mode == 'evac': sch_evac = ExponentialLR(opt_evac, gamma=0.9)
        elif self.lr_scheduler == ReduceLROnPlateau.__name__:
            sch = ReduceLROnPlateau(opt, factor=self.lr_sch_gamma4redOnPlat_and_stepLR, patience=self.lr_sch_patience4redOnPlat)
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

        return super().on_fit_start()
        print(f"\nFREEZE BACKBONE INITIALLY IN EPOCH {self.current_epoch}\n")

        for key, module in self.model.model._modules.items():
            if key == 'auxiliary_head': continue
            
            # if key != 'high_res_net':
            #     for p in module.parameters():
            #         p.requires_grad = False
            if key != 'evac_head': # 'classifier':
                for p in module.parameters():
                    p.requires_grad = False
            # else:
            #     for k, m in module._modules.items():
            #         if k == 'classifier':
            #             for pp in m.parameters():
            #                 pp.requires_grad = False
            continue
        return super().on_fit_start()

    def on_epoch_end(self) -> None:
        # if self.current_epoch == 1:
        #     print(f'\nUnfreezing all parameters in epoch {self.current_epoch}...')
        #     for param in self.parameters():
        #         param.requires_grad = True

        if self.trainer.state.stage in ['sanity_check', 'train']: return super().on_epoch_end()
        self.print_logs()
    

    def print_logs(self):
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
        train_string = ''
        train_vals = [val for val in self.train_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.train_losses_per_epoch.keys())):
            if id_k == 0:
                train_string += key+':'
            else:
                train_string += '\t\t\t' + key+':'
        for i_epoch in range(len(train_vals[0])):
            for i_loss in range(len(train_vals)):
                if i_loss == 0:
                    train_string += f'\n{train_vals[i_loss][i_epoch]:.5f}'
                else:
                    train_string += f'\t\t\t\t{train_vals[i_loss][i_epoch]:.5f}'
        print(train_string) 


        print('\nVALIDATION RESULT:')
        val_string = ''
        val_vals = [val for val in self.val_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.val_losses_per_epoch.keys())):
            if id_k == 0:
                val_string += key+':'
            else:
                val_string += '\t\t\t' + key+':'
        for i_epoch in range(len(val_vals[0])):
            for i_loss in range(len(val_vals)):
                if i_loss == 0:
                    val_string += f'\n{val_vals[i_loss][i_epoch]:.5f}'
                else:
                    val_string += f'\t\t\t\t{val_vals[i_loss][i_epoch]:.5f}'
        print(val_string) 
        