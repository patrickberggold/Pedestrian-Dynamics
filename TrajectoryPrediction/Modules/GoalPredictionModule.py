from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau
from torch.optim import Adam, AdamW
import pytorch_lightning as pl
import numpy as np
import torch
from Modules.GoalPredictionModels.goal_prediction_model import GoalPredictionModel
from TrajectoryPrediction.Modules.coca.vae_goal import ResNet18Adaption

class GoalPredictionModule(pl.LightningModule):
    def __init__(
        self, 
        config: dict,
        train_config: dict,
        ):
        super(GoalPredictionModule, self).__init__()
        self.mode = config['mode']
        self.config = config
        self.arch = config['goal_predictor']

        assert self.arch in ['unet', 'resnet18'], 'Unknown Goal Predictor!'
        
        self.learning_rate = train_config['learning_rate']
        self.lr_scheduler = train_config['lr_scheduler']
        self.lr_sch_step_size4lr_step = train_config['lr_sch_step_size4lr_step']
        self.lr_sch_step_size4cosAnneal = train_config['lr_sch_step_size4cosAnneal']
        self.lr_sch_gamma4redOnPlat_and_stepLR = train_config['lr_sch_gamma']
        self.lr_sch_gamma4expLR = train_config['lr_sch_gamma']
        self.lr_sch_patience4redOnPlat = train_config['lr_sch_patience4redOnPlat']
        self.opt = train_config['opt']
        self.init = train_config['init']
        self.save_results = config['save_results']
        self.txt_path = config['store_path'] if self.save_results and 'store_path' in config else None

        self.weight_decay = 0.0
        resize_factor = 1.0
        self.img_mean, self.img_std = 1280 // resize_factor / 2, 150 / resize_factor

        assert self.lr_scheduler in [CosineAnnealingLR.__name__, StepLR.__name__, ExponentialLR.__name__, ReduceLROnPlateau.__name__], 'Unknown LR Scheduler!'

        self.train_losses = {}
        self.train_losses_per_epoch = {}
        self.val_losses = {}
        self.val_losses_per_epoch = {}
        
        
        self.log_result = {'validation': [], 'training': []}
        self.backbone_frozen = False

        if self.mode == 'GOAL_PRED':
            self.model = GoalPredictionModel()
        elif self.mode == 'SEGMENTATION':
            self.model = ResNet18Adaption()
        
        self.model.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if hasattr(m, 'weight'):
            try:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    # torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                    # torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
                    torch.nn.init.xavier_uniform_(m.weight)
                else:
                    raise NotImplementedError(f'Weight initialization for {m.__repr__()} not implemented yet!')
            except ValueError:
                # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        elif hasattr(m, 'bias'):
            torch.nn.init.zeros_(m.bias)


    def forward(self, x, *args):
        return self.model(x, *args)

    def training_step(self, batch, batch_idx: int):
        image = batch['image'] if 'image' in batch else None
        obs_coords = batch['coords'].squeeze(0) if 'obs_coords' in batch else None
        goal_coords = batch['goal_coords'].squeeze(0) if 'goal_coords' in batch else None
        semanticMaps_egocentric = batch['semanticMaps_egocentric'].permute(1, 0, 2, 3) if 'semanticMaps_egocentric' in batch else None
        goalMaps_egocentric = batch['goalMaps_egocentric'].permute(3, 0, 1, 2) if 'goalMaps_egocentric' in batch else None
        transformed_agent_destinations = batch['transformed_agent_destinations'].squeeze(0) if 'transformed_agent_destinations' in batch else None
        semanticMaps_per_agent = batch['semanticMaps_per_agent'].permute(3, 0, 1, 2) if 'semanticMaps_per_agent' in batch else None
        globalOccupancyMap = batch['globalOccupancyMap'] if 'globalOccupancyMap' in batch else None
        
        if self.mode == 'GOAL_PRED':
            withGlobalSemMap = True
            if withGlobalSemMap:
                output = self.model(obs_coords, transformed_agent_destinations, semanticMaps_egocentric)
            else:
                output = self.model(obs_coords, goalMaps_egocentric, semanticMaps_egocentric)
            
            if isinstance(output, torch.Tensor):
                train_loss = torch.nn.MSELoss()(output, goal_coords)
                mean_abs_error = torch.abs((output.clone().detach()  * self.img_std + self.img_mean) - (goal_coords.clone() * self.img_std + self.img_mean)).mean()        
                self.internal_log({'train_loss': train_loss, 'abs_error': mean_abs_error}, stage='train')
            elif isinstance(output, tuple):
                reconstr, mu, logvar = output[0], output[1], output[2]
                KLE_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                mse_loss = torch.nn.MSELoss()(reconstr, goal_coords) 
                train_loss = mse_loss + KLE_loss
                mean_abs_error = torch.abs((reconstr.clone().detach()  * self.img_std + self.img_mean) - (goal_coords.clone() * self.img_std + self.img_mean)).mean()
                self.internal_log({'mse_loss': mse_loss, 'KLE_loss': KLE_loss, 'abs_error': mean_abs_error}, stage='train')
        
        elif self.mode == 'SEGMENTATION':
            """ output_dstMap, _ = self.model(semanticMaps_per_agent, mode=0)
            output_egocentric, _ = self.model(semanticMaps_egocentric, mode=1)
            dstMap_loss = torch.nn.MSELoss()(output_dstMap, semanticMaps_per_agent)
            egocentric_loss = torch.nn.MSELoss()(output_egocentric, semanticMaps_egocentric)
            train_loss = dstMap_loss + 100*egocentric_loss
            self.internal_log({'train_loss': train_loss, 'dstMap_loss': dstMap_loss, 'egocentric_loss': egocentric_loss}, stage='train') """

        self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
        return {'loss' : train_loss}

    def validation_step(self, batch, batch_idx: int) -> None:
        image = batch['image'] if 'image' in batch else None
        obs_coords = batch['coords'].squeeze(0) if 'obs_coords' in batch else None
        goal_coords = batch['goal_coords'].squeeze(0) if 'goal_coords' in batch else None
        semanticMaps_egocentric = batch['semanticMaps_egocentric'].permute(1, 0, 2, 3) if 'semanticMaps_egocentric' in batch else None
        goalMaps_egocentric = batch['goalMaps_egocentric'].permute(3, 0, 1, 2) if 'goalMaps_egocentric' in batch else None
        transformed_agent_destinations = batch['transformed_agent_destinations'].squeeze(0) if 'transformed_agent_destinations' in batch else None
        semanticMaps_per_agent = batch['semanticMaps_per_agent'].permute(3, 0, 1, 2) if 'semanticMaps_per_agent' in batch else None
        globalOccupancyMap = batch['globalOccupancyMap'] if 'globalOccupancyMap' in batch else None
        
        if self.mode == 'GOAL_PRED':
            withGlobalSemMap = True
            if withGlobalSemMap:
                output = self.model(obs_coords, transformed_agent_destinations, semanticMaps_egocentric)
            else:
                output = self.model(obs_coords, goalMaps_egocentric, semanticMaps_egocentric)

            if isinstance(output, torch.Tensor):
                val_loss = torch.nn.MSELoss()(output, goal_coords)
                mean_abs_error = torch.abs((output.clone().detach() * self.img_std + self.img_mean) - (goal_coords.clone() * self.img_std + self.img_mean)).mean()
                self.internal_log({'val_loss': val_loss, 'abs_error': mean_abs_error}, stage='val')
            elif isinstance(output, tuple):
                reconstr, mu, logvar = output[0], output[1], output[2]
                KLE_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                mse_loss = torch.nn.MSELoss()(reconstr, goal_coords) 
                val_loss = mse_loss + KLE_loss
                mean_abs_error = torch.abs((reconstr.clone().detach()  * self.img_std + self.img_mean) - (goal_coords.clone() * self.img_std + self.img_mean)).mean()
                self.internal_log({'mse_loss': mse_loss, 'KLE_loss': KLE_loss, 'abs_error': mean_abs_error}, stage='val')
            
        elif self.mode == 'SEGMENTATION':
            """ output_dstMap, _ = self.model(semanticMaps_per_agent, mode=0)
            output_egocentric, _ = self.model(semanticMaps_egocentric, mode=1)
            dstMap_loss = torch.nn.MSELoss()(output_dstMap, semanticMaps_per_agent)
            egocentric_loss = torch.nn.MSELoss()(output_egocentric, semanticMaps_egocentric)
            val_loss = dstMap_loss + 100*egocentric_loss
            self.internal_log({'val_loss': val_loss, 'dstMap_loss': dstMap_loss, 'egocentric_loss': egocentric_loss}, stage='val') """
      
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

        # return super().on_fit_start()
        print(f"\nFREEZE STRATEGY INITIALIZED IN EPOCH {self.current_epoch}\n")
        module_list = list(self.model._modules.keys())
        # for key, module in self.model._modules.items():
        #     # if key == 'auxiliary_head': continue
        #     # unfreeze
        #     if key in ['final_conv']:
        #         for p in module.parameters():
        #             p.requires_grad = True
        #     # freeze
        #     else:
        #         for p in module.parameters():
        #             p.requires_grad = False

        return super().on_fit_start()


    def on_train_epoch_start(self) -> None:        
        # if self.current_epoch == 10:
        #     print(f'\n\nUnfreezing all parameters at start of epoch {self.current_epoch}...')
        #     for param in self.parameters():
        #         param.requires_grad = True

        if self.trainer.state.stage in ['sanity_check']: return super().on_epoch_end()
        
        if self.current_epoch > 0: 
            self.print_logs()
    

    def print_logs(self):
        # Training Logs
        for key, val in self.train_losses.items():
            if key not in self.train_losses_per_epoch:
                mean = torch.as_tensor(val).nanmean()
                self.train_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.train_losses_per_epoch[key].append(torch.as_tensor(val).nanmean().item())

        # Validation logs
        for key, val in self.val_losses.items():
            if key not in self.val_losses_per_epoch:
                mean = torch.as_tensor(val).nanmean()
                self.val_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.val_losses_per_epoch[key].append(torch.as_tensor(val).nanmean().item())

        # Reset
        self.train_losses = {}
        self.val_losses = {}
        
        train_string = f'TRAINING RESULT:\nEpoch\t'
        train_vals = [val for val in self.train_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.train_losses_per_epoch.keys())):
            if id_k == 0:
                train_string += key+':'
            else:
                train_string += '\t\t' + key+':'
        for i_epoch in range(len(train_vals[0])):
            for i_loss in range(len(train_vals)):
                if i_loss == 0:
                    # train_string += f'\n{i_epoch}:\t{train_vals[i_loss][i_epoch]:.5f}'
                    train_string += f'\n{i_epoch}:\t{train_vals[i_loss][i_epoch]:.3e}'
                else:
                    train_string += f'\t\t{train_vals[i_loss][i_epoch]:.3e}'
                    # train_string += f'\t\t\t{train_vals[i_loss][i_epoch]:.5f}'
        print('\n\n'+train_string) 


        # print('\nVALIDATION RESULT:')
        val_string = f'\nVALIDATION RESULT:\nEpoch\t'
        val_vals = [val for val in self.val_losses_per_epoch.values()]
        for id_k, key in enumerate(list(self.val_losses_per_epoch.keys())):
            if id_k == 0:
                val_string += key+':'
            else:
                val_string += '\t\t' + key+':'
        for i_epoch in range(len(val_vals[0])):
            for i_loss in range(len(val_vals)):
                if i_loss == 0:
                    # val_string += f'\n{i_epoch}:\t{val_vals[i_loss][i_epoch]:.5f}'
                    val_string += f'\n{i_epoch}:\t{val_vals[i_loss][i_epoch]:.3e}'
                else:
                    # val_string += f'\t\t\t{val_vals[i_loss][i_epoch]:.5f}'
                    val_string += f'\t\t{val_vals[i_loss][i_epoch]:.3e}'
        print(val_string+'\n')

        if self.save_results and self.txt_path is not None:
            save_string = train_string+'\n\n'+val_string
            f = open(self.txt_path, 'w')
            f.write(f'Latest learning rate:{self.learning_rate}\n\n')
            f.write(save_string)
            f.close()