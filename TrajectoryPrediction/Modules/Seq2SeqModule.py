from torch import Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau
from torch.optim import Adam
import pytorch_lightning as pl
from Modules.goal.models.goal_sar import Goal_SAR
from Modules.coca.coca4traj import CoCa4Traj
from sys import path 
from Modules.coca.simple_goal import SimpleGoal
from Modules.coca.advanced_goal import AdvancedGoal
from Modules.coca.gan_goal import Generator
import numpy as np
from Modules.transformer.transformer import Transformer
import torch
import os
# from decimal import Decimal

class Seq2SeqModule(pl.LightningModule):
    def __init__(self, config: dict, train_config: dict):
        super(Seq2SeqModule, self).__init__()
        
        self.config = config
        self.mode = config['mode']
        self.arch = config['arch']
        self.img_arch = config['img_arch']
        self.learning_rate = train_config['learning_rate']
        self.lr_scheduler = train_config['lr_scheduler']
        self.lr_sch_step_size = train_config['lr_sch_step_size'] # 100 if lr_scheduler==CosineAnnealingLR.__name__ else 5 # lr_sch_step_size
        self.lr_sch_gamma = train_config['lr_sch_gamma']
        self.lr_sch_patience4redOnPlat = train_config['lr_sch_patience4redOnPlat']
        self.traj_quantity = config['traj_quantity']
        self.dim = train_config['dim']
        self.num_enc_layers = train_config['num_enc_layers']
        self.num_dec_layers = train_config['num_dec_layers']
        self.ff_mult = train_config['ff_mult']
        self.init = train_config['init']
        ###
        separate_obs_agent_batches = train_config['separate_obs_agent_batches']
        predict_additive = train_config['predict_additive']

        # decoder_mode: 0=teacher forcing, 1=running free, 2=scheduled sampling
        self.decoder_mode_training = 0
        self.decoder_mode_validation = 1 if self.mode == 'TRAJ_PRED' else 0

        assert self.lr_scheduler in [CosineAnnealingLR.__name__, StepLR.__name__, ExponentialLR.__name__, ReduceLROnPlateau.__name__], 'Unknown LR Scheduler!'
        assert self.arch in ['goal', 'tf', 'coca', 'coca_goal', 'simple_goal', 'gan_goal', 'adv_goal'], 'Unknown architecture!'
  
        if self.arch == 'goal':
            self.model = Goal_SAR(config=config, traj_quantity=self.traj_quantity)
        elif self.arch == 'tf':
            self.model = Transformer(enc_inp_size=2, dec_inp_size=3, dec_out_size=3, dropout=0.3, traj_quantity=self.traj_quantity)
        elif self.arch == 'simple_goal':
            self.model = SimpleGoal(
                config = config,
                dim = self.dim,
                num_enc_layers = self.num_enc_layers,
                num_dec_layers = self.num_dec_layers,
                ff_mult = self.ff_mult,
                init = self.init,
                ###
                pretrained_vision=True, predict_additive=predict_additive, separate_obs_agent_batches=separate_obs_agent_batches
            )
            if self.model.pretrained_vision and hasattr(self.model, 'img_encoder'):
                self.model._modules['img_encoder'].requires_grad_(False)
        elif self.arch == 'adv_goal':
            self.model = AdvancedGoal(
                config = config,
                dim = self.dim,
                num_enc_layers = self.num_enc_layers,
                num_dec_layers = self.num_dec_layers,
                ff_mult = self.ff_mult,
                init = self.init,
                fuse_option = train_config['fuse_option'],
                ###
                pretrained_vision=True, predict_additive=predict_additive, separate_obs_agent_batches=separate_obs_agent_batches,
            )
            if self.model.pretrained_vision and hasattr(self.model, 'img_encoder'):
                self.model._modules['img_encoder'].requires_grad_(False)
        elif self.arch == 'gan_goal':
            self.model = Generator(
                config = config,
                dim = self.dim,
                num_enc_layers = self.num_enc_layers,
                num_dec_layers = self.num_dec_layers,
                ff_mult = self.ff_mult,
                init = self.init,
                ###""" coords_normed = coords_normed, separate_obs_agent_batches = separate_obs_agent_batches, separate_fut_agent_batches = separate_fut_agent_batches, fuse_option = fuse_option """
                )
        elif self.arch == 'coca':
            from Modules.coca.coca4traj import CoCa4Traj
            # from Modules.coca.coca import CoCa
            self.model = CoCa4Traj(
                config=config,
                dim = 512,
                sequence_enc_blocks = 1,  # DEFAULT 6
                multimodal_blocks = 1,  # DEFAULT 6
                heads = 8,
                ff_mult = 1, # DEFAULT 4
            )

        self.save_results = config['save_results']
        self.txt_path = config['store_path'] if self.save_results and 'store_path' in config else None
        
        self.train_losses = {}
        self.train_losses_per_epoch = {}
        self.val_losses = {}
        self.val_losses_per_epoch = {}
        self.train_metrics = {}
        self.train_metrics_per_epoch = {} 
        self.val_metrics = {}
        self.val_metrics_per_epoch = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        # TODO maybe implement scheduler change once backbone is unfreezed

        prediction = self.model.forward(batch, decoder_mode=self.decoder_mode_training) # decoder_mode: 0=teacher forcing, 1=running free, 2=scheduled sampling

        train_loss, losses_it, metrics_it = self.model.compute_loss(prediction, batch, stage='train')
        # for a sequence: use self-attn(goal pred)[-1] as coord embed?
        # implement dst arrival, and corresponding last coord loss
        # find out how many future steps are still stable by providing goal as GT, then see which step number remains stable
        
        # AgentFormer: context_enc: kv_pairs and MLP(.mean()) --> mu, logvar --> q_z_dist_dlow = Normal(mu, logvar) with kl function

        # log losses and metrics internally
        # self.internal_log({'train_loss': losses_it.item()}, metrics_it, stage='train')
        self.internal_log(losses_it, metrics_it, stage='train')
        
        self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
        return {'loss' : train_loss}

    def validation_step(self, batch, batch_idx: int) -> None:

        prediction = self.model.forward(batch, decoder_mode=self.decoder_mode_validation) # decoder_mode: 0=teacher forcing, 1=running free, 2=scheduled sampling

        if self.txt_path is not None: # save images only if results are generally saved
            folder_id = self.current_epoch if self.trainer.state.stage != 'sanity_check' else 'sanityCheck'
            save_path = os.sep.join([self.trainer.checkpoint_callbacks[0].dirpath, f'images_epoch_{folder_id}']) if self.current_epoch % 1 == 0 else None
            if save_path is not None and not os.path.exists(save_path): os.mkdir(save_path)
        else:
            save_path = None
        val_loss, losses_it, metrics_it = self.model.compute_loss(prediction, batch, stage='val', save_path=save_path)

        # log losses and metrics internally
        # self.internal_log({'val_loss': losses_it}, metrics_it, stage='val')
        self.internal_log(losses_it, metrics_it, stage='val')
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, logger=False)
        return {'val_loss' : val_loss}


    def internal_log(self, losses_it, metrics_it, stage):
        if self.trainer.state.stage == 'sanity_check': return

        metrics_logger = self.train_metrics if stage=='train' else self.val_metrics
        losses_logger = self.train_losses if stage=='train' else self.val_losses

        if losses_it:
            for key, val in losses_it.items():
                if key not in losses_logger:
                    losses_logger.update({key: [val]})
                else:
                    losses_logger[key].append(val)
        if metrics_it:
            for key, val in metrics_it.items():
                if key not in metrics_logger:
                    metrics_logger.update({key: [val]})
                else:
                    metrics_logger[key].append(val)


    def configure_optimizers(self):
        if self.arch == 'gan_goal':
            opt_g = Adam(self.model.generator.parameters(), lr = self.learning_rate)
            opt_d = Adam(self.model.discriminator.parameters(), lr = self.learning_rate)
            sch_g = ReduceLROnPlateau(opt_g, factor=self.lr_sch_gamma, patience=self.lr_sch_patience4redOnPlat, min_lr=1e-7)
            sch_d = ReduceLROnPlateau(opt_d, factor=self.lr_sch_gamma, patience=self.lr_sch_patience4redOnPlat, min_lr=1e-7)
            return [opt_g, opt_d], [sch_g, sch_d]
        opt = Adam(self.model.parameters(), lr = self.learning_rate)
        if self.lr_scheduler == CosineAnnealingLR.__name__:
            sch = CosineAnnealingLR(opt, T_max = self.lr_sch_step_size)
        elif self.lr_scheduler == StepLR.__name__:
            sch = StepLR(opt, step_size=self.lr_sch_step_size, gamma=self.lr_sch_gamma)
        elif self.lr_scheduler == ExponentialLR.__name__:
            sch = ExponentialLR(opt, gamma=0.9)
        elif self.lr_scheduler == ReduceLROnPlateau.__name__:
            sch = ReduceLROnPlateau(opt, factor=self.lr_sch_gamma, patience=20, min_lr=1e-6)
            # Because of a weird issue with ReduceLROnPlateau, the monitored value needs to be returned... See https://github.com/PyTorchLightning/pytorch-lightning/issues/4454
            return {
            'optimizer': opt,
            'lr_scheduler': sch,
            'monitor': 'val_loss'
        }
        else:
            raise NotImplementedError('Scheduler has not been implemented yet!')
        return [opt], [sch]


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
        
        # Training metrics
        for key, val in self.train_metrics.items():
            if key not in self.train_metrics_per_epoch:
                mean = np.array(val).mean()
                self.train_metrics_per_epoch.update({key: [mean]})
                if "goal" in key:
                    self.train_metrics_per_epoch.update({key+'_std': np.array(val).std()})
            else:
                self.train_metrics_per_epoch[key].append(np.array(val).mean())
                if "goal" in key:
                    self.train_metrics_per_epoch[key+'_std'].append(np.array(val).std())

        # Validation logs
        for key, val in self.val_losses.items():
            if key not in self.val_losses_per_epoch:
                mean = torch.as_tensor(val).nanmean()
                self.val_losses_per_epoch.update({key: [mean.item()]})
            else:
                self.val_losses_per_epoch[key].append(torch.as_tensor(val).nanmean().item())

        # Validation metrics
        for key, val in self.val_metrics.items():
            if key not in self.val_metrics_per_epoch:
                mean = np.array(val).mean()
                self.val_metrics_per_epoch.update({key: [mean]})
                if "goal" in key:
                    self.val_metrics_per_epoch.update({key+'_std': np.array(val).std()})
            else:
                self.val_metrics_per_epoch[key].append(np.array(val).mean())
                if "goal" in key:
                    self.val_metrics_per_epoch[key+'_std'].append(np.array(val).std())

        # Reset
        self.train_losses = {}
        self.train_metrics = {}
        self.val_losses = {}
        self.val_metrics = {}
        
        train_string = f'TRAINING RESULT:\nEpoch\t'
        assert set(self.train_losses_per_epoch.keys()).isdisjoint(self.train_metrics_per_epoch.keys()), 'Loss and metric keys are not disjoint!'
        train_dict = self.train_losses_per_epoch.copy()
        train_dict.update(self.train_metrics_per_epoch)
        train_vals = [val for val in train_dict.values()]
        for id_k, key in enumerate(list(train_dict.keys())):
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
        assert set(self.val_losses_per_epoch.keys()).isdisjoint(self.val_metrics_per_epoch.keys()), 'Loss and metric keys are not disjoint!'
        val_dict = self.val_losses_per_epoch.copy()
        val_dict.update(self.val_metrics_per_epoch)
        val_vals = [val for val in val_dict.values()]
        for id_k, key in enumerate(list(val_dict.keys())):
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


    def on_fit_start(self) -> None:

        # return super().on_fit_start()
        print(f"\nFREEZE STRATEGY INITIALIZED IN EPOCH {self.current_epoch}\n")
        module_list = list(self.model._modules.keys())
        if self.model.pretrained_vision and hasattr(self.model, 'img_encoder'):
            for p in self.model._modules['img_encoder'].parameters():
                assert p.requires_grad == False, 'Vision encoder should be frozen!'
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


    # def on_before_zero_grad(self, *args, **kwargs):
    # def on_before_backward(self, loss: Tensor) -> None:
    #     return super().on_before_backward(loss)
    
    # def on_after_backward(self) -> None:
    #     return super().on_after_backward()

    # def on_train_batch_start(self, batch, batch_idx):
        # return super().on_train_batch_start(batch, batch_idx)
