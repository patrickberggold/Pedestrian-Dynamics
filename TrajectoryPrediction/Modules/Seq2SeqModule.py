from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau
from torch.optim import Adam
import pytorch_lightning as pl
from Modules.goal.models.goal_sar import Goal_SAR
from Modules.coca.coca4traj import CoCa4Traj
import numpy as np
from Modules.transformer.transformer import Transformer
import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class Seq2SeqModule(pl.LightningModule):
    def __init__(
        self,
        config: dict, 
        learning_rate: float = 5e-4, 
        lr_scheduler: str = ReduceLROnPlateau.__name__, 
        lr_sch_gamma: float = 0.3, 
        unfreeze_backbone_at_epoch: int = 8,
        alternate_unfreezing: int = False,
        ):
        super(Seq2SeqModule, self).__init__()
        
        self.arch = config['arch']
        self.img_arch = config['img_arch']
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_sch_step_size = 100 if lr_scheduler==CosineAnnealingLR.__name__ else 5 # lr_sch_step_size
        self.lr_sch_gamma = lr_sch_gamma
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.alternate_unfreezing = alternate_unfreezing
        self.traj_quantity = config['traj_quantity']
        self.normalize_dataset = config['normalize_dataset']

        assert self.lr_scheduler in [CosineAnnealingLR.__name__, StepLR.__name__, ExponentialLR.__name__, ReduceLROnPlateau.__name__], 'Unknown LR Scheduler!'
        assert self.arch in ['goal', 'tf', 'coca'], 'Unknown architecture!'
  
        if self.arch == 'goal':
            self.net = Goal_SAR(traj_quantity=self.traj_quantity)
        elif self.arch == 'tf':
            self.net = Transformer(enc_inp_size=2, dec_inp_size=3, dec_out_size=3, dropout=0.3, traj_quantity=self.traj_quantity)
        elif self.arch == 'coca':
            from Modules.coca.coca4traj import CoCa4Traj
            # from Modules.coca.coca import CoCa
            use_contrastive_loss = False
            self.net = CoCa4Traj(
                config=config,
                dim = 512, 
                num_tokens = 20000, 
                unimodal_depth = 6, # DEFAULT 6
                multimodal_depth = 6, # DEFAULT 6
                dim_head = 64, 
                heads = 8,
                ff_mult = 1, # DEFAULT 4
                caption_loss_weight = 1., 
                contrastive_loss_weight = 1., 
                forTraj=True, 
                use_contrastive_loss=use_contrastive_loss,
            )

            # self.net = CoCa(dim = 512, image_dim = 1024, num_tokens = 20000, unimodal_depth = 6, multimodal_depth = 6, dim_head = 64, heads = 8, caption_loss_weight = 1., contrastive_loss_weight = 1.)

            # # NLP example
            # from transformers import BertTokenizer, BertModel
            # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

            # example_text = '[PAD] I will watch Memento tonight'
            # bert_input = tokenizer(example_text,padding='max_length', max_length = 10, 
            #                     truncation=True, return_tensors="pt")

            # bert = BertModel.from_pretrained('bert-base-cased')
            # bert_input = bert_input['input_ids']
            # out_sentence = bert(bert_input)
            # hi = 5

            # use_traj = False
            # if use_traj:
            #     traj_obs_raw = torch.randn(15, 8, 2).cuda(0)
            #     traj_fut_raw = torch.randn(15, 12, 2).cuda(0)
            #     images = torch.randn(1, 3, 640, 640).cuda(0)

            #     labels = traj_fut_raw
            #     text = traj_obs_raw
            # else:
            #     text = torch.randint(0, 20000, (4, 512)).cuda(0) # --> if text tokens are 0, its a padding, basically this integer tensor is the output of the tokenizer (e.g., ['PAD'] --> 0)
            #     images = torch.randn(4, 3, 640, 640).cuda(0)
            #     labels = None
        
        # mock txt and img
        # cock = self.net.cuda(0)
        # loss = cock(
        #     text = text,
        #     labels = labels,
        #     images = images,
        #     return_loss = True  # set this to True to get the full caption + contrastive loss
        # )

        # for idx, child in enumerate(self.net.children()):
        #     if idx == 1:
        #         for param in child[1].parameters():
        #             param.requires_grad = False

        # from helper import count_parameters
        # count_parameters(self.net)
        # from torchsummary import summary
        # summary(self.net.to('cuda:0'), [(3,1024,1024), (12,2)], device='cuda') # IMPORTANT: INCLUDE non-Instance of torch.Tensor exclusion, otherwise exception
        # print('\n\n##########################################\n\n')
        # print(self.net)
        # output_tensor = self.net(torch.rand((2,3,800,800)).to('cuda:0'))   #   ([2, 256, 800, 800])

        # children_list = list(self.net.children())
        # children_backbone = list(children_list[0].children())
        # children_head = list(children_list[1].children())
        
        # first_part = list(self.net.children())[0]
        # first_result = first_part(torch.ones((1,3,800,800)).to('cuda:1'))['out']
        # summary(first_part, (3, 800, 800), device='cuda')

        self.train_losses = {}
        self.train_losses_per_epoch = {}
        self.val_losses = {}
        self.val_losses_per_epoch = {}
        self.train_metrics = {}
        self.train_metrics_per_epoch = {} 
        self.val_metrics = {}
        self.val_metrics_per_epoch = {}

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx: int):
        # TODO maybe implement scheduler change once backbone is unfreezed

        prediction = self.net.forward(batch, decoder_mode=0) # decoder_mode: 0=teacher forcing, 1=running free, 2=scheduled sampling

        train_loss, losses_it, metrics_it = self.net.compute_loss(prediction, batch, stage='train')

        # log losses and metrics internally
        # self.internal_log({'train_loss': losses_it.item()}, metrics_it, stage='train')
        self.internal_log(losses_it, metrics_it, stage='train')
        
        self.log('loss', train_loss, on_step=False, on_epoch=True, logger=False)
        return {'loss' : train_loss}

    def validation_step(self, batch, batch_idx: int) -> None:

        prediction = self.net.forward(batch, decoder_mode=0) # decoder_mode: 0=teacher forcing, 1=running free, 2=scheduled sampling

        val_loss, losses_it, metrics_it = self.net.compute_loss(prediction, batch, stage='val')

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
        opt = Adam(self.net.parameters(), lr = self.learning_rate)
        if self.lr_scheduler == CosineAnnealingLR.__name__:
            sch = CosineAnnealingLR(opt, T_max = self.lr_sch_step_size)
        elif self.lr_scheduler == StepLR.__name__:
            sch = StepLR(opt, step_size=self.lr_sch_step_size, gamma=self.lr_sch_gamma)
        elif self.lr_scheduler == ExponentialLR.__name__:
            sch = ExponentialLR(opt, gamma=0.9)
        elif self.lr_scheduler == ReduceLROnPlateau.__name__:
            sch = ReduceLROnPlateau(opt, factor=self.lr_sch_gamma, patience=3, min_lr=1e-6)
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
        return super().on_train_epoch_start()

        # determine whether backbone freezed at all
        if self.unfreeze_backbone_at_epoch:
            if self.current_epoch == 0:
                print(f"\nFREEZE BACKBONE INITIALLY IN EPOCH {self.current_epoch}\n")
                self.backbone_frozen = True
                # freeze backbone weights
                for idx, child in enumerate(self.net.children()):
                    if idx == 0:
                        for param in child.parameters():
                            param.requires_grad = False
            else:
                # if alternate freeze
                if self.alternate_unfreezing:
                    if self.current_epoch % self.unfreeze_backbone_at_epoch == 0:
                        if self.backbone_frozen:
                            print(f"\nUNFREEZING BACKBONE IN EPOCH {self.current_epoch}\n")
                            # unfreeze backbone
                            for idx, child in enumerate(self.net.children()):
                                if idx == 0:
                                    for param in child.parameters():
                                        param.requires_grad = True
                            self.backbone_frozen = False
                        else:
                            # freeze backbone
                            print(f"\nFREEZING BACKBONE IN EPOCH {self.current_epoch}\n")
                            for idx, child in enumerate(self.net.children()):
                                if idx == 0:
                                    for param in child.parameters():
                                        param.requires_grad = False
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


    def on_train_epoch_end(self) -> None:

        if self.trainer.state.stage in ['sanity_check']: return super().on_train_epoch_end()

        # Training Logs
        for key, val in self.train_losses.items():
            if key not in self.train_losses_per_epoch:
                mean = sum(val)/len(val)
                self.train_losses_per_epoch.update({key: [mean]})
            else:
                self.train_losses_per_epoch[key].append(sum(val)/len(val))

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
                mean = sum(val)/len(val)
                self.val_losses_per_epoch.update({key: [mean]})
            else:
                self.val_losses_per_epoch[key].append(sum(val)/len(val))

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
        self.val_losses = {}
        self.train_metrics = {}
        self.val_metrics = {}

        # Print log
        print('\n\nTRAINING RESULT:')
        for key, val in self.train_losses_per_epoch.items():
            print(key, ': ', val)
        # for result in self.train_losses_per_epoch['loss']:
        #     print(result) 
        if self.train_metrics_per_epoch:
            for key, val in self.train_metrics_per_epoch.items():
                print(key, ': ', val)
            print('\n')

        print('VALIDATION RESULT:')
        for key, val in self.val_losses_per_epoch.items():
            print(key, ': ', val)
        # for result in self.val_losses_per_epoch['val_loss']:
        #     print(result)
        if self.val_metrics_per_epoch:
            for key, val in self.val_metrics_per_epoch.items():
                print(key, ': ', val)
        print('')


    def on_fit_end(self):
        # return super().on_fit_end()
        self.train_losses_per_epoch = {}
        self.val_losses_per_epoch = {}
        self.train_metrics_per_epoch = {} 
        self.val_metrics_per_epoch = {}

        print('\nTRAIN STATISTICS:')
        for key in self.train_losses_per_epoch.keys():
            print('\n' + key + ':')
            for item in self.train_losses_per_epoch[key]:
                print(item)
        for key in self.train_metrics_per_epoch.keys():
            print('\n' + key + ':')
            for item in self.train_metrics_per_epoch[key]:
                print(item)
        
        print('\nVALIDATION STATISTICS:')
        for key in self.val_losses_per_epoch.keys():
            print('\n' + key + ':')
            for item in self.val_losses_per_epoch[key]:
                print(item)
        for key in self.val_metrics_per_epoch.keys():
            print('\n' + key + ':')
            for item in self.val_metrics_per_epoch[key]:
                print(item)
