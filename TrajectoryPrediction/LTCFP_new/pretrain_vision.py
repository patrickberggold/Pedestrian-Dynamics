from helper import SEP, OPSYS, PREFIX
import sys
sys.path.insert(0, SEP.join([PREFIX, 'Users', 'Remotey', 'Documents','Pedestrian-Dynamics']))
from Image2Image.models import Image2ImageModule
from Image2Image.Datamodules import FloorplanDataModule
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from collections import OrderedDict
import time


def pretrain_vision(cuda_device):

    pretrain_mode = 'grayscale' # implemented: grayscale, evac, evac_only, class_movie, density_reg, density_class, denseClass_wEvac
    pretrain_arch = 'SegFormer' # DeepLab, BeIT, SegFormer
    pretrain_batch_size = 4

    CONFIG = {
        'mode': pretrain_mode,
        'arch': pretrain_arch,
        'cuda_device': cuda_device,
        'batch_size': pretrain_batch_size,
        'from_ckpt_path': 'SegFormer_grayscale_pretrain_epoch=24-step=55094.ckpt',
        'load_to_ckpt_path': 'SegFormer_grayscale_pretrain_WSL_24+_',
        'early_stopping_patience': 6,
        'run_test_epoch': False,
        'vary_area_brightness': True,
        'limit_dataset': False, # or int-type
        'additional_info': False,
    }

    TRAIN_DICT = {
        # Learning rate and schedulers
        'learning_rate': 0.0004, 
        'lr_scheduler': 'CosineAnnealingLR',# 'ReduceLROnPlateau', 
        'lr_sch_step_size4lr_step': 8,
        'lr_sch_step_size4cosAnneal': 100,
        'lr_sch_gamma4redOnPlat_and_stepLR': 0.75,
        'lr_sch_gamma4expLR': 0.25,
        'lr_sch_patience4redOnPlat': 3,
        # optimizer
        'opt': 'Adam',
        'weight_decay': None, # 1e-6,
        # (un)freezing layers
        'unfreeze_backbone_at_epoch': None, # not implemented yet
        # loss params for tversky
        'loss_dict': {'alpha': 0.2, 'beta': 0.8} # 0.005//0.995 leads to lots of nonzero classifications, it's too much #### 0.05//0.95 better (but still too much) #### 0.2//0.8 is QUITE GOOD #### 0.8//0.2 leads to nonzero few classifications (it's too few...)
    }

    datamodule = FloorplanDataModule(config = CONFIG) # TODO add sparse timestamp thickness dataset
    # raise NotImplementedError

    module = Image2ImageModule(config=CONFIG, train_config=TRAIN_DICT)

    # Load from checkpoint
    if CONFIG['from_ckpt_path']:
        # trained_together_OPTIM_TWICE_2_1000dataset_epoch=16-step=2125.ckpt
        # CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'beit_whole_ds_finetuned_epoch=74-step=158925.ckpt'])
        # CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', CONFIG['from_ckpt_path']])
        CKPT_PATH = SEP.join(['TrajectoryPrediction', 'LTCFP_new', 'checkpoints', 'img2img_pretrain', CONFIG['from_ckpt_path']])
        state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])
        module_state_dict = module.state_dict()

        mkeys_missing_in_loaded = [module_key for module_key in list(module_state_dict.keys()) if module_key not in list(state_dict.keys())]
        lkeys_missing_in_module = [loaded_key for loaded_key in list(state_dict.keys()) if loaded_key not in list(module_state_dict.keys())]

        load_dict = OrderedDict()
        for key, tensor in module_state_dict.items():
            # if (key in state_dict.keys()) and ('decode_head' not in key):
            if key in state_dict.keys():
                load_dict[key] = state_dict[key]
            else:
                # if key == 'model.model.classifier.classifier.weight':
                #     load_dict[key] = state_dict['model.model.classifier.weight']
                # elif key == 'model.model.classifier.classifier.bias': 
                #     load_dict[key] = state_dict['model.model.classifier.bias']
                # else:
                #     load_dict[key] = tensor
                load_dict[key] = tensor

        module.load_state_dict(load_dict)
    
    if pretrain_arch == 'BeIT': module.model.model.auxiliary_head = None
    
    model_checkpoint = ModelCheckpoint(
        dirpath = SEP.join(['TrajectoryPrediction', 'LTCFP_new', 'checkpoints', 'img2img_pretrain']),
        filename = CONFIG['load_to_ckpt_path']+'{epoch}-{step}',
        save_top_k = 1,
        verbose = True, 
        monitor = 'val_loss',
        mode = 'min',
    )

    limit_batches = 2 if CONFIG['run_test_epoch'] else None
    trainer = pl.Trainer(
        gpus = [cuda_device], 
        devices=f'cuda:{str(cuda_device)}', 
        max_epochs = 500, 
        # checkpoint_callback = [checkpoint_callback], 
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=CONFIG['early_stopping_patience']), LearningRateMonitor(logging_interval='epoch'), model_checkpoint],
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        # progress_bar_refresh_rate=125,
        # logger = False,
        )

    start_training_time = time.time()
    trainer.fit(module, datamodule=datamodule)
    # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # IMPORTANT: this will delete the old checkpoint!!
    # trainer.fit(model, datamodule=datamodule, ckpt_path="Image2Image\checkpoints\checkpoints_DeepLab4Img2Img\model_grayscale_lineThickness5_CosAnn_Step5_Lr122_Gam42_epoch=36-step=5772.ckpt")
    print(f'Training took {(time.time() - start_training_time)/60./(module.current_epoch+1):.3f} minutes per epoch...')

    quit()

