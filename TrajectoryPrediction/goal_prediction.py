from helper import SEP, OPSYS, PREFIX
import os
import sys
# sys.path.insert(0, SEP.join([PREFIX, 'Users', 'Remotey', 'Documents','Pedestrian-Dynamics']))
# from Image2Image.models import Image2ImageModule
# from Image2Image.Datamodules import FloorplanDataModule
from Modules.GoalPredictionModule import GoalPredictionModule
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from collections import OrderedDict
import time
from helper import dir_maker
import matplotlib.pyplot as plt

def goal_prediction(config: dict, datamodule):

    do_training = True

    test_run = False
    save_model = False

    cuda_device = config['cuda_device']
    config.update({
        'from_ckpt_path': None, # 'Resnet18Adaption_pretrained_lr3e-4_withTanh_cont',
        'save_ckpt': 'Resnet18Adaption_pretrained_lr3e-4_withTanh_cont',
        'test_run': test_run,
        'save_model': save_model,
    })

    TRAIN_CONFIG = {
            'learning_rate': 1e-4, 
            'lr_scheduler': 'ReduceLROnPlateau',
            'lr_sch_gamma': 0.5,
            'lr_sch_step_size4lr_step': 8,
            'lr_sch_step_size4cosAnneal': 50,
            'lr_sch_patience4redOnPlat': 7,
            'opt': 'Adam',
            'weight_decay': 1e-6,
            'init': 'xavier',
            'early_stopping_patience': 100,
        }

    ckpt_folder = 'coca' if config['mode']=="SEGMENTATION" else 'GoalPredictionModels'
    if (not test_run) and save_model and do_training:
        store_folder_path = SEP.join(['TrajectoryPrediction', 'Modules', ckpt_folder, 'checkpoints', config['save_ckpt']])
        description_log = ''
        config.update({'store_path': store_folder_path+SEP+'results.txt'})
        dir_maker(store_folder_path, description_log, config, TRAIN_CONFIG)
    
    module = GoalPredictionModule(config=config, train_config=TRAIN_CONFIG)

    # Load from checkpoint
    if config['from_ckpt_path']:
        CKPT_PATH = SEP.join(['TrajectoryPrediction', 'Modules', ckpt_folder, 'checkpoints', config['from_ckpt_path']])
        model_file_path = [file for file in os.listdir(CKPT_PATH) if file.endswith('.ckpt') and not file.startswith('last')]
        assert len(model_file_path) == 1
        CKPT_PATH = SEP.join([CKPT_PATH, model_file_path[0]])
        state_dict = torch.load(CKPT_PATH)['state_dict']
        module_state_dict = module.state_dict()

        mkeys_missing_in_loaded = [module_key for module_key in list(module_state_dict.keys()) if module_key not in list(state_dict.keys())]
        lkeys_missing_in_module = [loaded_key for loaded_key in list(state_dict.keys()) if loaded_key not in list(module_state_dict.keys())]

        assert len(mkeys_missing_in_loaded) < 10 or len(lkeys_missing_in_module) < 10, 'Checkpoint loading went probably wrong...'

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

    if do_training:

        limit_batches = 2 if test_run else None

        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=TRAIN_CONFIG['early_stopping_patience']), LearningRateMonitor(logging_interval='epoch')]

        if save_model and not test_run:
            model_checkpoint = ModelCheckpoint(
                dirpath = store_folder_path,
                filename = config['save_ckpt']+'{epoch}-{step}',
                save_top_k = 1,
                verbose = True, 
                monitor = 'val_loss',
                mode = 'min',
            )
            callbacks.append(model_checkpoint)
            
        trainer = pl.Trainer(
            gpus = [cuda_device], 
            devices=f'cuda:{str(cuda_device)}', 
            max_epochs = 500, 
            # checkpoint_callback = [checkpoint_callback], 
            callbacks=callbacks,
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

    else:
        module.to(f'cuda:{cuda_device}')
        module.eval()

        datamodule.setup(stage='test')
        dataloader = datamodule.train_dataloader() # test_dataloader()
        # dataloader = datamodule.test_dataloader() # test_dataloader()
        # batchy = dataloader.dataset.__getitem__(15)

        test_result_folder = 'TrajectoryPrediction\\Modules\\coca\\checkpoints\\Resnet18Adaption_pretrained_lr3e-4_withTanh_cont\\Inference_results'
        if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

        for idx, batch in enumerate(dataloader):
            os.mkdir(test_result_folder+f'\\batch_id_{idx}')
            if config['mode'] == "SEGMENTATION":
                semanticMaps_per_agent = batch['semanticMaps_per_agent'].to(f'cuda:{cuda_device}').permute(3, 0, 1, 2)
                semanticMaps_egocentric = batch['semanticMaps_egocentric'].to(f'cuda:{cuda_device}').permute(1, 0, 2, 3)
                output_dstMap, _ = module.model(semanticMaps_per_agent, mode=0)
                output_egocentric, _ = module.model(semanticMaps_egocentric, mode=1)

                num_agents = semanticMaps_per_agent.shape[0]
                for i in range(num_agents):
                    # visualize semantic maps per agent
                    path = test_result_folder+f'\\batch_id_{idx}\\agent_{i}_dstMap.png'
                    path = None
                    visualize_occ_map(semanticMaps_per_agent[i], output_dstMap[i], path)

                    # visualize egocentric semantic maps
                    path = test_result_folder+f'\\batch_id_{idx}\\agent_{i}_egocentric.png'
                    path = None
                    visualize_occ_map(semanticMaps_egocentric[i], output_egocentric[i], path)

def visualize_occ_map(gt_img, out_img, path=None):
    gt_img = gt_img.squeeze().detach().cpu().numpy()
    out_img = out_img.squeeze().detach().cpu().numpy()

    out_min, out_max = out_img.min(), out_img.max()

    # normalize to the range 0-1
    gt_img = (gt_img + 1.) / 2.0
    out_img = (out_img - out_min) / (out_max - out_min)

    # plot both figures next to each other
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(gt_img)
    ax1.set_title('Ground Truth')
    ax2.imshow(out_img)
    ax2.set_title('Prediction')

    if path is not None:
        fig.savefig(path)
    
    # plot both figures next to each other
    # plt.show()
    # plt.close('all')
    
