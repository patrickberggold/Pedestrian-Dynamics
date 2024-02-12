import os
import time
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Modules.Seq2SeqModule import Seq2SeqModule
from Modules.Seq2SeqModule_GAN import Seq2SeqModule_GAN
from Datamodules import FloorplanDatamodule
from helper import SEP, dir_maker, load_module_from_checkpoint, visualize_trajectories
# from hyperparameter_optim import hyperparameter_optimization

MULIT_GPU = False
CUDA_DEVICE = 0
MODE = 'TRAJ_PRED' # TRAJ_PRED, MTM, GOAL_PRED, SEGMENTATION
DATA_FORMAT = 'by_frame' # by_frame, by_frame_masked, random, partial_tokenized_random, partial_tokenized_by_frame, full_tokenized_by_frame

do_vision_pretraining = False

do_training = True
test_run = False
save_model = False

CONFIG = {
    # GENERAL
    'mode': MODE, # TRAJ_PRED, MTM
    'arch': 'adv_goal', # tf, coc, goal, coca_goal, gan_goal, simple_goal, adv_goal
    'img_arch': 'SegFormer', # BeIT, SegFormer
    'cuda_device': CUDA_DEVICE,
    'imgs_per_batch': 1,
    'load_from_ckpt': None, # 'Advanced__Obs8_Pred12_Enc3_fus2__cont+ep29',
    'save_ckpt': None,
    'save_model': save_model,
    'save_results': True,
    # TESTRUN
    'test_run': test_run,
    'limit_dataset': 5,
    'limit_train_batches': 2 if test_run else None,
    'limit_val_batches': 2 if test_run else None,
    # DATALOADING
    'traj_quantity': 'pos', # pos, vel
    'data_format' : DATA_FORMAT, # by_frame, random, partial_tokenized_random, full_tokenized_by_frame, partial_tokenized_by_frame
    'normalize_dataset': True,
    'read_from_pickle': False,
    'grad_clip': 10.0,
    'resize_factor': 2,
    'ds_mean': 320., # 1280 // self.resize_factor / 2, 150 / self.resize_factor
    'ds_std': 320.,
    'goal_predictor': 'resnet18', # unet, resnet18
    'num_obs_steps': 12,
    'pred_length': 8,
}

TRAIN_CONFIG = {
    'learning_rate': 0.0003,
    'lr_scheduler': 'ReduceLROnPlateau', # ReduceLROnPlateau, CosineAnnealingLR, StepLR, ExponentialLR
    'lr_sch_step_size': 5, # 100 if lr_scheduler==CosineAnnealingLR.__name__ else 5 # lr_sch_step_size
    'lr_sch_gamma': 0.5,
    'lr_sch_patience4redOnPlat': 7,
    'dim': 512,
    'num_enc_layers': 3, # 1, 2, 3
    'num_dec_layers': 3, # 1, 2, 3
    'ff_mult': 1, # 1, 2, 4
    'init': 'xavier',
    ### optimization params:
    'coords_normed': False,
    'separate_fut_agent_batches': False, # not now
    'predict_additive': True,
    'separate_obs_agent_batches': True,
    'fuse_option': 2, # 1, 2, 3
}

# CONFIG['save_ckpt'] = 'Advanced__Obs'+str(CONFIG['num_obs_steps'])+'_Pred'+str(CONFIG['pred_length'])+'_Enc'+str(TRAIN_CONFIG['num_enc_layers'])+'_fus'+str(TRAIN_CONFIG['fuse_option'])

if (not test_run) and save_model and do_training:
    description_log = 'Standard CoCa for presentation purposes (random, TRAJ_PRED, 8-20 split)'
    store_folder_path = SEP.join(['TrajectoryPrediction', 'Modules', 'coca', 'checkpoints', CONFIG['save_ckpt']])
    CONFIG.update({'store_path': store_folder_path+SEP+'results.txt'})
    dir_maker(store_folder_path, description_log, CONFIG, TRAIN_CONFIG)


if __name__ == '__main__':

    if do_vision_pretraining:
        from pretrain_vision import pretrain_vision

        pretrain_vision(CUDA_DEVICE)

        quit()

    datamodule = FloorplanDatamodule(config=CONFIG)
    
    if MODE in ['GOAL_PRED', 'SEGMENTATION']:
        from goal_prediction import goal_prediction 
        goal_prediction(CONFIG, datamodule)        
        quit()

    module = Seq2SeqModule(config=CONFIG, train_config=TRAIN_CONFIG) if CONFIG['arch'] != 'gan_goal' else Seq2SeqModule_GAN(config=CONFIG, train_config=TRAIN_CONFIG)
    if CONFIG['load_from_ckpt']:
        # sss0 = module.state_dict().copy()
        module = load_module_from_checkpoint(module, CONFIG['load_from_ckpt'])
        # sss1 = module.state_dict().copy()
    
    if do_training:

        # callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=50), LearningRateMonitor(logging_interval='epoch')]
        callbacks = [LearningRateMonitor(logging_interval='epoch')]

        if (not CONFIG['test_run']) and save_model:
            model_checkpoint = ModelCheckpoint(
                dirpath = store_folder_path,
                filename = 'model_{epoch}-{step}',
                save_top_k = 1,
                verbose = True, 
                monitor = 'val_loss',
                mode = 'min',
                save_last=True
            )
            callbacks.append(model_checkpoint) # TODO print lr each epoch for checking + turn of lightning logger if possible

        # from pytorch_lightning.strategies import DDPStrategy
        # stategy = DDPStrategy(process_group_backend="gloo") # nccl
        trainer = pl.Trainer(
            gpus = [CUDA_DEVICE], 
            devices=f'cuda:{str(CUDA_DEVICE)}', 
            max_epochs = 500, 
            # checkpoint_callback = [checkpoint_callback], 
            callbacks=callbacks,
            limit_train_batches=CONFIG['limit_train_batches'],
            limit_val_batches=CONFIG['limit_val_batches'],
            # progress_bar_refresh_rate=125,
            gradient_clip_val=CONFIG['grad_clip'],
            )

        start_training_time = time.time()
        trainer.fit(module, datamodule=datamodule)
        # automatically restores model, epoch, step, LR schedulers, apex, etc... BUT will delete the old checkpoint!!
        # trainer.fit(model, datamodule=datamodule, ckpt_path="Image2Image\checkpoints\checkpoints_DeepLab4Img2Img\model_grayscale_lineThickness5_CosAnn_Step5_Lr122_Gam42_epoch=36-step=5772.ckpt")
        print(f'Training took {(time.time() - start_training_time)/60./(module.current_epoch+1):.3f} minutes per epoch...')

        quit()

    module.to(f'cuda:{CUDA_DEVICE}')
    module.eval()

    datamodule.setup(stage='test')
    dataloader = datamodule.train_dataloader() # test_dataloader()
    # batchy = dataloader.dataset.__getitem__(15)
    
    test_result_folder = 'TrajectoryPrediction'+SEP+'image_results'
    # if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

    for idx, batch in enumerate(dataloader):

        prediction = module.model.forward({key: tensor.to(f'cuda:{CUDA_DEVICE}') for key, tensor in batch.items()})
        pred_trajectories = prediction['pred_coords'].cpu()

        image = batch['image'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        abs_coordinates = batch['coords'].squeeze(0) if 'coords' in batch else None
        velocities = batch['transformed_velocities'].squeeze(0) if 'velocity' in batch else None
        occupancyMaps_egocentric = batch['occupancyMaps_egocentric'].permute(1, 0, 2, 3) if 'occupancyMaps_egocentric' in batch else None
        semanticMaps_per_agent = batch['semanticMaps_per_agent'].permute(3, 0, 1, 2) if 'semanticMaps_per_agent' in batch else None
        transformed_agent_destinations = batch['transformed_agent_destinations'].squeeze(0) if 'transformed_agent_destinations' in batch else None
        all_scene_destinations = batch['all_scene_destinations'].squeeze(0) if 'all_scene_destinations' in batch else None

        mse_loss, ade_loss, fde_loss = module.model.compute_mse_and_ade(pred_trajectories, abs_coordinates[:, CONFIG['num_obs_steps']:, :], abs_coordinates[:, CONFIG['num_obs_steps']-1, :])
        
        # hardcoded for now
        if not module.model.coords_normed: pred_trajectories = module.model.denormalize([pred_trajectories]) 
        # pred_trajectories += 320.
        # abs_coordinates += 320.

        # prepate coordinates
        abs_coordinates = abs_coordinates.numpy()
        pred_trajectories = pred_trajectories.numpy()
        obs_coords = abs_coordinates[:, :CONFIG['num_obs_steps'], :]
        gt_fut_coords = abs_coordinates[:, CONFIG['num_obs_steps']:, :]

        assert gt_fut_coords.shape == pred_trajectories.shape

        image_draw = visualize_trajectories(abs_coordinates, pred_trajectories, image, CONFIG['num_obs_steps'])
        plt.imshow(image_draw)
        plt.close('all')
