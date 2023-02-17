import os
import time
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from Modules.Seq2SeqModule import Seq2SeqModule
from Datamodules import FloorplanDatamodule
from collections import OrderedDict
from helper import SEP, linemaker, dir_maker
# from hyperparameter_optim import hyperparameter_optimization

MULIT_GPU = False
CUDA_DEVICE = 0
MODE = 'MTM' # TRAJ_PRED, MTM
DATA_FORMAT = 'tokenized' # by_frame, random, tokenized

do_vision_pretraining = False
do_training = True
test_run = False
save_model = False

CONFIG = {
    # GENERAL
    'mode': MODE, # TRAJ_PRED, MTM
    'arch': 'coca', # tf, coca, goal
    'img_arch': 'SegFormer', # BeIT, SegFormer
    'cuda_device': CUDA_DEVICE,
    'imgs_per_batch': 1,
    'load_from_ckpt': None, # 'coca_ff_mult1_wholeDS_pos_random_normalized_epoch=1-step=38654.ckpt',
    'save_ckpt': 'coca_ff_mult1_wholeDS_pos_random_normalized_WSL_+1',
    'save_model': save_model,
    # TESTRUN
    'test_run': test_run,
    'limit_dataset': 4 if test_run else 4,
    'limit_train_batches': 2 if test_run else None,
    'limit_val_batches': 2 if test_run else None,
    # DATALOADING
    'traj_quantity': 'pos', # pos, vel
    'data_format' : DATA_FORMAT, # by_frame, random, tokenized
    'normalize_dataset': True,
    'seq_length': 20,
    'num_obs_steps': 8,
    'read_from_pickle': False,
}

description_log = ''
if (not test_run) and save_model:
    store_folder_path = SEP.join(['TrajectoryPrediction', 'checkpoints', CONFIG['save_ckpt']])
    dir_maker(store_folder_path, description_log)


if __name__ == '__main__':

    if do_vision_pretraining:
        from pretrain_vision import pretrain_vision

        pretrain_vision(CUDA_DEVICE)

        quit()

    datamodule = FloorplanDatamodule(config=CONFIG)
    datamodule.setup('train')
    if do_training:

        module = Seq2SeqModule(config=CONFIG, learning_rate=5e-4, lr_scheduler='ReduceLROnPlateau', lr_sch_gamma=0.3)
        module_state_dict = module.state_dict()

        if CONFIG['load_from_ckpt']:
            CKPT_PATH = SEP.join(['TrajectoryPrediction', 'checkpoints', CONFIG['load_from_ckpt']])
            state_dict = OrderedDict([(key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])

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

        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=8), LearningRateMonitor(logging_interval='epoch')]

        if (not test_run) and save_model:
            model_checkpoint = ModelCheckpoint(
                dirpath = store_folder_path,
                filename = 'model_{epoch}-{step}',
                save_top_k = 1,
                save_last = False, # True,
                verbose = True, 
                monitor = 'val_loss',
                mode = 'min',
            )
            callbacks.append(model_checkpoint) # TODO print lr each epoch for checking + turn of lightning logger if possible

        # from pytorch_lightning.strategies import DDPStrategy
        # stategy = DDPStrategy(process_group_backend="gloo") # nccl
        trainer = pl.Trainer(
            gpus = [CUDA_DEVICE] if not MULIT_GPU else None, 
            devices=f'cuda:{str(CUDA_DEVICE)}' if not MULIT_GPU else [0,1],# '0, 1', 
            max_epochs = 500, 
            callbacks=callbacks,
            limit_train_batches=CONFIG['limit_train_batches'],
            limit_val_batches=CONFIG['limit_val_batches'],
            accelerator=None if not MULIT_GPU else 'gpu',
            # strategy=None if not MULIT_GPU else stategy # 'ddp'
            # logger=False,
            )

        start_training_time = time.time()
        trainer.fit(module, datamodule=datamodule)
        # automatically restores model, epoch, step, LR schedulers, apex, etc... BUT will delete the old checkpoint!!
        # trainer.fit(model, datamodule=datamodule, ckpt_path="Image2Image\checkpoints\checkpoints_DeepLab4Img2Img\model_grayscale_lineThickness5_CosAnn_Step5_Lr122_Gam42_epoch=36-step=5772.ckpt")
        print(f'Training took {(time.time() - start_training_time)/60./(module.current_epoch+1):.3f} minutes per epoch...')

        quit()

    CKPT_PATH = SEP.join(['TrajectoryPrediction', 'checkpoints', CONFIG['load_from_ckpt']])
    state_dict = OrderedDict([(key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])

    module = Seq2SeqModule(config=CONFIG, learning_rate=5e-4, lr_scheduler='ReduceLROnPlateau', lr_sch_gamma=0.3)
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

    module.to(f'cuda:{CUDA_DEVICE}')
    module.eval()

    datamodule.setup(stage='test')
    dataloader = datamodule.train_dataloader() # test_dataloader()
    # batchy = dataloader.dataset.__getitem__(15)
    
    test_result_folder = 'TrajectoryPrediction'+SEP+'image_results'
    if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

    for idx, batch in enumerate(dataloader):
        images = batch[0]
        plt.imshow(np.moveaxis(images.squeeze().detach().cpu().numpy(), 0, -1))
        plt.close('all')
        gt_coordinates = batch[1].squeeze(0)

        # batch = {k: v.squeeze(0).float().to(f'cuda:{CUDA_DEVICE}') for k, v in batch.items() if k not in ['scene_data', 'type']}
        # tensor_image = batch['tensor_image']
        batch = [batch_item.to(f'cuda:{CUDA_DEVICE}') for batch_item in batch]
        
        all_output = module.net.forward(batch, train=True) # TODO if False, why OOM error?
        pred_coordinates = all_output[0].squeeze().detach().cpu()

        if CONFIG['traj_quantity'] == 'pos' and CONFIG['normalize_dataset']:
            gt_coordinates = gt_coordinates * dataloader.dataset.overall_std + dataloader.dataset.overall_mean
            pred_coordinates = pred_coordinates * dataloader.dataset.overall_std + dataloader.dataset.overall_mean

        if CONFIG['arch'] == 'goal':
            pred_coordinates = pred_coordinates[0]
        import random
        from skimage.draw import line
        img_np = images.squeeze().permute(1, 2, 0).detach().numpy()
        img_input = img_np.copy()
        import torch.nn.functional as F
        mse = F.mse_loss(pred_coordinates, gt_coordinates[:, 8:])
        
        pred_coordinates = pred_coordinates.numpy()
        gt_coordinates = gt_coordinates.detach().cpu().numpy()
        limit_pred_agents = 4
        
        # trajs = [traj.detach().cpu().numpy() for traj in pred_coordinates]
        
        if CONFIG['arch'] == 'tf':
            trajs = [traj.detach().cpu().numpy() for traj in pred_coordinates[:, :limit_pred_agents]]
            gt_traj = [traj.detach().cpu().numpy() for traj in batch['abs_pixel_coord'][:, :limit_pred_agents]]

            trajs = gt_traj[:8] + trajs
            trajs = np.stack(trajs, axis=0)
            trajs = trajs.swapaxes(0, 1)
            trajs = [traj for traj in trajs]
        
        obs_cols = [
            np.array([1.0, 0.65, 0]), np.array([.8, 0.55, 0]), np.array([.9, .75, 0]),
            np.array([.7, 0.7, 0]), np.array([1.0, 0.9, 0]), np.array([.5, 0.85, 0]),
        ]

        for ida, agent_traj in enumerate(gt_coordinates[:limit_pred_agents]):
            old_point = None
            r_val = random.uniform(0.4, 1.0)
            b_val = random.uniform(0.7, 1.0)
            for idp, coord in enumerate(agent_traj):
                
                x_n, y_n = round(coord[0]), round(coord[1])
                # assert 0 <= x_n <= img_np.shape[1] and 0 <= y_n <= img_np.shape[0]
                
                if old_point != None:
                    # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
                    # c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, y_n)))]
                    c_line = linemaker((old_point[0], old_point[1]), (x_n, y_n), thickness=3)
                    c_line = [item for sublist in c_line for item in sublist]
                    for c in c_line:
                        if idp < 8:
                            col = obs_cols[ida]-np.array([0.15, 0.15, 0.]) # make observations a bit darker
                        else:
                            col = obs_cols[ida]
                        img_np[c[1], c[0]] = col
                    # plt.imshow(img_np)
                old_point = (x_n, y_n)

                if idp == 7:
                    old_pred_point = (x_n, y_n)
                if idp >= 8:
                    pred_traj_coord = pred_coordinates[ida, idp-8, :]
                    x_n, y_n = round(pred_traj_coord[0]), round(pred_traj_coord[1])

                    if 0 > x_n or x_n > img_np.shape[1]:
                        print(f'x value exceeded!! {x_n} not within bounds!')
                        x_n = np.clip(x_n, 0, img_np.shape[1])
                    if 0 > y_n or y_n > img_np.shape[0]:
                        print(f'y value exceeded!! {y_n} not within bounds!')
                        y_n = np.clip(y_n, 0, img_np.shape[0])
                    # c_line = [coord for coord in zip(*line(*(old_gt_point[0], old_gt_point[1]), *(x_n, y_n)))]
                    c_line = linemaker((old_pred_point[0], old_pred_point[1]), (x_n, y_n), thickness=3)
                    c_line = [item for sublist in c_line for item in sublist]
                    for c in c_line:
                        col = np.array([r_val, 0., b_val])
                        img_np[c[1], c[0]] = col

                    old_pred_point = (x_n, y_n)

            plt.imshow(img_np)
            plt.close('all')

        # for traj in trajs:
        #     old_point = None
        #     r_val = random.uniform(0.4, 1.0)
        #     b_val = random.uniform(0.7, 1.0)
        #     for idp, point in enumerate(traj):
        #         x_n, y_n = round(point[0]), round(point[1])
        #         assert 0 <= x_n <= img_np.shape[1] and 0 <= y_n <= img_np.shape[0]
        #         if old_point != None:
        #             # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
        #             c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, y_n)))]
        #             for c in c_line:
        #                 if idp < 8:
        #                     col = obs_col
        #                 else:
        #                     col = np.array([r_val, 0., b_val])
        #                 img_np[c[1], c[0]] = col
        #             # plt.imshow(img_np)
        #         old_point = (x_n, y_n)

        # cv2.imshow('namey', img_np)
        # plt.imshow(img_np)
        # plt.close('all')
        # plt.imshow(img_input)
        # plt.close('all')