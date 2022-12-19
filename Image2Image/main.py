import os
import time
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from models import Image2ImageModule
from Datamodules import FloorplanDataModule
from collections import OrderedDict
from torchvision import transforms
from helper import get_color_from_array, SEP, time_flow_creator
from hyperparameter_optim import hyperparameter_optimization
import imageio

CUDA_DEVICE = 0 # 0, 1 or 'cpu'
MODE = 'denseClass_wEvac' # implemented: grayscale, evac, evac_only, class_movie, density_reg, density_class, denseClass_wEvac
BATCH_SIZE = 4
ARCH = 'BeIT' # DeepLab, BeIT, SegFormer
ADD_INFO = True

CONFIG = {
    'mode': MODE, # implemented: grayscale, evac, evac_only, class_movie, density_reg, density_class, denseClass_wEvac
    'arch': ARCH, # DeepLab, BeIT, SegFormer
    'cuda_device': CUDA_DEVICE,
    'batch_size': BATCH_SIZE,
    'from_ckpt_path': 'beit_denseClass_wEvac_wholeDS_addInfo_Res6ConcatAtStart_B_ClassMovie_epoch=39-step=84760.ckpt',
    'load_to_ckpt_path': 'DELETE_ME_TESTRUN_',
    'early_stopping_patience': 10,
    'run_test_epoch': False,
    'vary_area_brightness': True,
    'limit_dataset': False, # or int-type
    'additional_info': ADD_INFO,
}

TRAIN_DICT = {
    # Learning rate and schedulers
    'learning_rate': 0.0008, 
    'lr_scheduler': 'ReduceLROnPlateau', 
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

do_training = True
do_hyperparameter_optim = False

if __name__ == '__main__':

    datamodule = FloorplanDataModule(config = CONFIG)

    if do_hyperparameter_optim:

        best_trial = hyperparameter_optimization(config=CONFIG, train_config=TRAIN_DICT, mode = MODE, arch=ARCH, datamodule=datamodule, n_trials = 20, epochs_per_trial = 3, folder_name='denseClass_rawTF', cuda_device = CUDA_DEVICE, test_run = False)

        if not do_training:
            quit()

    if do_training:
        module = Image2ImageModule(config=CONFIG, train_config=TRAIN_DICT)

        # Load from checkpoint
        if CONFIG['from_ckpt_path']:
            # trained_together_OPTIM_TWICE_2_1000dataset_epoch=16-step=2125.ckpt
            # CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'beit_whole_ds_finetuned_epoch=74-step=158925.ckpt'])
            # CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'beit_wholeDS_rawClassHead_epoch=19-step=42380.ckpt'])
            # CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'beit_wholeDS_rawTF_denseClass_v2_08_02_epoch=21-step=46618.ckpt'])
            # CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'beit_denseClass_wEvac_wholeDS_addInfo_NetworkD_ClassMovie_epoch=11-step=25428.ckpt'])
            CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', CONFIG['from_ckpt_path']])
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
        
        if ARCH == 'BeIT': module.model.model.auxiliary_head = None
        
        model_checkpoint = ModelCheckpoint(
            dirpath = SEP.join(['Image2Image','checkpoints', 'checkpoints_DeepLab4Img2Img']),
            filename = CONFIG['load_to_ckpt_path']+'{epoch}-{step}',
            save_top_k = 1,
            verbose = True, 
            monitor = 'val_loss',
            mode = 'min',
        )

        limit_batches = 2 if CONFIG['run_test_epoch'] else None
        trainer = pl.Trainer(
            gpus = [CUDA_DEVICE], 
            devices=f'cuda:{str(CUDA_DEVICE)}', 
            max_epochs = 500, 
            # checkpoint_callback = [checkpoint_callback], 
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=CONFIG['early_stopping_patience']), LearningRateMonitor(logging_interval='epoch'), model_checkpoint],
            limit_train_batches=limit_batches,
            limit_val_batches=limit_batches,
            # progress_bar_refresh_rate=125,
            )

        start_training_time = time.time()
        trainer.fit(module, datamodule=datamodule)
        # automatically restores model, epoch, step, LR schedulers, apex, etc...
        # IMPORTANT: this will delete the old checkpoint!!
        # trainer.fit(model, datamodule=datamodule, ckpt_path="Image2Image\checkpoints\checkpoints_DeepLab4Img2Img\model_grayscale_lineThickness5_CosAnn_Step5_Lr122_Gam42_epoch=36-step=5772.ckpt")
        print(f'Training took {(time.time() - start_training_time)/60./(module.current_epoch+1):.3f} minutes per epoch...')

        quit()

    # Load from checkpoint
    CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'beit_denseClass_wEvac_wholeDS_addInfo_Res6ConcatAtStart_B_ClassMovie_epoch=39-step=84760.ckpt'])
    module = Image2ImageModule(config=CONFIG, train_config=TRAIN_DICT)

    if ARCH == 'BeIT': module.model.model.auxiliary_head = None

    state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])
    module.load_state_dict(state_dict)
    
    # This only works when there are no required positional arguments that need to passed to the constructor
    module.to(f'cuda:{CUDA_DEVICE}')
    module.eval()

    datamodule.setup(stage='test')
    trainloader = datamodule.train_dataloader()
    testloader = datamodule.test_dataloader()

    test_result_folder = SEP.join(['Image2Image', 'image_results', 'image_results_presentation'])
    if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

    for idx, batch in enumerate(testloader):
        if MODE=='evac_only':
            img = batch[0][0].float().to(f'cuda:{CUDA_DEVICE}')
            evac_time = batch[0][1].float().to(f'cuda:{CUDA_DEVICE}') # 37.5000, 52.0000, 57.0000, 42.0000 // 36.1168, 51.1394, 65.4047, 45.4009
            evac_time_pred = module.forward(img) # 44.0000, 49.0000, 61.5000, 60.5000 // 47.8699, 48.8511, 53.3266, 60.4169
            mse_loss = torch.nn.MSELoss()(evac_time_pred.squeeze().unsqueeze(0), evac_time.unsqueeze(0))
        else:
            img = batch[0].float().to(f'cuda:{CUDA_DEVICE}')
            traj_pred = module.forward(img)#['out']
        traj = batch[1]
            
        print(f'Saving plots in testload: {idx}/{len(testloader)}')
        
        for i in range(BATCH_SIZE):
                    
            if MODE == 'grayscale':
                traj_pred_np = traj_pred[i].transpose(0,1).transpose(1, 2).cpu().detach().numpy()
                # Get GT floorplan RGB
                if ARCH != 'DeepLab': 
                    img_gt = transforms.Resize((800, 800))(img[i])
                    img_gt = img_gt.transpose(0,1).transpose(1, 2).detach().cpu().numpy()
                    img_gt = (img_gt+1)/2.
                else:
                    img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()#*255
                traj_max_timestamp = torch.max(traj[i]).item()
                traj_pred_max_timestamp = torch.max(traj_pred[i]).item()
                max_ts_overall = max(traj_max_timestamp, traj_pred_max_timestamp)
                # Draw predicted trajectories into floorplan
                traj_pred_np = traj_pred_np.squeeze()#*89.5 # Multiply by MAX_TIME
                traj_pred_bigger_thresh = np.argwhere(traj_pred_np > 0.0)
                pred_colors_from_timestamps = [get_color_from_array(traj_pred_np[x, y], max_ts_overall)/255. for x, y in traj_pred_bigger_thresh]
                traj_pred_img = img_gt.copy()
                # for i, (x, y) in enumerate(zip(traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1])):
                #     color = np.array(pred_colors_from_timestamps[i])
                #     traj_pred_img[x, y] = color

                if len(pred_colors_from_timestamps) > 0: traj_pred_img[traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1]] = np.array(pred_colors_from_timestamps)

                # Draw GT trajectories into floorplan
                traj_np = traj[i].detach().cpu().numpy()
                gt_colors_from_timestamps = [get_color_from_array(traj_np[x, y], max_ts_overall)/255. for x, y in np.argwhere(traj_np > 0.0)]
                img_gt[np.argwhere(traj_np > 0.0)[:,0], np.argwhere(traj_np > 0.0)[:,1]] = np.array(gt_colors_from_timestamps)
                
                # time_flow_creator(img_gt.copy(), [traj_np, traj_pred_np], traj_max_timestamp, idx*BATCH_SIZE+i)
                
                plt.imshow(img_gt)
                plt.close('all')
                plt.imshow(traj_pred_img)
                plt.close('all')

                # # 3D plot pred
                # fig = plt.figure(figsize=(6,6))
                # ax = fig.add_subplot(111, projection='3d')
                # X,Y = np.meshgrid(np.arange(800), np.arange(800))
                # ax.plot_surface(X, Y, traj_pred_np)
                # plt.show()

                # # 3D plot GT
                # fig = plt.figure(figsize=(6,6))
                # ax = fig.add_subplot(111, projection='3d')
                # X,Y = np.meshgrid(np.arange(800), np.arange(800))
                # ax.plot_surface(X, Y, traj_np)
                # plt.show()

                # fig, axes = plt.subplots(2, 1)
                # axes[0].imshow(img_gt)
                # axes[0].axis('off')
                # axes[1].imshow(traj_pred_img)
                # axes[1].axis('off')
                # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))
                # plt.show()
                # plt.close('all')
            
            elif MODE == 'class_movie':
                # Get GT floorplan RGB
                if ARCH != 'DeepLab': 
                    img_gt = transforms.Resize((800, 800))(img[i])
                    img_gt = img_gt.transpose(0,1).transpose(1, 2).detach().cpu().numpy()
                    img_gt = (img_gt+1)/2.
                else:
                    img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()#*255
                traj_np = traj[i].cpu().detach().numpy()
                traj_pred_np = traj_pred[i].cpu().detach().numpy()
                traj_pred_class = np.argmax(traj_pred_np, axis=0)
                for frame_id in range(len(traj_np)):
                    img_cp_gt = img_gt.copy()
                    img_cp_pred = img_gt.copy()
                    frame_mask_gt = traj_np[frame_id]
                    frame_mask_pred = traj_pred_class[frame_id]

                    coo_gt_slow = np.argwhere(frame_mask_gt == 1).squeeze()
                    coo_gt_fast = np.argwhere(frame_mask_gt == 2).squeeze()

                    coo_pred_slow = np.argwhere(frame_mask_pred == 1).squeeze()
                    coo_pred_fast = np.argwhere(frame_mask_pred == 2).squeeze()

                    img_cp_gt[coo_gt_slow[:, 0], coo_gt_slow[:, 1]] = np.array([0.6, 0.4, 0.])
                    img_cp_gt[coo_gt_fast[:, 0], coo_gt_fast[:, 1]] = np.array([0., 0.5, 1.])
                    
                    img_cp_pred[coo_pred_slow[:, 0], coo_pred_slow[:, 1]] = np.array([0.6, 0.4, 0.])
                    img_cp_pred[coo_pred_fast[:, 0], coo_pred_fast[:, 1]] = np.array([0., 0.5, 1.])

                    fig, axes = plt.subplots(2, 1)
                    axes[0].imshow(img_cp_gt)
                    axes[0].axis('off')
                    axes[1].imshow(img_cp_pred)
                    axes[1].axis('off')
                    # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))
                    plt.show()
                    plt.close('all')

            elif MODE in ['density_reg', 'density_class']:
                if ARCH != 'DeepLab': 
                    # img_gt = transforms.Resize((800, 800))(img[i])
                    img_gt = img[i].permute(1, 2, 0).detach().cpu().numpy()
                    img_gt = (img_gt+1)/2.
                else:
                    img_gt = img[i].permute(1, 2, 0).detach().cpu().numpy()#*255
                traj_np = traj[i].cpu().detach().numpy()
                traj_pred_np = traj_pred[i].cpu().detach().numpy()
                max_traj = np.max(traj_np)
                max_traj_pred = np.max(traj_pred_np)

                if MODE == 'density_class':
                    traj_pred_np = np.argmax(traj_pred_np, axis=0)
                    max_traj_pred = np.max(traj_pred_np)
                    traj_np = traj_np.transpose(1,2,0)

                max_bin_per_img = max(np.max(traj_np), np.max(traj_pred_np))
                cell_size = 4
                add_vector_template = np.zeros((cell_size**2,2), dtype=np.int32)
                for x in range(cell_size):
                    for y in range(cell_size):
                        add_vector_template[cell_size*x + y] = np.array([x, y], dtype=np.int32)
                
                for frame_id in range(traj_np.shape[-1]):
                    frame_np = traj_np[:,:,frame_id]
                    nnz_coords = np.argwhere(frame_np > 0).squeeze()
                    gt_counts = frame_np[nnz_coords[:,0], nnz_coords[:,1]]
                    # scale up
                    nnz_coords = np.repeat(nnz_coords, cell_size**2, axis=0) * cell_size
                    add_vector = np.tile(add_vector_template.transpose(1,0), np.argwhere(frame_np > 0).squeeze().shape[0]).transpose(1,0)
                    nnz_coords += add_vector
                    gt_counts = np.repeat(gt_counts, cell_size**2, axis=0)
                    gt_counts_colored = [get_color_from_array(count, max_bin_per_img)/255. for count in gt_counts]
                    binned_img = img_gt.copy()
                    if len(gt_counts_colored) > 0: binned_img[nnz_coords[:,0], nnz_coords[:,1]] = gt_counts_colored

                    frame_pred_np = traj_pred_np[:,:,frame_id]
                    nnz_coords = np.argwhere(frame_pred_np > 0).squeeze()
                    pred_counts = frame_pred_np[nnz_coords[:,0], nnz_coords[:,1]]
                    # scale up
                    nnz_coords = np.repeat(nnz_coords, cell_size**2, axis=0) * cell_size
                    add_vector = np.tile(add_vector_template.transpose(1,0), np.argwhere(frame_pred_np > 0).squeeze().shape[0]).transpose(1,0)
                    nnz_coords += add_vector
                    pred_counts = np.repeat(pred_counts, cell_size**2, axis=0)
                    pred_counts_colored = [get_color_from_array(count, max_bin_per_img)/255. for count in pred_counts]
                    binned_pred_img = img_gt.copy()
                    if len(pred_counts_colored) > 0: binned_pred_img[nnz_coords[:,0], nnz_coords[:,1]] = pred_counts_colored

                    # plt.imshow(binned_img)
                    # plt.close('all')
                    # plt.imshow(binned_pred_img)
                    # plt.close('all')

                    fig, axes = plt.subplots(2, 1)
                    axes[0].imshow(binned_img)
                    axes[0].axis('off')
                    axes[1].imshow(binned_pred_img)
                    axes[1].axis('off')
                    # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))
                    plt.show()
                    plt.close('all')