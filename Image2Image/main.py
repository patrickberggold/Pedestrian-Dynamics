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
from helper import get_color_from_array, SEP, load_img2img_into_state_dict
from hyperparameter_optim import hyperparameter_optimization
import imageio

CUDA_DEVICE = 1 # 0, 1 or 'cpu'
MODE = 'grayscale' # implemented: grayscale, grayscale_movie, evac
BATCH_SIZE = 4
NUM_HEADS = 8
PRED_EVAC_TIME = True
ARCH = 'BeIT' # DeepLab, BeIT, SegFormer

do_training = False
do_hyperparameter_optim = False

if __name__ == '__main__':

    datamodule = FloorplanDataModule(mode = MODE, arch=ARCH, cuda_index = CUDA_DEVICE, num_workers=2, batch_size = BATCH_SIZE, num_ts_per_floorplan=NUM_HEADS, vary_area_brightness=True)

    if do_hyperparameter_optim:

        best_trial = hyperparameter_optimization(mode = MODE, datamodule=datamodule, n_trials = 100, epochs_per_trial = 6, folder_name='img2img_finetune_2', cuda_device = CUDA_DEVICE, test_run = False)

        if not do_training:
            quit()

    if do_training:
        # TODO add last conv layer(s)
        module = Image2ImageModule(mode=MODE, arch=ARCH, unfreeze_backbone_at_epoch=None, learning_rate=0.0007, lr_sch_gamma=0.24946288393411667, lr_sch_step_size=100, lr_scheduler='ReduceLROnPlateau', alternate_unfreezing=True, num_heads=NUM_HEADS)
        # module.model.append_conv_layers()

        # Load from checkpoint
        # trained_together_OPTIM_TWICE_2_1000dataset_epoch=16-step=2125.ckpt
        CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'beit_whole_ds_epoch=11-step=25428.ckpt'])
        state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])
        module.model.load_state_dict(OrderedDict((k_module, v_loaded) for (k_loaded, v_loaded), (k_module, v_module) in zip(state_dict.items(), module.model.state_dict().items())))

        model_checkpoint = ModelCheckpoint(
            dirpath = SEP.join(['Image2Image','checkpoints', 'checkpoints_DeepLab4Img2Img']),
            filename = 'beit_whole_ds_finetuned_{epoch}-{step}',
            save_top_k = 1,
            verbose = True, 
            monitor = 'val_loss',
            mode = 'min',
        )

        trainer = pl.Trainer(
            gpus = [CUDA_DEVICE], 
            devices=f'cuda:{str(CUDA_DEVICE)}', 
            max_epochs = 100, 
            # checkpoint_callback = [checkpoint_callback], 
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10), LearningRateMonitor(logging_interval='epoch'), model_checkpoint],
            limit_train_batches=None,
            limit_val_batches=None,
            # progress_bar_refresh_rate=125,
            )

        start_training_time = time.time()
        trainer.fit(module, datamodule=datamodule)
        # automatically restores model, epoch, step, LR schedulers, apex, etc...
        # IMPORTANT: this will delete the old checkpoint!!
        # trainer.fit(model, datamodule=datamodule, ckpt_path="Image2Image\checkpoints\checkpoints_DeepLab4Img2Img\model_grayscale_lineThickness5_CosAnn_Step5_Lr122_Gam42_epoch=36-step=5772.ckpt")
        print(f'Training took {(time.time() - start_training_time)/60./(module.current_epoch+1):.3f} minutes per epoch...')

        quit()

    # Load stored model: keys in state_dict need to be adjusted
    CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'beit_whole_ds_finetuned_epoch=23-step=50856.ckpt'])
    # CKPT_PATH = SEP.join(['Image2Image', 'Optimization', 'evac_finetune_overfit', 'trained_together_OPTIM_TWICE_2_1000dataset_epoch=16-step=2125.ckpt'])
    # model_grayscale_scale_RELUatEnd_BBunfreezeAt8_epoch=51-step=8112
    # model_grayscale_lineThickness5_PRETRAINED_5datasets_CosAnn_Step5_Lr122_Gam42_epoch=15-step=12432
    # model_grayscalemovie_lineThickness5_PRETRAINED_5datasets_CosAnn_Step5_Lr122_Gam42_epoch=16-step=13209
    # new_model_only_img2img_ALT_FREEZE_epoch=51-step=110188.ckpt
    # trained_together_OPTIM_TWICE_2_1000dataset_epoch=16-step=2125.ckpt
    state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])

    model = Image2ImageModule(mode=MODE, arch=ARCH, num_heads=NUM_HEADS)
    model.load_state_dict(OrderedDict((k_module, v_loaded) for (k_loaded, v_loaded), (k_module, v_module) in zip(state_dict.items(), model.state_dict().items())))

    # This only works when there are no required positional arguments that need to passed to the constructor
    model.to(f'cuda:{CUDA_DEVICE}')
    model.eval()

    datamodule.setup(stage='test')
    trainloader = datamodule.train_dataloader()
    testloader = datamodule.test_dataloader()

    test_result_folder = SEP.join(['Image2Image', 'image_results', 'image_results_presentation'])
    if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

    for idx, batch in enumerate(testloader):
        if MODE=='evac':
            img = batch[0][0].float().to(f'cuda:{CUDA_DEVICE}')
            evac_time = batch[0][1].float().to(f'cuda:{CUDA_DEVICE}')
            # TODO ground truth trajectories are doing some weird stuff, basically at early timestamps they dont seem to be connected... -> Investigate by plotting same GT image + traj with increasing timestamp
            traj_pred, evac_time_pred = model.forward(img)
            MODE = 'grayscale'
        else:
            img = batch[0].float().to(f'cuda:{CUDA_DEVICE}')
            traj_pred = model.forward(img)#['out']
        traj = batch[1]

        if MODE == 'grayscale_movie':
            # Flip batch and heads
            traj = [traj_el.unsqueeze(0) for traj_el in traj]
            traj = torch.cat(traj, dim=0).permute(1, 0, 2, 3)
            traj_pred = torch.cat(traj_pred, dim=1)
            
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
                # Draw predicted trajectories into floorplan
                traj_pred_np = traj_pred_np.squeeze()#*89.5 # Multiply by MAX_TIME
                traj_pred_bigger_thresh = np.argwhere(traj_pred_np >= 1.0)
                pred_colors_from_timestamps = [get_color_from_array(traj_pred_np[x, y], traj_max_timestamp)/255. for x, y in traj_pred_bigger_thresh]
                traj_pred_img = img_gt.copy()
                # for i, (x, y) in enumerate(zip(traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1])):
                #     color = np.array(pred_colors_from_timestamps[i])
                #     traj_pred_img[x, y] = color

                if len(pred_colors_from_timestamps) > 0: traj_pred_img[traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1]] = np.array(pred_colors_from_timestamps)

                # Draw GT trajectories into floorplan
                traj_np = traj[i].detach().cpu().numpy()
                gt_colors_from_timestamps = [get_color_from_array(traj_np[x, y], traj_max_timestamp)/255. for x, y in np.argwhere(traj_np >= 1.0)]
                img_gt[np.argwhere(traj_np>= 1.0)[:,0], np.argwhere(traj_np >= 1.0)[:,1]] = np.array(gt_colors_from_timestamps)

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
                
            elif MODE == 'grayscale_movie':
                img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()
                traj_pred_imgs_np = []
                traj_imgs_np = []
                images = []
                for i_head in range(NUM_HEADS):
                    traj_pred_img_np = img_gt.copy()
                    traj_pred_np = traj_pred[i][i_head].cpu().detach().numpy()
                    traj_pred_bigger_thresh = np.argwhere(traj_pred_np >= 1.0)
                    # TODO (in case of normalization) visualize correctly... multiplication by 89.5 not correct because I am always calculating with the maximum timestep of the floorplan
                    pred_colors_from_timestamps = [get_color_from_array(traj_pred_np[x, y], 89.5)/255. for x, y in traj_pred_bigger_thresh]
                    if len(pred_colors_from_timestamps) > 0: traj_pred_img_np[traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1]] = np.array(pred_colors_from_timestamps)
                    traj_pred_imgs_np.append(traj_pred_img_np)

                    traj_img_np = img_gt.copy()
                    traj_np = traj[i][i_head].detach().cpu().numpy()
                    gt_colors_from_timestamps = [get_color_from_array(traj_np[x, y], 89.5)/255. for x, y in np.argwhere(traj_np >= 1.0)]
                    traj_img_np[np.argwhere(traj_np>= 1.0)[:,0], np.argwhere(traj_np >= 1.0)[:,1]] = np.array(gt_colors_from_timestamps)
                    traj_imgs_np.append(traj_img_np)

                    if i_head == 4:
                        # 3D plot pred
                        fig = plt.figure(figsize=(6,6))
                        ax = fig.add_subplot(111, projection='3d')
                        X,Y = np.meshgrid(np.arange(800), np.arange(800))
                        ax.plot_surface(X, Y, traj_pred_np)
                        plt.show()

                        # 3D plot GT
                        fig = plt.figure(figsize=(6,6))
                        ax = fig.add_subplot(111, projection='3d')
                        X,Y = np.meshgrid(np.arange(800), np.arange(800))
                        ax.plot_surface(X, Y, traj_np)
                        plt.show()

                    # plt.imshow(traj_img_np)
                    # plt.imshow(traj_pred_img_np)
                    # plt.imsave(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}_frame_{i_head}.png'), np.concatenate((traj_img_np, traj_pred_img_np), axis=0),dpi=900)
                    # fig, axes = plt.subplots(2, 1)
                    # axes[0].imshow(traj_img_np)
                    # axes[0].axis('off')
                    # axes[1].imshow(traj_pred_img_np)
                    # axes[1].axis('off')
                    # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}_frame_{i_head}.png'), dpi=900)
                    # plt.close('all')
                    # images.append(imageio.imread(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}_frame_{i_head}.png')))
            
                # imageio.mimsave(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}_movie.gif'), images)

                # fig, axes = plt.subplots(NUM_HEADS, 2)
                # for i_h in range(NUM_HEADS):
                #     axes[i_h,0].imshow(traj_imgs_np[i_h])
                #     axes[i_h,0].axis('off')
                #     axes[i_h,1].imshow(traj_pred_imgs_np[i_h])
                #     axes[i_h,1].axis('off')
                # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_movie_recon_{idx}_{i}_optim.png'))
                # plt.close('all')