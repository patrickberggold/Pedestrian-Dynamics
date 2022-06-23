import os
import time
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from image2image_module import Image2ImageModule
from FloorplanDatamodule import FloorplanDataModule
from collections import OrderedDict
from torchvision import transforms
from helper import get_color_from_array, SEP

CUDA_DEVICE = 0 # 0, 1 or 'cpu'
MODE = 'img2img' # implemented: grayscale, rgb, bool, timeAndId, grayscale_movie
BATCH_SIZE = 4

do_training = True

datamodule = FloorplanDataModule(mode = MODE, cuda_index = CUDA_DEVICE, batch_size = BATCH_SIZE)

if do_training:

    model = Image2ImageModule(mode=MODE, relu_at_end=False)
    # print(model)

    model_checkpoint = ModelCheckpoint(
        dirpath = SEP.join(['TrajectoryPrediction','sophie','feature_extractor','checkpoints']),
        filename = 'model_img2img_{epoch}-{step}',
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
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=8), LearningRateMonitor(logging_interval='epoch'), model_checkpoint],
        limit_train_batches=None,
        limit_val_batches=None,
        # progress_bar_refresh_rate=125,
        )

    start_training_time = time.time()
    trainer.fit(model, datamodule=datamodule)
    # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # IMPORTANT: this will delete the old checkpoint!!
    # trainer.fit(model, datamodule=datamodule, ckpt_path="Image2Image\checkpoints\checkpoints_DeepLab4Img2Img\model_grayscale_scale_RELUatEnd_BBunfreezeAt8_epoch=51-step=8112.ckpt")
    print(f'Training took {(time.time() - start_training_time)/60./(model.current_epoch+1):.3f} minutes per epoch...')
    
    quit()

# Load stored model: keys in state_dict need to be adjusted
CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'model_grayscale_movie_numHeads8_L3Loss_epoch=26-step=4212.ckpt'])
state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])
# CKPT_PATH = SEP.join(['Image2Image', 'Optimization', 'ReLU_activation_at_end__l1_loss', 'model_optuna_optim___non_traj_vals__unfreeze.ckpt'])
# state_dict = torch.load(CKPT_PATH)
model = Image2ImageModule(mode=MODE, relu_at_end=True)
model.net.load_state_dict(state_dict)

# This only works when there are no required positional arguments that need to passed to the constructor
# model = Image2ImageModule.load_from_checkpoint(CKPT_PATH) #'checkpoints_DeepLab/model_epoch=38-step=6942.ckpt')
model.to(f'cuda:{CUDA_DEVICE}')
model.net.eval()

datamodule.set_non_traj_vals(new_val=0.0)

datamodule.setup(stage='test')
trainloader = datamodule.train_dataloader()
testloader = datamodule.test_dataloader()

test_result_folder = 'Image2Image'+SEP+'image_results'+SEP+'image_results'
if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

for idx, batch in enumerate(testloader):
    img = batch[0].float().to(f'cuda:{CUDA_DEVICE}')
    traj = batch[1]
    # TODO ground truth trajectories are doing some weird stuff, basically at early timestamps they dont seem to be connected... -> Investigate by plotting same GT image + traj with increasing timestamp
    traj_pred = model.forward(img)['out']

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
            img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()#*255
            # Draw predicted trajectories into floorplan
            traj_pred_np = traj_pred_np.squeeze()#*89.5 # Multiply by MAX_TIME
            traj_pred_bigger_thresh = np.argwhere(traj_pred_np >= 1.0)
            pred_colors_from_timestamps = [get_color_from_array(traj_pred_np[x, y], 89.5)/255. for x, y in traj_pred_bigger_thresh]
            traj_pred_img = img_gt.copy()
            traj_pred_img[traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1]] = np.array(pred_colors_from_timestamps)

            # Draw GT trajectories into floorplan
            traj_np = traj[i].detach().cpu().numpy()
            gt_colors_from_timestamps = [get_color_from_array(traj_np[x, y], 89.5)/255. for x, y in np.argwhere(traj_np >= 1.0)]
            img_gt[np.argwhere(traj_np>= 1.0)[:,0], np.argwhere(traj_np >= 1.0)[:,1]] = np.array(gt_colors_from_timestamps)
            county = idx*BATCH_SIZE + i

            # 3D plot pred
            # fig = plt.figure(figsize=(6,6))
            # ax = fig.add_subplot(111, projection='3d')
            # X,Y = np.meshgrid(np.arange(800), np.arange(800))
            # ax.plot_surface(X, Y, traj_pred_np)
            # plt.show()

            # 3D plot GT
            # fig = plt.figure(figsize=(6,6))
            # ax = fig.add_subplot(111, projection='3d')
            # X,Y = np.meshgrid(np.arange(800), np.arange(800))
            # ax.plot_surface(X, Y, traj_np)
            # plt.show()
            
            # plt.imshow(img_gt)
            # plt.imshow(traj_pred_img.astype('uint8'), vmin=0, vmax=255)
            fig, axes = plt.subplots(2, 1)
            axes[0].imshow(img_gt)
            axes[0].axis('off')
            axes[1].imshow(traj_pred_img)
            axes[1].axis('off')
            plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))
            
        elif MODE == 'grayscale_movie':
            img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()
            traj_pred_imgs_np = []
            traj_imgs_np = []
            for i_head in range(NUM_HEADS):
                traj_pred_img_np = img_gt.copy()
                traj_pred_np = traj_pred[i][i_head].cpu().detach().numpy()
                traj_pred_bigger_thresh = np.argwhere(traj_pred_np >= 1.0)
                pred_colors_from_timestamps = [get_color_from_array(traj_pred_np[x, y], 89.5)/255. for x, y in traj_pred_bigger_thresh]
                if len(pred_colors_from_timestamps) > 0: traj_pred_img_np[traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1]] = np.array(pred_colors_from_timestamps)
                traj_pred_imgs_np.append(traj_pred_img_np)

                traj_img_np = img_gt.copy()
                traj_np = traj[i][i_head].detach().cpu().numpy()
                gt_colors_from_timestamps = [get_color_from_array(traj_np[x, y], 89.5)/255. for x, y in np.argwhere(traj_np >= 1.0)]
                traj_img_np[np.argwhere(traj_np>= 1.0)[:,0], np.argwhere(traj_np >= 1.0)[:,1]] = np.array(gt_colors_from_timestamps)
                traj_imgs_np.append(traj_img_np)

                # plt.imshow(traj_img_np)
                # plt.imshow(traj_pred_img_np)

            fig, axes = plt.subplots(NUM_HEADS, 2)
            for i_h in range(NUM_HEADS):
                axes[i_h,0].imshow(traj_imgs_np[i_h])
                axes[i_h,0].axis('off')
                axes[i_h,1].imshow(traj_pred_imgs_np[i_h])
                axes[i_h,1].axis('off')
            plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_movie_recon_{idx}_{i}.png'))

        elif MODE == 'timeAndId':
            # Get GT floorplan RGB
            img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()#*255
            # Draw predicted trajectories into floorplan
            traj_pred_time_np = traj_pred[i, 0, :, :].detach().cpu().numpy()
            traj_pred_bigger_thresh = np.argwhere(traj_pred_time_np >= 1.0)
            pred_colors_from_timestamps = [get_color_from_array(traj_pred_time_np[x, y]+3.5, 89.5)/255. for x, y in traj_pred_bigger_thresh]
            traj_pred_img = img_gt.copy()
            traj_pred_img[traj_pred_bigger_thresh[:,0], traj_pred_bigger_thresh[:,1]] = np.array(pred_colors_from_timestamps)

            # traj_pred_ids_np =  traj_pred[i, 1:, :, :].detach().cpu().numpy()
            # # traj_pred_ids_np = traj_pred[i]
            # traj_pred_ids_np = traj_pred_ids_np.argmax(dim=0, keepdim=True).squeeze().detach().cpu().numpy()
            # miny = traj_pred_ids_np.min()
            # maxy = traj_pred_ids_np.max()
            # traj_pred_bigger_thresh = np.argwhere(traj_pred_ids_np >= 1.0)

            # Draw GT trajectories into floorplan
            traj_pred_np = traj[i,:,:,0].detach().cpu().numpy()
            # traj_np = np.clip(traj_np, 0, traj_np.max())
            gt_colors_from_timestamps = [get_color_from_array(traj_np[x, y], 89.5)/255. for x, y in np.argwhere(traj_np >= 1.0)]
            img_gt[np.argwhere(traj_np >= 1.0)[:,0], np.argwhere(traj_np >= 1.0)[:,1]] = np.array(gt_colors_from_timestamps)

            # 3D plot
            # fig = plt.figure(figsize=(6,6))
            # ax = fig.add_subplot(111, projection='3d')
            # X,Y = np.meshgrid(np.arange(800), np.arange(800))
            # ax.plot_surface(X, Y, traj_np)
            # plt.show()

        elif MODE == 'bool': 
            labels = torch.argmax(traj_pred, dim = 0)
            raise NotImplementedError('Check if this works first')
        elif MODE == 'rgb':
            traj_pred_np = traj_pred[i].transpose(0,1).transpose(1, 2).cpu().detach().numpy()
            img_gt = traj[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()
            traj_pred_img = traj_pred_np
            # traj_pred_np *= 255
            # fig, axes = plt.subplots(2, 1)
            # axes[0].imshow(img_gt)
            # axes[1].imshow(np.clip(traj_pred, 0, 255))
            # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))