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
from helper import get_color_from_array, SEP
from hyperparameter_optim import hyperparameter_optimization

CUDA_DEVICE = 0 # 0, 1 or 'cpu'
MODE = 'grayscale' # implemented: grayscale, rgb, bool, timeAndId
BATCH_SIZE = 4

do_training = False
do_hyperparameter_optim = False

datamodule = FloorplanDataModule(mode = MODE, cuda_index = CUDA_DEVICE, batch_size = BATCH_SIZE, non_traj_vals=0.)

if do_hyperparameter_optim:

    best_trial = hyperparameter_optimization(mode = MODE, datamodule = datamodule, n_trials = 35, epochs_per_trial = 25, cuda_device = CUDA_DEVICE, limit_train_batches = None, limit_val_batches = None)

    if not do_training:
        quit()

if do_training:

    model = Image2ImageModule(mode=MODE, relu_at_end=True)
    # print(model)

    model_checkpoint = ModelCheckpoint(
        dirpath = SEP.join(['Image2Image','checkpoints', 'checkpoints_DeepLab4Img2Img']),
        filename = 'model_grayscale_scale_RELUatEnd_BBunfreezeAt8_CONTI_L4_{epoch}-{step}',
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
CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'model_grayscale_scale_RELUatEnd_BBunfreezeAt8_CONTI_L4_epoch=10-step=1716.ckpt'])
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

test_result_folder = 'Image2Image'+SEP+'image_results'
if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

for idx, batch in enumerate(testloader):
    img = batch[0].float().to(f'cuda:{CUDA_DEVICE}')
    traj = batch[1]
    if MODE == 'grayscale':
        traj = traj#*89.5 # type np, shape (800,800)
    elif MODE == 'bool':
        traj = traj # shape (800, 800)
    
    traj_pred = model.forward(img)['out']
    
    print(f'Saving plots in testload: {idx}/{len(testloader)}')
    
    for i in range(BATCH_SIZE):
        traj_pred_np = traj_pred[i].transpose(0,1).transpose(1, 2).cpu().detach().numpy()
        
        if MODE == 'grayscale':
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

        if MODE == 'timeAndId':
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
            img_gt = traj[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()
            traj_pred_img = traj_pred_np
            # traj_pred_np *= 255
            # fig, axes = plt.subplots(2, 1)
            # axes[0].imshow(img_gt)
            # axes[1].imshow(np.clip(traj_pred, 0, 255))
            # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))
        
        # plt.imshow(img_gt)
        # plt.imshow(traj_pred_img.astype('uint8'), vmin=0, vmax=255)
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(img_gt)
        axes[0].axis('off')
        axes[1].imshow(traj_pred_img)
        axes[1].axis('off')
        plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))