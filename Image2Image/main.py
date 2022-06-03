import os
import torch
from statistics import mode
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from models import Image2ImageModule
from Datamodules import FloorplanDataModule
from collections import OrderedDict
from torchvision import transforms
from helper import get_color_from_array, SEP

CUDA_DEVICE = 1 # 0, 1 or 'cpu'
MODE = 'grayscale' # implemented: grayscale, rgb, bool
BATCH_SIZE = 4

do_training = False

datamodule = FloorplanDataModule(mode=MODE, cuda_index=CUDA_DEVICE, batch_size=BATCH_SIZE)

if do_training:

    model = Image2ImageModule(mode=MODE)
    # print(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath = SEP.join(['Image2Image','checkpoints', 'checkpoints_DeepLab4Img2Img']),
        filename = 'model_grayscale_scale_-0.02_1.0_{epoch}-{step}',
        save_top_k = 1,
        verbose = True, 
        monitor = 'loss',
        mode = 'min',
    )

    trainer = pl.Trainer(gpus = [CUDA_DEVICE], devices=f'cuda:{str(CUDA_DEVICE)}', max_epochs = 70, checkpoint_callback = [checkpoint_callback], callbacks=[EarlyStopping(monitor="loss", mode="min", patience=8), checkpoint_callback])

    trainer.fit(model, datamodule=datamodule)
    # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

    quit()

# Load stored model
# keys in state_dict need to be adjusted
CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'model_grayscale_scale_-5_89.5_epoch=7-step=1424.ckpt']) # model_rgb_epoch=10-step=1958
state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])
model = Image2ImageModule(mode=MODE)
model.net.load_state_dict(state_dict)

# This only works when there are no required positional arguments that need to passed to the constructor
# model = Image2ImageModule.load_from_checkpoint(CKPT_PATH) #'checkpoints_DeepLab/model_epoch=38-step=6942.ckpt')

model.cuda()
model.net.eval()
datamodule.setup(stage='test')
trainloader = datamodule.train_dataloader()
testloader = datamodule.test_dataloader()

test_result_folder = 'Image2Image'+SEP+'image_results'
if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

for idx, batch in enumerate(testloader):
    img = batch[0].float().cuda()
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
            img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()#*255
            traj_pred_np = traj_pred_np.squeeze()#*89.5 # Multiply by MAX_TIME
            traj_pred_bigger_ones = np.argwhere(traj_pred_np > 0.0)
            pred_colors_from_timestamps = [get_color_from_array(traj_pred_np[x, y], 89.5)/255. for x, y in traj_pred_bigger_ones]

            # 3D plot
            # fig = plt.figure(figsize=(6,6))
            # ax = fig.add_subplot(111, projection='3d')
            # X,Y = np.meshgrid(np.arange(800), np.arange(800))
            # ax.plot_surface(X, Y, traj_pred_np)
            # plt.show()

            traj_pred_img = img_gt.copy()
            traj_pred_img[traj_pred_bigger_ones[:,0], traj_pred_bigger_ones[:,1]] = np.array(pred_colors_from_timestamps)

            traj_np = traj[i].detach().cpu().numpy()
            traj_np = np.clip(traj_np, 0, traj_np.max())
            gt_colors_from_timestamps = [get_color_from_array(traj_np[x, y], 89.5)/255. for x, y in np.argwhere(traj_np > 0.)]
            img_gt[np.argwhere(traj_np > 0.)[:,0], np.argwhere(traj_np > 0.)[:,1]] = np.array(gt_colors_from_timestamps)

        elif MODE == 'bool': 
            labels = torch.argmax(traj_pred, dim = 0)
            raise NotImplementedError('Check if this works first')
        elif MODE == 'rgb':
            img_gt = traj[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()
            traj_pred_img = traj_pred_np
            # traj_pred_np *= 255
            # fig, axes = plt.subplots(2, 1)
            # axes[0].imshow(img_gt)
            # axes[1].imshow(np.clip(traj_pred, 0, 255)) # TODO color-encoding missing
            # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))
        
        # plt.imshow(img_gt)
        # plt.imshow(traj_pred_img.astype('uint8'), vmin=0, vmax=255)
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(img_gt)
        axes[0].axis('off')
        axes[1].imshow(traj_pred_img)
        axes[1].axis('off')
        plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))