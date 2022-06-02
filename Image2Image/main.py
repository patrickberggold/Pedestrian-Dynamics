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
from helper import get_color_from_array

CUDA_DEVICE = 0 # 0, 1 or 'cpu'
MODE = 'grayscale' # implemented: grayscale, rgb, bool
BATCH_SIZE = 1

do_training = False

datamodule = FloorplanDataModule(mode=MODE, cuda_index=CUDA_DEVICE, batch_size=BATCH_SIZE)

if do_training:

    model = Image2ImageModule(mode=MODE)
    # print(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath = 'checkpoints/checkpoints_DeepLab4Img2Img',
        filename = 'model_grayscale_{epoch}-{step}',
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
CKPT_PATH = 'checkpoints/checkpoints_DeepLab4Img2Img/model_rgb_epoch=10-step=1958.ckpt' # 'checkpoints/checkpoints_DeepLab4Img2Img/model_grayscale_epoch=30-step=5518.ckpt' # 'checkpoints/checkpoints_DeepLab4Img2Img/model_boolean_epoch=10-step=1958.ckpt'
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

test_result_folder = '/home/Code/Image2Image/image_results'
if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

for idx, batch in enumerate(testloader):
    img = batch[0].float().cuda()
    traj = batch[1]
    if MODE == 'grayscale':
        traj *= 89.511 # type np, shape (800,800)
    elif MODE == 'rgb':
        traj = traj.detach().cpu().numpy() # shape (3, 800, 800)
    elif MODE == 'bool':
        traj = traj # shape (800, 800)
    
    traj_pred = model.forward(img)['out']
    
    print(f'Saving plots in testload: {idx}/{len(testloader)}')
    
    for i in range(BATCH_SIZE):
        traj_pred = traj_pred.transpose(0,1).transpose(1, 2).cpu().detach().numpy()[i].squeeze()
        if MODE == 'grayscale': traj_pred *= 89.5
        elif MODE == 'bool': 
            labels = torch.argmax(traj_pred, dim = 0)
            raise NotImplementedError('Check if this works first')
        elif MODE == 'rgb':
            traj_pred *= 255
            img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()*255
            traj = traj.squeeze()
            fig, axes = plt.subplots(2, 1)
            axes[0].imshow(img_gt)
            axes[1].imshow(np.clip(traj_pred, 0, 255)) # TODO color-encoding missing
            plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))

        
        smaller_equal_zero_args = np.argwhere(traj_pred <= 0.)
        bigger_thresh_args = np.argwhere(traj_pred > 20.)
        # mask_pred[smaller_equal_zero_args[:,0], smaller_equal_zero_args[:,1]] = 0
        # create rgb img from mask_pred
        # mask_pred[:,:]
        # mask_pred_bw = np.argmax(mask_pred[0], axis = 0)
        img_gt = img[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()*255
        mask_pred_show = np.ones_like(img_gt)*255
        mask_gt_non_zero = np.argwhere(traj[i] > 0)

        # img_gt[mask_gt_non_zero[:,0], mask_gt_non_zero[:,1]] = np.array([0, 187, 255]) # TODO color-encoding missing TODO run RGB
        for (x, y) in mask_gt_non_zero:
            time_val = traj[i][x, y]
            img_gt[x,y] = np.array([0, 187-time_val, 255])
        for (x,y) in bigger_thresh_args:
            pred_time_val = traj_pred[x,y]
            mask_pred_show[x,y] = np.array([0, 187-pred_time_val, 255])
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(img_gt)
        axes[1].imshow(traj_pred) # TODO color-encoding missing
        plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))


quit()
img_GT = next(iter(testloader))
img_GT = img_GT.float().cuda()
y = model.forward(img_GT)
mask_pred = y['out'].cpu().detach().numpy()
mask_pred_bw = np.argmax(mask_pred[0], axis = 0)

# unorm = UnNormalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
# img2 = unorm(img)
img_GT = img_GT.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
fig, axes = plt.subplots()
axes.imshow(img_GT[0])
plt.savefig('floorplan_GT.png')

fig, axes = plt.subplots(2, 1)
axes[0].imshow(img_GT[0])
axes[1].imshow(mask_pred_bw, cmap=get_customized_colormap())
plt.savefig('floorplan_output.png')
plt.show()