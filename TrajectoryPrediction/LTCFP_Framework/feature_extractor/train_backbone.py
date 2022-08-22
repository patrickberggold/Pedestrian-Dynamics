import os
import time
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from .image2image_module import Image2ImageModule
from .FloorplanDatamodule import FloorplanDataModule
from collections import OrderedDict
from helper import SEP

def train_backbone():
    CUDA_DEVICE = 0 # 0, 1 or 'cpu'
    MODE = 'img2img' # implemented: img2img
    BATCH_SIZE = 4

    datamodule = FloorplanDataModule(mode = MODE, cuda_index = CUDA_DEVICE, batch_size = BATCH_SIZE)

    model = Image2ImageModule(mode=MODE, relu_at_end=True)#, lr_sch_step_size=len(datamodule.train_dataloader)/BATCH_SIZE)
    # print(model)

    model_checkpoint = ModelCheckpoint(
        dirpath = SEP.join(['TrajectoryPrediction','LTCFP_Framework','feature_extractor','checkpoints']),
        filename = 'model_deepLab_img2img_SMALL_{epoch}-{step}',
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
    # trainer.fit(model, datamodule=datamodule, ckpt_path="TrajectoryPrediction\\sophie\\feature_extractor\\checkpoints\\model_vgg_img2img_epoch=19-step=3120.ckpt")
    print(f'Training took {(time.time() - start_training_time)/60./(model.current_epoch+1):.3f} minutes per epoch...')
   

def backbone_inference():
    CUDA_DEVICE = 0 # 0, 1 or 'cpu'
    MODE = 'img2img' # implemented: img2img
    BATCH_SIZE = 4

    datamodule = FloorplanDataModule(mode = MODE, cuda_index = CUDA_DEVICE, batch_size = BATCH_SIZE)
    
    # Load stored model: keys in state_dict need to be adjusted
    CKPT_PATH = SEP.join(['TrajectoryPrediction','LTCFP_Framework','feature_extractor','checkpoints', 'model_deepLab_img2img_V2_epoch=10-step=11660.ckpt'])
    state_dict = OrderedDict([(key, tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])
    # CKPT_PATH = SEP.join(['Image2Image', 'Optimization', 'ReLU_activation_at_end__l1_loss', 'model_optuna_optim___non_traj_vals__unfreeze.ckpt'])
    # state_dict = torch.load(CKPT_PATH)
    model = Image2ImageModule(mode=MODE, relu_at_end=True)
    model.load_state_dict(state_dict)

    # This only works when there are no required positional arguments that need to passed to the constructor
    # model = Image2ImageModule.load_from_checkpoint(CKPT_PATH) #'checkpoints_DeepLab/model_epoch=38-step=6942.ckpt')
    model.to(f'cuda:{CUDA_DEVICE}')
    model.net.eval()

    datamodule.set_non_traj_vals(new_val=0.0)

    datamodule.setup(stage='test')
    trainloader = datamodule.train_dataloader()
    testloader = datamodule.test_dataloader()

    # test_result_folder = 'Image2Image'+SEP+'image_results'+SEP+'image_results'
    # if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

    for idx, batch in enumerate(testloader):
        img = batch.float().to(f'cuda:{CUDA_DEVICE}')

        # TODO ground truth trajectories are doing some weird stuff, basically at early timestamps they dont seem to be connected... -> Investigate by plotting same GT image + traj with increasing timestamp
        img_pred = model.forward(img)['out']
        
        print(f'Saving plots in testload: {idx}/{len(testloader)}')
        
        for i in range(BATCH_SIZE):

            if MODE == 'img2img':
                img_np = img[i].transpose(0,1).transpose(1, 2).cpu().detach().numpy()
                img_pred_np = img_pred[i].transpose(0,1).transpose(1, 2).detach().cpu().numpy()
                img_total = np.concatenate((img_np, img_pred_np), axis=1)
                # plt.imshow(img_total)
                # plt.imshow(img_pred_np)

                # traj_pred_np *= 255
                # fig, axes = plt.subplots(2, 1)
                # axes[0].imshow(img_gt)
                # axes[1].imshow(np.clip(traj_pred, 0, 255))
                # plt.savefig(os.path.join(test_result_folder, f'floorplan_traj_recon_{idx}_{i}.png'))
                plt.imshow(img_total)

if __name__=='__main__':
    train_backbone()