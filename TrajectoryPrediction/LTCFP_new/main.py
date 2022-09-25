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
from torchvision import transforms
from helper import SEP
# from hyperparameter_optim import hyperparameter_optimization
import imageio

MODE = 'undefined'
CUDA_DEVICE = 1
BATCH_SIZE = 1

do_training = True

if __name__ == '__main__':

    datamodule = FloorplanDatamodule(mode=MODE, cuda_index=CUDA_DEVICE, batch_size=BATCH_SIZE)

    if do_training:

        model = Seq2SeqModule(mode=MODE, unfreeze_backbone_at_epoch=3, lr_sch_step_size=100, alternate_unfreezing=True)

        model_checkpoint = ModelCheckpoint(
            dirpath = SEP.join(['TrajectoryPrediction', 'LTCFP_new', 'checkpoints']),
            filename = 'goal_{epoch}-{step}',
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
        # automatically restores model, epoch, step, LR schedulers, apex, etc... BUT will delete the old checkpoint!!
        # trainer.fit(model, datamodule=datamodule, ckpt_path="Image2Image\checkpoints\checkpoints_DeepLab4Img2Img\model_grayscale_lineThickness5_CosAnn_Step5_Lr122_Gam42_epoch=36-step=5772.ckpt")
        print(f'Training took {(time.time() - start_training_time)/60./(model.current_epoch+1):.3f} minutes per epoch...')

        quit()

    CKPT_PATH = SEP.join(['TrajectoryPrediction', 'LTCFP_new', 'checkpoints', 'goal_epoch=16-step=3264.ckpt'])
    state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])

    model = Seq2SeqModule(mode=MODE, unfreeze_backbone_at_epoch=3, lr_sch_step_size=100, alternate_unfreezing=True)

    model.load_state_dict(OrderedDict((k_module, v_loaded) for (k_loaded, v_loaded), (k_module, v_module) in zip(state_dict.items(), model.state_dict().items())))

    model.to(f'cuda:{CUDA_DEVICE}')
    model.eval()

    datamodule.setup(stage='test')
    dataloader = datamodule.train_dataloader() # test_dataloader()
    
    test_result_folder = 'TrajectoryPrediction'+SEP+'LTCFP_new'+SEP+'image_results'
    if not os.path.isdir(test_result_folder): os.mkdir(test_result_folder)

    for idx, batch in enumerate(dataloader):
        batch = {k: v.squeeze(0).float().to(f'cuda:{CUDA_DEVICE}') for k, v in batch.items() if k != 'scene_data'}
        tensor_image = batch['tensor_image']
        all_output = model.net.forward(batch, if_test=False)[0]

        import random
        from skimage.draw import line
        img_np = tensor_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        all_output = all_output.squeeze().permute(1, 0, 2) # 1, 20, 16, 2])
        trajs = [traj.detach().cpu().numpy() for traj in all_output]
        obs_col = np.array([1.0, 0.65, 0])
        for traj in trajs:
            old_point = None
            r_val = random.uniform(0.4, 1.0)
            b_val = random.uniform(0.7, 1.0)
            for idp, point in enumerate(traj):
                x_n, y_n = round(point[0]), round(point[1])
                assert 0 <= x_n <= img_np.shape[1] and 0 <= y_n <= img_np.shape[0]
                if old_point != None:
                    # cv2.line(img_np, (old_point[1], old_point[0]), (y_n, x_n), (0, 1.,0 ), thickness=5)
                    c_line = [coord for coord in zip(*line(*(old_point[0], old_point[1]), *(x_n, y_n)))]
                    for c in c_line:
                        if idp < 8:
                            col = obs_col
                        else:
                            col = np.array([r_val, 0., b_val])
                        img_np[c[1], c[0]] = col
                    # plt.imshow(img_np)
                old_point = (x_n, y_n)

        # cv2.imshow('namey', img_np)
        plt.imshow(img_np)
        plt.close('all')