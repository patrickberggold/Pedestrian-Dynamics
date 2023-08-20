import os
from collections import OrderedDict
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ObjDetDatamodule import ObjDetDatamodule
from ObjDetModule import ObjDetModule, metrics
from tqdm import tqdm
from helper import SEP, dir_maker, update_config

CUDA_DEVICE = 0
BATCH_SIZE = 4
ARCH = 'EfficientDet' # Detr, FasterRCNN, FasterRCNN_custom, EfficientDet, YoloV5

test_run = False # limit_batches -> 2 
save_model = True # create folder and save the model

CONFIG = {
    'arch': ARCH, # DeepLab, BeIT, SegFormer
    'version': '5', # '0', '4', 's', 'm'
    'cuda_device': CUDA_DEVICE,
    'batch_size': BATCH_SIZE,
    'from_ckpt_path': None, # 'EffDet-D4_onYellow_1024p_pretrained_black', # 'Yolo_onOrigins', # 'FirstTry_FasterRCNN_wMAP_corrupted', # None, FirstTry_FasterRCNN
    'load_to_ckpt_path': 'EffDet-D5_custom_1024p_pretrained_black_larger', # FirstTry_FasterRCNN_onlyBoxes
    'early_stopping_patience': 50,
    'run_test_epoch': test_run,
    'limit_dataset': None, 
    'save_model': save_model,
    'img_max_size': (1024, 1024), # (640, 1536), # (1344, 3072), # next highest 64-divisible
    'num_classes': 1,
}
update_config(CONFIG)
# TODO: massively extend the black dataset
TRAIN_DICT = {
    # Learning rate and schedulers
    'learning_rate': 0.005, 
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
    'model_ema_decay': 0.999966, # None
    'customized_optim': True
}

do_training = True

if __name__ == '__main__':

    if (not test_run) and save_model and do_training:
        store_folder_path = SEP.join(['ObjectDetection', 'checkpoints', CONFIG['load_to_ckpt_path']])
        description_log = ''
        dir_maker(store_folder_path, description_log, CONFIG, TRAIN_DICT)

    datamodule = ObjDetDatamodule(config=CONFIG)

    if do_training:
        module = ObjDetModule(config=CONFIG, train_config=TRAIN_DICT)

        if CONFIG['from_ckpt_path']:
            CKPT_PATH = SEP.join(['ObjectDetection', 'checkpoints', CONFIG['from_ckpt_path']])
            model_file_path = [file for file in os.listdir(CKPT_PATH) if file.endswith('.ckpt') and not file.startswith('last')]
            assert len(model_file_path) == 1
            CKPT_PATH = SEP.join([CKPT_PATH, model_file_path[0]])
            state_dict = torch.load(CKPT_PATH)['state_dict']
            module_state_dict = module.state_dict()

            mkeys_missing_in_loaded = [module_key for module_key in list(module_state_dict.keys()) if module_key not in list(state_dict.keys())]
            lkeys_missing_in_module = [loaded_key for loaded_key in list(state_dict.keys()) if loaded_key not in list(module_state_dict.keys())]

            load_dict = OrderedDict()
            for key, tensor in module_state_dict.items():
                # if (key in state_dict.keys()) and ('decode_head' not in key):
                if key in state_dict.keys():
                    load_dict[key] = state_dict[key]
                else:
                    if key.startswith('model_ema.module.'):
                        if key.replace('model_ema.module.', 'model.') in state_dict.keys(): 
                            load_dict[key] = state_dict[key.replace('model_ema.module.', 'model.')]
                            continue
                    # if key == 'model.model.classifier.classifier.weight':
                    #     load_dict[key] = state_dict['model.model.classifier.weight']
                    # elif key == 'model.model.classifier.classifier.bias': 
                    #     load_dict[key] = state_dict['model.model.classifier.bias']
                    # else:
                    #     load_dict[key] = tensor
                    load_dict[key] = tensor

            module.load_state_dict(load_dict)
        
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=CONFIG['early_stopping_patience']), LearningRateMonitor(logging_interval='epoch')]
        
        if (not CONFIG['run_test_epoch']) and save_model:
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

        limit_batches = 2 if CONFIG['run_test_epoch'] else None
        trainer = pl.Trainer(
            gpus = [CUDA_DEVICE], 
            devices=f'cuda:{str(CUDA_DEVICE)}', 
            max_epochs = 500, 
            # checkpoint_callback = [checkpoint_callback], 
            callbacks=callbacks,
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
    
    module = ObjDetModule(config=CONFIG, train_config=TRAIN_DICT)

    if CONFIG['from_ckpt_path']:
        CKPT_PATH = SEP.join(['ObjectDetection', 'checkpoints', CONFIG['from_ckpt_path']])
        model_file_path = [file for file in os.listdir(CKPT_PATH) if file.endswith('.ckpt') and not file.startswith('last')]
        assert len(model_file_path) == 1
        CKPT_PATH = SEP.join([CKPT_PATH, model_file_path[0]])
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

        module.model.metrics_mode = True

        module.to(f'cuda:{CUDA_DEVICE}')
        module.eval()

        datamodule.setup(stage='test')
        trainloader = datamodule.train_dataloader()
        testloader = datamodule.test_dataloader()

        true_boxes, true_labels, pred_boxes, pred_labels, confidences = [], [], [], [], []

        for idx, batch in tqdm(enumerate(trainloader), total=len(trainloader.dataset.img_paths)//BATCH_SIZE):
            img, labels_b, bboxes_b, numAgentsIds_b = batch
            
            if CONFIG['arch']=='FasterRCNN':
                images = list(image.to(f'cuda:{CUDA_DEVICE}') for image in img)
                targets = []
                for i in range(len(img)):
                    d = {}
                    d['boxes'] = bboxes_b[i].to(f'cuda:{CUDA_DEVICE}')
                    d['labels'] = labels_b[i].to(f'cuda:{CUDA_DEVICE}')
                    targets.append(d)
                output = module.model(images, numAgentsIds_b.to(f'cuda:{CUDA_DEVICE}'), targets)
                pred_boxes += [o["boxes"].detach().cpu() for o in output]
                confidences += [o["scores"].detach().cpu() for o in output]
                pred_labels += [o["labels"].detach().cpu() for o in output]

            elif CONFIG['arch'] == 'Detr':
                target = [{"class_labels": labels, "boxes": bboxes} for labels, bboxes in zip(labels_b, bboxes_b)]
                prediction = module.model(img, labels=target)
            
                val_loss = prediction.loss
            
            true_labels += labels_b
            true_boxes += bboxes_b

            metric = metrics(true_boxes, true_labels, pred_boxes, pred_labels, confidences)

    hie = 3