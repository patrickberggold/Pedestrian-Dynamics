import os
from collections import OrderedDict
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ObjDetDatamodule import ObjDetDatamodule
from ObjDetModule import ObjDetModule, metrics_sklearn
from tqdm import tqdm
from helper import SEP, dir_maker, update_config
import cv2
import matplotlib.pyplot as plt
from helper import xywhn2xyxy

CUDA_DEVICE = 0
BATCH_SIZE = 4
ARCH = 'Detr_custom' # Detr, Detr_custom, FasterRCNN, FasterRCNN_custom, EfficientDet, YoloV5

test_run = True # limit_batches -> 2 
save_model = False # create folder and save the model

do_training = True

CONFIG = {
    'arch': ARCH, # DeepLab, BeIT, SegFormer
    'version': '4', # '0', '4', 's', 'm'
    'cuda_device': CUDA_DEVICE,
    'batch_size': BATCH_SIZE,
    'from_ckpt_path': 'Detr_custom_numQenc=100_after_encoder_custom', # 'FasterRCNN_custom_1024p_RealData_skipConn_pretr', # 'Detr_custom_1024p_RealData_numQenc=100', 'FasterRCNN_custom_1024p_RealData_skipConn_pretr', # 'FasterRCNN_1024p_ArtifData', Detr_1024p_ArtifData
    'load_to_ckpt_path': 'Detr_custom_numQenc=100_vanilla_custom', #'Detr_custom_1024p_RealData_numQenc=100', # 'FasterRCNN_custom_1024p_RealData_skipConn_pretr', #'FasterRCNN_custom_1024p_ArtifData_skipConn', Detr_custom_1024p_ArtifData_numQdec=100, # 'FasterRCNN_1024p_ArtifData_batchAugm_1024longest'
    'early_stopping_patience': 55,
    'run_test_epoch': test_run,
    'limit_dataset': None, 
    'save_model': save_model,
    'img_max_size': (1024, 1024), # (1024, 1024), # (640, 1536), # (1344, 3072), # next highest 64-divisible
    'num_classes': 1,
    'top_k': 100,
    'augment_batch_level': False,
    'merge_mech': 'linear+skip', # cross_attn, cross_attn_large, linear+skip, linear+skip_large --> in the long-term, linear+skip seems to perform quite better
    'num_extra_queries': ['after_encoder', 100], # vanilla / vanilla_imgAugm / before_encoder / after_encoder, 50 --> enc seems to perform better, but pre-training withou custom is necessary
    'facebook_pretrained': True
}
update_config(CONFIG)

TRAIN_DICT = {
    # Learning rate and schedulers
    'learning_rate': 0.0001,
    'lr_scheduler': 'ReduceLROnPlateau', 
    'lr_sch_step_size4lr_step': 8,
    'lr_sch_step_size4cosAnneal': 100,
    'lr_sch_gamma4redOnPlat_and_stepLR': 0.75,
    'lr_sch_gamma4expLR': 0.25,
    'lr_sch_patience4redOnPlat': 10,
    # optimizer
    'opt': 'Adam',
    'weight_decay': None, # 1e-6,
    'gradient_clip': 100.,
    # (un)freezing layers
    'unfreeze_backbone_at_epoch': None, # not implemented yet
    'model_ema_decay': None, # 0.999966, # None
    'customized_optim': True
}

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
            assert len(mkeys_missing_in_loaded) < 10 or len(lkeys_missing_in_module) < 10, 'Checkpoint loading went probably wrong...'

            load_dict = OrderedDict()
            for key, tensor in module_state_dict.items():
                # if (key in state_dict.keys()) and ('decode_head' not in key):
                if key in state_dict.keys() and tensor.size()==state_dict[key].size():
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
            gradient_clip_val=TRAIN_DICT['gradient_clip'],
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
        state_dict = torch.load(CKPT_PATH)['state_dict']
        module_state_dict = module.state_dict()

        mkeys_missing_in_loaded = [module_key for module_key in list(module_state_dict.keys()) if module_key not in list(state_dict.keys())]
        lkeys_missing_in_module = [loaded_key for loaded_key in list(state_dict.keys()) if loaded_key not in list(module_state_dict.keys())]
        assert len(mkeys_missing_in_loaded) < 10 or len(lkeys_missing_in_module) < 10, 'Checkpoint loading went probably wrong...'

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
    loaders = [datamodule.val_dataloader(), datamodule.train_dataloader()]
    loader_names = ['val', 'train']
    # loaders = [datamodule.test_dataloader(), datamodule.val_dataloader(), datamodule.train_dataloader()]
    # loader_names = ['test', 'val', 'train']
    # loaders = [datamodule.test_dataloader()]
    # loader_names = ['test']

    show_pred_images = False
    save_pred_images = False
    save_curve = True

    for stage, loader in zip(loader_names, loaders):
        true_boxes, true_labels, pred_boxes, pred_labels, confidences = [], [], [], [], []
        for id_batch, batch in tqdm(enumerate(loader), total=len(loader.dataset.img_paths)//BATCH_SIZE):

            # if id_batch < 11:
            #     continue
            img, labels_b, bboxes_b, numAgentsIds_b = batch
            img, labels_b, bboxes_b, numAgentsIds_b = img.to(f'cuda:{CUDA_DEVICE}'), [l.to(f'cuda:{CUDA_DEVICE}') for l in labels_b], [b.to(f'cuda:{CUDA_DEVICE}') for b in bboxes_b], numAgentsIds_b.to(f'cuda:{CUDA_DEVICE}')
            
            if CONFIG['arch'] in ['FasterRCNN', 'FasterRCNN_custom']:
                images = list(image for image in img)
                targets = []
                for i in range(len(img)):
                    d = {}
                    d['boxes'] = bboxes_b[i]
                    d['labels'] = labels_b[i]
                    targets.append(d)
                losses, prediction = module.model(images, targets) if ARCH == 'FasterRCNN' else module.model(images, numAgentsIds_b, targets)
                val_loss = sum([losses[k] for k in losses])

                # pred_boxes, confidences, pred_labels = [],[],[]
                for pred in prediction:
                    selected_ids = torch.argwhere(pred["labels"] == 1).squeeze()
                    b = pred["boxes"][selected_ids]
                    s = pred["scores"][selected_ids]
                    l = pred["labels"][selected_ids]
                    if s.ndim == 0:
                        b, s, l = b.unsqueeze(0), s.unsqueeze(0), l.unsqueeze(0)
                    pred_boxes.append(b.detach().cpu())
                    confidences.append(s.detach().cpu())
                    pred_labels.append(l.detach().cpu())

            elif CONFIG['arch'] in ['Detr', 'Detr_custom']:

                target = [{"class_labels": l, "boxes": b} for l, b in zip(labels_b, bboxes_b)]
                prediction = module.model(img, labels=target) if ARCH == 'Detr' else module.model(img, numAgentsIds_b, labels=target)
                val_loss = prediction.loss

                for i in range(prediction.logits.size(0)):
                    scores = torch.softmax(prediction.logits[i].detach(), dim=-1)
                    argmaxes = torch.argmax(scores, dim=-1)
                    box_detection_indices = torch.argwhere((argmaxes == 0) & (scores[:, 0] > 0.7)).squeeze() # select only boxes with class 0 (class 1 == 'no-object')
                    selected_scores = scores[box_detection_indices, 0]
                    assert torch.all(selected_scores > scores[box_detection_indices, 1])
                    box_proposals_per_batch = prediction.pred_boxes[i].detach()
                    selected_boxes = box_proposals_per_batch[box_detection_indices]
                    selected_labels = argmaxes[box_detection_indices].detach()

                    if selected_scores.ndim==0:
                        selected_boxes, selected_labels, selected_scores = selected_boxes.unsqueeze(0), selected_labels.unsqueeze(0), selected_scores.unsqueeze(0)
                    # print(f'selection length: {selected_scores.size(0)}')

                    pred_boxes.append(xywhn2xyxy(selected_boxes, img.size(3), img.size(2)).detach().cpu()) # 2048, 789
                    confidences.append(selected_scores.cpu())
                    pred_labels.append(selected_labels.long().cpu())
                
                bboxes_b = [xywhn2xyxy(box, img.size(3), img.size(2)) for box in bboxes_b]
            
            true_labels += [l.detach().cpu() for l in labels_b]
            true_boxes += [b.detach().cpu() for b in bboxes_b]
            # continue

            if save_pred_images or show_pred_images:
                # visualize the boxes
                def insert_bbox(img_, bbox, class_name, score=None, color=(255, 0, 0), thickness=2):

                    x_min, y_min, x_max, y_max = bbox
                    x_min, y_min, x_max, y_max = round(x_min), round(y_min), round(x_max), round(y_max)
                
                    img_ = np.zeros((img_.shape[0], img_.shape[1], 3), dtype=np.uint8) + img_
                    cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color, thickness=thickness)
                    
                    boxText = f'{class_name}'
                    if score is not None: boxText += f': {round(score*100)}%'
                    ((text_width, text_height), _) = cv2.getTextSize(boxText, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
                    cv2.rectangle(img_, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
                    cv2.putText(
                        img_,
                        text=boxText,
                        org=(x_min, y_min - int(0.3 * text_height)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.35, 
                        color=(255, 255, 255), 
                        lineType=cv2.LINE_AA,
                    )
                    return img_

                import numpy as np
                import random
                plt.rcParams['figure.figsize'] = [20, 14]
                category_id_to_name_gt = {0: 'Background', 1: 'critArea'} if ARCH in ['FasterRCNN', 'FasterRCNN_custom'] else {0: 'critArea', 1: 'Background'}
                category_id_to_name_pr = {0: 'Background', 1: 'Prediction'} if ARCH in ['FasterRCNN', 'FasterRCNN_custom'] else {0: 'Prediction', 1: 'Background'}
                color_gt = (0, 170, 0)
                color_pr = (170, 0, 0)
                
                for b in range(img.size(0)):
                    img_np = img[b].cpu().numpy().copy().transpose(1,2,0)
                    img_np = (img_np * 255).astype(np.uint8)
                    labels_gt = true_labels[id_batch*BATCH_SIZE + b].cpu().numpy()
                    bboxes_gt = true_boxes[id_batch*BATCH_SIZE + b].cpu().numpy()

                    labels_pr = pred_labels[id_batch*BATCH_SIZE + b].cpu().numpy()
                    bboxes_pr = pred_boxes[id_batch*BATCH_SIZE + b].detach().cpu().numpy()
                    scores_pr = confidences[id_batch*BATCH_SIZE + b].detach().cpu().numpy()

                    # draw gt boxes individually
                    for lgt, bgt in zip(labels_gt, bboxes_gt):
                        class_name_gt = category_id_to_name_gt[lgt]
                        img_np = insert_bbox(img_np, bgt, class_name_gt, score=None, color=color_gt)

                    # if show_pred_images:
                    #     plt.imshow(img_np)
                    #     plt.close('all')
                    # if save_pred_images:
                    #     plt.imsave(f"ObjectDetection\\results\\{ARCH}\\OD_result_stage={stage}_batch={id_batch}_i={b}_gt.png", img_np)
                    
                    # draw pr boxes individually
                    for lpr, bpr, score in zip(labels_pr, bboxes_pr, scores_pr):
                        class_name_pr = category_id_to_name_pr[lpr]

                        img_np = insert_bbox(img_np, bpr, class_name_pr, score=score, color=color_pr)

                    if show_pred_images:
                        plt.imshow(img_np)
                        plt.close('all')
                    if save_pred_images:
                        plt.imsave(f"ObjectDetection\\results\\{ARCH}\\OD_result_stage={stage}_batch={id_batch}_i={b}_pred.png", img_np)
           
        if stage == 'test':
            metric_sbahn, f1_score_sbahn, statistics_sbahn = metrics_sklearn(true_boxes[:3], true_labels[:3], pred_boxes[:3], pred_labels[:3], confidences[:3], return_curve=False)
            metric_u9, f1_score_u9, statistics_u9 = metrics_sklearn(true_boxes[3:], true_labels[3:], pred_boxes[3:], pred_labels[3:], confidences[3:], return_curve=False)

            tps_sb, tps_ious_sb, fps_sb, fns_sb =  statistics_sbahn['tps'], statistics_sbahn['tp_ious'], statistics_sbahn['fps'], statistics_sbahn['fns']
            recall_sb, precision_sb = tps_sb/(tps_sb+fns_sb+1e-8), tps_sb/(tps_sb+fps_sb+1e-8)
            iou_mean_sb = sum(tps_ious_sb) / len(tps_ious_sb) if len(tps_ious_sb) > 0 else 0
            print(f'Arch: {ARCH}, stage: {stage}\nmAP SBAHN: {metric_sbahn}, recall: {recall_sb}, precision: {precision_sb}, mean IOU: {iou_mean_sb}\n\n')

            tps_u9, tps_ious_u9, fps_u9, fns_u9 =  statistics_u9['tps'], statistics_u9['tp_ious'], statistics_u9['fps'], statistics_u9['fns']
            recall_u9, precision_u9 = tps_u9/(tps_u9+fns_u9+1e-8), tps_u9/(tps_u9+fps_u9+1e-8)
            iou_mean_u9 = sum(tps_ious_u9) / len(tps_ious_u9) if len(tps_ious_u9) > 0 else 0
            print(f'Arch: {ARCH}, stage: {stage}\nmAP U9: {metric_u9}, recall: {recall_u9}, precision: {precision_u9}, mean IOU: {iou_mean_u9}\n\n')
        
        else:
            if save_curve:
                metric, f1_score, statistics, (fig, ax) = metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences, return_curve=save_curve)
                ax.set_ylim(0.0, 1.05)
                # fig.savefig(f"ObjectDetection\\results\\{ARCH}\\OD_curve_stage={stage}_PR_curve_075.png", dpi=900.) # .savefig("testy.png") # 
        
            else:
                metric, f1_score, statistics = metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences, return_curve=False)
            
            tps, tps_ious, fps, fns =  statistics['tps'], statistics['tp_ious'], statistics['fps'], statistics['fns']
            recall = tps/(tps+fns)
            precision = tps/(tps+fps)
            iou_mean = sum(tps_ious) / len(tps_ious) if len(tps_ious) > 0 else 0
            print(f'Arch: {ARCH}, stage: {stage}')
            print(f'mAP: {metric}, recall: {recall}, precision: {precision}, mean IOU: {iou_mean}\n\n')
            
        

    quit()
