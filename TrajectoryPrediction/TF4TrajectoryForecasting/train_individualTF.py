""" Based on https://github.com/FGiuliari/Trajectory-Transformer """

import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle
import individual_TF
from tqdm import tqdm
# from ..trajectory_visualization import read_trajs_from_seqArray

from torch.utils.tensorboard import SummaryWriter

# python train_individualTF.py --dataset_name eth --name eth --max_epoch 240 --batch_size 100 --name eth_train --factor 1
CUDA_DEVICE = 0
do_floorplans = True
train = False
inference = True

parser=argparse.ArgumentParser(description='Train the individual Transformer model')
parser.add_argument('--name', type=str, default="eth_train")
parser.add_argument('--dataset_name',type=str,default='eth')
parser.add_argument('--max_epoch',type=int, default=240)
parser.add_argument('--batch_size',type=int,default=100)
parser.add_argument('--dataset_folder',type=str,default='datasets')
parser.add_argument('--obs',type=int,default=8)
parser.add_argument('--preds',type=int,default=12)
# Network parameters
parser.add_argument('--emb_size',type=int,default=512)
parser.add_argument('--heads',type=int, default=8)
parser.add_argument('--layers',type=int,default=6)
parser.add_argument('--dropout',type=float,default=0.1)

parser.add_argument('--cpu',action='store_true')
parser.add_argument('--val_size',type=int, default=0)
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--validation_epoch_start', type=int, default=30)
parser.add_argument('--resume_train',action='store_true')
parser.add_argument('--delim',type=str,default='\t')
parser.add_argument('--factor', type=float, default=1.)
parser.add_argument('--save_step', type=int, default=0)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--model_pth', type=str)


file_dir = os.path.dirname(os.path.realpath(__file__))

args=parser.parse_args()
model_name=args.name
args.dataset_folder = os.path.join(file_dir, args.dataset_folder)

if do_floorplans:
    args.dataset_folder = 'C:\\Users\\Remotey\\Documents\\Datasets\CSV_SIMULATION_DATA_numAgents_50'
    args.dataset_name = '0__floorplan_zPos_0.0_roomWidth_0.24_numRleft_2.0_numRright_2.0'
    model_name = args.name = 'floorplans_0'
    args.delim = ','

try:
    os.mkdir(os.path.join(file_dir, 'models'))
except:
    pass
try:
    os.mkdir(os.path.join(file_dir, 'output'))
except:
    pass
try:
    os.mkdir(os.path.join(file_dir, 'output', 'Individual'))
except:
    pass
try:
    os.mkdir(os.path.join(file_dir, 'models', 'Individual'))
except:
    pass

try:
    os.mkdir(os.path.join(file_dir, 'output', 'Individual', f'{args.name}'))
except:
    pass

try:
    os.mkdir(os.path.join(file_dir, 'models', 'Individual', f'{args.name}'))
except:
    pass

log=SummaryWriter('logs/Ind_%s'%model_name)

log.add_scalar('eval/mad', 0, 0)
log.add_scalar('eval/fad', 0, 0)
device=torch.device(f"cuda:{CUDA_DEVICE}")

if args.cpu or not torch.cuda.is_available():
    device=torch.device("cpu")

args.verbose=True

model=individual_TF.IndividualTF(2, 3, 3, N=args.layers,
                d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)
if args.resume_train:
    model.load_state_dict(torch.load(f'models/Individual/{args.name}/{args.model_pth}'))

## creation of the dataloaders for train and validation
if args.val_size==0:
    train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose, do_floorplans=do_floorplans)
    # val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs,
                                                                # args.preds, delim=args.delim, train=False,
                                                                # verbose=args.verbose, do_floorplans=do_floorplans)
else:
    train_dataset, val_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs,
                                                            args.preds, delim=args.delim, train=True,
                                                            verbose=args.verbose, do_floorplans=do_floorplans)

test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose, do_floorplans=do_floorplans)



tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
# val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

#optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
#sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
epoch=0


#mean=train_dataset[:]['src'][:,1:,2:4].mean((0,1))
mean=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).mean((0,1))
#std=train_dataset[:]['src'][:,1:,2:4].std((0,1))
std=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).std((0,1))
means=[]
stds=[]
# Extract different means and stds for different scenes
for i in np.unique(train_dataset[:]['dataset']):
    ind=train_dataset[:]['dataset']==i
    means.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1))) # concat obs (except present) and future trajectories, mean over all sequences and timesteps-per-sequence in one scene/layout in the train dataset
    stds.append(
        torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
# Mean over all scenes in the train set
mean=torch.stack(means).mean(0)
std=torch.stack(stds).mean(0)

scipy.io.savemat(os.path.join(file_dir, 'models', 'Individual', f'{args.name}', 'norm.mat'), {'mean':mean.cpu().numpy(),'std':std.cpu().numpy()})

if inference:

    model_path = os.path.join('TrajectoryPrediction\\Trajectory-Transformer\\models\\Individual', args.name)
    pth_filename = sorted([filename for filename in os.listdir(model_path) if filename.endswith('.pth')], key=lambda x: int(x.replace('.pth', '')))[-1]
    model.load_state_dict(torch.load(os.path.join(model_path, pth_filename)))

    model.eval()
    gt = []
    pr = []
    inp_ = []
    peds = []
    frames = []
    dt = []
    pbar = tqdm(total=len(test_dl))
    for id_b,batch in enumerate(test_dl):
        pbar.update(1)
        inp_.append(batch['src'])
        gt.append(batch['trg'][:,:,0:2])
        frames.append(batch['frames'])
        peds.append(batch['peds'])
        dt.append(batch['dataset'])

        inp_np = batch['src'][:,:,0:2].numpy()
        gt_np = batch['trg'][:,:,0:2].numpy()
        frames_np = batch['frames'].numpy()                

        inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
        start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
            device)
        dec_inp=start_of_seq

        for i in range(args.preds):
            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
            out = model(inp, dec_inp, src_att, trg_att)
            dec_inp=torch.cat((dec_inp,out[:,-1:,:]),1)

        preds_tr_b=(dec_inp[:,1:,0:2]*std.to(device)+mean.to(device)).detach().cpu().numpy().cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
        pr.append(preds_tr_b)
        # print("test epoch %03i/%03i  batch %04i / %04i" % (
        # epoch, args.max_epoch, id_b, len(test_dl)))

        # Create GT and pred. pedestrian dicts
        x_min = float('inf')
        y_min = float('inf')
        x_max = 0.
        y_max = 0.
        ped_Coords = {}
        for idx, pedID in enumerate(batch['peds'].numpy()):
            key = 'ped_'+str(int(pedID))
            if key not in ped_Coords:                
                ## get mins/maxs ##
                if preds_tr_b[idx,:,0].min() < x_min:
                        x_min = preds_tr_b[idx,:,0].min()
                if inp_np[idx,:,0].min() < x_min:
                        x_min = inp_np[idx,:,0].min()
                if preds_tr_b[idx,:,1].min() < y_min:
                        y_min = preds_tr_b[idx,:,1].min()
                if inp_np[idx,:,1].min() < y_min:
                        y_min = inp_np[idx,:,1].min()
                
                if preds_tr_b[idx,:,0].max() > x_max:
                        x_max = preds_tr_b[idx,:,0].max()
                if inp_np[idx,:,0].max() > x_max:
                        x_max = inp_np[idx,:,0].max()
                if preds_tr_b[idx,:,1].max() > y_max:
                        y_max = preds_tr_b[idx,:,1].max()
                if inp_np[idx,:,1].max() > y_max:
                        y_max = inp_np[idx,:,1].max()
                ## get mins/maxs ##
                ped_Coords.update({key: [np.concatenate((inp_np[idx], gt_np[idx]), 0), np.concatenate((inp_np[idx], preds_tr_b[idx]), 0), frames_np[idx]]})
            else:
                curr_coord = np.expand_dims(gt_np[idx, -1], 0)
                ped_Coords[key][0] = np.concatenate((ped_Coords[key][0], curr_coord), 0)

                # TODO for different sequences different predictions, even if same pedId in subsequent timesteps....
                pred_coord = np.expand_dims(preds_tr_b[idx, -1], 0)
                ped_Coords[key][1] = np.concatenate((ped_Coords[key][1], pred_coord), 0)

                curr_frame = np.expand_dims(frames_np[idx, -1], 0)
                ped_Coords[key][2] = np.concatenate((ped_Coords[key][2], curr_frame), 0)

        def linear_interpolation(curr_point_or, lim_min_or, lim_max_or, lim_min_proj, lim_max_proj):
            return lim_min_proj + (curr_point_or - lim_min_or) * (lim_max_proj - lim_min_proj) / (lim_max_or - lim_min_or)

        for _pedID in ped_Coords:
            for list_id, coord_list in enumerate(ped_Coords[_pedID][:2]):
                for coord_id, coord in enumerate(coord_list):
                    x,y = coord
                    x_proj = round(linear_interpolation(x, x_min, x_max, 0, 800))
                    y_proj = round(linear_interpolation(y, y_min, y_max, 0, 800))
                    ped_Coords[_pedID][list_id][coord_id] = np.array([x_proj, y_proj])
        
        import cv2
        import matplotlib.pyplot as plt
        img = np.ones((800,800,3))

        for pedId_c in ped_Coords:
            # color GT
            for gt_id in range(1, ped_Coords[pedId_c][0].shape[0]):
                start_point = (ped_Coords[pedId_c][0][gt_id-1])
                end_point = (ped_Coords[pedId_c][0][gt_id])
                color = (0,0,1)
                if gt_id < 8:
                    color = (0,1,0)
                cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), color=color, thickness=2)
            for pred_id in range(8, ped_Coords[pedId_c][1].shape[0]):
                start_point = (ped_Coords[pedId_c][1][pred_id-1])
                end_point = (ped_Coords[pedId_c][1][pred_id])
                cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), color=(1,0,0), thickness=1)
        plt.imshow(img)
        hi = 1
        

    peds = np.concatenate(peds, 0)
    frames = np.concatenate(frames, 0)
    dt = np.concatenate(dt, 0)
    gt = np.concatenate(gt, 0)
    dt_names = test_dataset.data['dataset_name']
    pr = np.concatenate(pr, 0)
    mad, fad, errs = baselineUtils.distance_metrics(gt, pr) 
    # last_model: 1.6672620397873765, 3.1573270799436917 ::::: sixth model: 1.7412602195747557, 3.234687785197249 ::::: first model: 1.9025136932044424, 3.6679819893402357

if train:
    best_loss = float('inf')
    epochs_since_improvement = 0

    while epoch<args.max_epoch:
        epoch_loss=0
        model.train()
        pbar = tqdm(total=len(tr_dl))
        for id_b,batch in enumerate(tr_dl):
            pbar.update(1)
            optim.optimizer.zero_grad()
            inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device) # OUT: (100, 7, 2) -> 100 peds from different scenes, 7 (past) observed 2D normalized velocities (except first one which is the present one)
            target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device) # OUT: (100, 11, 2) -> 100 peds from different scenes, 11 future 2D normalized velocites (except last one)
            target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device) # OUT: zeros (100, 11, 1)
            target=torch.cat((target,target_c),-1)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device) # OUT: 100x[[0, 0, 1]]

            dec_inp = torch.cat((start_of_seq, target), 1) # OUT: (100, 12, 3)

            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)


            pred=model(inp, dec_inp, src_att, trg_att)
            # Pairwise distance from normalized (pred-GT) velocities
            loss = F.pairwise_distance(pred[:, :,0:2].contiguous().view(-1, 2),
                                        ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(pred[:,:,2]))
            loss.backward()
            optim.step()
            # print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(tr_dl), loss.item()))
            epoch_loss += loss.item()
        pbar.close()
        print(f'Loss in epoch {epoch}/{args.max_epoch}: {epoch_loss/len(tr_dl):.8f}')
        #sched.step()
        log.add_scalar('Loss/train', epoch_loss / len(tr_dl), epoch)
        with torch.no_grad():
            model.eval()

            val_loss=0
            step=0
            model.eval()
            gt = []
            pr = []
            inp_ = []
            peds = []
            frames = []
            dt = []

            for id_b, batch in enumerate(val_dl):
                inp_.append(batch['src'])
                gt.append(batch['trg'][:, :, 0:2])
                frames.append(batch['frames'])
                peds.append(batch['peds'])
                dt.append(batch['dataset'])

                inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
                src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                    device)
                dec_inp = start_of_seq

                for i in range(args.preds):
                    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                    out = model(inp, dec_inp, src_att, trg_att)
                    dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

                preds_tr_b = (dec_inp[:, 1:, 0:2] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + batch[
                                                                                                                    'src'][
                                                                                                                :, -1:,
                                                                                                                0:2].cpu().numpy()
                pr.append(preds_tr_b)
                # print("val epoch %03i/%03i  batch %04i / %04i" % (
                #     epoch, args.max_epoch, id_b, len(val_dl)))


            peds = np.concatenate(peds, 0)
            frames = np.concatenate(frames, 0)
            dt = np.concatenate(dt, 0)
            gt = np.concatenate(gt, 0)
            dt_names = test_dataset.data['dataset_name']
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
            log.add_scalar('validation/MAD', mad, epoch)
            log.add_scalar('validation/FAD', fad, epoch)




            if args.evaluate:

                model.eval()
                gt = []
                pr = []
                inp_ = []
                peds = []
                frames = []
                dt = []
                
                for id_b,batch in enumerate(test_dl):
                    inp_.append(batch['src'])
                    gt.append(batch['trg'][:,:,0:2])
                    frames.append(batch['frames'])
                    peds.append(batch['peds'])
                    dt.append(batch['dataset'])

                    inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
                    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                        device)
                    dec_inp=start_of_seq

                    for i in range(args.preds):
                        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                        out = model(inp, dec_inp, src_att, trg_att)
                        dec_inp=torch.cat((dec_inp,out[:,-1:,:]),1)


                    preds_tr_b=(dec_inp[:,1:,0:2]*std.to(device)+mean.to(device)).cpu().numpy().cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
                    pr.append(preds_tr_b)
                    # print("test epoch %03i/%03i  batch %04i / %04i" % (
                    # epoch, args.max_epoch, id_b, len(test_dl)))

                peds = np.concatenate(peds, 0)
                frames = np.concatenate(frames, 0)
                dt = np.concatenate(dt, 0)
                gt = np.concatenate(gt, 0)
                dt_names = test_dataset.data['dataset_name']
                pr = np.concatenate(pr, 0)
                mad, fad, errs = baselineUtils.distance_metrics(gt, pr)


                log.add_scalar('eval/DET_mad', mad, epoch)
                log.add_scalar('eval/DET_fad', fad, epoch)


                # log.add_scalar('eval/DET_mad', mad, epoch)
                # log.add_scalar('eval/DET_fad', fad, epoch)

                scipy.io.savemat(os.path.join(file_dir, 'output', 'Individual', f'{args.name}', f'det_{epoch}.mat'),
                                    {'input': inp, 'gt': gt, 'pr': pr, 'peds': peds, 'frames': frames, 'dt': dt,
                                    'dt_names': dt_names})

        if args.save_step != 0:
            if epoch%args.save_step==0:

                torch.save(model.state_dict(), os.path.join(file_dir, 'models', 'Individual', f'{args.name}', f'{epoch:05d}.pth'))
        else:
            if epoch_loss < best_loss:
                print(f'Overall loss improved from {best_loss/len(tr_dl):.6f} to {epoch_loss/len(tr_dl):.6f}, saving model...')
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(file_dir, 'models', 'Individual', f'{args.name}', f'{epoch:05d}.pth'))

                epochs_since_improvement = 0
            else:
                if epochs_since_improvement >= 8:
                    print('No loss improvement for 8 epochs, quitting...')
                    epoch = args.max_epoch

        epochs_since_improvement+=1
        epoch+=1