""" Based on https://github.com/FGiuliari/Trajectory-Transformer """

import copy
import torch.nn as nn
import numpy as np
from .layers import MultiHeadAttention, PointerwiseFeedforward
from .embedding import PositionalEncoding, LinearEmbedding
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
import torch.nn.functional as F
import torch
from .utils import subsequent_mask

class Transformer(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, traj_quantity='pos', N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0,0], std=[0,0]):
        super(Transformer, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.mean=np.array(mean)
        self.std=np.array(std)
        self.pred_length = 12
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

        self.src_embed = nn.Sequential(LinearEmbedding(enc_inp_size,d_model), c(position))
        self.tgt_embed = nn.Sequential(LinearEmbedding(dec_inp_size,d_model), c(position))

        self.generator = nn.Linear(d_model, dec_out_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.traj_quantity = traj_quantity

    def forward(self, batch, train):

        inputs = {k: v.squeeze(0).float() for k, v in batch.items() if k not in ['scene_data', 'type']}
        
        batch_coords = inputs["abs_pixel_coord"]
        if self.traj_quantity == 'vel':
            starting_pos, batch_coords = batch_coords[0], batch_coords[1:]

        src = batch_coords[:-self.pred_length].transpose(0, 1) # num_agents, obs steps, coord
        src_mask = torch.ones((src.size(0), 1, src.size(1))).to(src.device)

        # Teacher forcing
        if train:
            tgt = batch_coords[-self.pred_length:].transpose(0, 1) # num_agents, pred steps, coord
            # dec_inp = torch.cat((start_of_seq, target), 1) # OUT: (100, 12, 3)
            tgt_mask = subsequent_mask(tgt.size(1)).repeat(tgt.size(0), 1, 1).to(tgt.device)

            # actual forward pass
            src_emb = self.src_embed(src)
            tgt_emb = self.tgt_embed(tgt)

            x = self.encoder(src_emb, src_mask)
            x = self.decoder(tgt_emb, x, src_mask, tgt_mask)

            x = self.generator(x)

        else:
            if self.traj_quantity == 'vel':
                # start with current velocity
                tgt = torch.zeros((src.size(0), 1, src.size(2))).to(src.device)
            else:
                # start with current position
                tgt = src[:, -1, :].unsqueeze(1)
            for i in range(self.pred_length):
                tgt_mask = subsequent_mask(tgt.size(1)).repeat(tgt.size(0), 1, 1).to(tgt.device)

                # actual forward pass
                src_emb = self.src_embed(src)
                tgt_emb = self.tgt_embed(tgt)

                x = self.encoder(src_emb, src_mask)
                x = self.decoder(tgt_emb, x, src_mask, tgt_mask)

                x = self.generator(x)
                # prediction is fed as input, only last state appended: https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452
                tgt = torch.cat((tgt, x[:, -1:, :]), 1)
            x = tgt[:, 1:, :]  
        return x

    def compute_loss(self, prediction, batch, stage):

        batch_coords = batch["abs_pixel_coord"].squeeze()
        if self.traj_quantity == 'vel':
            starting_pos, batch_coords = batch_coords[0], batch_coords[1:]
        ground_truth = batch_coords[-self.pred_length:].transpose(0, 1)
        # pw_distance = F.pairwise_distance(prediction, ground_truth) # + torch.mean(torch.abs(pred[:,:,2]))
        pw_distance = F.mse_loss(prediction, ground_truth) # + torch.mean(torch.abs(pred[:,:,2]))

        return pw_distance, pw_distance, None
        # F.pairwise_distance(prediction.contiguous().view(-1, 2),
        #                     ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(pred[:,:,2]))

        # # mean and std maybe
        # pred = tgt[:,1:,0:2].detach().cpu().numpy()
        # pred_traj = pred.cumsum(1) # +batch['src'][:,-1:,0:2].cpu().numpy()

# TODO here 
# - Goal SAR on bigger dataset (trains)
# - slides
# - Beit evac prediction + smaller dataset
# - new approach