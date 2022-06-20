import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *

def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def get_noise(shape):
    return torch.randn(*shape).cuda(CUDA_DEVICE)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM

        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1).cuda(CUDA_DEVICE)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim).cuda(CUDA_DEVICE)

    def init_hidden(self, batch):
        h = torch.zeros(1, batch, self.h_dim).cuda(CUDA_DEVICE)
        c = torch.zeros(1, batch, self.h_dim).cuda(CUDA_DEVICE)
        return (h, c)

    def forward(self, obs_traj):

        padded = len(obs_traj.shape) == 4
        npeds = obs_traj.size(1)
        total = npeds * (MAX_PEDS if padded else 1)

        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, total, self.embedding_dim)
        state = self.init_hidden(total)
        output, state = self.encoder(obs_traj_embedding, state)
        final_h = state[0]
        if padded:
            final_h = final_h.view(npeds, MAX_PEDS, self.h_dim)
        else:
            final_h = final_h.view(npeds, self.h_dim)
        return final_h

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.seq_len = PRED_LEN
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.hidden2pos = nn.Linear(self.h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple):
        npeds = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, npeds, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos
            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, npeds, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(npeds, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel

class PhysicalAttention(nn.Module):
    def __init__(self):
        super(PhysicalAttention, self).__init__()

        self.L = ATTN_L
        self.D = ATTN_D
        self.D_down = ATTN_D_DOWN
        self.bottleneck_dim = BOTTLENECK_DIM
        self.embedding_dim = EMBEDDING_DIM

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.pre_att_proj = nn.Linear(self.D, self.D_down)

        mlp_pre_dim = self.embedding_dim + self.D_down
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims).to(CUDA_DEVICE)

        self.attn = nn.Linear(self.L*self.bottleneck_dim, self.L)

    def forward(self, vgg, end_pos):
        # vgg IN: torch.tensor(n_ped_per_seq*batch_size x 15 x 15 x 512)
        # end_pos IN: torch.tensor(batch_size*n_ped_per_seq, 64, 2)
        npeds = end_pos.size(0)
        end_pos = end_pos[:, 0, :] # end positions of current (considered) peds in the sequence, eject their respective neighbors
        curr_rel_embedding = self.spatial_embedding(end_pos)
        curr_rel_embedding = curr_rel_embedding.view(-1, 1, self.embedding_dim).repeat(1, self.L, 1)

        vgg_new = vgg.view(-1, self.D)
        features_proj = self.pre_att_proj(vgg_new) # TODO maybe it is problematic that I am mixing layouts here... go through each layout separately?
        features_proj = features_proj.view(-1, self.L, self.D_down)

        mlp_h_input = torch.cat([features_proj, curr_rel_embedding], dim=2)
        attn_h = self.mlp_pre_attn(mlp_h_input.view(-1, self.embedding_dim+self.D_down))
        attn_h = attn_h.view(npeds, self.L, self.bottleneck_dim)

        attn_w = F.softmax(self.attn(attn_h.view(npeds, -1)), dim=1)
        attn_w = attn_w.view(npeds, self.L, 1)

        attn_h = torch.sum(attn_h * attn_w, dim=1)
        return attn_h

class SocialAttention(nn.Module):
    def __init__(self):
        super(SocialAttention, self).__init__()

        self.h_dim = H_DIM
        self.bottleneck_dim = BOTTLENECK_DIM
        self.embedding_dim = EMBEDDING_DIM

        mlp_pre_dim = self.embedding_dim + self.h_dim
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)
        self.attn = nn.Linear(MAX_PEDS*self.bottleneck_dim, MAX_PEDS)

    def repeat(self, tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, end_pos):

        npeds = h_states.size(0)
        curr_rel_pos = end_pos[:, :, :] - end_pos[:, 0:1, :]
        curr_rel_embedding = self.spatial_embedding(curr_rel_pos.view(-1, 2))
        curr_rel_embedding = curr_rel_embedding.view(npeds, MAX_PEDS, self.embedding_dim)

        mlp_h_input = torch.cat([h_states, curr_rel_embedding], dim=2)
        attn_h = self.mlp_pre_attn(mlp_h_input.view(-1, self.embedding_dim+self.h_dim))
        attn_h = attn_h.view(npeds, MAX_PEDS, self.bottleneck_dim)
        
        attn_w = F.softmax(self.attn(attn_h.view(npeds, -1)), dim=1)
        attn_w = attn_w.view(npeds, MAX_PEDS, 1)

        attn_h = torch.sum(attn_h * attn_w, dim=1)
        return attn_h

class TrajectoryGenerator(nn.Module):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.mlp_dim = MLP_DIM
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.bottleneck_dim = BOTTLENECK_DIM
        self.noise_dim = NOISE_DIM

        self.encoder = Encoder().to(CUDA_DEVICE)
        self.sattn = SocialAttention().to(CUDA_DEVICE)
        self.pattn = PhysicalAttention().to(CUDA_DEVICE)
        self.decoder = Decoder().to(CUDA_DEVICE)

        input_dim = self.h_dim + 2*self.bottleneck_dim
        mlp_decoder_context_dims = [input_dim, self.mlp_dim, self.h_dim - self.noise_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims).to(CUDA_DEVICE)

    def add_noise(self, _input):
        npeds = _input.size(0)
        noise_shape = (self.noise_dim,)
        z_decoder = get_noise(noise_shape)
        vec = z_decoder.view(1, -1).repeat(npeds, 1)
        return torch.cat((_input, vec), dim=1)

    def forward(self, obs_traj, obs_traj_rel, vgg_list):
        # vgg IN: torch.tensor(batch_size*n_ped_per_seq x batch_size*225 x 512)
        # obs_traj IN: torch.tensor(8, batch_size*n_ped_per_seq, 64, 2)
        npeds = obs_traj_rel.size(1) # number of considered pedestrians in current sequence

        final_encoder_h = self.encoder(obs_traj_rel)

        end_pos = obs_traj[-1, :, :, :] # first row in the 64-column is the considered pedestrian, the following are its neighbors until only zeros
        attn_s = self.sattn(final_encoder_h, end_pos)
        attn_p = self.pattn(vgg_list, end_pos) # torch.rand(12, 32).cuda(CUDA_DEVICE), 12 important, 32 not?
        mlp_decoder_context_input = torch.cat([final_encoder_h[:, 0, :], attn_s, attn_p], dim=1)

        noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        decoder_h = self.add_noise(noise_input)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(1, npeds, self.h_dim).cuda(CUDA_DEVICE)
        state_tuple = (decoder_h, decoder_c)

        last_pos = obs_traj[-1, :, 0, :]
        last_pos_rel = obs_traj_rel[-1, :, 0, :]
        pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
        return pred_traj_fake_rel

class TrajectoryDiscriminator(nn.Module):
    def __init__(self):
        super(TrajectoryDiscriminator, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = H_DIM

        self.encoder = Encoder()
        real_classifier_dims = [self.h_dim, self.mlp_dim, 1]
        self.real_classifier = make_mlp(real_classifier_dims).cuda(CUDA_DEVICE)

    def forward(self, traj, traj_rel):

        final_h = self.encoder(traj_rel)
        scores = self.real_classifier(final_h)
        return scores
