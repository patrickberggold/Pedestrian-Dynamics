""" Based on https://github.com/lucidrains/CoCa-pytorch/blob/main/coca_pytorch/coca_pytorch.py """

import torch.nn as nn

import torch
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.transforms.functional import resize, InterpolationMode
# from einops import rearrange, repeat

# from .img_backbone import ImgBackbonePretrained
# from .vit import SimpleViT, Extractor

from Modules.GoalPredictionModels.goal_prediction_model import GoalPredictionModel
# from TrajectoryPrediction.Modules.GoalPredictionModels.goal_prediction_model import GoalPredictionModel
import numpy as np
import os
import torchvision.models as models
from helper import visualize_trajectories
import random
def check_agents_in_destination(agent_coords, destinations):
    agents_x, agents_y = agent_coords[:, 0], agent_coords[:, 1]
    agents_within_destinations = torch.zeros_like(agents_x, dtype=torch.bool)
    any_agent_terminated = False

    for dest in destinations:
        # dest = dest.unsqueeze(0)
        # Check if agents are within the bounding boxes
        within_x = torch.logical_and(agents_x >= dest[0], agents_x <= dest[2])
        within_y = torch.logical_and(agents_y >= dest[1], agents_y <= dest[3])

        # Combine the conditions to find agents within the current destination
        agents_within_dest = torch.logical_and(within_x, within_y)

        # Combine the conditions to find agents within the destinations
        agents_within_destinations = torch.logical_or(agents_within_destinations, agents_within_dest)
    
    if agents_within_destinations.sum() > 0:
        # print(f'\n{agents_within_destinations.sum()} new agents within destinations.')
        non_traj_mask = torch.logical_or(agents_within_destinations, torch.isinf(agent_coords[:, 0]))
        agent_coords[non_traj_mask, :] = torch.full((2,), float('inf'), dtype=torch.float32, device=agent_coords.device)
        any_agent_terminated = True
    return agent_coords, any_agent_terminated


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.fc = nn.Linear((2) * d_model, d_model)
        self.pe = self.build_pos_enc(max_t_len)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe
   
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[0] // num_a
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        feat = [x, pos_enc.repeat(1, x.size(1), 1).to(x.device)]
        x = torch.cat(feat, dim=-1)
        x = self.fc(x)
        return self.dropout(x)


class TransformerPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model) -> None:
        self.max_len = max_len
        self.d_model = d_model
    def forward(self, x):
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe = np.zeros((self.max_len, self.d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return x + pe.to(x.device)


def pos_encoding(num_agents=1, max_len=200, d_model=512):
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).repeat(num_agents, 1, 1)
    return pe


class CoCaGoal(nn.Module):
    def __init__(
        self,
        config,
        dim,
        num_enc_layers,
        num_dec_layers,
        heads=8,
        ff_mult=4,
        init='xavier',
        pretrained_vision=True,
        coords_normed=True, separate_obs_agent_batches=True, separate_fut_agent_batches=False, fuse_option=3
    ):

        super().__init__()
        self.dim = dim

        self.config = config
        self.normalize_dataset = config['normalize_dataset'] 
        self.img_arch = config['img_arch']
        self.mode = config['mode']
        self.num_obs_steps = config['num_obs_steps']
        self.seq_length = config['pred_length']+self.num_obs_steps
        self.init = init

        resize_factor = config['resize_factor']
        # self.ds_mean, self.ds_std = 1280 // resize_factor / 2, 150 / resize_factor # 320, 75
        self.ds_mean, self.ds_std = config['ds_mean'], config['ds_std']
        self.pix_to_real = 0.03125
        self.heads = heads
        self.pretrained_vision = pretrained_vision
        self.reconstr_loss_factor = 0.001 if not coords_normed else 10.
        coord_dims = 2

        # token embeddings
        self.check_inf_values = torch.isinf
        self.check_trajectory = torch.isfinite

        # positional encoding
        # self.pos_encoder = PositionalEncoding(dim)
        # self.pos_encoder = TransformerPositionalEncoding(max_len=200, d_model=dim)

        # some options...
        # different positional encoding options, try out AgentAwareAttention, velocity input, simulation parameters
        self.coords_normed = coords_normed
        self.separate_obs_agent_batches = separate_obs_agent_batches
        self.separate_fut_agent_batches = separate_fut_agent_batches
        self.fuse_option = fuse_option
        # self.rec_loss_fct = nn.MSELoss(reduction='none')
       
        self.coord_emb = nn.Sequential(nn.LayerNorm(coord_dims), nn.Linear(coord_dims, dim))
        # self.destination_emb = nn.Linear(coord_dims, dim)
        # self.goal_emb = nn.Linear(coord_dims, dim)
        # self.vel_emb = nn.Linear(coord_dims, dim)

        if self.fuse_option == 1:
            self.sequence_destination_fusion = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=dim, nhead=heads), num_layers=1)
        elif self.fuse_option == 2:
            self.sequence_destination_fusion = nn.Sequential(nn.Linear(dim*2, dim), nn.LayerNorm(dim), nn.ReLU())
        elif self.fuse_option == 3:
            self.context_emb_semMap = nn.Linear(dim, dim//2)
            self.obs_sequence_embed = nn.Linear(dim, dim//2)
        self.context_emb_obs_sem = nn.Linear(dim, dim//2) 
        # self.context_emb_goals = nn.Linear(dim, dim//4)
        self.context_emb_occMap = nn.Linear(dim//4, dim//2)
        self.semantic_agent_embeds = nn.Linear(dim//2, dim)

        # first runs, with goal distance: train and valADE: sepTrue_fuse3, sepTrue_fuse1, sepTrue_fuse2, sepFalse_fuse1, sepFalse_fuse3, sepFalse_fuse2

        # load the image encoder
        self.img_encoder = ResNet18Adaption()
        if self.pretrained_vision:
            ckpt_path = os.sep.join(['TrajectoryPrediction', 'Modules', 'coca', 'checkpoints', 'Resnet18Adaption_pretrained_lr3e-4_withTanh_cont'])
            self.img_encoder = self.checkpoint_loader(self.img_encoder, ckpt_path)

        self.z_dim = 32
        self.q_mlp = nn.Sequential(
            nn.Linear(3*(dim//2), dim), nn.ReLU(), nn.LayerNorm(dim), 
            nn.Linear(dim, dim//2), nn.ReLU(), nn.LayerNorm(dim//2))
        self.q_A = nn.Linear(dim//2, self.z_dim)
        self.q_b = nn.Linear(dim//2, self.z_dim)

        self.z_to_hidden = nn.Linear(self.z_dim, dim//4) # -> relu -> to dim -> to coord
        # normalize to radius via sigmoid/tanh etc.?

        # decoder networks
        self.decoder_emb = nn.Linear(self.z_dim + coord_dims + coord_dims, dim)
        self.to_logits = nn.Sequential(
            nn.Linear(dim, coord_dims, bias=False)
        )

        self.sequence_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*ff_mult), num_layers=num_enc_layers)
        self.decoder_cross_attn = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=dim, nhead=heads), num_layers=num_dec_layers)

        # initialize weights
        self.apply(self._initialize_weights)
    

    def _initialize_weights(self, m):
        if hasattr(m, 'weight'):
            try:
                nn.init.xavier_normal_(m.weight)
                # nn.init.uniform_(m.weight, -0.02, 0.02)
                # nn.init.normal_(m.weight, std=0.02)
            except ValueError:
                # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                m.weight.data.uniform_(-0.02, 0.02)
                # print("Bypassing ValueError...")
        elif hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.zero_()


    def checkpoint_loader(self, model: nn.Module, ckpt_path):
        model_file_path = [file for file in os.listdir(ckpt_path) if file.endswith('.ckpt') and not file.endswith('last.ckpt')]
        assert len(model_file_path) == 1
        ckpt_path = os.sep.join([ckpt_path, model_file_path[0]])
        state_dict = OrderedDict({key.replace('model.', ''): tensor for key, tensor in torch.load(ckpt_path)['state_dict'].items()})
        module_state_dict = model.state_dict()

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

        model.load_state_dict(load_dict)
        return model


    def normalize(self, in_list, onlyStd=True):
        raise ValueError('clone the tensors, otherwise bad things happen!')
        out_list = []
        for in_tensor in in_list:
            if in_tensor is None:
                out_list.append(None)
                continue
            out_tensor = in_tensor / self.ds_std if onlyStd else (in_tensor-self.ds_mean) / self.ds_std
            out_list.append(out_tensor)
        if len(out_list) == 1: return out_list[0]
        return out_list

    
    def denormalize(self, in_list, onlyStd=True):
        raise ValueError('clone the tensors, otherwise bad things happen!')
        out_list = []
        for in_tensor in in_list:
            if in_tensor is None:
                out_list.append(None)
                continue
            out_tensor = in_tensor * self.ds_std if onlyStd else in_tensor * self.ds_std + self.ds_mean
            out_list.append(out_tensor)
        if len(out_list) == 1: return out_list[0]
        return out_list


    def determine_goals(self, abs_coordinates):
        # determine goal coordinates by extracting the last finite coordinates of the trajectories
        infinite_idx = torch.isinf(abs_coordinates[:,:,0]).long()
        first_infinite_idx = torch.argmax(infinite_idx, axis=1)
        # select obs_coords
        # select goal_coords
        goal_coords = abs_coordinates[np.arange(abs_coordinates.shape[0]), first_infinite_idx-1, :]
        assert torch.all(torch.isfinite(goal_coords)), 'goal coordinates must be finite!'
        inf_coords = abs_coordinates[np.arange(abs_coordinates.shape[0]), first_infinite_idx, :][first_infinite_idx != 0]
        assert torch.all(torch.isinf(inf_coords)), 'all next to last coordinates must be infinite (except where the entire trajectory is finite, meaning first_infinite_idx == 0)!'
        return goal_coords


    def forward(
        self,
        batch,
        decoder_mode=0,
    ):       
        #### inputs ####
        image = batch['image']
        abs_coordinates = batch['coords'].squeeze(0) if 'coords' in batch else None
        velocities = batch['transformed_velocities'].squeeze(0) if 'velocity' in batch else None
        occupancyMaps_egocentric = batch['occupancyMaps_egocentric'].permute(1, 0, 2, 3) if 'occupancyMaps_egocentric' in batch else None
        semanticMaps_per_agent = batch['semanticMaps_per_agent'].permute(3, 0, 1, 2) if 'semanticMaps_per_agent' in batch else None
        transformed_agent_destinations = batch['transformed_agent_destinations'].squeeze(0) if 'transformed_agent_destinations' in batch else None
        all_scene_destinations = batch['all_scene_destinations'].squeeze(0) if 'all_scene_destinations' in batch else None

        if self.coords_normed:
            abs_coordinates, velocities, transformed_agent_destinations, all_scene_destinations = self.normalize([abs_coordinates, velocities, transformed_agent_destinations, all_scene_destinations], onlyStd=True)

        obs_coords = abs_coordinates[:, :self.num_obs_steps]
        # teacher forcing
        label_coords = abs_coordinates[:, self.num_obs_steps:]
        # running free
        # label_coords = obs_coords[:, -1].unsqueeze(1) if abs_coordinates is not None else None
        num_agents = obs_coords.shape[0]
                    
        # obs_coords_mask = self.check_trajectory(obs_coords)[:,:,0:1]
        obs_coords_inf_mask_b = self.check_inf_values(obs_coords)[:,:,0:1]
        obs_coords_inf_mask_f = obs_coords_inf_mask_b.float().masked_fill(obs_coords_inf_mask_b, -1e25) # avoid the nan-problem...
        
        # context 1: agent observations	
        obs_coords = obs_coords.masked_fill(obs_coords_inf_mask_b.repeat(1, 1, 2), 0)
        coord_embeds = self.coord_emb(obs_coords)
        coord_embeds = coord_embeds.masked_fill(obs_coords_inf_mask_b.repeat(1, 1, self.dim), 0)       
        assert torch.all(torch.isfinite(coord_embeds))
        
        if self.separate_obs_agent_batches:
            attn_mask = torch.repeat_interleave(torch.repeat_interleave(obs_coords_inf_mask_f, obs_coords_inf_mask_f.shape[1], dim=-1), self.heads, dim=0) # shape = BS*num_heads, T_len, S_len
            pos_enc = pos_encoding(num_agents=num_agents, max_len=coord_embeds.shape[1]).to(coord_embeds.device).permute(1,0,2)
            coord_embeds = coord_embeds.permute(1,0,2) + pos_enc
            coord_embeds = self.sequence_encoder(coord_embeds, mask=attn_mask).permute(1,0,2)
        else:
            attn_mask = obs_coords_inf_mask_f.squeeze(-1).view(1, -1)
            attn_mask = torch.repeat_interleave(torch.repeat_interleave(attn_mask.unsqueeze(-1), attn_mask.shape[1], dim=-1), self.heads, dim=0)
            pos_enc = pos_encoding(num_agents=1, max_len=coord_embeds.shape[1]*num_agents).to(coord_embeds.device).permute(1,0,2)
            coord_embeds = coord_embeds.view(-1, 1, coord_embeds.shape[-1]) + pos_enc
            coord_embeds = self.sequence_encoder(coord_embeds, mask=attn_mask).view(num_agents, self.num_obs_steps, coord_embeds.shape[-1])
        
        # image encoder
        semanticMaps_per_agent_recon, semanticMaps_per_agent_enc = self.img_encoder(semanticMaps_per_agent)
        semanticMaps_per_agent_enc = semanticMaps_per_agent_enc.flatten(-2).permute(0,2,1)
        
        if self.fuse_option == 1: # cross-attention
            context_agent_map = self.sequence_destination_fusion(coord_embeds.mean(dim=1).unsqueeze(0), semanticMaps_per_agent_enc.permute(1,0,2)).squeeze()
        elif self.fuse_option == 2: # concatenate
            context_agent_map = torch.cat((coord_embeds.mean(dim=1), semanticMaps_per_agent_enc.mean(dim=1)), dim=-1)
            context_agent_map = self.sequence_destination_fusion(context_agent_map)
        elif self.fuse_option == 3: # separate
            context_emb_semMap = self.context_emb_semMap(semanticMaps_per_agent_enc.mean(dim=1))
            obs_sequence_embed = self.obs_sequence_embed(coord_embeds.mean(dim=1))
            # context_agent_map = torch.cat((context_emb_semMap, obs_sequence_embed), dim=-1)

        # context 2: current ego-centric maps
        occupancyMaps_egocentric_recon, occupancyMaps_egocentric_enc = self.img_encoder(occupancyMaps_egocentric, mode=1)
        context_egoOccMap = occupancyMaps_egocentric_enc.flatten(-2) # N, 128, 64 

        # context 3: agent destination
        # agent_goals = self.determine_goals(abs_coordinates)
        # agent_goals_embed = self.goal_emb(agent_goals)
        # assert torch.all(torch.isfinite(agent_goals_embed))

        # encode and concatenate all context vectors
        # context_emb_obs_sem = self.context_emb_obs_sem(context_agent_map)
        context_egoOccMap = self.context_emb_occMap(context_egoOccMap.mean(dim=-1))
        # context_goals = self.context_emb_goals(agent_goals_embed)
        context_vector = torch.cat((obs_sequence_embed, context_emb_semMap, context_egoOccMap), dim=-1) # , context_goals

        last_pos = obs_coords[:,-1,:]
        # last_coords_inf_mask_b = obs_coords_inf_mask_b[:,-1,:]
        # this is basically check_agents_in_destination(...), must be done to know which agents terminate after the last observed time step
        last_coords_inf_mask_b = torch.isinf(label_coords[:, 0, 0:1]) 
        # TODO check in the Dataset creation how many timesteps are passed within destination until termination 
        future_coords_inf_mask_b = last_coords_inf_mask_b.clone()

        qnet_h = self.q_mlp(context_vector)
        A = self.q_A(qnet_h)
        logvar = (A ** 2 + 1e-8).log()
        mu = self.q_b(qnet_h)

        q_z_dist = Normal(mu=mu, logvar=logvar, valid_agents=~last_coords_inf_mask_b)
        z = q_z_dist.sample()        

        # distance_to_goals = agent_goals - last_pos
        distance_to_destinations = (transformed_agent_destinations[:,:2] + transformed_agent_destinations[:,2:])/2. - last_pos
        # dec_in = torch.cat([last_pos, distance_to_goals, z], dim=-1)
        dec_in = torch.cat([last_pos, distance_to_destinations, z], dim=-1)
        # context_vector = context_vector.unsqueeze(0) if self.separate_agent_batches else context_vector.unsqueeze(1)
        semantic_agent_embeds = self.semantic_agent_embeds(context_emb_semMap).unsqueeze(0) if self.separate_fut_agent_batches else self.semantic_agent_embeds(context_emb_semMap).unsqueeze(1)

        # possibility to mean over all timesteps and use this as input
        # TODO for an extension: max_step_size, egocentric heatmaps every timestep, intermediate goal predictor, better loss function
        # TODO network that predicts from (normalized?) coord to heatmap (mean at coord center), maybe semantic heatmap?
        pred_coords = []
        # decode
        for i_step in range(self.num_obs_steps, self.seq_length):
            num_future_steps = i_step + 1 - self.num_obs_steps

            decoder_emb = self.decoder_emb(dec_in)
            decoder_emb = decoder_emb.masked_fill(future_coords_inf_mask_b.repeat(1, decoder_emb.shape[-1]), 0)
            decoder_emb = decoder_emb.unsqueeze(1)
            assert torch.all(torch.isfinite(decoder_emb)), 'decoder_emb must be finite!'
            
            tgt_mask_b = future_coords_inf_mask_b
            if i_step != self.num_obs_steps :assert torch.all(self.check_inf_values(last_pos[:, 0:1]) == future_coords_inf_mask_b[-num_agents:]), 'tgt_mask_b must be equal to last_obs_pos[:, 0:1]!'
            tgt_mask = tgt_mask_b.float().masked_fill(tgt_mask_b, -1e25) 
            if self.separate_fut_agent_batches:
                decoder_emb = decoder_emb.permute(1,0,2)
                # context_vector_rep = context_vector.repeat(1, num_future_steps,1)
                semantic_agent_embeds_rep = semantic_agent_embeds.repeat(1, num_future_steps,1)
                tgt_mask = torch.repeat_interleave(tgt_mask.unsqueeze(-1), self.heads, dim=0) # shape = BS*num_heads, T_len, S_len
                pos_enc = pos_encoding(num_agents=num_agents * num_future_steps, max_len=decoder_emb.shape[0]).to(coord_embeds.device).permute(1,0,2)
                assert pos_enc.shape == decoder_emb.shape and decoder_emb.shape == semantic_agent_embeds_rep.shape
                decoder_emb += pos_enc
            else:
                tgt_mask = torch.repeat_interleave(torch.repeat_interleave(tgt_mask.view(1, -1).unsqueeze(-1), tgt_mask.shape[0], dim=-1), self.heads, dim=0) # shape = BS*num_heads, T_len, S_len
                pos_enc = pos_encoding(num_agents=1, max_len=decoder_emb.shape[1] * num_agents * num_future_steps).to(coord_embeds.device).permute(1,0,2)
                # context_vector_rep = context_vector.repeat(num_future_steps, 1, 1)
                semantic_agent_embeds_rep = semantic_agent_embeds.repeat(num_future_steps, 1, 1)
                assert pos_enc.shape == decoder_emb.shape and decoder_emb.shape == semantic_agent_embeds_rep.shape
                decoder_emb += pos_enc
            memory_mask = tgt_mask.clone()
            # mask shape check
            T_len, S_len, bs = decoder_emb.shape[0], semantic_agent_embeds_rep.shape[0], decoder_emb.shape[1]
            assert tgt_mask.shape == (bs*self.heads, T_len, S_len) and decoder_emb.shape[1]==semantic_agent_embeds_rep.shape[1], 'incorrect shapes!'
            
            # cross-attention between dec_in and context
            decoder_out = self.decoder_cross_attn(decoder_emb, semantic_agent_embeds_rep, tgt_mask=tgt_mask, memory_mask=memory_mask)

            # coordinate decoding
            out = self.to_logits(decoder_out).squeeze() if decoder_out.shape[0]==num_agents else self.to_logits(decoder_out.permute(1,0,2)).squeeze()
            outputs_current = out[-num_agents:].clone().detach()
            if self.coords_normed:
                outputs_current += last_pos

            # mask out agents that had already reached a destination before
            outputs_current = outputs_current.masked_fill(last_coords_inf_mask_b.repeat(1, 2), float('inf'))
            pred_coords.append(outputs_current.clone()) # this ensures always at least one prediction
            # mask out agents that have reached any destination within the current timestep
            outputs_current, any_agent_terminated = check_agents_in_destination(outputs_current, all_scene_destinations)
            
            # append out_in together with z_in (again), plus adjust the mask
            last_pos = outputs_current
            # distance_to_goals = agent_goals - last_pos
            distance_to_destinations = (transformed_agent_destinations[:,:2] + transformed_agent_destinations[:,2:])/2. - last_pos
            # adjust the current and the total masks: last_coords_inf_mask_b gets filled with True slowly, while future_coords_inf_mask_b concatenates over all time steps for mask calculations above
            if not any_agent_terminated: 
                assert torch.all(last_coords_inf_mask_b == self.check_inf_values(last_pos[:, 0:1])), 'if no agent has reached a destination, then last_coords_inf_mask_b must be equal to inf positions in las_pos'
            last_coords_inf_mask_b = self.check_inf_values(last_pos[:, 0:1])
            future_coords_inf_mask_b = torch.cat([future_coords_inf_mask_b, last_coords_inf_mask_b], dim=0)
            # dec_in_current = torch.cat([last_pos, distance_to_goals, z], dim=-1)
            dec_in_current = torch.cat([last_pos, distance_to_destinations, z], dim=-1)
            dec_in = torch.cat([dec_in, dec_in_current], dim=0) 
        
        # auxilliary MSE image reconstruction loss
        if self.pretrained_vision:
            aux_img_loss = None
        else:
            aux_img_loss_1 = (semanticMaps_per_agent_recon - semanticMaps_per_agent)**2
            aux_img_loss_2 = (occupancyMaps_egocentric_recon - occupancyMaps_egocentric)**2
            aux_img_loss = (aux_img_loss_1.sum() + aux_img_loss_2.sum()) / (aux_img_loss_1.numel() + aux_img_loss_2.numel()) 
            # aux_img_loss = F.mse_loss(occupancyMaps_egocentric_recon, occupancyMaps_egocentric)
        pred_coords = torch.stack(pred_coords, dim=1)

        return {'pred_coords': pred_coords, 'aux_img_loss': aux_img_loss, 'q_z_dist': q_z_dist}


    def compute_loss(self, prediction, batch, stage, save_path=None):

        abs_coordinates = batch['coords'].squeeze(0) if 'coords' in batch else None
        velocities = batch['transformed_velocities'].squeeze(0) if 'velocity' in batch else None
        occupancyMaps_egocentric = batch['occupancyMaps_egocentric'].permute(1, 0, 2, 3) if 'occupancyMaps_egocentric' in batch else None
        semanticMaps_per_agent = batch['semanticMaps_per_agent'].permute(3, 0, 1, 2) if 'semanticMaps_per_agent' in batch else None
        transformed_agent_destinations = batch['transformed_agent_destinations'].squeeze(0) if 'transformed_agent_destinations' in batch else None
        all_scene_destinations = batch['all_scene_destinations'].squeeze(0) if 'all_scene_destinations' in batch else None

        pred_coords = prediction['pred_coords']
        aux_img_loss = prediction['aux_img_loss']
        q_z_dist = prediction['q_z_dist']

        label_coords = abs_coordinates[:, self.num_obs_steps:]

        reconstruction_loss, ade_loss, fde_loss = self.compute_mse_and_ade(pred_coords, label_coords, abs_coordinates[:, self.num_obs_steps-1])

        if save_path is not None:
            abs_coordinates_draw = abs_coordinates.clone() * self.ds_std + self.ds_mean
            # pred_coords_draw = self.denormalize([prediction['pred_coords'].clone().detach()])
            pred_coords_draw = prediction['pred_coords'].clone().detach() * self.ds_std + self.ds_mean
            file_name = 'img_'+str(len(os.listdir(save_path)))+f'__ADE{float(ade_loss.clone().detach().item()):.5f}__FDE{float(fde_loss.clone().detach().item()):.5f}.png'
            save_path = os.path.join(save_path, file_name)
            visualize_trajectories(abs_coordinates_draw.cpu().numpy(), pred_coords_draw.cpu().numpy(), batch['image'].squeeze(0).permute(1, 2, 0).cpu().numpy(), self.num_obs_steps, save_image=True, save_path=save_path)

        # reconstruction_loss = torch.sum((torch.abs(pred_coords - label_coords).masked_fill(~is_traj_mask, 0))**2) / torch.sum(is_traj_mask)
        # TODO implement closest to final loss
        kl_loss = q_z_dist.kl().mean()

        total_loss = reconstruction_loss + kl_loss
        loss_log = {'total_loss': total_loss, 'MSE_loss': reconstruction_loss, 'kl_loss': kl_loss}
        if aux_img_loss is not None: 
            total_loss += aux_img_loss
            loss_log.update({'aux_img_loss': aux_img_loss})

        # ade_loss = torch.sum((torch.abs(prediction_cl - labels_cl).masked_fill(~is_traj_mask, 0)))  / torch.sum(is_traj_mask)
        metrics = {'ADE_caption': ade_loss.item(), 'FDE_caption': fde_loss.item()}

        return total_loss, loss_log, metrics
    

    def compute_mse_and_ade(self, outputs, ground_truth, last_obs_gt):
        """ Compute MSE and ADE loss based on the last non-inf coordinate of the predicted trajectory. """
        future_length = outputs.shape[1]
        # assuming that no new agents spawn during the sequence
        finite_mask_outputs = torch.isfinite(outputs)
        finite_mask_ground_truth = torch.isfinite(ground_truth)
        
        # both outputs and ground_truth are inf
        inf_mask = torch.logical_and(~finite_mask_outputs, ~finite_mask_ground_truth)
        no_seq_per_agent = (~inf_mask[:,:,0]).sum(dim=-1) == 0 # no coords present over entire sequence per agent
        mse_items = []
        # rec_items = []

        # for metrics calculation
        outputs_clone = outputs.clone().detach() * self.ds_std + self.ds_mean
        ground_truth_clone = ground_truth.clone().detach() * self.ds_std + self.ds_mean
        if self.coords_normed:
            ground_truth, last_obs_gt = self.normalize([ground_truth, last_obs_gt], onlyStd=True)
            outputs_clone = self.denormalize([outputs_clone], onlyStd=True)
        ade_items = []
        fde_items = []

        valid_agents = torch.arange(outputs.shape[0])[~no_seq_per_agent]
        for agent_id in valid_agents:
            out_agent_coords = outputs[agent_id]
            finite_out_agent_mask = finite_mask_outputs[agent_id]
            out_agent_last_finite_id = future_length - torch.argmax((finite_out_agent_mask[:,0]).flip(dims=[-1]).long(), dim=-1) - 1

            gt_agent_coords = ground_truth[agent_id]
            finite_gt_agent_mask = finite_mask_ground_truth[agent_id]
            gt_agent_last_finite_id = future_length - torch.argmax((finite_gt_agent_mask[:,0]).flip(dims=[-1]).long(), dim=-1) - 1

            # metrics
            out_agent_coords_clone = outputs_clone[agent_id]
            gt_agent_coords_clone = ground_truth_clone[agent_id]

            fde_items.append(torch.abs(out_agent_coords_clone[out_agent_last_finite_id] - gt_agent_coords_clone[gt_agent_last_finite_id]))
            # one is all inf, the other is not
            if torch.all(~finite_out_agent_mask) != torch.all(~finite_gt_agent_mask):
                if torch.all(~finite_gt_agent_mask):
                    assert torch.all(torch.isfinite(last_obs_gt[agent_id]))
                    mse_loss_item = out_agent_coords[:out_agent_last_finite_id+1] - last_obs_gt[agent_id]
                    mse_items.append(mse_loss_item.pow(2))
                    ade_items.append(torch.abs(out_agent_coords_clone[:out_agent_last_finite_id+1] - gt_agent_coords_clone[agent_id]))
                else:
                    raise ValueError('This should not happen! There cannot be a case where all predictions are already terminated, but the ground truth is not!')
            # both end at different timesteps
            elif out_agent_last_finite_id.item() != gt_agent_last_finite_id.item():
                min_id = min(out_agent_last_finite_id, gt_agent_last_finite_id)
                max_id = max(out_agent_last_finite_id, gt_agent_last_finite_id)
                mse_loss_item = out_agent_coords[:min_id+1] - gt_agent_coords[:min_id+1]
                mse_items.append(mse_loss_item.pow(2))
                ade_items.append(torch.abs(out_agent_coords_clone[:min_id+1] - gt_agent_coords_clone[:min_id+1]))
                if out_agent_last_finite_id < gt_agent_last_finite_id:
                    mse_per_agent = (gt_agent_coords[min_id+1:max_id+1] - out_agent_coords[min_id])**2
                    mse_items.append(mse_per_agent)
                    ade_items.append(torch.abs(gt_agent_coords_clone[min_id+1:max_id+1] - out_agent_coords_clone[min_id]))
                else:
                    mse_per_agent = (out_agent_coords[min_id+1:max_id+1] - gt_agent_coords[min_id])**2
                    mse_items.append(mse_per_agent)
                    ade_items.append(torch.abs(out_agent_coords_clone[min_id+1:max_id+1] - gt_agent_coords_clone[min_id]))
            # both end at the same timestep
            else:
                # rec_items.append([out_agent_coords[:out_agent_last_finite_id+1].view(-1), gt_agent_coords[:gt_agent_last_finite_id+1].view(-1)])
                mse_items.append((out_agent_coords[:out_agent_last_finite_id+1] - gt_agent_coords[:gt_agent_last_finite_id+1])**2)
                ade_items.append(torch.abs(out_agent_coords_clone[:out_agent_last_finite_id+1] - gt_agent_coords_clone[:gt_agent_last_finite_id+1]))
        # TODO maybe reward for reaching the (correct) destination? e.g. by concatenating zeros to the end of the sequence when both are inf
        # manual reward for reaching the correct destination?

        mse_items_t = torch.cat(mse_items, dim=0)
        ade_items_t = torch.cat(ade_items, dim=0)
        assert torch.all(torch.isfinite(mse_items_t)) and torch.all(torch.isfinite(ade_items_t))
        mse_loss = mse_items_t.mean()
        ade_loss = ade_items_t.mean()
        fde_loss = torch.cat(fde_items, dim=0).mean()
        return mse_loss, ade_loss, fde_loss



class ResNet18Adaption(nn.Module):
    def __init__(self):
        super(ResNet18Adaption, self).__init__()

        # Load pre-trained ResNet18 model
        resnet18 = models.resnet18(pretrained=False)
        conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Extract layers from the pre-trained ResNet18 model
        self.conv1 = conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        
        # Define segmentation head
        self.conv_t1 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=4, bias=False)
        self.conv_t2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, bias=False)
        self.conv_t3 = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, bias=False)

        self.final_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False)
        self.use_activation = True


    def forward(self, x_in, mode=0):
        # Encoder (backbone)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x_in))))
        x = self.layer1(x)
        x_enc = self.layer2(x)
        if mode == 0:
            x_enc = self.layer3(x_enc)
            x_enc = self.layer4(x_enc)
            x_enc_out = nn.AdaptiveAvgPool2d(output_size=(8,8))(x_enc)
        else:
            x_enc_out = x_enc

        # Segmentation head
        if mode == 0:
            x_up = self.conv_t1(x_enc)
        else:
            x_up = x_enc
        x_up = self.conv_t2(x_up)
        x_rec = self.conv_t3(x_up)
        if x_rec.shape[-2:] != x_in.shape[-2:]:
            x_rec = resize(x_rec, size=x_in.shape[-2:], interpolation=InterpolationMode.BILINEAR)

        x_rec = self.final_conv(x_rec)
        if self.use_activation:
            x_rec = torch.tanh(x_rec)

        return x_rec, x_enc_out



class Normal:
    def __init__(self, mu=None, logvar=None, params=None, valid_agents:torch.Tensor=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)
        assert not torch.all(torch.isinf(valid_agents))
        self.valid_agents = valid_agents.repeat(1, self.mu.shape[-1]) # only KL divergence for agents that are present in the last observed timestep

    def sample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        kl = kl[self.valid_agents]
        return kl

    def mode(self):
        return self.mu
