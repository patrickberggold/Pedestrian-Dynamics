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
from helper import visualize_trajectories, relative_to_abs
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


class Generator(nn.Module):
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

        self.embedding_dim = 128
        self.encoder_h_dim_g = 128
        self.dropout = 0.1
        self.mlp_dim = 512
        self.num_layers = 1

        self.encoder = Encoder(
            embedding_dim=self.embedding_dim,
            h_dim=self.encoder_h_dim_g,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            dropout=0.1
        )

        self.decoder_h_dim_g = 128
        self.bottleneck_dim = 64
        self.decoder = Decoder(
            config['pred_length'],
            embedding_dim=self.embedding_dim,
            h_dim=self.decoder_h_dim_g,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            pool_every_timestep=False,
            dropout=self.dropout,
            bottleneck_dim=self.bottleneck_dim,
            activation='relu',
            batch_norm=0,
            pooling_type=None,
            grid_size=8,
            neighborhood_size=2
        )

        # token embeddings
        self.check_inf_values = torch.isinf
        self.check_trajectory = torch.isfinite

        # positional encoding
        # self.pos_encoder = PositionalEncoding(dim)
        # self.pos_encoder = TransformerPositionalEncoding(max_len=200, d_model=dim)

        """ self.coords_normed = coords_normed
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
        self.decoder_cross_attn = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=dim, nhead=heads), num_layers=num_dec_layers) """

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

        # if self.coords_normed:
        #     abs_coordinates, velocities, transformed_agent_destinations, all_scene_destinations = self.normalize([abs_coordinates, velocities, transformed_agent_destinations, all_scene_destinations], onlyStd=True)

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
        # obs_coords = obs_coords.masked_fill(obs_coords_inf_mask_b.repeat(1, 1, 2), 0)
        # coord_embeds = self.coord_emb(obs_coords)
        # coord_embeds = coord_embeds.masked_fill(obs_coords_inf_mask_b.repeat(1, 1, self.dim), 0)       
        # assert torch.all(torch.isfinite(coord_embeds))
        
        # if self.separate_obs_agent_batches:
        #     attn_mask = torch.repeat_interleave(torch.repeat_interleave(obs_coords_inf_mask_f, obs_coords_inf_mask_f.shape[1], dim=-1), self.heads, dim=0) # shape = BS*num_heads, T_len, S_len
        #     pos_enc = pos_encoding(num_agents=num_agents, max_len=coord_embeds.shape[1]).to(coord_embeds.device).permute(1,0,2)
        #     coord_embeds = coord_embeds.permute(1,0,2) + pos_enc
        #     coord_embeds = self.sequence_encoder(coord_embeds, mask=attn_mask).permute(1,0,2)
        # else:
        #     attn_mask = obs_coords_inf_mask_f.squeeze(-1).view(1, -1)
        #     attn_mask = torch.repeat_interleave(torch.repeat_interleave(attn_mask.unsqueeze(-1), attn_mask.shape[1], dim=-1), self.heads, dim=0)
        #     pos_enc = pos_encoding(num_agents=1, max_len=coord_embeds.shape[1]*num_agents).to(coord_embeds.device).permute(1,0,2)
        #     coord_embeds = coord_embeds.view(-1, 1, coord_embeds.shape[-1]) + pos_enc
        #     coord_embeds = self.sequence_encoder(coord_embeds, mask=attn_mask).view(num_agents, self.num_obs_steps, coord_embeds.shape[-1])
        
        # discriminator_step --> if self.d_steps > 0:
        # GENERATOR CALL: self.generator(obs_traj, obs_traj_rel, seq_start_end)
        # encoder (together with masking)
        final_encoder_h = self.encoder(obs_coords, obs_coords_inf_mask_b)
        
        # social pooling here for all agents present in the last observed time frame.. (N,1,hidden_enc_dim) -> (N, pooling_dim)
        mlp_decoder_context_input = final_encoder_h.view(-1, self.encoder_h_dim_g)
        noise_input = mlp_decoder_context_input

        # decoder_h = self.add_noise(noise_input, None, user_noise=None)
        decoder_h = torch.unsqueeze(noise_input, 1)
        decoder_c = torch.zeros(self.num_layers, num_agents, self.decoder_h_dim_g)
        state_tuple = (decoder_h, decoder_c)
        obs_coords_rel = torch.cat(torch.zeros((num_agents, 1, 2), dtype=torch.float32, device=obs_coords.device), obs_coords[:, 1:] - obs_coords[:, :-1], dim=1)
        label_coords_rel = torch.cat(label_coords[:, 0] - obs_coords[:, -1], label_coords[:, 1:] - label_coords[:, :-1], dim=1)
        
        assert obs_coords_rel.shape == obs_coords.shape
        last_pos = obs_coords[:, -1]
        last_pos_rel = obs_coords_rel[:, -1]

        pred_traj_fake_rel, _ = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            None,
        )

        # TODO apply some masks
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_coords[:, -1])

        return pred_traj_fake, pred_traj_fake_rel


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
        # kl_loss = q_z_dist.kl().mean()

        total_loss = reconstruction_loss # + kl_loss
        loss_log = {'total_loss': total_loss, 'MSE_loss': reconstruction_loss}
        if aux_img_loss is not None: 
            total_loss += aux_img_loss
            loss_log.update({'aux_img_loss': aux_img_loss})

        # ade_loss = torch.sum((torch.abs(prediction_cl - labels_cl).masked_fill(~is_traj_mask, 0)))  / torch.sum(is_traj_mask)
        metrics = {'ADE_caption': ade_loss.item(), 'FDE_caption': fde_loss.item()}

        return total_loss, loss_log, metrics
    

    def compute_mse_and_ade(self, outputs, ground_truth, last_obs_gt):
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



class Discriminator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.normalize_dataset = config['normalize_dataset'] 
        self.img_arch = config['img_arch']
        self.mode = config['mode']
        self.num_obs_steps = config['num_obs_steps']
        self.seq_length = config['pred_length']+self.num_obs_steps

        self.hidden_dim = 128
        self.mlp_dim = 512
        self.encoder = Encoder(
            embedding_dim=128,
            h_dim=self.hidden_dim,
            mlp_dim=512,
            num_layers=1,
            dropout=0.1
        )
        real_classifier_dims = [self.hidden_dim, self.mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation='relu',
            batch_norm=0,
            dropout=0.1
        )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        classifier_input = final_h.squeeze()
        scores = self.real_classifier(classifier_input)
        return scores


import torch
import torch.nn as nn


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj, inf_mask):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(0)
        obs_traj = obs_traj.masked_fill(inf_mask.repeat(1, 1, 2), 0)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            batch, -1, self.embedding_dim
        )
        obs_traj_embedding = obs_traj_embedding.masked_fill(inf_mask.repeat(1, 1, self.embedding_dim), 0)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores



"""
self.discriminator = TrajectoryDiscriminator(
            obs_len=config['num_obs_steps'],
            pred_len=config['pred_length'],
            embedding_dim=self.embedding_dim,
            h_dim=self.encoder_h_dim_d,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_norm=0,
            d_type='local')
"""


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)