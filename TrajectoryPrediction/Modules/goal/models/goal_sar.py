import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from Modules.goal.models.base_model import Base_Model
from Modules.goal.models.unet import UNet
from Modules.goal.metrics import ADE_best_of, KDE_negative_log_likelihood, \
    FDE_best_of_goal, FDE_best_of_goal_world
from Datamodules.SceneLoader import Scene_floorplan
from Modules.goal.metrics import compute_metric_mask
from Modules.goal.goal_helper import MSE_loss, MSE_loss_new, Goal_BCE_loss, sampling, TTST_test_time_sampling_trick, check_agents_in_destination

class Goal_SAR(Base_Model):
    def __init__(self, config, traj_quantity = 'pos'):
        super().__init__()
        self.args = None
        self.traj_quantity = traj_quantity
        # self.device = device
        # self.full_dataset = full_dataset
        ##################
        # MODEL PARAMETERS
        ##################

        # set parameters for network architecture
        self.input_size = 2  # size of the input 2: (x,y)
        self.embedding_size = 32  # embedding dimension
        self.nhead = 8  # number of heads in multi-head attentions TF
        self.d_hidden = 2048  # hidden dimension in the TF encoder layer
        self.n_layers_temporal = 6  # number of TransformerEncoderLayers  ### ORIGINAL: 6
        self.dropout_prob = 0  # the dropout probability value ### ORIGINAL: 0.1
        self.noise_size = 16  # size of random noise vector
        self.output_size = 2  # output size

        self.config = config
        self.masked = True if config['data_format'] in ['by_frame_masked'] else False
        self.normalize = config['normalize_dataset']

        # Goal args
        self.add_noise_traj = True
        self.obs_length = 8
        self.pred_length = 12
        self.seq_length = 20
        self.num_test_samples = 20
        self.num_valid_samples = 20
        self.compute_valid_nll = False
        self.compute_test_nll = True
        self.down_factor = 1
        self.sampler_temperature = 1
        self.use_ttst = True

        # GOAL MODULE PARAMETERS
        self.num_image_channels = 3 # if args.dataset == 'floorplan' else 6

        # U-net encoder channels
        self.enc_chs = (self.num_image_channels + self.obs_length,
                        32, 32, 64, 64, 64)
        # U-net decoder channels
        self.dec_chs = (64, 64, 64, 32, 32)

        self.extra_features = 4  # extra information to concat goals: time,
        # last positions, predicted final positions, distance to predicted goals

        ##################
        # MODEL LAYERS
        ##################

        ################## GOAL ##################

        in_goal_channels = self.num_image_channels + self.obs_length if not self.masked else self.num_image_channels + self.seq_length
        self.goal_module = UNet(
            enc_chs=(in_goal_channels,
                     32, 32, 64, 64, 64),
            dec_chs=(64, 64, 64, 32, 32),
            out_chs=self.pred_length)

        ################## TEMP BACKBONE ##################

        # linear layer to map input to embedding
        if config['data_format'] in ['partial_tokenized_random', 'partial_tokenized_by_frame']:
            self.tokens = ['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[TRAJ]', '[UNK]']
            dict_size = len(self.tokens) if self.tokens is not None else 0
            self.input_size += len(self.tokens)
            self.output_size += len(self.tokens)
            self.input_embedding_layer_temporal = TrajectoryEmbedding(self.input_size, self.tokens, self.embedding_size, data_format = config['data_format']) # nn.Linear(self.input_size, self.embedding_size)
            self.output_layer = ToLogitsNet(self.embedding_size, self.output_size, dict_size)
        else:
            self.tokens = None
            # original
            self.input_embedding_layer_temporal = nn.Linear(self.input_size, self.embedding_size, bias=False)
            if self.add_noise_traj:
                self.output_layer = nn.Linear(
                    self.embedding_size + self.noise_size, self.output_size)
            else:
                self.output_layer = nn.Linear(
                    self.embedding_size, self.output_size)

        # self.input_embedding_layer_temporal = nn.Linear(
        #     self.input_size, self.embedding_size)

        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_input_temporal = nn.Dropout(self.dropout_prob)

        # temporal encoder layer for temporal sequence
        self.temporal_encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_size + self.extra_features * self.nhead,
            nhead=self.nhead,
            dim_feedforward=self.d_hidden)

        # temporal encoder for temporal sequence
        self.temporal_encoder = TransformerEncoder(
            self.temporal_encoder_layer,
            num_layers=self.n_layers_temporal)

        # fusion layer
        self.fusion_layer = nn.Linear(
            self.embedding_size + self.nhead * self.extra_features + \
            self.extra_features * 2 - 1, self.embedding_size)

        # FC decoder
        # if self.add_noise_traj:
        #     self.output_layer = nn.Linear(
        #         self.embedding_size + self.noise_size, self.output_size)
        # else:
        #     self.output_layer = nn.Linear(
        #         self.embedding_size, self.output_size)
        
        self.clip = 1

        self.losses_coeffs = {
            "traj_MSE_loss": 1,
            "goal_BCE_loss": 1e6,
        }

    def init_losses(self):
        losses = {
            "traj_MSE_loss": 0,
            "goal_BCE_loss": 0,
        }
        return losses

    def init_train_metrics(self):
        train_metrics = {
            "ADE": [],
            # "FDE": [],
        }
        return train_metrics

    # def init_test_metrics(self):
    #     test_metrics = {
    #         "ADE": [],
    #         "FDE": [],
    #         "ADE_world": [],
    #         "FDE_world": [],
    #         "NLL": [],
    #     }
    #     return test_metrics

    # def init_best_metrics(self):
    #     best_metrics = {
    #         "ADE": 1e9,
    #         "FDE": 1e9,
    #         "ADE_world": 1e9,
    #         "FDE_world": 1e9,
    #         "goal_BCE_loss": 1e9,
    #     }
    #     return best_metrics


    def compute_masked_metrics(self, all_outputs, abs_coordinates, prediction_length):
        assert all_outputs.shape[0] == 1, "Only 1 sample implemented!"
        outputs = all_outputs.squeeze(0)[:, -prediction_length:]
        ground_truth = abs_coordinates[:, -prediction_length:]

        inf_mask_outputs = torch.isinf(outputs)
        inf_mask_ground_truth = torch.isinf(ground_truth)

        inf_mask = torch.logical_and(inf_mask_outputs, inf_mask_ground_truth)
        # if output ends earlier than gt, use last distance (and vice versa)
        # revert inf_mask_outputs and inf_mask_ground_truth along dim=1
        last_finite_coord_outputs = torch.argmax((~inf_mask_outputs[:,:,0]).flip(dims=[1]).long(), dim=1)
        last_finite_coord_ground_truth = torch.argmax((~inf_mask_ground_truth[:,:,0]).flip(dims=[1]).long(), dim=1)

        for agent_id in torch.arange(outputs.shape[0]):
            outputs_old_np, ground_truth_old_np = outputs.detach().cpu().numpy(), ground_truth.detach().cpu().numpy() # NUMPY CHECK
            if last_finite_coord_outputs[agent_id] != 0:
                outputs[agent_id, (-last_finite_coord_outputs[agent_id]-1):, :] = outputs[agent_id, (-last_finite_coord_outputs[agent_id]-1), :]
            
            if last_finite_coord_ground_truth[agent_id] != 0:
                ground_truth[agent_id, (-last_finite_coord_ground_truth[agent_id]-1):, :] = ground_truth[agent_id, (-last_finite_coord_ground_truth[agent_id]-1), :]
            outputs_new_np, ground_truth_new_np = outputs.detach().cpu().numpy(), ground_truth.detach().cpu().numpy() # NUMPY CHECK

        non_inf_error = (outputs - ground_truth).masked_fill(inf_mask, 0)


    def compute_loss(self, prediction, batch, stage):

        losses_it = self.init_losses()
        metrics_it = self.init_train_metrics()
        
        all_outputs, all_aux_outputs = prediction
        # self.device = all_outputs.device

        images = batch['images']
        abs_coordinates = batch['coords'].squeeze(0) if 'coords' in batch else None
        tokens = batch['tokens'].squeeze(0) if 'tokens' in batch else None
        input_traj_maps  = batch['input_traj_maps'].squeeze(0) if 'input_traj_maps' in batch else None
        # scene = batch['scene']

        # inputs = {k: v.squeeze(0).float() for k, v in batch.items() if k not in ['type', 'scene_data']}
        # inputs['scene_data'] = batch['scene_data']

        # compute loss_mask
        seq_list = torch.ones((abs_coordinates.shape[1], abs_coordinates.shape[0]), device=self.device)
        loss_mask = self.compute_loss_mask(seq_list).to(self.device)

        # compute metric_mask
        metric_mask = compute_metric_mask(seq_list)

        # compute model losses
        losses = self.compute_model_losses(all_outputs, abs_coordinates, loss_mask, input_traj_maps, all_aux_outputs)

        # overall loss
        loss = torch.zeros(1).to(self.device)
        for loss_name, loss_value in losses.items():
            loss += self.losses_coeffs[loss_name]*loss_value
            losses_it[loss_name] += loss_value.item()
            if loss_name == "ADE":
                metrics_it[loss_name].extend(loss_value)
            # if stage == 'train':
            #     self.losses_train_epoch[loss_name] += loss_value.item()
            # elif stage == 'val':
            #     self.losses_val_epoch[loss_name] += loss_value.item()

        # torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
            
        with torch.no_grad():
            if not self.masked:
                for metric_name in self.init_train_metrics():
                    metrics_it[metric_name] = self.compute_model_metrics(
                        metric_name=metric_name,
                        phase=stage,
                        predictions=all_outputs,
                        ground_truth=abs_coordinates,
                        metric_mask=metric_mask,
                        all_aux_outputs=all_aux_outputs,
                        # scene=scene,
                        obs_length=self.obs_length,
                    ) 
                # metrics_it[metric_name].extend(metric)

                # if stage == 'train':
                #     self.metrics_train_epoch[metric_name].extend(
                #         metric
                #     )
                # elif stage == 'val':
                #     self.metrics_val_epoch[metric_name].extend(
                #         metric
                #     )

        return loss, losses_it, metrics_it


    def init_losses(self):
        losses = {
            "traj_MSE_loss": 0,
            "goal_BCE_loss": 0,
        }
        return losses

    def set_losses_coeffs(self):
        losses_coeffs = {
            "traj_MSE_loss": 1,
            "goal_BCE_loss": 1e6,
        }
        return losses_coeffs

    def init_train_metrics(self):
        train_metrics = {
            "ADE": [],
            "FDE": [],
        }
        return train_metrics

    def init_test_metrics(self):
        test_metrics = {
            "ADE": [],
            "FDE": [],
            "ADE_world": [],
            "FDE_world": [],
            "NLL": [],
        }
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            "ADE": 1e9,
            "FDE": 1e9,
            "ADE_world": 1e9,
            "FDE_world": 1e9,
            "goal_BCE_loss": 1e9,
        }
        return best_metrics

    def best_valid_metric(self):
        return "FDE"

    def compute_model_losses(self,
                             outputs,
                             ground_truth,
                             loss_mask,
                             input_traj_maps,
                             aux_outputs):
        """
        Compute loss for a generic model.
        """
        out_maps_GT_goal = input_traj_maps[:, self.obs_length:]
        goal_logit_map = aux_outputs["goal_logit_map"]
        goal_logit_map_masked_out = aux_outputs["goal_logit_map_masked_out"]
        # check whether are all non-traj maps zero
        non_traj = torch.argwhere(torch.isinf(ground_truth[:, self.obs_length:][:, :, 0]))
        assert torch.all(input_traj_maps[non_traj[:, 0], non_traj[:, 1], :, :] == 0)
        loss_mask_np = loss_mask.detach().cpu().numpy() # NUMPY CHECK

        goal_BCE_loss = Goal_BCE_loss(
            goal_logit_map, out_maps_GT_goal, loss_mask, goal_logit_map_masked_out)

        if self.masked:
            mse_loss, ade_loss = MSE_loss_new(outputs, ground_truth, prediction_length=self.pred_length)
        else:
            mse_loss = MSE_loss(outputs, ground_truth, loss_mask.permute(1, 0))
            ade_loss = None
        losses = {
            "traj_MSE_loss": mse_loss,
            "goal_BCE_loss": goal_BCE_loss,
            "ADE": ade_loss
        }

        return losses

    def compute_model_metrics(self,
                              metric_name,
                              phase,
                              predictions,
                              ground_truth,
                              metric_mask,
                              all_aux_outputs,
                            #   scene,
                              obs_length=8):
        """
        Compute model metrics for a generic model.
        Return a list of floats (the given metric values computed on the batch)
        """
        if phase == 'test':
            compute_nll = self.compute_test_nll
            num_samples = self.num_test_samples
        elif phase == 'val':
            compute_nll = self.compute_valid_nll
            num_samples = self.num_valid_samples
        else:
            compute_nll = False
            num_samples = 1

        # scale back to original dimension
        predictions = predictions.detach() * self.down_factor
        ground_truth = ground_truth.detach() * self.down_factor
      
        pred_world = []
        image_res_x, image_res_y = 640, 640
        floorplan_min_x, floorplan_max_x = 0, 640
        floorplan_min_y, floorplan_max_y = 0, 640
        for i in range(predictions.shape[0]):
            pred_world.append(Scene_floorplan.make_world_coord_torch_static(predictions[i], image_res_x, image_res_y, floorplan_min_x, floorplan_min_y, floorplan_max_x, floorplan_max_y))
        pred_world = torch.stack(pred_world)

        GT_world = Scene_floorplan.make_world_coord_torch_static(ground_truth, image_res_x, image_res_y, floorplan_min_x, floorplan_min_y, floorplan_max_x, floorplan_max_y)

        if metric_name == 'ADE':
            return ADE_best_of(
                predictions, ground_truth, metric_mask, obs_length)
        elif metric_name == 'FDE':
            return FDE_best_of_goal(all_aux_outputs, ground_truth,
                                    metric_mask, down_factor=1)
        if metric_name == 'ADE_world':
            return ADE_best_of(
                pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'FDE_world':
            raise NotImplementedError('scene parameter is input to calculation, first check what is needed since scene passing is not easy')
            return FDE_best_of_goal_world(all_aux_outputs, scene,
                                          GT_world, metric_mask, down_factor=1)
        elif metric_name == 'NLL':
            if compute_nll and num_samples > 1:
                return KDE_negative_log_likelihood(
                    predictions, ground_truth, metric_mask, obs_length)
            else:
                return [0, 0, 0]
        else:
            raise ValueError("This metric has not been implemented yet!")

    def forward(self, batch, decoder_mode=0, num_samples=1):

        train = True if decoder_mode == 0 else False

        # inputs = {k: v.squeeze(0).float() for k, v in batch.items() if k not in ['scene_data', 'type']}
        images = batch['images']
        abs_coordinates = batch['coords'].squeeze(0) if 'coords' in batch else None
        tokens = batch['tokens'].squeeze(0) if 'tokens' in batch else None
        input_traj_maps  = batch['input_traj_maps'].squeeze(0) if 'input_traj_maps' in batch else None
        destinations = batch['destinations'].squeeze(0) if 'destinations' in batch else None
        
        # if self.traj_quantity == 'vel':
        #     for key in inputs.keys():
        #         if key in ['abs_pixel_coord', 'tensor_image', 'input_traj_maps']:
        #             raise NotImplementedError('check neglection of first element before running... and are all importantant keys considered?')
        #             inputs[key] = inputs[key][1:]

        batch_coords = abs_coordinates
        # seq_list =  np.ones((abs_pixel_coord.shape[0], abs_pixel_coord.shape[1]))
        # Number of agent in current batch_abs_world
        num_agents, seq_length, _ = abs_coordinates.shape

        self.device = batch_coords.device

        ##################
        # PREDICT GOAL
        ##################
        # extract precomputed map for goal goal_idx
        tensor_image = images.repeat(num_agents, 1, 1, 1)
        obs_traj_maps = input_traj_maps[:, 0:self.obs_length] if not self.masked else input_traj_maps
        if self.masked:
            inf_idx = torch.argwhere(torch.isinf(batch_coords[:, :, 0]))
            assert obs_traj_maps[inf_idx[:, 0], inf_idx[:, 1], :, :].sum() == 0

            obs_traj_maps_masked_out = torch.ones_like(obs_traj_maps, dtype=torch.bool, device=self.device)
            fut_traj_mask = torch.isfinite(batch_coords[:, self.obs_length:, 0]).long()
            no_agents_present_in_future = fut_traj_mask.sum(dim=1) == 0
            seq_starters_per_agent = torch.argmax(fut_traj_mask, dim=1) + self.obs_length
            if no_agents_present_in_future.sum() > 0:
                # no future trajectories for some agents
                seq_starters_per_agent[no_agents_present_in_future] = 0
            for agent_id, start_id in zip(torch.arange(num_agents), seq_starters_per_agent):
                # do not apply the mask (turn False) if trajectory is observed
                obs_traj_maps_masked_out[agent_id, :start_id, :, :] = False
                if start_id==0: assert torch.all(obs_traj_maps[agent_id, self.obs_length:, :, :] == 0) # if entire future traj is missing, all traj maps should be zero
                elif start_id!=self.obs_length: assert torch.all(obs_traj_maps[agent_id, :start_id, :, :] == 0) # if some future traj is missing, all traj maps before that should be zero
            obs_traj_maps = obs_traj_maps.masked_fill(obs_traj_maps_masked_out, 0)
            obs_traj_maps_masked_out_np = obs_traj_maps_masked_out.long().detach().cpu().numpy() # NUMPY CHECK
            seq_starters_per_agent_np = seq_starters_per_agent.detach().cpu().numpy() # NUMPY CHECK
            obs_traj_maps_np = obs_traj_maps.detach().cpu().numpy() # NUMPY CHECK
        else:
            obs_traj_maps_masked_out = None

        input_goal_module = torch.cat((tensor_image, obs_traj_maps), dim=1) # 20, 3, 1000, 1000 // 32, 8, 1000, 1000
        # compute goal maps
        goal_logit_map_start = self.goal_module(input_goal_module) # output: num_agenten, pred_length, res, res
        goal_prob_map = torch.sigmoid(
            goal_logit_map_start[:, -1:] / self.sampler_temperature)

        if not train:
            if self.use_ttst and num_samples > 3:
                goal_point_start = TTST_test_time_sampling_trick(
                    goal_prob_map,
                    num_goals=num_samples,
                    device=self.device)
                goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2)
            else:
                goal_point_start = sampling(goal_prob_map, num_samples=num_samples)
                goal_point_start = goal_point_start.squeeze(1)
            
            if self.normalize:
                goal_point_start = (goal_point_start - 320.) / 75.
        
        # START SAMPLES LOOP
        all_outputs = []
        all_aux_outputs = []
        
        assert num_samples == 1, 'for num_samples > 1: must be adapted in the losses (which are only adjusted for num_samples = 1)'
        for sample_idx in range(num_samples):
            # Output tensor of shape (seq_length,N,2)
            outputs = torch.zeros(num_agents, seq_length,
                                  self.output_size).to(self.device)
            # add observation as first output
            outputs[:, 0:self.obs_length] = batch_coords[:, 0:self.obs_length]

            # create noise vector to promote different trajectories
            noise = torch.randn((1, self.noise_size)).to(self.device)
            
            if train:
                # teacher forcing: learn with ground truth instead of predictions
                if torch.any(torch.isinf(batch_coords[:, -1])):
                    lasties_np = batch_coords[:, -1].detach().cpu().numpy() # NUMPY CHECK
                    future_coords = batch_coords[:, self.obs_length:]
                    non_inf_indices = torch.where(torch.isfinite(future_coords[:, :, 0]))

                    # Get the last non-inf index along axis=1
                    unique_x = torch.unique(non_inf_indices[0])
                    # Find the highest y value for each unique x
                    max_y_for_each_x = torch.maximum.reduceat(non_inf_indices[1], torch.searchsorted(non_inf_indices[0], unique_x))
                    if unique_x.shape[0] != num_agents:
                        missing_idx = torch.where(~torch.isin(max_y_for_each_x, torch.arange(num_agents)))[0]
                        missing_idx_np = missing_idx.detach().cpu().numpy() # NUMPY CHECK
                        unique_x = torch.cat((unique_x, missing_idx))
                        max_y_for_each_x = torch.cat((max_y_for_each_x, torch.full((missing_idx.shape[0],), torch.inf)))
                        unique_x, indices = torch.sort(unique_x)
                        max_y_for_each_x = max_y_for_each_x[indices]

                    # Display the result
                    goal_point = torch.column_stack((unique_x, max_y_for_each_x))
                else:
                    goal_point = batch_coords[:, -1]
            else:
                goal_point = goal_point_start[:, sample_idx]

            # list of auxiliary outputs
            aux_outputs = {
                "goal_logit_map": goal_logit_map_start,
                "goal_point": goal_point,
                "goal_logit_map_masked_out": obs_traj_maps_masked_out}

            ##################
            # loop over seq_length-1 frames, starting from frame 8
            ##################
            for frame_idx in range(self.obs_length, self.seq_length):
                # If testing phase and frame >= obs_length (prediction)
                if not train and frame_idx >= self.obs_length:
                    # Get current agents positions: from 0 to obs_length take
                    # GT, then previous predicted positions
                    current_agents = torch.cat((
                        batch_coords[:, :self.obs_length],
                        outputs[:, self.obs_length:frame_idx]), dim=1)
                else:  # Train phase or frame < obs_length (observation)
                    # Current agents positions
                    current_agents = batch_coords[:, :frame_idx]

                ##################
                # RECURRENT MODULE
                ##################
                # TODO check all use cases: what if traj begins after obs_length?

                non_traj_mask = torch.isinf(current_agents) if self.masked else None
                if self.masked: assert torch.all(non_traj_mask[:,:,0] == non_traj_mask[:,:,1])
                current_agents = current_agents.masked_fill(non_traj_mask, 0) if self.masked else current_agents

                # Input Embedding
                temporal_input_embedded = self.dropout_input_temporal(  # deleted relu here and turned bias = 0
                    self.input_embedding_layer_temporal(current_agents)) # nn.Linear(self.input_size, self.embedding_size)

                # compute current positions and current time step
                # and distance to goal
                last_positions = current_agents[:, -1]
                current_time_step = torch.full(size=(last_positions.shape[0], 1),
                                               fill_value=frame_idx).to(self.device)
                # cancel out all agents' last positions that are infinite (not there any more due to destination reached)
                goal_point_curr_frame = goal_point.masked_fill(non_traj_mask[:, -1], 0)
                current_time_step = current_time_step.masked_fill(non_traj_mask[:, -1, 0].unsqueeze(-1), 0)
                distance_to_goal = goal_point_curr_frame - last_positions
                last_positions_np, goal_point_curr_frame_np, distance_to_goal_np, current_time_step_np = last_positions.detach().cpu().numpy(), goal_point_curr_frame.detach().cpu().numpy(), distance_to_goal.detach().cpu().numpy(), current_time_step.detach().cpu().numpy() # SOME NUMPY CHECKS
                assert torch.all((goal_point_curr_frame[:, 0].unsqueeze(-1) == 0) == (current_time_step == 0))
                assert torch.all((distance_to_goal == 0) == (last_positions == 0))
                assert torch.all((distance_to_goal == 0) == (goal_point_curr_frame == 0))
                assert torch.all(torch.isfinite(distance_to_goal)) and torch.all(torch.isfinite(current_time_step)) and torch.all(torch.isfinite(last_positions)) and torch.all(torch.isfinite(goal_point_curr_frame))
                # prepare everything for concatenation
                # Transformers need everything to be multiple of nhead
                last_positions_to_cat = last_positions.repeat(
                    frame_idx, 1, self.nhead//2)
                current_time_step_to_cat = current_time_step.repeat(
                    frame_idx, 1, self.nhead)
                final_positions_pred_to_cat = goal_point.repeat(
                    frame_idx, 1, self.nhead//2)
                distance_to_goal_to_cat = distance_to_goal.repeat(
                    frame_idx, 1, self.nhead//2)

                # concat additional info BEFORE temporal transformer
                temporal_input_cat = torch.cat(
                    (temporal_input_embedded.permute(1, 0, 2),
                     final_positions_pred_to_cat,
                     last_positions_to_cat,
                     distance_to_goal_to_cat,
                     current_time_step_to_cat,
                     ), dim=2)

                # temporal transformer encoding
                encoder_mask = non_traj_mask[:, :, 0]
                encoder_mask_float = encoder_mask.float().masked_fill(non_traj_mask[:, :, 0], -1e25) if self.masked else None # cant do boolean mask as NaN would be returned through softmax(-torch.inf) error: https://github.com/pytorch/pytorch/issues/25110

                assert torch.all(torch.isfinite(temporal_input_cat))
                temporal_output = self.temporal_encoder(
                    temporal_input_cat, src_key_padding_mask=encoder_mask_float)
                
                # temporal_output = temporal_output.masked_fill(check_inf_encoder, 0)
                # Take last temporal encoding
                temporal_output_last = temporal_output[-1]

                # concat additional info AFTER temporal transformer
                fusion_feat = torch.cat((
                    temporal_output_last,
                    last_positions,
                    goal_point,
                    distance_to_goal,
                    current_time_step,
                ), dim=1)

                # fusion FC layer
                fusion_feat = self.fusion_layer(fusion_feat)

                if self.add_noise_traj:
                    # Concatenate noise to fusion output
                    noise_to_cat = noise.repeat(fusion_feat.shape[0], 1)
                    fusion_feat = torch.cat((fusion_feat, noise_to_cat), dim=1)

                # Output FC decoder
                outputs_current = self.output_layer(fusion_feat)
                # write inf back to outputs and inf to agents that reached their destination
                outputs_current = outputs_current.masked_fill(non_traj_mask[:, -1], torch.inf) if self.masked else outputs_current
                outputs_current = check_agents_in_destination(outputs_current, destinations)
                # insert all agents that are initialized after self.obs_length
                if not train:
                    curr_gt_frame_mask = torch.isfinite(batch_coords[:, frame_idx])[:, 0]
                    prev_gt_frame_mask = torch.isfinite(batch_coords[:, frame_idx-1])[:, 0]
                    # find idx where prev_gt_frame_mask is true and curr_gt_frame_mask is false
                    new_agent_coord_mask = torch.logical_and(torch.logical_not(prev_gt_frame_mask), curr_gt_frame_mask)
                    if new_agent_coord_mask.sum() > 0:
                        # get all True indices from new_agent_coord_mask
                        new_agent_idx = new_agent_coord_mask.detach().cpu().nonzero(as_tuple=True)[0]
                        print(f'Found new agents {new_agent_idx} in frame {frame_idx}.')
                        new_agent_inits = batch_coords[new_agent_coord_mask, frame_idx, :]
                        old_agent_spots = batch_coords[new_agent_coord_mask, frame_idx-1, :]
                        outputs_current[new_agent_coord_mask, :] = batch_coords[new_agent_coord_mask, frame_idx, :]
                        new_agent_inits_np = new_agent_inits.detach().cpu().numpy() # SOME NUMPY CHECKS
                        old_agent_spots_np = old_agent_spots.detach().cpu().numpy() # SOME NUMPY CHECKS
                        assert torch.all(torch.isfinite(new_agent_inits)) and torch.all(torch.isinf(old_agent_spots)) and torch.all(torch.isfinite(outputs_current[new_agent_coord_mask, :]))

                # append to outputs
                outputs[:, frame_idx, :] = outputs_current

            all_outputs.append(outputs)
            all_aux_outputs.append(aux_outputs)

        # stack predictions
        all_outputs = torch.stack(all_outputs)
        # from list of dict to dict of list (and then tensors)
        all_aux_outputs = {k: torch.stack([d[k] for d in all_aux_outputs])
                           for k in all_aux_outputs[0].keys()}
        return all_outputs, all_aux_outputs


class TrajectoryEmbedding(nn.Module):
    def __init__(self, coord_dims, tokens, dim, data_format='') -> None:
        super().__init__()
        self.coord_dims = coord_dims
        self.tokens = tokens
        self.dim = dim
        self.data_format = data_format
        self.check_for_fill = torch.isinf

        self.embed_coord = nn.Linear(coord_dims, dim)
        if self.tokens is not None:           
            self.embed_token = nn.Embedding(len(tokens), dim)
        
        if self.data_format == 'full_tokenized_by_frame':
            raise NotImplementedError
            self.coord_network = nn.Sequential(nn.Linear(10, 100), nn.Linear(100, 1))
            # every x- and y-digit with the same weight/influence as [BOS], [EOS], ... does not make sense --> full/partial hybrid?: same but no regression problem, but actually classification

    def tokenize(self, token):
        return self.tokens.index(token)


    def forward(self, obs, all_tokens):

        if self.data_format == 'full_tokenized_by_frame':
            # get all traj tokens
            is_traj_mask = torch.argwhere(all_tokens[:,:,1] != -1)
            traj_tokens = all_tokens[is_traj_mask[:,0], is_traj_mask[:,1], :]
            assert torch.all(traj_tokens != -1)
            # get all non traj tokens
            is_not_traj_mask = torch.argwhere(all_tokens[:,:,1] == -1)
            non_traj_tokens = all_tokens[is_not_traj_mask[:,0], is_not_traj_mask[:,1], 0]
            assert torch.all(all_tokens[is_not_traj_mask[:,0], is_not_traj_mask[:,1], 1:] == -1)

            traj_tokens = self.embed_token(traj_tokens)
            non_traj_tokens = self.embed_token(non_traj_tokens)

            traj_tokens = self.coord_network(traj_tokens.permute(0,2,1)).squeeze(2)

            # is_traj_mask_np = is_traj_mask.detach().cpu().numpy()
            # is_not_traj_mask_np = is_not_traj_mask.detach().cpu().numpy()
            tokens_tensor = torch.full((all_tokens.shape[0], all_tokens.shape[1], self.dim), fill_value=float('-inf'), dtype=torch.float32, device=all_tokens.device)

            # Fill the tensor using indices from indices1 and indices2
            tokens_tensor[is_traj_mask[:, 0], is_traj_mask[:, 1], :] = traj_tokens
            tokens_tensor[is_not_traj_mask[:, 0], is_not_traj_mask[:, 1], :] = non_traj_tokens

            assert torch.all(torch.isfinite(tokens_tensor))
            return tokens_tensor

        is_traj_mask = self.check_for_fill(obs)[:,:,0]
        
        # TODO turn off assertions after one batch
        if all_tokens is not None:
            assert torch.equal(self.check_for_fill(obs)[:,:,0], self.check_for_fill(obs)[:,:,1]), 'obs masks on x- and y-direction need to be equal!'
            # assert torch.equal(is_traj_mask.unsqueeze(-1).repeat(1, 1, self.dim)[:,:,0], all_tokens.ne(4)), 'non-4-mask in tokens needs to be equal to inf mask in traj!'
        else:
            assert torch.count_nonzero(self.check_for_fill(obs)) == 0, 'If no tokens supplied, obs cannot contain any inf values!'
        
        obs = obs.masked_fill(is_traj_mask.unsqueeze(-1).repeat(1, 1, obs.size(-1)), 0)
        coord_emb = self.embed_coord(obs)
        # check inf coords before and after pass
        assert torch.all(torch.isinf(coord_emb)[:,:,0] == torch.isinf(obs)[:,:,0]) and torch.all(torch.isinf(coord_emb)[:,:,443] == torch.isinf(obs)[:,:,1])

        if self.tokens is not None:
            token_emb = self.embed_token(all_tokens)

            # only mask out non-trajectories, but not TRAJ token (since it needs to be learned)
            coord_emb = coord_emb.masked_fill(is_traj_mask.unsqueeze(-1).repeat(1, 1, self.dim), 0)
            # token_emb = token_emb.masked_fill(~is_traj_mask, 0)

            output = coord_emb + token_emb

            # diff = time.process_time() - start
            # print(f'Forward pass time with inf: {diff} s with {torch.count_nonzero(torch.isinf(obs)[:,:,0])} Falses')
        else:
            output = coord_emb
        
        return output
    

class ToLogitsNet(nn.Module):
    def __init__(self, dim, coord_dims, dict_size) -> None:
        super().__init__()
        self.dim = dim
        self.coord_dims = coord_dims
        self.dict_size = dict_size

        self.to_token_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dict_size, bias=False)
        )
        self.to_coord_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, coord_dims, bias=False)
        )
    
    def forward(self, x):
        token_logits = self.to_token_logits(x)
        coord_logits = self.to_coord_logits(x)
        return torch.cat((coord_logits, token_logits), dim=-1)