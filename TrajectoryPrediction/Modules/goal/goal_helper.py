import numpy as np
import torch
import torch.nn as nn
from Modules.goal.models.model_utils import kmeans
import torchvision.ops as ops

def argmax_over_map(x):
    """
    From probability maps of shape (B, T, H, W), extract the
    coordinates of the maximum values (i.e. argmax).
    Hint: you need to use numpy.amax
    Output shape is (B, T, 2)
    """

    def indexFunc(array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx

    B, T, _, _ = x.shape
    device = x.device
    x = x.detach().cpu().numpy()
    maxVals = np.amax(x, axis=(2, 3))
    max_indices = np.zeros((B, T, 2), dtype=np.int64)
    for index in np.ndindex(x.shape[0], x.shape[1]):
        max_indices[index] = np.asarray(
            indexFunc(x[index], maxVals[index]), dtype=np.int64)[::-1]
    max_indices = torch.from_numpy(max_indices)
    return max_indices.to(device)


def MSE_loss_new(outputs, ground_truth, prediction_length):
    """
    Compute an averaged Mean-Squared-Error, only on the positions
    in which loss_mask is True.
    outputs.shape = num_samples, seq_length, num_agent, n_coords(2)
    """
    assert outputs.shape[0] == 1, "Only 1 sample implemented!"
    outputs = outputs.squeeze(0)[:, -prediction_length:]
    ground_truth = ground_truth[:, -prediction_length:]

    inf_mask_outputs = torch.isinf(outputs)
    inf_mask_ground_truth = torch.isinf(ground_truth)
    inf_mask_outputs_np, inf_mask_ground_truth_np = inf_mask_outputs.detach().cpu().numpy(), inf_mask_ground_truth.detach().cpu().numpy() # NUMPY CHECK
    # x & y must both be fininte or infinite
    assert torch.all(inf_mask_ground_truth[:,:,0] == inf_mask_ground_truth[:,:,1])
    assert torch.all(inf_mask_outputs[:,:,0] == inf_mask_outputs[:,:,1])
    # both outputs and ground_truth must start their trajectories at the same time
    seq_starters_per_agent_out = torch.argmax(torch.isfinite(outputs).long(), dim=1) 
    seq_starters_per_agent_gt = torch.argmax(torch.isfinite(ground_truth).long(), dim=1) 
    assert torch.all(seq_starters_per_agent_out == seq_starters_per_agent_gt)

    # assert that all newly initialized agents are equal for ground_truth and outputs
    assert torch.all(torch.argmax((~inf_mask_outputs[:,:,0]).long(), dim=1) == torch.argmax((~inf_mask_ground_truth[:,:,0]).long(), dim=1))

    # both outputs and ground_truth are inf
    inf_mask = torch.logical_and(inf_mask_outputs, inf_mask_ground_truth)
    no_seq_per_agent = (~inf_mask[:,:,0]).sum(dim=-1) == 0 # no coords present over entire sequence per agent

    # if output ends earlier than gt, use last distance (and vice versa)
    # revert inf_mask_outputs and inf_mask_ground_truth along dim=1
    last_finite_coord_outputs_mask = (~inf_mask_outputs[:,:,0]).flip(dims=[1]).long()
    last_finite_coord_outputs = torch.argmax(last_finite_coord_outputs_mask, dim=1)
    last_finite_coord_ground_truth_mask = (~inf_mask_ground_truth[:,:,0]).flip(dims=[1]).long()
    last_finite_coord_ground_truth = torch.argmax(last_finite_coord_ground_truth_mask, dim=1)
    last_finite_coord_outputs_mask_np, last_finite_coord_ground_truth_mask_np = last_finite_coord_outputs_mask.detach().cpu().numpy(), last_finite_coord_ground_truth_mask.detach().cpu().numpy() # NUMPY CHECK

    # fill all inf values along dim=1 with the last finite value
    valid_agents = torch.arange(outputs.shape[0])[~no_seq_per_agent]
    for agent_id in valid_agents:
        outputs_old_np, ground_truth_old_np = outputs.detach().cpu().numpy(), ground_truth.detach().cpu().numpy() # NUMPY CHECK
        if last_finite_coord_outputs[agent_id] != 0:
            outputs[agent_id, (-last_finite_coord_outputs[agent_id]-1):, :] = outputs[agent_id, (-last_finite_coord_outputs[agent_id]-1), :]
        
        if last_finite_coord_ground_truth[agent_id] != 0:
            ground_truth[agent_id, (-last_finite_coord_ground_truth[agent_id]-1):, :] = ground_truth[agent_id, (-last_finite_coord_ground_truth[agent_id]-1), :]
        outputs_new_np, ground_truth_new_np = outputs.detach().cpu().numpy(), ground_truth.detach().cpu().numpy() # NUMPY CHECK

    assert torch.all(inf_mask == torch.isinf(outputs - ground_truth)), "there should not be any infs except the ones that are in both outputs and ground_truth"
    non_inf_error = (outputs - ground_truth).masked_fill(inf_mask, 0)

    assert not torch.any(torch.isinf(non_inf_error))

    squared_error = non_inf_error**2
    # sum over coordinates --> Shape becomes: num_samples * seq_len * n_agents
    loss = torch.sum(squared_error, dim=-1)

    # Take a weighted loss, but only on places where loss_mask=True.
    # Divide by loss_mask.sum() instead of seq_len*N_pedestrians (or mean).
    # This is an average loss per time-step per pedestrian.
    denominator = (~inf_mask).sum()
    loss = loss.sum() / denominator

    # # minimum loss over samples (only 1 sample during training)
    # loss, _ = loss.min(dim=0)
    outputs_clone = outputs.clone().detach() * 75 + 320 # denormalize
    ground_truth_clone = ground_truth.clone().detach() * 75 + 320 # denormalize
    ade_loss = (outputs_clone - ground_truth_clone).masked_fill(inf_mask, 0)
    ade_loss = ade_loss.sum() / denominator

    return loss, ade_loss


def MSE_loss(outputs, ground_truth, loss_mask):
    """
    Compute an averaged Mean-Squared-Error, only on the positions
    in which loss_mask is True.
    outputs.shape = num_samples, seq_length, num_agent, n_coords(2)
    """
    squared_error = (outputs - ground_truth)**2
    # sum over coordinates --> Shape becomes: num_samples * seq_len * n_agents
    squared_error = torch.sum(squared_error, dim=-1)

    # compute error only where mask is True
    loss = squared_error * loss_mask

    # Take a weighted loss, but only on places where loss_mask=True.
    # Divide by loss_mask.sum() instead of seq_len*N_pedestrians (or mean).
    # This is an average loss per time-step per pedestrian.
    loss = loss.sum(dim=-1).sum(dim=-1) / loss_mask.sum()

    # minimum loss over samples (only 1 sample during training)
    loss, _ = loss.min(dim=0)

    return loss


def Goal_BCE_loss(logit_map, goal_map_GT, loss_mask, goal_logit_map_masked_out):
    """
    Compute the Binary Cross-Entropy loss for the probability distribution
    of the goal. Prediction and GT are two pixel maps.
    """
    losses_samples = []
    for logit_map_sample_i in logit_map:
        loss = BCE_loss_sample(logit_map_sample_i, goal_map_GT, loss_mask, goal_logit_map_masked_out)
        losses_samples.append(loss)
    losses_samples = torch.stack(losses_samples)

    # minimum loss over samples (only 1 sample during training)
    loss, _ = losses_samples.min(dim=0)

    return loss


def BCE_loss_sample(logit_map, goal_map_GT, loss_mask, goal_logit_map_masked_out):
    """
    Compute the Binary Cross-Entropy loss for the probability distribution
    of the goal maps. logit_map is a logit map, goal_map_GT a probability map.
    """
    batch_size, T, H, W = logit_map.shape
    # reshape across space and time
    if goal_logit_map_masked_out is not None:
        applied_map_mask = goal_logit_map_masked_out.squeeze(0)[:, -logit_map.shape[1]:]
        applied_map_mask_np = applied_map_mask.detach().cpu().numpy() # NUMPY CHECK
        logit_map = logit_map.masked_fill(~applied_map_mask, 0)
        goal_map_GT = goal_map_GT.masked_fill(~applied_map_mask, 0)
    output_reshaped = logit_map.view(batch_size, -1)
    target_reshaped = goal_map_GT.view(batch_size, -1)

    # takes as input computed logit and GT probabilities
    BCE_criterion = nn.BCEWithLogitsLoss(reduction='sum')

    # compute the Goal CE loss for each agent and sample
    loss = BCE_criterion(output_reshaped, target_reshaped)
    num_valid_cells = applied_map_mask.sum()
    loss /= num_valid_cells

    # Mean over maps (T, W, H)
    # loss = loss.mean(dim=-1)

    # # Take a weighted loss, but only on places where loss_mask=True
    # # Divide by full_agents.sum() instead of seq_len*N_pedestrians (or mean)
    # full_agents = loss_mask[-1]
    # loss = (loss * full_agents).sum(dim=0) / full_agents.sum()

    return loss


def sampling(probability_map,
             num_samples=10000,
             rel_threshold=0.05,
             replacement=True):
    """Given probability maps of shape (B, T, H, W) sample
    num_samples points for each B and T"""
    # new view that has shape=[batch*timestep, H*W]
    prob_map = probability_map.view(probability_map.size(0) * probability_map.size(1), -1)
    if rel_threshold is not None:
        # exclude points with very low probability
        thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(-1, prob_map.size(1))
        mask = prob_map < thresh_values * rel_threshold
        prob_map = prob_map * (~mask).int()
        prob_map = prob_map / prob_map.sum()

    # samples.shape=[batch*timestep, num_samples]
    samples = torch.multinomial(prob_map,
                                num_samples=num_samples,
                                replacement=replacement)

    # unravel sampled idx into coordinates of shape [batch, time, sample, 2]
    samples = samples.view(probability_map.size(0), probability_map.size(1), -1)
    idx = samples.unsqueeze(3)
    preds = idx.repeat(1, 1, 1, 2).float()
    preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
    preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))
    return preds


def TTST_test_time_sampling_trick(x, num_goals, device):
    """
    From a probability map of shape (B, 1, H, W), sample num_goals
    goals so that they cover most of the space (thanks to k-means).
    Output shape is (num_goals, B, 1, 2).
    """
    assert x.shape[1] == 1
    # first sample is argmax sample
    num_clusters = num_goals - 1
    goal_samples_argmax = argmax_over_map(x)

    # sample a large amount of goals to be clustered
    goal_samples = sampling(x[:, 0:1], num_samples=10000)
    # from (B, 1, num_samples, 2) to (num_samples, B, 1, 2)
    goal_samples = goal_samples.permute(2, 0, 1, 3)

    # Iterate through all person/batch_num, as this k-Means implementation
    # doesn't support batched clustering
    goal_samples_list = []
    for person in range(goal_samples.shape[1]):
        goal_sample = goal_samples[:, person, 0]

        # Actual k-means clustering, Outputs:
        # cluster_ids_x -  Information to which cluster_idx each point belongs
        # to cluster_centers - list of centroids, which are our new goal samples
        cluster_ids_x, cluster_centers = kmeans(X=goal_sample,
                                                num_clusters=num_clusters,
                                                distance='euclidean',
                                                device=device, tqdm_flag=False,
                                                tol=0.001, iter_limit=1000)
        goal_samples_list.append(cluster_centers)

    goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
    goal_samples = torch.cat([goal_samples_argmax.unsqueeze(0), goal_samples],
                             dim=0)
    return goal_samples


def check_agents_in_destination(agent_coords, destinations):
    agents_x, agents_y = agent_coords[:, 0], agent_coords[:, 1]
    agents_within_destinations = torch.zeros_like(agents_x, dtype=torch.bool)
    agent_coords_cp = agent_coords.clone()

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
        print(f'\n{agents_within_destinations.sum()} new agents within destinations.')
        non_traj_mask = torch.logical_or(agents_within_destinations, torch.isinf(agent_coords[:, 0]))
        agent_coords[non_traj_mask, :] = torch.full((2,), float('inf'), dtype=torch.float32, device=agent_coords.device)

    return agent_coords
