import numpy as np
from geo_sim.cli.env import CSSMovementsEnv
# Obs constants
from geo_sim.config.env import KEY_OBS_FEATURES_LOCATION, KEY_OBS_FEATURES_GROUP, KEY_OBS_OCCUPANCY
# Action constants
from geo_sim.config.env import ActionIdx, ACTION_VALUE_DEFAULT, MASK_RADIUS
env = CSSMovementsEnv()
# Features
from geo_sim.config.features import GeoFeatureIdx
## Feature accumulation kernels
from geo_sim.cli.spatial_accumulation import (
    get_gauss_kernel,
    get_gauss_dijkstra_kernel
)

observations, infos = env.reset()

def group_policy(obs, info, group, compute_padding=True):
    cell_occupancy = obs[KEY_OBS_OCCUPANCY]          # (G, H, W), np.float32 in [0,1]
    cell_features  = obs[KEY_OBS_FEATURES_LOCATION]  # (F_W, H, W), np.float32
    group_features = obs[KEY_OBS_FEATURES_GROUP]     # (G, F_G), np.float32 (per-group features)
    i_group = env._agent_index(group)                # int

    # Default: Assign ACTION_VALUE_DEFAULT to all movement direction logits.
    # For occupied cells, these defaults will be overriden 
    action_space = env.action_space(group) # (H, W, 5), np.float32
    action = np.full(action_space.shape, ACTION_VALUE_DEFAULT, dtype=np.float32)

    occupied_by_us = cell_occupancy[i_group] > 0 # (H,W), bool
    if not np.any(occupied_by_us):
        # Group doesn't exist anymore
        return action
    
    # Zero padding for bound-safe observation masking
    pad = MASK_RADIUS
    
    # TODO(L): Padding is already precomputed
    # cell_occupancy_padded = env.world_occupancy_padded
    cell_occupancy_padded = np.pad(
        cell_occupancy,
        pad_width=((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )
    # cell_features_padded = env.world_occupancy_padded
    cell_features_padded = np.pad(
        cell_features,
        pad_width=((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )

    occupied_positions = np.argwhere(occupied_by_us)
    
    for h,w in occupied_positions:
        hp = h + pad
        wp = w + pad

        masked_occupancy = cell_occupancy_padded[
            :,
            hp - pad : hp + pad + 1,
            wp - pad : wp + pad + 1,
        ] # (G, D_MASK, D_MASK)
        # D_MASK = 2*MASK_RADIUS + 1

        masked_features = cell_features_padded[
            :,
            hp - pad : hp + pad + 1,
            wp - pad : wp + pad + 1,
        ] # (F_W, D_MASK, D_MASK)

        # Override default action with actual behaviour
        local_group_policy(
            action[h, w], masked_occupancy, masked_features, group_features, i_group, h, w
        )

    return action

def local_group_policy(action, masked_occupancy, masked_features, group_features, i_group, h, w):
    # action: shape (5), write your action there
    # masked_occupancy: (G, D_MASK, D_MASK), masked occupancy information
    # masked_features: (F_W, D_MASK, D_MASK), masked location features
    # group_features: (G, F_G), all group features
    # i_group: int32, current group index
    # h: int32, current x location on the grid
    # w: int32, current y location on the grid

    D_MASK = 2*MASK_RADIUS + 1
    coords_centered = np.arange(D_MASK) - MASK_RADIUS
    I, J = np.meshgrid(coords_centered, coords_centered, indexing="ij")
    diag1 = I+J
    diag2 = I-J

    direction_masks = np.zeros((4,D_MASK,D_MASK),dtype=bool)
    direction_masks[ActionIdx.UP]    = (diag1<=0) & (diag2<=0)
    direction_masks[ActionIdx.LEFT]  = (diag1<=0) & (diag2>=0)
    direction_masks[ActionIdx.RIGHT] = (diag1>=0) & (diag2<=0)
    direction_masks[ActionIdx.DOWN]  = (diag1>=0) & (diag2>=0)
    # The above vectorized version is equivalent to this scalar code:
    # for i in range(D_MASK):
    #     for j in range(D_MASK):
    #         i_centered = i - MASK_RADIUS
    #         j_centered = j - MASK_RADIUS
    #         # i=0, j=0 is top left, i is column (h dim), j is row (w dim)
    #         direction_masks[ActionIdx.UP,   i,j] = (i_centered+j_centered < 0) and (i_centered-j_centered < 0)
    #         direction_masks[ActionIdx.LEFT, i,j] = (i_centered+j_centered < 0) and (i_centered-j_centered > 0)
    #         direction_masks[ActionIdx.RIGHT,i,j] = (i_centered+j_centered > 0) and (i_centered-j_centered < 0)
    #         direction_masks[ActionIdx.DOWN, i,j] = (i_centered+j_centered > 0) and (i_centered-j_centered > 0)

    # Depreciate resources by distance
    K = get_gauss_kernel(MASK_RADIUS) # (D_MASK, D_MASK)
    
    # Consider occupied resources    
    resources = masked_features[GeoFeatureIdx.RESOURCES] # (D_MASK, D_MASK)
    G = masked_occupancy.shape[0]
    occupancy_total = np.sum(masked_occupancy, axis=0) # (D_MASK, D_MASK)
    unoccupied = 1-occupancy_total # (D_MASK, D_MASK)
    resources_unoccupied = K * unoccupied * resources
    
    # Reset all actions to 0
    action[:] = 0
    our_strength = env._get_group_strength_local(i_group, h, w, compute_padding=False)

    ## Onccupied resources are an attractor
    action[ActionIdx.UP]    += np.sum(direction_masks[ActionIdx.UP]    * resources_unoccupied) 
    action[ActionIdx.RIGHT] += np.sum(direction_masks[ActionIdx.RIGHT] * resources_unoccupied) 
    action[ActionIdx.LEFT]  += np.sum(direction_masks[ActionIdx.LEFT]  * resources_unoccupied) 
    action[ActionIdx.DOWN]  += np.sum(direction_masks[ActionIdx.DOWN]  * resources_unoccupied) 
    # STAY is just the average across directions
    action[ActionIdx.STAY]  += np.sum(0.25                             * resources_unoccupied) 
    
    ## Occupied resources attract/detract based on the other group's relative strength
    for g in range(G):
        if g == i_group:
            continue
        their_strength = env._get_group_strength_local(g, h, w, compute_padding=False)
        attraction_weight = _get_group_strength_ratio_weight(their_strength, our_strength) # (-infty, 1]
        their_resources = resources * masked_occupancy[g] # (D_MASK, D_MASK)
        their_resources_weighted = K * their_resources * attraction_weight
        action[ActionIdx.UP]    += np.sum(direction_masks[ActionIdx.UP]    * their_resources_weighted) 
        action[ActionIdx.RIGHT] += np.sum(direction_masks[ActionIdx.RIGHT] * their_resources_weighted) 
        action[ActionIdx.LEFT]  += np.sum(direction_masks[ActionIdx.LEFT]  * their_resources_weighted) 
        action[ActionIdx.DOWN]  += np.sum(direction_masks[ActionIdx.DOWN]  * their_resources_weighted) 
        # STAY is just the average across directions
        action[ActionIdx.STAY]  += np.sum(0.25                             * their_resources_weighted)

    _convert_to_action_probabilities(action)

    return action

def _convert_to_action_probabilities(action):
    # Redirect negative action flows into the opposite direction
    action_weights = np.copy(action)
    action_weights[ActionIdx.RIGHT] -= min(0.0, action[ActionIdx.LEFT])
    action_weights[ActionIdx.LEFT]  -= min(0.0, action[ActionIdx.RIGHT])
    action_weights[ActionIdx.UP]    -= min(0.0, action[ActionIdx.DOWN])
    action_weights[ActionIdx.DOWN]  -= min(0.0, action[ActionIdx.UP])

    action_weights[ActionIdx.STAY]  = max(0.0, action[ActionIdx.STAY]) 
    action_weights[ActionIdx.RIGHT] = max(0.0, action_weights[ActionIdx.RIGHT])
    action_weights[ActionIdx.LEFT]  = max(0.0, action_weights[ActionIdx.LEFT])
    action_weights[ActionIdx.UP]    = max(0.0, action_weights[ActionIdx.UP])
    action_weights[ActionIdx.DOWN]  = max(0.0, action_weights[ActionIdx.DOWN])

    # Linear normalization
    action_weight_sum = np.sum(action)
    action[:] /= action_weight_sum

def _get_group_strength_ratio_weight(strength_g1, strength_g2):
    return 1.0 - strength_g1/strength_g2

done = False
while True:
    # 1. Get actions from each agent’s policy
    actions = {}
    base_group = env.agents[0]
    obs = observations[base_group]
    info = infos[base_group]

    # Precompute padding once
    env._prepare_padded_views(pad=MASK_RADIUS)
    for agent in env.agents:              
        obs = observations[agent]
        info = infos[agent]
        actions[agent] = group_policy(obs, info, agent)

    # 2. Step the environment with those actions
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # 3. Optional: render / log / train
    env.render()

    # 4. Check if episode is over
    if any(terminations.values()) or all(truncations.values()):
        break
