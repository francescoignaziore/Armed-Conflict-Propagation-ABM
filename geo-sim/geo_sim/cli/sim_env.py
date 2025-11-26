from geo_sim.cli.env import CSSMovementsEnv
# Obs constants
from geo_sim.config.env import KEY_OBS_FEATURES_LOCATION, KEY_OBS_FEATURES_GROUP, KEY_OBS_OCCUPANCY
# Action constants
from geo_sim.config.env import ACTION_VALUE_DEFAULT, MASK_RADIUS
env = CSSMovementsEnv()

observations, infos = env.reset()

def group_policy(obs, info, group):
    cell_occupancy = obs[KEY_OBS_OCCUPANCY] # (G, H, W), np.float32 in [0,1]
    cell_features = obs[KEY_OBS_FEATURES_LOCATION] # (F_W, H, W), np.float32
    group_features = obs[KEY_OBS_FEATURES_GROUP] # (G, F_G), np.float32
    i_group = env._agent_index(group) # int

    # Default: Assign 1s to all movement direction logits.
    # For occupied cells, these defaults will be overriden 
    action_space = env.action_space(group) # (H, W, 5), np.float32
    action = np.full(action_space.shape, ACTION_VALUE_DEFAULT, dtype=np.float32)

    occupied_by_us = cell_occupancy[i_group] > 0 # (H,W), bool
    H,W = occupied_by_us.shape

    # Zero padding for bound-safe observation masking
    F_W = cell_features.shape[0]
    pad = MASK_RADIUS
    mask_diameter = 2 * MASK_RADIUS + 1
    # D_MASK = 2*MASK_RADIUS + 1
    
    cell_occupancy_padded = np.pad(
        cell_occupancy,
        pad_width=((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )
    cell_features_padded = np.pad(
        cell_features,
        pad_width=((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )

    for h in range(H):
        for w in range(W):
            if not occupied_by_us[h][w]:
                continue
            
            hp = h + pad
            wp = w + pad

            masked_occupancy = cell_occupancy_padded[
                :,
                hp - pad : hp + pad + 1,
                wp - pad : wp + pad + 1,
            ] # (G, D_MASK, D_MASK)

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

    # Dummy behaviour: do nothing
    pass

done = False
while True:
    # 1. Get actions from each agent’s policy
    actions = {}
    base_group = env.agents[0]
    obs = observations[base_group]
    info = infos[base_group]
    
    occupancy_mask = occupancy > 0 # (G, H, W) in {0,1}
    is_occupied = np.sum(, axis=0)  # (H, W)
    assert len(occupancy.shape) == 3, occupancy.shape
    G,H,W = occupancy.shape
    for x in range(H):
        for y in range(W):


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
