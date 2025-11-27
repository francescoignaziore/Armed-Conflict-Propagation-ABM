import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from pettingzoo import ParallelEnv

from geo_sim.config.env import SEED
from geo_sim.config.env import MAX_INIT_VAL
from geo_sim.config.env import KEY_OBS_FEATURES, KEY_OBS_OCCUPANCY, KEY_OBS_FEATURES_GROUP
from geo_sim.config.env import MAX_OCCUPANCY_GAIN
from geo_sim.config.features import FEATURES_SPEC, FeatureKey, GeoFeatureIdx, GrpFeatureIdx, GeoGrpFeatureIdx

class CSSMovementsEnv(ParallelEnv):
    metadata = {
        "name": "css_movements_v0",
    }

    def __init__(self, G, H, W, seed=SEED):
        """
        G   : number of groups (agents)
        H,W : grid height/width
        F_GRP : features per group
        F_GEO : features per world cell
        """
        self.G = G
        self.H = H
        self.W = W
        self.F_GRP = len(GrpFeatureIdx)
        self.F_GEO = len(GeoFeatureIdx)
        self.F_GEO_GRP = len(GeoGrpFeatureIdx)

        self.time = None
        self.seed = seed

        # World state
        self.world_occupancy = None  # (G, H, W)
        self.geo_features = None     # (F_GEO, H, W)
        self.grp_features = None     # (G, F_GRP)
        self.geo_grp_features = None # (G, F_GEO_GRP, H, W)

        # RNG
        self.np_random = np.random.default_rng(self.seed)

        # Agents (PettingZoo style)
        self.agents = [f"grp_{g}" for g in range(self.G)]
        self.possible_agents = self.agents[:]

        # For simplicity, we assume max strength per cell = 100
        self.max_strength = 100 # TODO(L): Replace, dummy only
        self._make_action_space()

    # ---------------------------------------------------------------------
    # Helpers to build spaces
    # ---------------------------------------------------------------------
    def _make_action_space(self):
        """
        Each group action is a continuous vector over all cells and 5
        directions (up/right/down/left/stay).

        For each cell (x, y), the 5 actions are softmax logits that
        indicate how much to move in which direction 
        from the available resources at the current cell.
        """
        
        # NOTE (L):
        self._action_space = Box(low=0.0, high=1.0, shape=(self.H, self.W, 5), dtype=np.float32)

    # ---------------------------------------------------------------------
    # PettingZoo API: reset
    # ---------------------------------------------------------------------
    def reset(self, seed=SEED, options=None):
        """
        1. initialize world_occupancy:
           for each group, randomly sample K=H*W/sparsity non-zero coords
           and an occupancy value uniform from 1 to 100.

        2. initialize geo features: dummy normal initialization

        3. initialize group features: dummy normal initialization
        """
        if seed is not None:
            self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

        self.time = 0

        # Occupancy
        self.world_occupancy = np.zeros((self.G, self.H, self.W), dtype=np.float32)

        def _reset_world_occupancy():
            # Choose some sparsity factor; tweak as you wish
            sparsity = 100  # K ≈ (H*W)/100 non-zero cells per group
            K = max(1, (self.H * self.W) // sparsity)

            for g in range(self.G):
                # Sample K distinct flat indices
                flat_indices = self.np_random.choice(self.H * self.W, size=K, replace=False)
                xs = flat_indices // self.W
                ys = flat_indices % self.W

                # Occupancy uniform from 1 to 100
                vals = self.np_random.integers(
                    low=1, high=self.max_strength + 1, size=K, dtype=np.int32
                )
                self.world_occupancy[g, xs, ys] = vals

        _reset_world_occupancy()

        # World features: dummy normal
        self.geo_features = self.np_random.integers(
            low=0, high=MAX_INIT_VAL + 1, size=(self.F_GEO, self.H, self.W), dtype=np.int32
        ).astype(np.float32)

        # Group features: dummy normal
        self.grp_features = self.np_random.integers(
            low=0, high=MAX_INIT_VAL + 1, size=(self.G, self.F_GRP), dtype=np.int32
        ).astype(np.float32)

        # All agents active at reset
        self.agents = self.possible_agents[:]

        # Return (observations, infos) at t=0
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    # ---------------------------------------------------------------------
    # Internal helpers for strength & similarity
    # ---------------------------------------------------------------------
    def _group_index(self, agent):
        # agent is "group_k"
        if isinstance(agent, int):
            return agent
        return int(agent.split("_")[-1])

    def _get_group_strength(self, g, x, y):
        # TODO: Replace dummy accumulation
        resources = np.sum(self.geo_features[:, x, y])
        return self.world_occupancy[g, x, y] * resources

    def _set_group_strength(self, g, x, y, val):
        self.world_occupancy[g, x, y] = max(0, val)

    def _group_similarity(self, g1, g2):
        """
        Example similarity: cosine similarity of group features.
        Just a dummy implementation for now.
        """
        v1 = self.grp_features[g1]
        v2 = self.grp_features[g2]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    # ---------------------------------------------------------------------
    # Action space per group
    # ---------------------------------------------------------------------
    def _group_actions(self, g):
        # same action space for all groups
        return self._action_space

    def _get_absorption_rate(self, key: FeatureKey) -> float | None:
        return FEATURES_SPEC.get(key).absorption_rate()

    def _resource_absorption(self):
        occupancy = self.world_occupancy # (G,H,W)
        occupancy_total = np.sum(occupancy, axis=0, keepdims=True) # (1,H,W)
        remaining = 1 - occupancy_total # (1,H,W)

        occupied = occupancy_total > 0
        occupancy_ratio = np.divide(
            occupancy, 
            occupancy_total, 
            out=np.zeros_like(occupancy), 
            where=occupied
            )
        max_absorption_rate = self._get_absorption_rate(FeatureKey.RESOURCES)
        max_absorption = max_absorption_rate * remaining * occupancy_ratio

        self.world_occupancy = np.minimum(MAX_OCCUPANCY_GAIN*occupancy, occupancy + max_absorption)

    
    # ---------------------------------------------------------------------
    # Group interaction for collisions
    # ---------------------------------------------------------------------
    def group_interaction(self, x, y):
        """
        Tournament mode at position (x, y):

        1. Gather all non-zero group indices into an array a
        2. Randomly permute that array a
        3. Tournament:
           - a[0] vs a[1], a[2] vs a[3], ...
           - For each fight, compare strength, update as:
             s1' = max(0, s1 - s2)
             s2' = max(0, s2 - s1)
           - Winner (if still non-zero) goes to next round
           - Repeat until only one left with non-zero strength (or none)
        """
        strengths = self.world_occupancy[:, x, y]
        active_groups = np.nonzero(strengths > 0)[0]
        if active_groups.size < 2:
            return  # no collision

        # TODO: Implement cooperation vs. conflict decision

        # Random tournament bracket
        permuted = self.np_random.permutation(active_groups)
        current = permuted.tolist()

        # Dummy fight
        while len(current) > 1:
            next_round = []
            # Pairwise fights
            for i in range(0, len(current), 2):
                if i + 1 >= len(current):
                    # Odd one out advances automatically
                    g_odd = current[i]
                    if self._get_group_strength(g_odd, x, y) > 0:
                        next_round.append(g_odd)
                    continue

                g1 = current[i]
                g2 = current[i + 1]

                s1 = self._get_group_strength(g1, x, y)
                s2 = self._get_group_strength(g2, x, y)

                # Resolve fight
                new_s1 = max(0, s1 - s2)
                new_s2 = max(0, s2 - s1)

                self._set_group_strength(g1, x, y, new_s1)
                self._set_group_strength(g2, x, y, new_s2)

                # Winner (if non-zero) advances
                if new_s1 > new_s2 and new_s1 > 0:
                    next_round.append(g1)
                elif new_s2 > new_s1 and new_s2 > 0:
                    next_round.append(g2)
                # if equal & both zero -> no one advances

            current = next_round

        # At the end, either 0 or 1 non-zero group remains at (x, y)
        # In either case, world_occupancy is already updated.

    def _move_resources(self, actions):
        """
        1. Each group moves the resources it owns.
        2a. The resources of each group get updated.
        2b. Accordingly, the occupancy ratios are re-evaluated.
        """
        for group, ratios in actions.items():
            # ratios: direction ratios with shape (H,W,5)
            g = self._group_index(group)

            # Occupancy ratios for the current group 
            g_occupancy = self.world_occupancy[g] # (H,W)

            # Ratio for each direction
            ratio_sum = np.sum(ratios, axis=-1) # (H,W)

            ## 0. Assert that the five ratios sum to 1
            # assert ratio_sum <= 1, ratio_sum # NOTE (L): For debugging.
            # NOTE (L): We could enforce aswell ratio_sum == 1
            rescale_mask = ratio_sum > 1
            if np.any(rescale_mask):
                # NOTE (L): Div by zero cannot occur here, by definition of rescale_mask
                ratios[rescale_mask] /= ratio_sum[rescale_mask]
                
            ratio_up    = ratios[:, :, 0] # (H,W)
            ratio_right = ratios[:, :, 1] # (H,W)
            ratio_down  = ratios[:, :, 2] # (H,W)
            ratio_left  = ratios[:, :, 3] # (H,W)
            ratio_stay  = ratios[:, :, 4] # (H,W)

            ## 1. Groups move their resources 
            g_resources = self.geo_features[GeoFeatureIdx.RESOURCES] * g_occupancy # (H,W), float32
            ### Store moved resources
            g_resources_next = self.geo_grp_features[g, GeoGrpFeatureIdx.RESOURCES] # (H,W)

            g_resources_next[:,:] = ratio_stay[:,:] * g_resources[:,:] # (H,W)
            
            # NOTE (L): We could enforce that no resources "leave" the world. 
            # up: from (x, y) -> (x-1, y)
            g_resources_next[:-1,:] += ratio_up[1:,:] * g_resources[1:,:]
            # down: from (x, y) -> (x+1, y)
            g_resources_next[1:,:] += ratio_down[:-1,:] * g_resources[:-1,:]
            # left: from (x, y) -> (x, y-1)
            g_resources_next[:,:-1] += ratio_left[:,1:] * g_resources[:,1:]
            # right: from (x, y) -> (x, y+1)
            g_resources_next[:,1:] += ratio_right[:,:-1] * g_resources[:,:-1]
            
        # 2. Update total resources and occupancy distribution of each cell
        group_resources = self.geo_grp_features[:, GeoGrpFeatureIdx.RESOURCES] # (G,H,W)
        group_resources_total = np.sum(group_resources, axis=0) # (H,W)

        ## Remaining geo resources that are not possessed by any group yet
        occupancy = self.world_occupancy # (G,H,W)
        occupancy_total = np.sum(occupancy, axis=0) # (H,W)
        remaining = 1 - occupancy_total # (H,W)
        remaining_geo_resources = self.geo_features[GeoFeatureIdx.RESOURCES] * remaining # (H,W), float32
        resources_total = group_resources_total + remaining_geo_resources # (H,W)

        ## Overwrite geo resources with new resources after groups moved some of their resources
        self.geo_features[GeoFeatureIdx.RESOURCES] = resources_total
        
        ## Recompute occupancy ratios
        for agent in self.possible_agents:
            g = self._group_index(agent)
            self.world_occupancy[g] = np.divide(
                group_resources[g],
                resources_total,
                out=self.world_occupancy[g],
                where=resources_total > 0
            )

        # Sanity check: no negative occupancies
        assert np.all(self.world_occupancy >= 0)

    # ---------------------------------------------------------------------
    # Step: apply actions and resolve collisions
    # ---------------------------------------------------------------------
    def step(self, actions):
        """
        1. Group movement
        For each group g, for each coord x,y,
        add integral quantities 0 <= Q1,Q2,Q3,Q4 <= self.world_occupancy[g][x][y]
        to their corresponding neighboring quantities.

        2. Group interaction
        For each coord x,y, if multiple groups occupy it (occupancy > 0),
        invoke group_interaction(x,y).

        3. Resource absorption
        Groups absorp resources from the cells they occupy
        """
        
        if not self.agents:
            # No active agents; environment is done
            return {}, {}, {}, {}, {}

        self.time += 1

        # 1. Groups move their resources
        self._move_resources(actions)    

        # 2. Group interaction
        ## Resolve collisions: cells where ≥2 groups have non-zero occupancy
        is_occupied = self.world_occupancy > 0  # (G, H, W)
        num_occupiers = np.sum(is_occupied, axis=0)  # (H, W)
        xs, ys = np.nonzero(num_occupiers >= 2)
        for x, y in zip(xs, ys):
            self.group_interaction(x, y)

        # 3. Resource absorption
        self._resource_absorption()
        
        # Build return values (simple global observation, dummy rewards)
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents} # TODO(L): Total global strength per agent
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    # ---------------------------------------------------------------------
    # Rendering & (pseudo) observation / action space accessors
    # ---------------------------------------------------------------------
    def render(self):
        # Minimal textual render: sum occupancy per group
        print(f"Time step: {self.time}")
        for g in range(self.G):
            total = int(self.world_occupancy[g].sum())
            print(f"  Group {g}: total strength = {total}")

    def _get_observation(self, agent):
        # For now: each agent sees the entire occupancy grid and world features
        i_agent = self._group_index(agent)
        return {
            KEY_OBS_OCCUPANCY: self.world_occupancy.copy(),
            KEY_OBS_FEATURES: self.geo_features.copy(),
            KEY_OBS_FEATURES_GROUP: self.grp_features.copy()
        }

    def observation_space(self, agent):
        """
        NOTE:
        In proper PettingZoo, this should return a Gymnasium Space, not the
        actual observation values. 
        TODO: Implement masked observation
        """
        return self._get_observation(agent)

    def action_space(self, agent):
        return self._group_actions(agent)