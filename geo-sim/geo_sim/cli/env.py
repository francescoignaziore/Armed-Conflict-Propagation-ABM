import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from pettingzoo import ParallelEnv

from geo_sim.config.env import SEED
from geo_sim.config.env import KEY_OBS_FEATURES, KEY_OBS_OCCUPANCY

class CSSMovementsEnv(ParallelEnv):
    metadata = {
        "name": "css_movements_v0",
    }

    def __init__(self, G, H, W, F_G, F_W, seed=SEED):
        """
        G   : number of groups (agents)
        H,W : grid height/width
        F_G : features per group
        F_W : features per world cell
        """
        self.G = G
        self.H = H
        self.W = W
        self.F_G = F_G
        self.F_W = F_W

        self.time = None
        self.seed = seed

        # World state
        self.world_occupancy = None  # shape (G, H, W)
        self.world_features = None   # shape (F_W, H, W)
        self.group_features = None   # shape (G, F_G)

        # RNG
        self.np_random = np.random.default_rng(self.seed)

        # Agents (PettingZoo style)
        self.agents = [f"group_{g}" for g in range(self.G)]
        self.possible_agents = self.agents[:]

        # For simplicity, we assume max strength per cell = 100
        self.max_strength = 100
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
        
        # NOTE (L): We can also have
        self._action_space = Box(low=-1000.0, high=1000.0, shape=(self.H, self.W, 5), dtype=np.float32)

    # ---------------------------------------------------------------------
    # PettingZoo API: reset
    # ---------------------------------------------------------------------
    def reset(self, seed=SEED, options=None):
        """
        1. initialize world_occupancy:
           for each group, randomly sample K=H*W/sparsity non-zero coords
           and an occupancy value uniform from 1 to 100.

        2. initialize world_features: dummy normal initialization

        3. initialize group_features: dummy normal initialization
        """
        if seed is not None:
            self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

        self.time = 0

        # Occupancy
        self.world_occupancy = np.zeros((self.G, self.H, self.W), dtype=np.float32)

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

        # World features: dummy normal
        self.world_features = self.np_random.normal(
            loc=0.0, scale=1.0, size=(self.F_W, self.H, self.W), dtype=np.float32
        )

        # Group features: dummy normal
        self.group_features = self.np_random.normal(
            loc=0.0, scale=1.0, size=(self.G, self.F_G), dtype=np.float32
        )

        # All agents active at reset
        self.agents = self.possible_agents[:]

        # In a real ParallelEnv, reset should return (observations, infos)
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    # ---------------------------------------------------------------------
    # Internal helpers for strength & similarity
    # ---------------------------------------------------------------------
    def _agent_index(self, agent):
        # agent is "group_k"
        if isinstance(agent, int):
            return agent
        return int(agent.split("_")[-1])

    def _get_group_strength(self, g, x, y):
        # TODO: Replace dummy accumulation
        resources = np.sum(world_features[:, x, y])
        return self.world_occupancy[g, x, y] * resources

    def _set_group_strength(self, g, x, y, val):
        self.world_occupancy[g, x, y] = max(0, val)

    def _group_similarity(self, g1, g2):
        """
        Example similarity: cosine similarity of group_features.
        Just a dummy implementation for now.
        """
        v1 = self.group_features[g1]
        v2 = self.group_features[g2]
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

    def _world_process(self):
        # TODO Resource processing
        pass
    
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

    # ---------------------------------------------------------------------
    # Step: apply actions and resolve collisions
    # ---------------------------------------------------------------------
    def step(self, actions):
        """
        1.
        For each group g, for each coord x,y,
        add integral quantities 0 <= Q1,Q2,Q3,Q4 <= self.world_occupancy[g][x][y]
        to their corresponding neighboring quantities.

        2.
        For each coord x,y, if multiple groups occupy it (occupancy > 0),
        invoke group_interaction(x,y).
        """
        if not self.agents:
            # No active agents; environment is done
            return {}, {}, {}, {}, {}

        self.time += 1

        # 1. Simulate resource consumption and processing
        self._world_process()

        # Start from current occupancy
        new_occ = self.world_occupancy.copy()

        # Directions: 0 = up, 1 = right, 2 = down, 3 = left
        for agent, act in actions.items():
            g = self._agent_index(agent)

            # Convert action to (H, W, 4)
            act = np.asarray(act, dtype=np.int32)
            act = act.reshape(self.H, self.W, 4)

            # Ensure non-negative and within per-direction cap
            act = np.clip(act, 0, self.max_strength)

            # Current available occupancy
            available = new_occ[g]

            # Separate each direction
            q_up = act[:, :, 0]
            q_right = act[:, :, 1]
            q_down = act[:, :, 2]
            q_left = act[:, :, 3]

            # Zero out flows that would go out of bounds
            q_up[0, :] = 0
            q_down[-1, :] = 0
            q_left[:, 0] = 0
            q_right[:, -1] = 0

            # Total outgoing per cell
            total_out = q_up + q_right + q_down + q_left

            # Enforce sum(Q) <= available per cell by rescaling if needed
            mask = total_out > available
            if np.any(mask):
                # NOTE (L): We could also throw an error here.
                # Avoid division by zero
                scale = np.zeros_like(available, dtype=np.float32)
                # NOTE (L): Div by zero should not occur, as the mask condition would otherwise not be satisfied
                scale[mask] = available[mask] / (total_out[mask] + 1e-9).astype(
                    np.float32
                )

                # Scale flows
                q_up = (q_up * scale).astype(np.int32)
                q_right = (q_right * scale).astype(np.int32)
                q_down = (q_down * scale).astype(np.int32)
                q_left = (q_left * scale).astype(np.int32)

                total_out = q_up + q_right + q_down + q_left  # recompute

            # Subtract outgoing from source cells
            new_occ[g] = available - total_out

            # Add incoming to destination cells via slicing
            # up: from (x, y) -> (x-1, y)
            new_occ[g, 0:-1, :] += q_up[1:, :]

            # down: from (x, y) -> (x+1, y)
            new_occ[g, 1:, :] += q_down[:-1, :]

            # left: from (x, y) -> (x, y-1)
            new_occ[g, :, 0:-1] += q_left[:, 1:]

            # right: from (x, y) -> (x, y+1)
            new_occ[g, :, 1:] += q_right[:, :-1]

        # Update occupancy
        self.world_occupancy = new_occ

        # Sanity check: no negative occupancies
        assert np.all(self.world_occupancy >= 0)

        # Resolve collisions: cells where ≥2 groups have non-zero occupancy
        occ_mask = self.world_occupancy > 0  # (G, H, W)
        collision_counts = np.sum(occ_mask, axis=0)  # (H, W)
        xs, ys = np.nonzero(collision_counts >= 2)

        for x, y in zip(xs, ys):
            self.group_interaction(x, y)

        # Build return values (simple global observation, dummy rewards)
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
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
        i_agent = self._agent_index(agent)
        return {
            KEY_OBS_OCCUPANCY: self.world_occupancy.copy(),
            KEY_OBS_FEATURES: self.world_features.copy(),
            KEY_OBS_FEATURES_GROUP: self.group_features.copy()
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