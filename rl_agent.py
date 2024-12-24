import gymnasium as gym
import jax
import numpy as np
import torch as T
from gymnasium.spaces import Box

from agent import Agent
from base import OBSERVATION_SPACE
from core import ActorNetwork, CriticNetwork, PPOMemory

config = {
    "model_path": "",
    "gamma": 0.99,
    "alpha": 0.0001,
    "gae_lambda": 0.97,
    "policy_clip": 0.2,
    "batch_size": 500,
    "N": 20,
    "n_epochs": 10,
}


class Agent(Agent):
    def __init__(self, player: str, env_cfg):
        super().__init__(player, env_cfg)
        self.gamma = config["gamma"]
        self.policy_clip = config["policy_clip"]
        self.n_epochs = config["n_epochs"]
        self.gae_lambda = config["gae_lambda"]

        self.observation_space = OBSERVATION_SPACE
        self.obs_dim = gym.spaces.utils.flatdim(self.observation_space)
        self.action_space = self._determine_action_space()
        self.act_dim = gym.spaces.utils.flatdim(self.action_space)

        self.actor = ActorNetwork(
            input_dims=self.obs_dim,
            alpha=config["alpha"],
            sap_range=self.env_cfg["unit_sap_range"],
            player_id=self.player,
        )
        self.critic = CriticNetwork(
            input_dims=self.obs_dim, alpha=config["alpha"], player_id=self.player
        )
        self.memory = PPOMemory(config["batch_size"])

    def update_env_config(self, env_cfg):
        """
        Update the environment configuration.
        """
        self.env_cfg = env_cfg
        self.actor.update_action_space(env_cfg)

    def remember(self, state, action, probs, vals, reward, done):
        state = self._preprocess_obs(state).cpu().numpy().flatten()
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = self._preprocess_obs(observation)
        value = self.critic(state)
        action_dists = self.actor(state)

        actions = np.zeros((16, 3), dtype=np.int16)
        log_probs = []

        for i in range(16):
            action_type_dist, dx_dist, dy_dist = action_dists[i]

            # Sample raw values
            action_type = action_type_dist.sample()
            dx_raw = dx_dist.sample()  # [0, offset_range-1]
            dy_raw = dy_dist.sample()  # [0, offset_range-1]

            log_prob = action_type_dist.log_prob(action_type)
            if action_type.item() == 5:
                log_prob = (
                    log_prob + dx_dist.log_prob(dx_raw) + dy_dist.log_prob(dy_raw)
                )
            log_probs.append(log_prob)

            if action_type.item() == 5:
                dx = dx_raw.item() - self.actor.sap_range
                dy = dy_raw.item() - self.actor.sap_range

                unit_x = state[0, i * 2].item()
                unit_y = state[0, i * 2 + 1].item()
                target_x = unit_x + dx
                target_y = unit_y + dy

                if not (0 <= target_x < 24 and 0 <= target_y < 24):
                    action_type = T.tensor(0, device=state.device)
                    dx = 0
                    dy = 0
            else:
                dx = 0
                dy = 0

            actions[i] = [action_type.item(), dx, dy]

        return actions, sum(log_probs).item(), value.item()

    def learn(self):
        if len(self.memory.states) == 0:  # If memory is empty, nothing to learn from
            return None

        actor_loss = None
        critic_loss = None
        critic_value = None

        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            if len(batches) == 0:  # If no batches, nothing to learn from
                return None

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float).to(
                    self.actor.device
                )
                actions = T.tensor(action_arr[batch], dtype=T.float).to(
                    self.actor.device
                )

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = T.zeros_like(old_probs)
                for unit in range(self.actor.max_units):
                    action_type_dist, dx_dist, dy_dist = dist[unit]
                    
                    # Clamp action types to valid range [0, 5] and convert to long
                    action_types = T.clamp(actions[:, unit, 0], 0, 5).long()
                    
                    # For dx and dy, we need to ensure they're in [0, offset_range-1]
                    # The original actions are in [-sap_range, sap_range]
                    # We need to shift them to [0, 2*sap_range-1] for the categorical distribution
                    dx = actions[:, unit, 1].clamp(-self.actor.sap_range, self.actor.sap_range)
                    dy = actions[:, unit, 2].clamp(-self.actor.sap_range, self.actor.sap_range)
                    
                    # Shift from [-sap_range, sap_range] to [0, 2*sap_range-1]
                    dx_shifted = (dx + self.actor.sap_range).long()
                    dy_shifted = (dy + self.actor.sap_range).long()
                    
                    # Double check the ranges
                    dx_shifted = dx_shifted.clamp(0, self.actor.offset_range - 1)
                    dy_shifted = dy_shifted.clamp(0, self.actor.offset_range - 1)
                    
                    new_probs[:, unit, 0] = action_type_dist.log_prob(action_types)
                    # Only include dx, dy probs if action type is sap (5)
                    sap_mask = (action_types == 5).float()
                    new_probs[:, unit, 1] = dx_dist.log_prob(dx_shifted) * sap_mask
                    new_probs[:, unit, 2] = dy_dist.log_prob(dy_shifted) * sap_mask

                # sum probabilities of each action component
                old_probs = old_probs.sum(-1).sum(-1)
                new_probs = new_probs.sum(-1).sum(-1)

                prob_ratio = (new_probs - old_probs).exp()

                # Clip probability ratios to prevent extremely large values
                prob_ratio = T.clamp(prob_ratio, 1e-10, 10.0)

                # Convert advantage to tensor and normalize it
                advantage_tensor = T.tensor(advantage[batch], dtype=T.float).to(self.actor.device)
                # Normalize advantages
                advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
                
                weighted_probs = advantage_tensor * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                ) * advantage_tensor

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage_tensor + T.tensor(values[batch], dtype=T.float).to(
                    self.actor.device
                )
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                # Clip gradients to prevent exploding gradients
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                
                # Add gradient clipping
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                # Check for invalid loss values
                if not T.isfinite(total_loss):
                    print(f"Warning: Non-finite loss detected! Actor loss: {actor_loss.item()}, Critic loss: {critic_loss.item()}")
                    continue

        self.memory.clear_memory()

        # Only return metrics if we actually performed learning
        if actor_loss is not None:
            return (
                actor_loss.item(),
                critic_loss.item(),
                critic_value.mean().item(),
                advantage.mean(),
            )
        return None

    def _determine_observation_space(self):
        pass

    def _determine_action_space(self) -> Box:
        """
        Returns the action space of the environment based on the environment configuration.
        """
        low = np.zeros((self.env_cfg["max_units"], 3))
        low[:, 1:] = -self.env_cfg["unit_sap_range"]
        high = np.ones((self.env_cfg["max_units"], 3)) * 6
        high[:, 1:] = self.env_cfg["unit_sap_range"]
        return gym.spaces.Box(low=low, high=high, dtype=np.int16)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Returns actions for all units in a single turn.
        Format: numpy array of shape (max_units, 3) where each row is [action_type, dx, dy]
        """
        # Check for rule based start
        # if self.model is None:
        #     return super().act(step, obs, remainingOverageTime)
        # if step < 100:
        actions = super().act(step, obs, remainingOverageTime)
        actions = self.choose_action(obs)
        return actions

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def _preprocess_obs(self, obs):
        flattened_tree = jax.tree_util.tree_flatten(obs)
        tensor_list = jax.tree_util.tree_map(
            lambda x: T.from_numpy(x).to(dtype=T.float32).to(self.actor.device),
            flattened_tree[0],
        )
        flattened = [t.flatten() for t in tensor_list]
        x = T.cat(flattened, dim=0).unsqueeze(0)
        return x
        # for tensor in tensor_list:
        #     print(tensor.shape)
        # unit_positions = torch.tensor(obs["units"]["position"][self.team_id]).to(
        #     self.device
        # )
        # enemy_unit_positions = torch.from_numpy(
        #     obs["units"]["position"][self.opp_team_id]
        # ).to(self.device)
        # unit_energys = torch.tensor(obs["units"]["energy"]).to(self.device)
        # unit_mask = obs["units_mask"][self.team_id]
        # enemy_unit_mask = obs["units_mask"][self.opp_team_id]
        # sensor_mask = obs["sensor_mask"]
        # energy_map = obs["map_features"]["energy"]
        # tile_type_map = obs["map_features"]["tile_type"]
        # relic_nodes = obs["relic_nodes"]
        # relic_nodes_mask = obs["relic_nodes_mask"]
        # team_points = obs["team_points"]
        # team_wins = obs["team_wins"]
        # steps = obs["steps"]
        # match_steps = obs["match_steps"]
        pass
