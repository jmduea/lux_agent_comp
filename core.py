import os

import numpy as np
import scipy.signal
import torch as T
import torch.nn as nn
import torch.optim as optim
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


class PPOMemory:
    def __init__(self, batch_size, capacity=1000):
        self.batch_size = batch_size
        self.capacity = capacity
        self.cursor = 0
        self.current_size = 0
        self.states = np.zeros((capacity, 1880), dtype=np.float32)
        self.probs = np.zeros((capacity, 16, 3), dtype=np.float32)
        self.vals = np.zeros((capacity,), dtype=np.float32)
        self.actions = np.zeros((capacity, 16, 3), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def generate_batches(self):
        indices = np.random.permutation(self.current_size)
        n_batches = self.current_size // self.batch_size
        batches = indices[: n_batches * self.batch_size].reshape(
            n_batches, self.batch_size
        )
        return (
            self.states[: self.current_size],
            self.actions[: self.current_size],
            self.probs[: self.current_size],
            self.vals[: self.current_size],
            self.rewards[: self.current_size],
            self.dones[: self.current_size],
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        idx = self.current_size
        if idx < self.capacity:
            self.states[idx] = state
            self.actions[idx] = action
            self.probs[idx] = probs
            self.vals[idx] = vals
            self.rewards[idx] = reward
            self.dones[idx] = done
            self.current_size += 1

    def clear_memory(self):
        self.current_size = 0


class ActorNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        alpha,
        max_units=16,
        n_action_types=6,
        sap_range=3,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="tmp/ppo",
        player_id="player_0",
    ):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, f"actor_torch_ppo.{player_id}")
        self.tensorboard_dir = os.path.join(chkpt_dir, "tensorboard", player_id)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Initialize tensorboard write
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.train_step = 0

        self.max_units = max_units
        self.n_action_types = n_action_types
        self.sap_range = sap_range
        self.offset_range = 2 * sap_range
        self.valid_actions_cache = {}
        self.weights_cache = {"dx": {}, "dy": {}}
        self.features = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        # Seperate heads for each unit's actions
        self.action_type_heads = nn.ModuleList(
            [nn.Linear(fc2_dims, n_action_types) for _ in range(max_units)]
        )
        self.dx_heads = nn.ModuleList(
            [nn.Linear(fc2_dims, self.offset_range) for _ in range(max_units)]
        )
        self.dy_heads = nn.ModuleList(
            [nn.Linear(fc2_dims, self.offset_range) for _ in range(max_units)]
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def log_metrics(self, metrics_dict, step=None):
        if step is None:
            step = self.train_step
        for name, value in metrics_dict.items():
            self.writer.add_scalar(name, value, step)
        self.train_step += 1

    def log_distributions(self, step=None):
        """
        Log nework parameter distributions
        """
        if step is None:
            step = self.train_step
        for name, param in self.named_parameters():
            self.writer.add_histogram(f"params/{name}", param.data, step)
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad, step)

    def cache_weights(self, heads, cache_dict):
        """
        Store weights for each dimension in the cache
        """
        dim = heads[0].weight.size(0)  # get output dimension
        if dim not in cache_dict:
            cache_dict[dim] = []
            for head in heads:
                cache_dict[dim].append(
                    (head.weight.data.clone(), head.bias.data.clone())
                )

    def load_cached_weights(self, heads, cache_dict, target_dim):
        """Load weights from cache if available"""
        if target_dim in cache_dict:
            for head, (cached_weight, cached_bias) in zip(
                heads, cache_dict[target_dim]
            ):
                head.weight.data.copy_(cached_weight)
                head.bias.data.copy_(cached_bias)
            return True
        return False

    def update_action_space(self, env_cfg):
        """
        Update action space based on new environment config
        """
        old_range = self.offset_range
        self.sap_range = env_cfg["unit_sap_range"]
        self.offset_range = 2 * self.sap_range
        self.valid_actions_cache.clear()

        # Only recreate heads if the range has changed
        if old_range != self.offset_range:
            # Store old heads
            self.cache_weights(self.dx_heads, self.weights_cache["dx"])
            self.cache_weights(self.dy_heads, self.weights_cache["dy"])
            old_dx_heads = self.dx_heads
            old_dy_heads = self.dy_heads
            # Create new heads with correct output dimensions
            self.dx_heads = nn.ModuleList(
                [
                    nn.Linear(self.features[-2].out_features, self.offset_range)
                    for _ in range(self.max_units)
                ]
            ).to(self.device)
            self.dy_heads = nn.ModuleList(
                [
                    nn.Linear(self.features[-2].out_features, self.offset_range)
                    for _ in range(self.max_units)
                ]
            ).to(self.device)

            # Try to load cached weights for the new dimension
            if not self.load_cached_weights(
                self.dx_heads, self.weights_cache["dx"], self.offset_range
            ):
                # If not available, copy weights from old heads
                min_dim = min(old_range, self.offset_range)
                for i in range(self.max_units):
                    # Copy weights for dx heads
                    self.dx_heads[i].weight.data[:min_dim, :] = old_dx_heads[
                        i
                    ].weight.data[:min_dim, :]
                    self.dx_heads[i].bias.data[:min_dim] = old_dx_heads[i].bias.data[
                        :min_dim
                    ]

                    if self.offset_range > old_range:
                        nn.init.xavier_uniform_(
                            self.dx_heads[i].weight.data[min_dim:, :]
                        )
                        self.dx_heads[i].bias.data[min_dim:].zero_()

            if not self.load_cached_weights(
                self.dy_heads, self.weights_cache["dy"], self.offset_range
            ):
                # If not available, copy weights from old heads
                min_dim = min(old_range, self.offset_range)
                for i in range(self.max_units):
                    # Copy weights for dy heads
                    self.dy_heads[i].weight.data[:min_dim, :] = old_dy_heads[
                        i
                    ].weight.data[:min_dim, :]
                    self.dy_heads[i].bias.data[:min_dim] = old_dy_heads[i].bias.data[
                        :min_dim
                    ]

                    if self.offset_range > old_range:
                        nn.init.xavier_uniform_(
                            self.dy_heads[i].weight.data[min_dim:, :]
                        )
                        self.dy_heads[i].bias.data[min_dim:].zero_()

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.optimizer.param_groups[0]["lr"]
        )

    def get_valid_positions(self, unit_positions):
        """
        Get valid positions for each unit
        """
        cache_key = tuple(unit_positions.flatten().tolist())
        if cache_key in self.valid_actions_cache:
            return self.valid_actions_cache[cache_key]

        dx_offsets = T.arange(
            -self.sap_range, self.sap_range + 1, device=unit_positions.device
        )
        dy_offsets = T.arange(
            -self.sap_range, self.sap_range + 1, device=unit_positions.device
        )

        target_x = unit_positions[..., 0:1] + dx_offsets.view(1, 1, -1)
        target_y = unit_positions[..., 1:2] + dy_offsets.view(1, 1, -1)

        valid_x = (target_x >= 0) & (target_x < 24)
        valid_y = (target_y >= 0) & (target_y < 24)

        result = (valid_x, valid_y)
        self.valid_actions_cache[cache_key] = result
        return result

    @T.no_grad()  # Optimization for inference
    def forward(self, state):
        features = self.features(state)
        batch_size = state.shape[0]

        action_type_logits = T.stack(
            [head(features) for head in self.action_type_heads], dim=1
        )
        dx_logits = T.stack([head(features) for head in self.dx_heads], dim=1)
        dy_logits = T.stack([head(features) for head in self.dy_heads], dim=1)

        # Get all unit positions [batch_size, max_units, 2]
        unit_positions = state[:, : self.max_units * 2].view(
            batch_size, self.max_units, 2
        )

        enemy_positions = state[:, self.max_units * 2 : self.max_units * 4].view(
            batch_size, self.max_units, 2
        )

        dx_offsets = T.arange(-self.sap_range, self.sap_range + 1, device=state.device)
        dy_offsets = T.arange(-self.sap_range, self.sap_range + 1, device=state.device)

        target_x = unit_positions[..., 0:1] + dx_offsets.view(1, 1, -1)
        target_y = unit_positions[..., 1:2] + dy_offsets.view(1, 1, -1)

        valid_x = (target_x >= 0) & (target_x < 24)
        valid_y = (target_y >= 0) & (target_y < 24)

        valid_targets_x = T.zeros(
            (batch_size, self.max_units, self.offset_range),
            dtype=T.bool,
            device=state.device,
        )
        valid_targets_y = T.zeros(
            (batch_size, self.max_units, self.offset_range),
            dtype=T.bool,
            device=state.device,
        )

        # Create target positions tensor [batch, units, offsets, 2]
        target_positions = T.stack(
            [
                target_x[..., : self.offset_range],
                target_y[..., : self.offset_range],
            ],
            dim=-1,
        )
        valid_enemies = (enemy_positions[..., 0] >= 0) & (enemy_positions[..., 1] >= 0)
        for b in range(batch_size):
            enemy_coords = enemy_positions[b, valid_enemies[b]]
            if len(enemy_coords) > 0:
                matches = (target_positions[b, :, :, None] == enemy_coords).all(dim=-1)
                has_match = matches.any(dim=-1)
                valid_targets_x[b] = has_match
                valid_targets_y[b] = has_match

        # for b in range(batch_size):
        #     for u in range(self.max_units):
        #         # Only consider valid enemy positions (not -1)
        #         valid_enemies = (enemy_positions[b, :, 0] >= 0) & (
        #             enemy_positions[b, :, 1] >= 0
        #         )
        #         enemy_coords = enemy_positions[b, valid_enemies]
        #         if len(enemy_coords) > 0:
        #             # check each possible dx and dy offset
        #             for i in range(self.offset_range):
        #                 dx = i - self.sap_range
        #                 target_x = unit_positions[b, u, 0] + dx
        #                 for j in range(self.offset_range):
        #                     dy = j - self.sap_range
        #                     target_y = unit_positions[b, u, 1] + dy
        #                     target = T.tensor([target_x, target_y], device=state.device)
        #                     matches = (enemy_coords == target).all(dim=1).any()
        #                     if matches:
        #                         valid_targets_x[b, u, i] = True
        #                         valid_targets_y[b, u, j] = True

        # Combine validity masks
        valid_x = valid_x[..., : self.offset_range] & valid_targets_x
        valid_y = valid_y[..., : self.offset_range] & valid_targets_y

        # valid_x = valid_x.view(batch_size, self.max_units, -1)
        # valid_y = valid_y.view(batch_size, self.max_units, -1)
        LARGE_NEG = -1e9
        dx_mask = T.where(
            valid_x,
            T.zeros_like(dx_logits),
            T.ones_like(dx_logits) * LARGE_NEG,
        )
        dy_mask = T.where(
            valid_y,
            T.zeros_like(dy_logits),
            T.ones_like(dy_logits) * LARGE_NEG,
        )

        # Apply masks
        dx_logits = dx_logits + dx_mask
        dy_logits = dy_logits + dy_mask

        has_valid_positions = (valid_x.any(-1) & valid_y.any(-1)).view(
            batch_size, self.max_units, 1
        )
        sap_mask = T.zeros_like(action_type_logits)
        sap_mask[..., 5:6] = T.where(
            has_valid_positions,
            T.zeros_like(action_type_logits[..., 5:6]),
            T.ones_like(action_type_logits[..., 5:6]) * LARGE_NEG,
        )
        action_type_logits = action_type_logits + sap_mask
        # Create distributions for all units
        action_dists = [
            (
                Categorical(logits=action_type_logits[:, i]),
                Categorical(logits=dx_logits[:, i]),
                Categorical(logits=dy_logits[:, i]),
            )
            for i in range(self.max_units)
        ]
        # # Get distributions for each unit
        # action_dists = []
        # for i in range(self.max_units):
        #     # Action Type distribution
        #     action_type_logits = self.action_type_heads[i](features)

        #     # Offsets distribution
        #     dx_logits = self.dx_heads[i](features)
        #     dy_logits = self.dy_heads[i](features)

        #     # Create position validity masks
        #     # For each unit in the batch, check if dx, dy would lead to a valid board position
        #     LARGE_NEG = -1e9
        #     dx_mask = T.ones_like(dx_logits) * LARGE_NEG
        #     dy_mask = T.ones_like(dy_logits) * LARGE_NEG

        #     for b in range(batch_size):
        #         # Get unit positions from state
        #         unit_x = state[b, i * 2].item()
        #         unit_y = state[b, i * 2 + 1].item()

        #         for dx_idx in range(self.offset_range):
        #             dx_val = dx_idx - self.sap_range
        #             target_x = unit_x + dx_val
        #             if 0 <= target_x < 24:  # Valid
        #                 dx_mask[b, dx_idx] = 0  # Allow this offset

        #         for dy_idx in range(self.offset_range):
        #             dy_val = dy_idx - self.sap_range
        #             target_y = unit_y + dy_val
        #             if 0 <= target_y < 24:  # Valid
        #                 dy_mask[b, dy_idx] = 0  # Allow this offset

        #     dx_logits += dx_mask
        #     dy_logits += dy_mask

        #     action_type_dist = Categorical(logits=action_type_logits)
        #     dx_dist = Categorical(logits=dx_logits)
        #     dy_dist = Categorical(logits=dy_logits)

        #     action_dists.append((action_type_dist, dx_dist, dy_dist))

        return action_dists

    def save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "weights_cache": self.weights_cache,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        T.save(checkpoint, self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            checkpoint = T.load(self.checkpoint_file)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.weights_cache = checkpoint["weights_cache"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.eval()


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        alpha,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="tmp/ppo",
        player_id="player_0",
    ):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, f"critic_torch_ppo.{player_id}")
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        self.eval()


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(T.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = T.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return T.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with T.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
