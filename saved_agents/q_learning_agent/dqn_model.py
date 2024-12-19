import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from base import SPACE_SIZE, Global, NodeType


class DQN(nn.Module):
    def __init__(self, state_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Network for processing individual ship features
        self.ship_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Network for processing global state
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_size // 2 * Global.MAX_UNITS, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Action prediction heads for each ship
        self.action_heads = nn.ModuleList(
            [nn.Linear(hidden_size, output_size) for _ in range(Global.MAX_UNITS)]
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state_batch):
        batch_size = state_batch.shape[0]
        ship_states = state_batch.view(batch_size, Global.MAX_UNITS, -1)

        # Process each ship's state
        ship_encodings = []
        for i in range(Global.MAX_UNITS):
            ship_state = ship_states[:, i, :]
            ship_encoding = self.ship_encoder(ship_state)
            ship_encodings.append(ship_encoding)

        # Concatenate ship encodings
        combined_encoding = torch.cat(ship_encodings, dim=1)

        # Process global state
        global_features = self.global_encoder(combined_encoding)

        # Predict actions for each ship
        action_predictions = []
        for i in range(Global.MAX_UNITS):
            action_prediction = self.action_heads[i](global_features)
            action_predictions.append(action_prediction)

        return torch.stack(action_predictions, dim=1)


class ImitationReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.expert_buffer = deque(maxlen=capacity // 2)
        self.reward_history = deque(maxlen=10000)

    def push(self, state, actions, reward, next_state, done, is_expert=False):
        if is_expert:
            self.expert_buffer.append((state, actions, reward, next_state, done))
        else:
            self.buffer.append((state, actions, reward, next_state, done))
        self.reward_history.append(reward)

    def sample(self, batch_size):
        """Sample a batch of experiences from memory"""
        # Calculate available samples
        total_samples = len(self.buffer) + len(self.expert_buffer)
        if total_samples == 0:
            return None

        # Adjust batch size if we don't have enough samples
        actual_batch_size = min(batch_size, total_samples)

        # Calculate proportions for expert and regular samples
        if len(self.expert_buffer) > 0:
            expert_ratio = len(self.expert_buffer) / total_samples
            expert_size = min(
                int(actual_batch_size * expert_ratio), len(self.expert_buffer)
            )
        else:
            expert_size = 0

        regular_size = min(actual_batch_size - expert_size, len(self.buffer))

        # Sample from available experiences
        expert_batch = []
        if expert_size > 0:
            expert_batch = random.sample(list(self.expert_buffer), expert_size)

        regular_batch = []
        if regular_size > 0:
            regular_batch = random.sample(list(self.buffer), regular_size)

        combined_batch = expert_batch + regular_batch
        if not combined_batch:
            return None

        random.shuffle(combined_batch)

        # Scale rewards
        scaled_batch = []
        for state, actions, reward, next_state, done in combined_batch:
            scaled_reward = self.scale_reward(reward)
            scaled_batch.append((state, actions, scaled_reward, next_state, done))
        return scaled_batch

    def scale_reward(self, reward):
        if len(self.reward_history) < 100:  # Wait for enough samples
            return reward * 0.1  # Simple scaling at start

        # Use percentile-based scaling
        rewards = np.array(self.reward_history)
        p95 = np.percentile(np.abs(rewards), 95)
        if p95 == 0:
            return reward * 0.1
        return reward / (p95 + 1e-8)  # Avoid division by zero

    def __len__(self):
        return len(self.buffer) + len(self.expert_buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, player, env_cfg):
        self.state_size = state_size
        self.action_size = action_size
        self.player = player
        self.env_cfg = env_cfg
        self.memory = ImitationReplayBuffer(10000)

        # Hyperparameters
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update = 10  # how often to update target network

        # Networks
        if torch.cuda.is_available():
            print(f"GPU:{torch.cuda.get_device_name(0)} is available.")
        else:
            print("No GPU available. Training will run on cpu.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(SPACE_SIZE * SPACE_SIZE * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        ).to(self.device)

        # Local feature encoder
        self.local_encoder = nn.Sequential(
            nn.Linear(25 * 5, 128),  # 5 local 5x5 windows
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        ).to(self.device)

        # Ship feature encoder
        self.ship_encoder = nn.Sequential(
            nn.Linear(
                15, 32
            ),  # position(2) + energy(1) + node_type(3) + distances(4) + task(3) + nearby_enemies(1) + energy_efficiency(1)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        ).to(self.device)

        # Action heads
        self.advantage_net = nn.Sequential(
            nn.Linear(256 + 64 + 32, 128),  # Combined features
            nn.ReLU(),
            nn.Linear(128, action_size),
        ).to(self.device)

        self.value_net = nn.Sequential(
            nn.Linear(256 + 64 + 32, 128), nn.ReLU(), nn.Linear(128, 1)
        ).to(self.device)

        # Target networks
        self.target_global_encoder = copy.deepcopy(self.global_encoder)
        self.target_local_encoder = copy.deepcopy(self.local_encoder)
        self.target_ship_encoder = copy.deepcopy(self.ship_encoder)
        self.target_advantage_net = copy.deepcopy(self.advantage_net)
        self.target_value_net = copy.deepcopy(self.value_net)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.global_encoder.parameters())
            + list(self.local_encoder.parameters())
            + list(self.ship_encoder.parameters())
            + list(self.advantage_net.parameters())
            + list(self.value_net.parameters()),
            lr=self.learning_rate,
        )

        # Metrics
        self.total_steps = 0
        self.loss_history = []
        self.reward_history = []
        self.q_value_history = []

    def encode_state(self, state_dict):
        """
        Encode the state dictionary into a format suitable for the neural network

        Args:
            state_dict: Dictionary containing global and ship-specific features

        Returns:
            torch.Tensor: Encoded state
        """
        # Process global features
        global_features = torch.cat(
            [
                torch.FloatTensor(state_dict["global_features"][k])
                .flatten()
                .to(self.device)
                for k in [
                    "energy_map",
                    "nebula_map",
                    "void_map",
                    "relic_map",
                    "visibility_map",
                ]
            ]
        )

        # Process local features
        local_features = torch.cat(
            [
                torch.FloatTensor(state_dict["ship_features"][k])
                .flatten()
                .to(self.device)
                for k in [
                    "local_energy",
                    "local_nebula",
                    "local_void",
                    "local_relic",
                    "local_visibility",
                ]
            ]
        )

        # Process ship features
        ship_feature_list = []
        # Position (2)
        ship_feature_list.extend(state_dict["ship_features"]["position"])
        # Energy (1)
        ship_feature_list.append(state_dict["ship_features"]["energy"])
        # Node type (3)
        ship_feature_list.extend(state_dict["ship_features"]["node_type"])
        # Distances (4)
        ship_feature_list.extend(state_dict["ship_features"]["distances"])
        # Task (3)
        ship_feature_list.extend(state_dict["ship_features"]["task"])
        # Nearby enemies (1)
        ship_feature_list.append(state_dict["ship_features"]["nearby_enemies"])
        # Energy efficiency (1)
        ship_feature_list.append(state_dict["ship_features"]["energy_efficiency"])

        ship_features = torch.FloatTensor(ship_feature_list).to(self.device)

        # Encode features through respective networks
        global_encoded = self.global_encoder(global_features)
        local_encoded = self.local_encoder(local_features)
        ship_encoded = self.ship_encoder(ship_features)

        # Combine all features
        combined_features = torch.cat([global_encoded, local_encoded, ship_encoded])

        return combined_features

    def forward(self, state_dict):
        """
        Forward pass through the network using Dueling DQN architecture

        Args:
            state_dict: Dictionary containing state features

        Returns:
            torch.Tensor: Q-values for each action
        """
        combined_features = self.encode_state(state_dict)

        # Dueling DQN
        advantage = self.advantage_net(combined_features)
        value = self.value_net(combined_features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean())

        return q_values

    def get_state_features(self, fleet, space, opp_fleet):
        """Convert game state to neural network input"""
        state = []
        for ship in fleet.ships:
            if ship.node:
                ship_state = self._get_ship_features(ship, space, fleet, opp_fleet)
            else:
                ship_state = np.zeros(self.state_size)  # padding for inactive ships
            state.append(ship_state)
        return np.concatenate(state)

    def _get_ship_features(self, ship, space, fleet, opp_fleet):
        features = []

        # Ship features
        features.extend(
            [
                ship.energy / 100.0,  # Normalize energy
                ship.node.x / Global.SPACE_SIZE,  # Normalize position
                ship.node.y / Global.SPACE_SIZE,
                len(ship.sap_targets) / len(opp_fleet.ships),
            ]
        )

        features.extend(
            [
                Global.UNIT_MOVE_COST / 5.0,
                Global.UNIT_SAP_COST / 50.0,
                Global.UNIT_SENSOR_RANGE / 4.0,
                Global.MAX_UNITS / 16.0,
            ]
        )
        # Local environment features
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = ship.node.x + dx, ship.node.y + dy
                if 0 <= x < Global.SPACE_SIZE and 0 <= y < Global.SPACE_SIZE:
                    node = space.get_node(x, y)
                    features.extend(
                        [
                            1.0 if node.type == NodeType.asteroid else 0.0,
                            1.0 if node.relic else 0.0,
                            1.0 if node.reward else 0.0,
                            node.energy / 100.0 if node.energy is not None else 0.0,
                        ]
                    )
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                self.policy_net.eval()
                q_values = self.policy_net(state)
                self.policy_net.train()
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_size)

    def train(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)

        # Convert batch into separate arrays
        batch = list(zip(*transitions))

        try:
            # Convert state dictionaries to tensors
            state_batch = [
                {
                    "global_features": {
                        k: torch.FloatTensor(v).to(self.device)
                        for k, v in state["global_features"].items()
                    },
                    "ship_features": {
                        k: torch.FloatTensor(v).to(self.device)
                        if isinstance(v, (list, np.ndarray))
                        else torch.tensor(v, dtype=torch.float32).to(self.device)
                        for k, v in state["ship_features"].items()
                    },
                }
                for state in batch[0]
            ]

            next_state_batch = [
                {
                    "global_features": {
                        k: torch.FloatTensor(v).to(self.device)
                        for k, v in state["global_features"].items()
                    },
                    "ship_features": {
                        k: torch.FloatTensor(v).to(self.device)
                        if isinstance(v, (list, np.ndarray))
                        else torch.tensor(v, dtype=torch.float32).to(self.device)
                        for k, v in state["ship_features"].items()
                    },
                }
                for state in batch[3]
            ]

            # Convert other elements to tensors
            action_batch = torch.LongTensor(batch[1]).to(self.device)
            reward_batch = torch.FloatTensor(batch[2]).to(self.device)
            done_batch = torch.BoolTensor(batch[4]).to(self.device)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = torch.zeros(self.batch_size).to(self.device)
            for i, state in enumerate(state_batch):
                q_values = self.forward(state)
                state_action_values[i] = q_values[action_batch[i]]

            # Compute V(s_{t+1}) for all next states
            next_state_values = torch.zeros(self.batch_size).to(self.device)
            with torch.no_grad():
                for i, next_state in enumerate(next_state_batch):
                    next_state_values[i] = self.forward(next_state).max()

            # Compute the expected Q values
            expected_state_action_values = (
                next_state_values * self.gamma * (~done_batch) + reward_batch
            )

            # Compute loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), 100)
            self.optimizer.step()

        except Exception as e:
            print(f"Error during training: {str(e)}")
            print(f"Batch contents: {batch}")

    def get_expert_action(self, step, obs, remainingOverageTime=60):
        return self.expert.act(step, obs, remainingOverageTime)

    def save(self, path):
        torch.save(
            {
                "global_encoder_state_dict": self.global_encoder.state_dict(),
                "local_encoder_state_dict": self.local_encoder.state_dict(),
                "ship_encoder_state_dict": self.ship_encoder.state_dict(),
                "advantage_net_state_dict": self.advantage_net.state_dict(),
                "value_net_state_dict": self.value_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.global_encoder.load_state_dict(checkpoint["global_encoder_state_dict"])
        self.local_encoder.load_state_dict(checkpoint["local_encoder_state_dict"])
        self.ship_encoder.load_state_dict(checkpoint["ship_encoder_state_dict"])
        self.advantage_net.load_state_dict(checkpoint["advantage_net_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]

    def end_episode(self):
        """Call this at the end of each episode to reset episode-specific metrics"""
        pass
