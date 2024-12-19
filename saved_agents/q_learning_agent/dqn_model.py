import copy
import os
import random
from collections import deque
from sys import stderr

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from base import Global, NodeType

from .logger import TrainingLogger


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
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.reward_history = deque(maxlen=100)  # Keep track of last 100 rewards for scaling

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Extract reward value from dict if needed
        if isinstance(reward, dict):
            if "reward" in reward:
                reward_value = reward["reward"]
            else:
                print(f"Warning: Unexpected reward format: {reward}, using 0")
                reward_value = 0
        else:
            reward_value = reward

        # Store experience
        self.buffer.append((state, action, reward_value, next_state, done))
        self.reward_history.append(reward_value)

    def sample(self, batch_size):
        """Sample a batch of experiences from memory."""
        if len(self.buffer) < batch_size:
            return None

        # Sample random batch
        combined_batch = random.sample(self.buffer, batch_size)

        # Separate into individual components
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for state, action, reward, next_state, done in combined_batch:
            states.append(state)  # Keep state as dictionary
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)  # Keep next_state as dictionary
            dones.append(done)

        return (
            states,
            torch.tensor(actions),
            torch.tensor(rewards),
            next_states,
            torch.tensor(dones),
        )

    def scale_reward(self, reward):
        """Scale reward based on recent history."""
        if len(self.reward_history) < 100:  # Wait for enough samples
            return reward

        mean_reward = np.mean(self.reward_history)
        std_reward = np.std(self.reward_history) + 1e-8  # Add small epsilon to avoid division by zero
        
        return (reward - mean_reward) / std_reward


class DQNAgent(nn.Module):
    """Deep Q-Network Agent"""

    def __init__(self, state_size, action_size, player, env_cfg):
        super(DQNAgent, self).__init__()

        # Model parameters
        self.state_size = state_size
        self.action_size = action_size
        self.player = player
        self.env_cfg = env_cfg

        print(f"State size: {state_size}, Action size: {action_size}")  # Debug print

        # Training parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10  # how often to update target network

        # Device setup
        if torch.cuda.is_available():
            print(f"GPU:{torch.cuda.get_device_name(0)} is available.")
            self.device = torch.device("cuda")
            self.train_device = torch.device("cuda")
        else:
            print("No GPU available. Training will run on cpu.")
            self.device = torch.device("cpu")
            self.train_device = torch.device("cpu")

        # Networks
        self.hidden_dim = 256

        # Single network to process all features
        self.feature_net = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        ).to(self.train_device)

        # Advantage and value networks for dueling architecture
        self.advantage_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.action_size),
        ).to(self.train_device)

        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        ).to(self.train_device)

        # Create target networks
        self.target_feature_net = copy.deepcopy(self.feature_net)
        self.target_advantage_net = copy.deepcopy(self.advantage_net)
        self.target_value_net = copy.deepcopy(self.value_net)

        # Optimizer
        params = (
            list(self.feature_net.parameters())
            + list(self.advantage_net.parameters())
            + list(self.value_net.parameters())
        )
        self.optimizer = optim.Adam(params, lr=0.001)

        # Memory
        self.memory = ImitationReplayBuffer(10000)

        # Metrics tracking
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.episodes = 0
        self.logger = TrainingLogger(os.path.join(os.path.dirname(__file__), "logs"))

    def to_device(self, device):
        """Move the model to the specified device"""
        self.device = device
        self.feature_net = self.feature_net.to(device)
        self.advantage_net = self.advantage_net.to(device)
        self.value_net = self.value_net.to(device)
        self.target_feature_net = self.target_feature_net.to(device)
        self.target_advantage_net = self.target_advantage_net.to(device)
        self.target_value_net = self.target_value_net.to(device)
        return self

    def encode_state(self, state_dict):
        """
        Encode the state dictionary into a format suitable for the neural network

        Args:
            state_dict: Dictionary containing global and ship-specific features

        Returns:
            torch.Tensor: Encoded state
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(next(iter(state_dict["global_features"].values())), np.ndarray):
            state_dict = {
                "global_features": {
                    k: torch.from_numpy(v).float().to(self.device)
                    for k, v in state_dict["global_features"].items()
                },
                "ship_features": {
                    k: torch.from_numpy(v).float().to(self.device)
                    if isinstance(v, np.ndarray)
                    else torch.tensor(v, dtype=torch.float32).to(self.device)
                    for k, v in state_dict["ship_features"].items()
                },
            }

        # Concatenate all features
        features = []

        # Global features
        for v in state_dict["global_features"].values():
            if v.dim() == 1:
                v = v.unsqueeze(0)
            features.append(v.flatten())

        # Ship features
        for v in state_dict["ship_features"].values():
            if isinstance(v, list):
                v = torch.tensor(v, dtype=torch.float32).to(self.device)
            if v.dim() == 0:
                v = v.unsqueeze(0)
            features.append(v.flatten())

        x = torch.cat(features, dim=0)

        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return x

    def forward(self, state):
        """
        Forward pass through the network using Dueling DQN architecture

        Args:
            state: Either a state dictionary or a pre-encoded tensor

        Returns:
            torch.Tensor: Q-values for each action
        """
        # If state is a dictionary, encode it first
        if isinstance(state, dict):
            x = self.encode_state(state)
        else:
            x = state
            # Add batch dimension if needed
            if x.dim() == 1:
                x = x.unsqueeze(0)

        # Forward pass through networks
        features = self.feature_net(x)
        advantage = self.advantage_net(features)
        value = self.value_net(features)

        # Combine using dueling architecture
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
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

    def train(self, state_dict, action, reward, next_state_dict, done):
        """Train the model on a single step of experience."""
        # Convert states to tensors
        state = self.encode_state(state_dict)
        next_state = self.encode_state(next_state_dict)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done = torch.tensor([done], device=self.device, dtype=torch.float32)

        # Get current Q values
        current_q_values = self(state)
        current_q_value = current_q_values.gather(1, action.unsqueeze(-1))

        # Compute target Q values
        with torch.no_grad():
            # Get next state values using target network
            next_features = self.target_feature_net(next_state)
            next_advantage = self.target_advantage_net(next_features)
            next_value = self.target_value_net(next_features)
            next_q_values = next_value + (
                next_advantage - next_advantage.mean(dim=1, keepdim=True)
            )

            # Get max Q value for next state
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)

            # Compute target Q value using Bellman equation
            target_q_value = (
                reward.unsqueeze(1)
                + (1 - done.unsqueeze(1)) * self.gamma * next_q_value
            )

        # Compute loss
        loss = F.smooth_l1_loss(current_q_value, target_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.total_steps += 1
        if self.total_steps % self.target_update == 0:
            self.target_feature_net.load_state_dict(self.feature_net.state_dict())
            self.target_advantage_net.load_state_dict(self.advantage_net.state_dict())
            self.target_value_net.load_state_dict(self.value_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Log metrics
        self.episode_reward += reward.item()
        self.episode_steps += 1

        metrics = {
            "loss": loss.item(),
            "q_value": current_q_value.mean().item(),
            "reward": reward.item(),
            "epsilon": self.epsilon,
        }
        self.logger.log_metrics(metrics, self.total_steps)

    def end_episode(self):
        """Log episode metrics and reset episode-specific counters."""
        episode_metrics = {
            "episode_reward": self.episode_reward,
            "episode_steps": self.episode_steps,
            "episodes": self.episodes,
        }
        self.logger.log_episode_metrics(episode_metrics, self.episodes)

        # Reset episode counters
        self.episode_reward = 0
        self.episode_steps = 0
        self.episodes += 1

    def save(self, path):
        """
        Save the model's state dictionaries to the specified path

        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        try:
            os.makedirs(os.path.dirname(path))
        except Exception as e:
            print(f"Error creating directory: {e}", file=stderr)
        # Save all network states
        torch.save(
            {
                "feature_net_state_dict": self.feature_net.state_dict(),
                "advantage_net_state_dict": self.advantage_net.state_dict(),
                "value_net_state_dict": self.value_net.state_dict(),
                "target_feature_net_state_dict": self.target_feature_net.state_dict(),
                "target_advantage_net_state_dict": self.target_advantage_net.state_dict(),
                "target_value_net_state_dict": self.target_value_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "episodes": self.episodes,
                "epsilon": self.epsilon,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load the model's state dictionaries from the specified path

        Args:
            path (str): Path to load the model from
        """
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return False

        checkpoint = torch.load(path)

        # Load network states
        self.feature_net.load_state_dict(checkpoint["feature_net_state_dict"])
        self.advantage_net.load_state_dict(checkpoint["advantage_net_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.target_feature_net.load_state_dict(
            checkpoint["target_feature_net_state_dict"]
        )
        self.target_advantage_net.load_state_dict(
            checkpoint["target_advantage_net_state_dict"]
        )
        self.target_value_net.load_state_dict(checkpoint["target_value_net_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load training state
        self.total_steps = checkpoint["total_steps"]
        self.episodes = checkpoint["episodes"]
        self.epsilon = checkpoint["epsilon"]

        print(f"Model loaded from {path}")
        return True

    def __del__(self):
        """Cleanup when the agent is destroyed."""
        if hasattr(self, "logger"):
            self.logger.close()
