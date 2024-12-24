import copy
import os
import random
from collections import deque

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
        self.reward_history = deque(
            maxlen=100
        )  # Keep track of last 100 rewards for scaling

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Extract reward value from dict if needed
        if isinstance(reward, dict):
            if "reward" in reward:
                reward_value = float(reward["reward"])  # Ensure it's a float
            elif "nodes" in reward:
                # Handle the case where reward contains nodes
                if isinstance(reward["nodes"], (set, list)):
                    # Use the number of nodes as a reward component
                    nodes_reward = float(len(reward["nodes"]))
                    # If there's an explicit reward value, use that too
                    explicit_reward = float(reward.get("reward", 0))
                    reward_value = nodes_reward + explicit_reward
                else:
                    print(
                        f"Warning: Unexpected nodes format in reward: {reward}, using 0"
                    )
                    reward_value = 0.0
            else:
                print(f"Warning: Unexpected reward format: {reward}, using 0")
                reward_value = 0.0
        else:
            reward_value = float(reward)  # Ensure it's a float

        # Clip reward to reasonable range to prevent instability
        reward_value = max(min(reward_value, 100.0), -100.0)

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
        std_reward = (
            np.std(self.reward_history) + 1e-8
        )  # Add small epsilon to avoid division by zero

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
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 100  # how often to update target network

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
        self.optimizer = optim.Adam(params, lr=0.005)

        # Memory
        self.memory = ImitationReplayBuffer(50000)

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

    def forward(self, state, valid_actions_mask=None):
        """
        Forward pass through the network using Dueling DQN architecture

        Args:
            state: Either a state dictionary or a pre-encoded tensor
            valid_actions_mask: Optional boolean mask for valid actions (batch_size, action_size)

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

        # Apply action masking if provided
        if valid_actions_mask is not None:
            # Set Q-values of invalid actions to a large negative number
            invalid_actions_mask = ~valid_actions_mask
            q_values = q_values.masked_fill(invalid_actions_mask, float("-inf"))

        return q_values

    def select_action(self, state, valid_actions_mask=None):
        """Select action using epsilon-greedy policy with action masking"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if valid_actions_mask is not None:
                    valid_actions_mask = (
                        torch.FloatTensor(valid_actions_mask)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                q_values = self(state_tensor, valid_actions_mask)
                return q_values.argmax().item()
        else:
            if valid_actions_mask is not None:
                # Only select from valid actions during exploration
                valid_indices = np.where(valid_actions_mask)[0]
                return np.random.choice(valid_indices)
            return random.randrange(self.action_size)

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

    def _calculate_state_size(self):
        """
        Calculate the size of the state space with enhanced features.
        Features per ship:

        Ship Features (4):
        - Energy level (1)
        - Position x, y (2)
        - Number of sap targets (1)

        Global Parameters (4):
        - Unit move cost (1)
        - Unit sap cost (1)
        - Unit sensor range (1)
        - Max units (1)

        Enemy Information (7):
        - Distance to closest enemy (1)
        - Energy of closest enemy (1)
        - Enemy centroid position x, y (2)
        - Number of visible enemies (1)
        - Average enemy energy (1)
        - Total enemy energy (1)

        Local Environment Features per cell in 3x3 grid (6 * 9 = 54):
        - Is asteroid (1)
        - Has relic (1)
        - Has reward (1)
        - Energy value (1)
        - Enemy presence (1)
        - Enemy energy (1)

        Total Features: 4 + 4 + 7 + 54 = 69

        Returns:
            int: Size of the state space
        """
        # Ship features
        ship_features = 4  # energy, x, y, sap_targets

        # Global parameters
        global_params = 4  # move_cost, sap_cost, sensor_range, max_units

        # Enemy information
        enemy_info = 7  # closest_dist, closest_energy, centroid_x, centroid_y, num_visible, avg_energy, total_energy

        # Local environment features (3x3 grid)
        features_per_cell = (
            6  # asteroid, relic, reward, energy, enemy_present, enemy_energy
        )
        local_grid_size = 9  # 3x3 grid
        local_features = features_per_cell * local_grid_size

        total_size = ship_features + global_params + enemy_info + local_features
        return total_size

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
        # Enhanced enemy information
        closest_enemy_dist = float("inf")
        closest_enemy_energy = 0
        enemy_centroid_x = 0
        enemy_centroid_y = 0
        num_visible_enemies = 0
        total_enemy_energy = 0

        for enemy_ship in opp_fleet.ships:
            if enemy_ship.node:
                dist = abs(ship.node.x - enemy_ship.node.x) + abs(
                    ship.node.y - enemy_ship.node.y
                )
                if dist < closest_enemy_dist:
                    closest_enemy_dist = dist
                    closest_enemy_energy = enemy_ship.energy

                enemy_centroid_x += enemy_ship.node.x
                enemy_centroid_y += enemy_ship.node.y
                num_visible_enemies += 1
                total_enemy_energy += enemy_ship.energy

        if num_visible_enemies > 0:
            enemy_centroid_x /= num_visible_enemies
            enemy_centroid_y /= num_visible_enemies
            avg_enemy_energy = total_enemy_energy / num_visible_enemies
        else:
            enemy_centroid_x = Global.SPACE_SIZE / 2
            enemy_centroid_y = Global.SPACE_SIZE / 2
            avg_enemy_energy = 0

        # Add enemy features
        features.extend(
            [
                closest_enemy_dist
                / Global.SPACE_SIZE,  # Normalized distance to closest enemy
                closest_enemy_energy / 100.0,  # Normalized energy of closest enemy
                enemy_centroid_x
                / Global.SPACE_SIZE,  # Normalized enemy centroid position
                enemy_centroid_y / Global.SPACE_SIZE,
                num_visible_enemies
                / Global.MAX_UNITS,  # Normalized count of visible enemies
                avg_enemy_energy / 100.0,  # Normalized average enemy energy
                total_enemy_energy
                / (100.0 * Global.MAX_UNITS),  # Normalized total enemy energy
            ]
        )

        # Local environment features with enemy presence
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = ship.node.x + dx, ship.node.y + dy
                if 0 <= x < Global.SPACE_SIZE and 0 <= y < Global.SPACE_SIZE:
                    node = space.get_node(x, y)
                    # Check for enemy presence in this cell
                    enemy_present = any(
                        enemy.node and enemy.node.x == x and enemy.node.y == y
                        for enemy in opp_fleet.ships
                    )
                    enemy_energy = sum(
                        enemy.energy
                        for enemy in opp_fleet.ships
                        if enemy.node and enemy.node.x == x and enemy.node.y == y
                    )

                    features.extend(
                        [
                            1.0 if node.type == NodeType.asteroid else 0.0,
                            1.0 if node.relic else 0.0,
                            1.0 if node.reward else 0.0,
                            node.energy / 100.0 if node.energy is not None else 0.0,
                            1.0 if enemy_present else 0.0,  # Enemy presence indicator
                            enemy_energy / 100.0,  # Normalized enemy energy in cell
                        ]
                    )
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def train(
        self,
        state_dict,
        action,
        reward,
        next_state_dict,
        done,
        valid_actions_mask=None,
        next_valid_actions_mask=None,
    ):
        """Train the model on a single step of experience using Double DQN."""
        # Convert states to tensors
        state = self.encode_state(state_dict)
        next_state = self.encode_state(next_state_dict)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done = torch.tensor([done], device=self.device, dtype=torch.float32)

        if valid_actions_mask is not None:
            valid_actions_mask = torch.tensor(valid_actions_mask, device=self.device)
        if next_valid_actions_mask is not None:
            next_valid_actions_mask = torch.tensor(
                next_valid_actions_mask, device=self.device
            )

        # Get current Q values
        current_q_values = self(state, valid_actions_mask)
        current_q_value = current_q_values.gather(1, action.unsqueeze(-1))

        # Compute target Q values using Double DQN
        with torch.no_grad():
            # Get next action using online network
            next_features_online = self.feature_net(next_state)
            next_advantage_online = self.advantage_net(next_features_online)
            next_value_online = self.value_net(next_features_online)
            next_q_values_online = next_value_online + (
                next_advantage_online - next_advantage_online.mean(dim=1, keepdim=True)
            )

            if next_valid_actions_mask is not None:
                next_q_values_online = next_q_values_online.masked_fill(
                    ~next_valid_actions_mask, float("-inf")
                )

            # Get best action from online network
            next_action = next_q_values_online.argmax(1, keepdim=True)

            # Get Q values from target network
            next_features_target = self.target_feature_net(next_state)
            next_advantage_target = self.target_advantage_net(next_features_target)
            next_value_target = self.target_value_net(next_features_target)
            next_q_values_target = next_value_target + (
                next_advantage_target - next_advantage_target.mean(dim=1, keepdim=True)
            )

            if next_valid_actions_mask is not None:
                next_q_values_target = next_q_values_target.masked_fill(
                    ~next_valid_actions_mask, float("-inf")
                )

            # Use Q-value of best action from target network
            next_q_value = next_q_values_target.gather(1, next_action)

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
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

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
