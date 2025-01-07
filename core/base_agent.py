import random
from abc import ABC, abstractmethod
from collections import deque

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state

from core.space import Space


class DQN(nn.Module):
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class TrainStateDQN(train_state.TrainState):
    pass


@struct.dataclass
class AgentState:
    team_id: int
    opp_team_id: int
    rng_key: jnp.ndarray
    training: bool
    debug: bool

    # # DQN Hyperparameters
    # epsilon: float
    # epsilon_min: float
    # epsilon_decay: float
    # gamma: float
    # learning_rate: float

    # # DQN Model states
    # policy_state: TrainStateDQN
    # target_state: TrainStateDQN

    # Replay Buffer
    # memory: ReplayBuffer = struct.field(pytree_node=False)

    # Space representation
    space: Space

    # reward tracking
    match_score: int = 0
    game_score: int = 0
    match_step: int = 0
    game_step: int = 0

    # Game State
    # Parameters that may change game to game
    # these will be provided by the environment
    unit_sap_cost: int = 30  # OPTIONS: list(range(30, 51))
    unit_sap_range: int = 3  # OPTIONS: list(range(3, 8))
    unit_move_cost: int = 1  # OPTIONS: list(range(1, 6))
    unit_sensor_range: int = 2  # OPTIONS: list(range(2, 5))
    # these will need to be inferred from the environment
    unit_sap_dropoff_factor: float = -1  # OPTIONS: [0.25, 0.5, 1]
    unit_energy_void_factor: float = -1  # OPTIONS: [0.0625, 0.125, 0.25, 0.375]

    # Constants
    MAX_STEPS_IN_MATCH: int = 100
    MATCH_COUNT_PER_EPISODE: int = 5
    MAX_UNITS: int = 16
    SPAWN_RATE: int = 3

    INIT_UNIT_ENERGY: int = 100
    MIN_UNIT_ENERGY: int = 0
    MAX_UNIT_ENERGY: int = 400

    # Flags

    OBSTACLE_MOVEMENT_PERIOD_FOUND: bool = False
    OBSTACLE_MOVEMENT_DIRECTION_FOUND: bool = False

    # Others:

    # The energy on the unknown tiles will be used in the pathfinding
    HIDDEN_NODE_ENERGY: int = 0

    @classmethod
    def create(cls, env_cfg, agent_config, team_id, batch_size=1):
        opp_team_id = 1 - team_id
        rng_key = jax.random.PRNGKey(agent_config["seed"])

        space = Space.create()

        return cls(
            team_id=team_id,
            opp_team_id=opp_team_id,
            rng_key=rng_key,
            training=agent_config["training"],
            debug=agent_config["debug"],
            space=space,
            unit_move_cost=env_cfg["unit_move_cost"],
            unit_sap_cost=env_cfg["unit_sap_cost"],
            unit_sap_range=env_cfg["unit_sap_range"],
            unit_sensor_range=env_cfg["unit_sensor_range"],
        )


# @struct.dataclass
# class AgentConfig:
#     seed: jnp.random.PRNGKey
#     training: bool
#     debug: bool
#     eps: float
#     eps_min: float
#     eps_decay: float
#     gamma: float
#     lr: float
#     state_size: int
#     action_size: int
#     hidden_size: int
#     batch_size: int
#     mem_capacity: int

#     @classmethod
#     def create(cls, *args, **kwargs):
#         return cls(**kwargs)

# for holding agent specific configuration, in order to avoid passing them as arguments
agent_config = {
    "seed": 0,
    "training": False,
    "debug": False,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "gamma": 0.99,
    "learning_rate": 1e-4,
    "state_size": 6,
    "action_size": 6,
    "hidden_size": 128,
    "batch_size": 64,
    "memory_capacity": 10000,
}


class BaseAgent(ABC):
    def __init__(self, player, env_cfg, batch_size=1):
        self.player = player
        self.opp_player = "player_0" if player == "player_1" else "player_1"
        team_id = 0 if player == "player_0" else 1
        self.state = AgentState.create(env_cfg, agent_config, team_id, batch_size)

    @property
    def team_id(self):
        return self.state.team_id

    @property
    def opp_team_id(self):
        return self.state.opp_team_id

    @abstractmethod
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        raise NotImplementedError
