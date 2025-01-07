import dataclasses
from functools import partial
from typing import Dict, Optional, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
from luxai_s3.params import EnvParams
from luxai_s3.state import EnvState
from luxai_s3.utils import to_numpy


def get_action_dim(action_sample):
    """Calculate total action dimension from action sample"""
    player_0_actions = action_sample["player_0"]
    action_shape = player_0_actions.shape
    return action_shape[0] * action_shape[1]


@jax.jit
def extract_unit_state(unit):
    """Extract and flatten unit state into a 1D array.

    Args:
        unit: UnitState object with position (shape: [T,N,2] for T teams, N units)
              and energy (shape: [T,N] for T teams, N units)

    Returns:
        1D JAX array containing flattened [position, energy] for all units
    """
    # Reshape position to (-1, 2) and energy to (-1)
    pos_flat = unit.position.reshape(-1, 2)
    energy_flat = jnp.array(unit.energy, dtype=jnp.float32).reshape(-1)

    # Combine position and energy for each unit
    return jnp.concatenate([pos_flat.reshape(-1), energy_flat])


@jax.jit
def extract_map_tile(tile):
    """Extract and flatten map tile into a 1D array.

    Args:
        tile: MapTile object with energy and tile_type (both shape: [H,W])

    Returns:
        1D JAX array containing flattened [energy, tile_type] for all tiles
    """
    energy_flat = jnp.array(tile.energy, dtype=jnp.float32).reshape(-1)
    type_flat = jnp.array(tile.tile_type, dtype=jnp.float32).reshape(-1)
    return jnp.concatenate([energy_flat, type_flat])


@jax.jit
def flatten_env_obs(player_0_obs, player_1_obs):
    """Flatten environment observations into a single array.

    Args:
        player_0_obs: EnvObs object for player 0
        player_1_obs: EnvObs object for player 1

    Returns:
        1D JAX array containing concatenated observations
    """
    # Pre-compute vmapped functions
    v_extract_unit = jax.vmap(extract_unit_state)
    v_extract_map = jax.vmap(extract_map_tile)

    def flatten_single_obs(obs):
        """Flatten a single player's observations."""
        arrays = [
            v_extract_unit(obs.units).reshape(-1),
            obs.units_mask.reshape(-1),
            obs.sensor_mask.reshape(-1),
            v_extract_map(obs.map_features).reshape(-1),
            obs.relic_nodes.reshape(-1),
            obs.relic_nodes_mask.reshape(-1),
            obs.team_points.reshape(-1),
            obs.team_wins.reshape(-1),
            jnp.array([obs.steps, obs.match_steps], dtype=jnp.float32),
        ]
        return jnp.concatenate(arrays)

    # Flatten each player's observations
    player_0_flat = flatten_single_obs(player_0_obs)
    player_1_flat = flatten_single_obs(player_1_obs)

    # Concatenate both players' observations
    return jnp.concatenate([player_0_flat, player_1_flat])


class LuxWrapper:
    """Base wrapper class for LuxAIS3Env."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


class LuxFlattenWrapper(LuxWrapper):
    """Wrapper that flattens the observation space of the LuxAI environment."""

    def __init__(self, env):
        super().__init__(env)
        # Calculate observation dimension from a sample
        obs_dict, _ = env.reset(jax.random.PRNGKey(0))
        sample_obs = obs_dict["player_0"]
        self.flat_obs_dim = (
            sample_obs.units.position.size  # [16,2,16,2]
            + sample_obs.units.energy.size  # [16,2,16]
            + sample_obs.units_mask.size  # [16,2,16]
            + sample_obs.sensor_mask.size  # [16,24,24]
            + sample_obs.map_features.energy.size  # [16,24,24]
            + sample_obs.map_features.tile_type.size  # [16,24,24]
            + sample_obs.relic_nodes.size  # [16,6,2]
            + sample_obs.relic_nodes_mask.size  # [16,6]
            + sample_obs.team_points.size  # [16,2]
            + sample_obs.team_wins.size  # [16,2]
            + sample_obs.steps.size  # [16]
            + sample_obs.match_steps.size  # [16]
        )

    def _flatten_obs(self, obs):
        """Flatten a single observation."""
        # Concatenate all observation components and ensure 1D
        flat = jnp.concatenate(
            [
                obs.units.position.reshape(-1),
                obs.units.energy.reshape(-1),
                obs.units_mask.reshape(-1),
                obs.sensor_mask.reshape(-1),
                obs.map_features.energy.reshape(-1),
                obs.map_features.tile_type.reshape(-1),
                obs.relic_nodes.reshape(-1),
                obs.relic_nodes_mask.reshape(-1),
                obs.team_points.reshape(-1),
                obs.team_wins.reshape(-1),
                obs.steps.reshape(-1),
                obs.match_steps.reshape(-1),
            ]
        )
        return flat.ravel()  # Ensure 1D array

    def reset(self, key, params=None):
        obs_dict, state = self._env.reset(key, params)
        # Flatten observations for each agent
        flat_obs = {agent: self._flatten_obs(obs) for agent, obs in obs_dict.items()}
        return flat_obs, state

    def step(self, key, state, action):
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action
        )
        # Flatten observations for each agent
        flat_obs = {agent: self._flatten_obs(o) for agent, o in obs.items()}
        return flat_obs, state, reward, terminated, truncated, info


class LuxMultiAgentWrapper(LuxWrapper):
    def __init__(
        self, env, num_agents: int = 2, agents=None, numpy_output: bool = False
    ):
        super().__init__(env)
        self.numpy_output = numpy_output
        self.rng_key = jax.random.key(0)
        self.num_agents = num_agents
        self.env_cfg = None
        if agents is None:
            self.agents = [f"player_{i}" for i in range(num_agents)]
        else:
            self.agents = agents
        self.action_spaces = self._env.action_space().spaces
        self.observation_spaces = {
            k: self._env.observation_space(self._env.fixed_env_params)
            for k in self.agents
        }

    def step(self, key, state, actions, params=None):
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, actions, params
        )
        reward = {}
        for player in obs:
            team_points = obs[player].team_points
            reward[player] = team_points[0] - team_points[1]
        return obs, state, reward, terminated, truncated, info

    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """
        Reset the environment to its initial state and return the initial observations.

        Args:
            key (int | None): The seed to use for the random number generator.
            options (dict[str, Any] | None): The options to use for the environment.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observations and the options.
        """
        if params is None:
            params = self._env.default_params
        obs, self.state = self._env.reset(key, params=params)
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
        params_dict = dataclasses.asdict(params)
        params_dict_kept = dict()
        for k in [
            "max_units",
            "match_count_per_episode",
            "max_steps_in_match",
            "map_height",
            "map_width",
            "num_teams",
            "unit_move_cost",
            "unit_sap_cost",
            "unit_sap_range",
            "unit_sensor_range",
        ]:
            params_dict_kept[k] = params_dict[k]
        self.env_cfg = params_dict_kept
        return obs, self.state


class VecEnv:
    """Vectorized environment wrapper that runs multiple environments in parallel."""

    def __init__(self, env_single, num_envs: int):
        """Initialize vectorized environment.

        Args:
            env_single: Single environment to vectorize
            num_envs: Number of parallel environments
        """
        self._env = env_single
        self.num_envs = num_envs

        # Get environment properties from single env
        self.flat_obs_dim = getattr(env_single, "flat_obs_dim", None)
        self.action_space = env_single.action_space

        # Create vectorized functions
        self.v_reset = jax.vmap(env_single.reset)
        self.v_step = jax.vmap(env_single.step)

    def reset(self, keys):
        """Reset all environments.

        Args:
            keys: JAX random keys for each environment

        Returns:
            Tuple of (observations, states)
        """
        return self.v_reset(keys)

    def step(self, keys, states, actions):
        """Step all environments.

        Args:
            keys: JAX random keys for each environment
            states: Current states of all environments
            actions: Actions for all environments

        Returns:
            Tuple of (next_observations, next_states, rewards, terminated, truncated, info)
        """
        return self.v_step(keys, states, actions)


class LuxLogWrapper(LuxWrapper):
    """Tracks episode stats."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, Dict]:
        obs, state = self._env.reset(key)
        log_state = {
            "env_state": state,
            "episode_returns": 0.0,
            "episode_lengths": 0,
            "timestep": 0,
        }
        return obs, log_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: Dict,
        action: Dict,
    ) -> Tuple[chex.Array, Dict, float, bool, bool, Dict]:
        obs, env_state, reward, term, trunc, info = self._env.step(
            key, state["env_state"], action
        )
        done = term or trunc

        new_state = {
            "env_state": env_state,
            "episode_returns": state["episode_returns"] + reward,
            "episode_lengths": state["episode_lengths"] + 1,
            "timestep": state["timestep"] + 1,
        }

        info.update(
            {
                "episode_return": new_state["episode_returns"] if done else None,
                "episode_length": new_state["episode_lengths"] if done else None,
                "timestep": new_state["timestep"],
            }
        )

        return obs, new_state, reward, term, trunc, info


def test_observation_flattening(env):
    """Test the observation flattening functions and print shape information.

    This function:
    1. Gets a sample observation from the environment
    2. Flattens it using our functions
    3. Verifies the structure and shapes
    4. Prints detailed information about the flattened arrays

    Args:
        env: LuxAI environment instance

    Returns:
        None, but prints validation information
    """
    # Get a sample observation
    obs, _ = env.reset(jax.random.PRNGKey(0))

    # Test unit state extraction
    unit = obs["player_0"].units
    unit_flat = extract_unit_state(unit)
    print("\nUnit state flattening:")
    print(f"Original unit - position: {unit.position.shape}, energy: {unit.energy}")
    print(f"Flattened unit shape: {unit_flat.shape}")

    # Test map tile extraction
    tile = obs["player_0"].map_features
    tile_flat = extract_map_tile(tile)
    print("\nMap tile flattening:")
    print(f"Original tile - energy: {tile.energy}, type: {tile.tile_type}")
    print(f"Flattened tile shape: {tile_flat.shape}")

    # Test player observation flattening
    player_flat = flatten_env_obs(obs["player_0"], obs["player_1"])
    print("\nPlayer observation flattening:")
    print(f"Flattened player obs shape: {player_flat.shape}")

    # Verify consistency
    assert player_flat.shape[0] == 2 * (
        player_flat.shape[0] // 2
    ), "Player observations should have same shape"
    print("\nAll shapes verified successfully!")
