import cProfile
import pstats
import time

import jax
import jax.numpy as jnp
import numpy as np
from luxai_s3.env import LuxAIS3Env
from luxai_s3.wrappers import LuxAIS3GymEnv


def print_obs_structure(obs):
    """Print the structure of the observation dictionary."""

    def print_array_info(name, arr):
        if isinstance(arr, (np.ndarray, jax.Array)):
            print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
        elif hasattr(arr, "position") and hasattr(arr, "energy"):  # UnitState
            print(f"{name} (UnitState):")
            print(f"  position: shape={arr.position.shape}, dtype={arr.position.dtype}")
            print(f"  energy: {arr.energy}")
        elif hasattr(arr, "energy") and hasattr(arr, "tile_type"):  # MapTile
            print(f"{name} (MapTile):")
            print(f"  energy: {arr.energy}")
            print(f"  tile_type: {arr.tile_type}")
        else:
            print(f"{name}: type={type(arr)}")

    print("\nObservation structure for each player:")
    for player in ["player_0", "player_1"]:
        print(f"\n{player}:")
        env_obs = obs[player]
        print_array_info("  units", env_obs.units)
        print_array_info("  units_mask", env_obs.units_mask)
        print_array_info("  sensor_mask", env_obs.sensor_mask)
        print_array_info("  map_features", env_obs.map_features)
        print_array_info("  relic_nodes", env_obs.relic_nodes)
        print_array_info("  relic_nodes_mask", env_obs.relic_nodes_mask)
        print_array_info("  team_points", env_obs.team_points)
        print_array_info("  team_wins", env_obs.team_wins)
        print(f"  steps: {env_obs.steps}")
        print(f"  match_steps: {env_obs.match_steps}")


def run_pure_jax_environment(num_steps=100):
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)
    env = LuxAIS3Env()
    obs, state = env.reset(key_reset)

    start_time = time.time()
    step_times = []
    for i in range(num_steps):
        step_start = time.time()
        action = env.action_space().sample(key_act)
        n_obs, n_state, reward, terminated, truncated, info = env.step(
            key_step, state, action
        )
        step_end = time.time()
        step_times.append(step_end - step_start)

        if terminated or truncated:
            obs, state = env.reset(key_reset)

    total_time = time.time() - start_time
    return {
        "total_time": total_time,
        "avg_step_time": sum(step_times) / len(step_times),
        "min_step_time": min(step_times),
        "max_step_time": max(step_times),
        "steps_per_second": (num_steps) / total_time,
    }


def run_environment(num_steps=1000):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset()

    start_time = time.time()
    step_times = []

    for i in range(num_steps):
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        step_end = time.time()
        step_times.append(step_end - step_start)

        if terminated or truncated:
            obs, info = env.reset()

    total_time = time.time() - start_time
    env.close()
    return {
        "total_time": total_time,
        "avg_step_time": sum(step_times) / len(step_times),
        "min_step_time": min(step_times),
        "max_step_time": max(step_times),
        "steps_per_second": (num_steps) / total_time,
    }


def run_vectorized_jax_environment(num_envs=10, num_steps=1000):
    # Create a single PRNG key and split it for all envs
    rng = jax.random.PRNGKey(0)
    reset_keys = jax.random.split(rng, num_envs)

    # Initialize environments
    env = LuxAIS3Env(auto_reset=True)
    # env = LuxLogWrapper(env)
    # Vectorize over key, state, action

    v_sample = jax.jit(jax.vmap(env.action_space.sample))

    # Reset all environments
    obs_batch, state_batch = env.reset(reset_keys)

    start_time = time.time()
    step_times = []

    for i in range(num_steps):
        step_start = time.time()

        # Generate new keys for actions and steps
        rng, *step_keys = jax.random.split(rng, num_envs + 1)
        step_keys = jnp.array(step_keys)
        # Sample actions for all environments for player_0
        actions = v_sample(step_keys)

        # Step all environments - with auto_reset=True, this will handle resets automatically
        step_outputs = env.step(step_keys, state_batch, actions)
        n_obs_batch, n_state_batch, rewards, terminated, truncated, infos = step_outputs

        step_end = time.time()
        step_times.append(step_end - step_start)

        # Update state batch - no need to handle resets manually
        state_batch = n_state_batch

    total_time = time.time() - start_time
    return {
        "total_time": total_time,
        "avg_step_time": sum(step_times) / len(step_times),
        "min_step_time": min(step_times),
        "max_step_time": max(step_times),
        "steps_per_second": (num_steps * num_envs) / total_time,
    }


def run_profiling(
    num_envs: int = 10,
    num_steps: int = 1000,
    mode: str = "timing",
    envs: list[str] = ["pure_jax", "vec_jax"],
) -> None:
    """
    Run profiling on different environments.

    Parameters
    ----------

    num_envs (int) = 10: Number of environments to run in parallel.
    num_steps (int) = 1000: Number of steps to run per environment.
    mode (str) = "timing": 'timing' or 'profiling'.
    envs (list[str]) = ["pure_jax", "vec_jax"]: List of environments to run profiling on.

    Returns
    -------
        None
    """
    # # Choose mode: 'timing' or 'profiling'

    if mode == "timing" and "pure_jax" in envs:
        print("Running single environment:")
        stats = run_pure_jax_environment(num_steps=num_steps)
        print(f"Total time: {stats['total_time']:.4f} seconds")
        print(f"Average step time: {stats['avg_step_time']:.4f} seconds")
        print(f"Min step time: {stats['min_step_time']:.4f} seconds")
        print(f"Max step time: {stats['max_step_time']:.4f} seconds")
        print(f"Steps per second: {stats['steps_per_second']:.4f}")
    elif mode == "profiling" and "pure_jax" in envs:
        profiler = cProfile.Profile()
        profiler.enable()
        run_pure_jax_environment(num_steps=num_steps)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(20)

    if mode == "timing" and "vec_jax" in envs:
        print("\nRunning vectorized JAX environments:")
        stats = run_vectorized_jax_environment(num_envs=num_envs, num_steps=num_steps)
        print(f"Total time: {stats['total_time']:.4f} seconds")
        print(f"Average step time: {stats['avg_step_time']:.4f} seconds")
        print(f"Min step time: {stats['min_step_time']:.4f} seconds")
        print(f"Max step time: {stats['max_step_time']:.4f} seconds")
        print(f"Steps per second: {stats['steps_per_second']:.4f}")
    elif mode == "profiling" and "vec_jax" in envs:
        print("\nRunning vectorized JAX environments:")
        profiler = cProfile.Profile()
        profiler.enable()
        run_vectorized_jax_environment(num_envs=num_envs, num_steps=num_steps)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(20)


if __name__ == "__main__":
    run_profiling(num_steps=1000, num_envs=10, envs=["pure_jax", "vec_jax"])
