import os
import time

import jax
import jax.numpy as jnp
from luxai_s3.env import LuxAIS3Env

from core.space import Space, update_space_batch
from utils.wrappers import LuxMultiAgentWrapper


def create_space_batch(num_envs):
    space_list = [Space.create() for _ in range(num_envs)]
    batched_space = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *space_list
    )
    return batched_space


def make_benchmark(config):
    env = LuxAIS3Env(auto_reset=True)
    env = LuxMultiAgentWrapper(env)
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]

    # Create spaces for each player
    spaces = {
        "player_0": create_space_batch(config["NUM_ENVS"]),
        "player_1": create_space_batch(config["NUM_ENVS"]),
    }
    obs, info = env.reset(jax.random.PRNGKey(0))

    def benchmark(rng):
        def env_step(runner_state, unused):
            env_state, last_obs, rng, spaces = runner_state
            step = env_state.steps[0]  # Get the scalar value from the array

            spaces = {
                "player_0": update_space_batch(
                    spaces["player_0"], last_obs["player_0"]
                ),
                "player_1": update_space_batch(
                    spaces["player_1"], last_obs["player_1"]
                ),
            }

            # Rng for agent actions
            rng, rng_player_0, rng_player_1 = jax.random.split(rng, 3)

            # Split rng for each env
            rng_p0_batch = jax.random.split(rng_player_0, config["NUM_ENVS"])
            rng_p1_batch = jax.random.split(rng_player_1, config["NUM_ENVS"])

            def sample_p0_action(key):
                raw_action = env.action_spaces["player_0"].sample(key)
                # arr = jnp.array(
                #     [
                #         raw_action["action_type"],
                #         raw_action["dx"],
                #         raw_action["dy"],
                #     ],
                #     dtype=jnp.int32,
                # )
                return raw_action

            def sample_p1_action(key):
                raw_action = env.action_spaces["player_1"].sample(key)
                # arr = jnp.array(
                #     [
                #         raw_action["action_type"],
                #         raw_action["dx"],
                #         raw_action["dy"],
                #     ],
                #     dtype=jnp.int32,
                # )
                return raw_action

            p0_actions = jax.vmap(sample_p0_action)(rng_p0_batch)
            p1_actions = jax.vmap(sample_p1_action)(rng_p1_batch)

            # Get actions from agents
            actions = {"player_0": p0_actions, "player_1": p1_actions}

            # Step env
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, _, _, _, info = jax.vmap(env.step)(
                rng_step, env_state, actions
            )
            runner_state = (env_state, obsv, rng, spaces)
            return runner_state, None

        def init_runner_state(rng):
            # Init env
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset)(reset_rng)
            spaces = {
                "player_0": create_space_batch(config["NUM_ENVS"]),
                "player_1": create_space_batch(config["NUM_ENVS"]),
            }
            return (env_state, obsv, rng, spaces)

        rng, init_rng = jax.random.split(rng)
        runner_state = init_runner_state(init_rng)
        runner_state = jax.lax.scan(env_step, runner_state, None, config["NUM_STEPS"])
        return runner_state

    return benchmark


if __name__ == "__main__":
    try:
        os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
    except Exception as e:
        print(e)
    print(jax.devices())
    config = {
        "NUM_STEPS": 1000,
        "NUM_ENVS": 1000,
        "ACTIVATION": "relu",
        "ENV_KWARGS": {},
        "NUM_SEEDS": 1,
        "SEED": 0,
    }

    num_envs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    jaxmarl_sps = []
    for num in num_envs:
        config["NUM_ENVS"] = num
        benchmark_fn = jax.jit(make_benchmark(config))
        rng = jax.random.PRNGKey(config["SEED"])
        rng, _rng = jax.random.split(rng)
        benchmark_jit = jax.jit(benchmark_fn).lower(_rng).compile()
        before = time.perf_counter_ns()
        runner_state = jax.block_until_ready(benchmark_jit(_rng))
        after = time.perf_counter_ns()
        total_time = (after - before) / 1e9
        sps = config["NUM_STEPS"] * config["NUM_ENVS"] / total_time
        jaxmarl_sps.append(sps)

        print(f"JAXMARL, Num Envs: {num}, Total Time (s): {total_time}")
        print(
            f"JAXMARL, Num Envs: {num}, Total Steps: {config["NUM_STEPS"] * config["NUM_ENVS"]}"
        )
        print(f"JAXMARL, Num Envs: {num}, SPS: {sps}")
        print(f"JAXMARL, Num Envs: {num}, SPS: {sps}")
        print(f"JAXMARL, Num Envs: {num}, SPS: {sps}")
