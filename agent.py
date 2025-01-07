from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from luxai_s3.state import EnvObs

from core.base_agent import BaseAgent
from core.space import get_tile_info, preprocess_obs, update_space
from lux.utils import direction_to


def select_action(rng_key):
    rng_key, subkey = jax.random.split(rng_key)

    def random_action(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        rand_act = jax.random.randint(subkey, shape=(), minval=0, maxval=6)
        del subkey
        return rand_act, rng_key

    return random_action(rng_key)


def _act(
    state,
    obs,
):
    actions = np.zeros((state.MAX_UNITS, 3), dtype=int)
    if jnp.any(jnp.logical_not(state.space.explored_for_relic)):
        unknown_tiles = jnp.argwhere(jnp.logical_not(state.space.explored_for_relic))
        for unit in range(state.MAX_UNITS):
            x = unknown_tiles[unit][1]
            y = unknown_tiles[unit][0]
            pos = obs["units"]["position"][state.team_id][unit]
            action = direction_to(pos, [x, y])
            if action is not None:
                if action == 1:
                    pos[1] -= 1
                elif action == 2:
                    pos[0] += 1
                elif action == 3:
                    pos[1] += 1
                elif action == 4:
                    pos[0] -= 1
            next_tile_info = get_tile_info(state.space, pos[0], pos[1])
            if next_tile_info.is_walkable:
                actions[unit] = [action, x, y]
                continue
            else:
                actions[unit], rng_key = select_action(state.rng_key)
                state = state.replace(rng_key=rng_key)
                continue
    return actions, state


def _dummy_act(state, obs):
    actions = jnp.zeros((state.MAX_UNITS, 3), dtype=jnp.int32)
    rng_key = state.rng_key

    def random_action(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        rand_act = jax.random.randint(subkey, shape=(), minval=0, maxval=6)
        del subkey
        return rand_act, rng_key

    def for_i_body(i, carry):
        actions, rng_key = carry
        action, rng_key = random_action(rng_key)
        actions = actions.at[i].set([action, 0, 0])
        return (actions, rng_key)

    actions, rng_key = jax.lax.fori_loop(
        0, state.MAX_UNITS, for_i_body, (actions, rng_key)
    )
    state = state.replace(rng_key=rng_key)
    return actions, state


class Agent(BaseAgent):
    def __init__(self, player: str, env_cfg: dict):
        super().__init__(player, env_cfg, batch_size=1)

    def act(
        self, step: int, obs: Optional[dict | EnvObs], remainingOverageTime: int = 60
    ):
        obs = preprocess_obs(obs)
        state = self.state
        match_step = step % state.MAX_STEPS_IN_MATCH
        state = state.replace(
            match_score=obs["team_points"][state.team_id],
            match_step=match_step,
            game_step=step,
            space=update_space(state.space, obs),
        )
        actions, new_state = _dummy_act(state, obs)
        self.state = new_state
        return actions
