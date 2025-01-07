from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
from luxai_s3.state import EnvObs, MapTile, UnitState
from numpy import roll

from .base import SPACE_SIZE
from .node import Node


@jax.jit
def parse_envobs_to_obs(envobs: EnvObs) -> dict:
    return {
        "units": {
            "position": envobs.units.position,
            "energy": envobs.units.energy,
        },
        "units_mask": envobs.units_mask,
        "sensor_mask": envobs.sensor_mask,
        "map_features": {
            "energy": envobs.map_features.energy,
            "tile_type": envobs.map_features.tile_type,
        },
        "relic_nodes": envobs.relic_nodes,
        "relic_nodes_mask": envobs.relic_nodes_mask,
        "team_points": envobs.team_points,
        "team_wins": envobs.team_wins,
        "steps": envobs.steps,
        "match_steps": envobs.match_steps,
    }


@jax.jit
def parse_obs_to_envobs(obs: dict) -> EnvObs:
    unit_state = UnitState(
        position=obs["units"]["position"],
        energy=obs["units"]["energy"],
    )

    map_tile = MapTile(
        energy=obs["map_features"]["energy"],
        tile_type=obs["map_features"]["tile_type"],
    )

    return EnvObs(
        units=unit_state,
        units_mask=obs["units_mask"],
        sensor_mask=obs["sensor_mask"],
        map_features=map_tile,
        relic_nodes=obs["relic_nodes"],
        relic_nodes_mask=obs["relic_nodes_mask"],
        team_points=obs["team_points"],
        team_wins=obs["team_wins"],
        steps=obs["steps"],
        match_steps=obs["match_steps"],
    )


@jax.jit
def preprocess_obs(obs) -> dict:
    if isinstance(obs, EnvObs):
        return parse_envobs_to_obs(obs)
    return obs


@jax.jit
def validate_coords(x, y):
    return jax.numpy.logical_not(
        jax.numpy.logical_or(
            jax.numpy.logical_or(x < 0, x >= SPACE_SIZE),
            jax.numpy.logical_or(y < 0, y >= SPACE_SIZE),
        )
    )


@struct.dataclass
class Space:
    tile_type: jnp.ndarray
    energy: jnp.ndarray
    is_visible: jnp.ndarray
    relic_mask: jnp.ndarray
    drift_speed_found: bool = False
    drift_speed: float = 0.0
    step: int = 0

    # Some map related constants
    SPACE_SIZE = 24
    MAX_ENERGY_NODES: int = 6
    MAX_ENERGY_PER_TILE: int = 20
    MIN_ENERGY_PER_TILE: int = -20
    MAX_RELIC_NODES: int = 6
    RELIC_CONFIG_SIZE: int = 5
    ALL_RELICS_FOUND: bool = False
    ALL_REWARDS_FOUND: bool = False

    @classmethod
    def create(cls, space_size=SPACE_SIZE):
        shape = (space_size, space_size)

        # -1 to represent that the tile is unknown
        tile_type = jnp.full(shape, -1, dtype=jnp.int32)
        # 0 for unknown energy
        energy = jnp.zeros(shape, dtype=jnp.int32)
        # False for not visible, True for visible
        is_visible = jnp.zeros(shape, dtype=jnp.bool)
        relic_mask = jnp.zeros(shape, dtype=jnp.bool)
        drift_speed_found = jnp.full((), dtype=jnp.bool, fill_value=False)
        drift_speed = jnp.full((), dtype=jnp.float32, fill_value=0.0)
        step = jnp.full((), dtype=jnp.int32, fill_value=0)
        args = (
            tile_type,
            energy,
            is_visible,
            relic_mask,
            drift_speed_found,
            drift_speed,
            step,
        )
        return cls(*args)


@jax.jit
def _setup_for_update(obs):
    obs = preprocess_obs(obs)
    new_tile_types, new_tile_energies = _preprocess_map_features(obs["map_features"])
    # Transpose the sensor mask to match the space representation
    # We don't simulate vision across the diagonal so we don't need to flip and combine the sensor mask
    sensor_mask = obs["sensor_mask"].mT
    relic_nodes = jnp.asarray(obs["relic_nodes"])
    step = jnp.asarray(obs["steps"])
    return (
        new_tile_types,
        new_tile_energies,
        sensor_mask,
        relic_nodes,
        step,
    )


@jax.jit
def _move_obstacles(space):
    tile_types = jnp.roll(
        space.tile_type,
        [
            (1 * jnp.sign(space.drift_speed)),
            (-1 * jnp.sign(space.drift_speed)),
        ],
        axis=[0, 1],
    )
    tile_types = jnp.where(
        space.step * space.drift_speed % 1 == 0, tile_types, space.tile_type
    )
    return space.replace(tile_type=tile_types)


@jax.jit
def _update_drift_if_needed(space, new_tile_types):
    condition = jnp.logical_and(
        jnp.logical_not(space.drift_speed_found),
        jnp.logical_not(
            jnp.all(
                jnp.where(space.is_visible, new_tile_types, -1)
                == jnp.where(space.is_visible, space.tile_type, -1)
            )
        ),
    )

    def _update_drift_speed(space):
        def _check_drift_speed(speed):
            """This function checks if the drift speed is correct by predicting the tile types after the drift speed is applied and comparing them to the observed tile types.

            The logic for making the prediction is based on the LuxAIS3Env implementation for movement of obstacles.
            link: luxai_s3/env.LuxAIS3Env.step_env line 640:652

            Given the predicted speed, we roll the current observed tile types to the right and up 1 tile (1 * speed for right, -1 * speed for up).
            We then compare the predicted tile types to the observed tile types. If they match, the drift speed is correct and we return the speed as is.
            If they don't match, we return the opposite speed since we've already confirmed the drift speed, we just need to know the direction indicated by the sign.

            Args:
                speed (float): The speed at which the obstacles are drifting.

            Returns:
                float: The corrected drift speed (positive for up and right, negative for down and left).
            """
            visible_new = jnp.where(space.is_visible, new_tile_types, -1)
            visible_last = jnp.where(space.is_visible, space.tile_type, -1)

            # jnp.array_str(visible_new, precision=0)
            # jnp.array_str(visible_last, precision=0)
            def _roll_drift_forward():
                rolled = jnp.roll(
                    visible_last, shift=jnp.where(speed >= 0, 1, -1).astype(int), axis=0
                )
                rolled = jnp.roll(
                    rolled, shift=jnp.where(speed >= 0, -1, 1).astype(int), axis=1
                )
                return rolled

            # roll_visible_last = jnp.roll(
            #     visible_last,
            #     shift=((1 * jnp.sign(speed)), (-1 * jnp.sign(speed))),
            #     axis=[0, 1],
            # )
            roll_visible_last = _roll_drift_forward()
            matched_visible_new = jnp.where(roll_visible_last == -1, -1, visible_new)
            matched_rolled_visible_last = jnp.where(
                matched_visible_new == -1, -1, roll_visible_last
            )
            # jnp.array_str(matched_visible_new, precision=0)
            # jnp.array_str(matched_rolled_visible_last, precision=0)

            matched = jnp.all(matched_visible_new == matched_rolled_visible_last)

            def _check_neg_speed(speed):
                def _roll_drift_backward():
                    rolled = jnp.roll(
                        visible_last,
                        shift=jnp.where(speed >= 0, -1, 1).astype(int),
                        axis=0,
                    )
                    rolled = jnp.roll(
                        rolled, shift=jnp.where(speed >= 0, 1, -1).astype(int), axis=1
                    )
                    return rolled

                # roll_visible_last = jnp.roll(
                #     visible_last,
                #     shift=(-1 * jnp.sign(speed), 1 * jnp.sign(speed)),
                #     axis=(0, 1),
                # )
                roll_visible_last = _roll_drift_backward()
                matched_visible_new = jnp.where(
                    roll_visible_last == -1, -1, visible_new
                )
                matched_rolled_visible_last = jnp.where(
                    matched_visible_new == -1, -1, roll_visible_last
                )
                matched_back = jnp.all(
                    matched_visible_new == matched_rolled_visible_last
                )
                return jax.lax.cond(matched_back, lambda x: -x, lambda x: x - x, speed)

            return jax.lax.cond(matched, lambda x: x, _check_neg_speed, speed)

        drift_speed = jnp.where(((space.step - 21) % 40 < 20), 1 / 20, 1 / 40)
        # drift_speed = jnp.squeeze(drift_speed)
        # drift_speed = jax.lax.cond(
        #     ((space.step - 21) % 40 < 20), lambda x: 1 / x, lambda x: 1 / (x * 2), 20
        # )  # 1/20 or 1/40, if we got here at step 20, then it's 20, otherwise 40
        speed = _check_drift_speed(drift_speed)
        speed = jnp.squeeze(speed)
        condition_speed = jnp.squeeze(speed == 0)
        return jax.lax.cond(
            condition_speed,
            lambda x: x.replace(drift_speed_found=False, drift_speed=speed),
            lambda x: x.replace(drift_speed_found=True, drift_speed=speed),
            space,
        )

    return jax.lax.cond(
        condition,
        _update_drift_speed,
        lambda space: space,
        space,
    )


@jax.jit
def _update_relic_mask_with_nodes(
    space: Space,
    relic_nodes: jnp.ndarray,
    space_size=SPACE_SIZE,
) -> Space:
    """
    Updates the relic_mask in the Space object based on relic_nodes and their mirrored positions.

    Args:
        space_size (int): The size of the space grid.
        relic_nodes (jnp.ndarray): Array of relic node positions, shape (N, 2).

    Returns:
        Space: Updated Space object with modified relic_mask.
    """
    relic_mask = jnp.zeros((space_size, space_size), dtype=jnp.bool)
    relic_mask = relic_mask.at[relic_nodes[:, 0], relic_nodes[:, 1]].set(True)
    # Hack for removing -1, -1 nodes since they are not valid relic nodes
    relic_mask = relic_mask.at[-1, -1].set(False)
    relic_mask_opposite = jnp.flip(relic_mask)
    relic_mask = jnp.transpose(relic_mask)
    relic_mask = jnp.logical_or(relic_mask, relic_mask_opposite)
    relic_mask = jnp.logical_or(relic_mask, space.relic_mask)
    return space.replace(relic_mask=relic_mask)


@jax.jit
def _preprocess_map_features(map_features):
    """Takes the observed tile types and tile energys and preprocesses them for updating the space.

    Tile types and tile energies need to be transposed to match the space representation. The original tile types and tile energies are flipped to get the positions across the diagonal.

    The tile types and tile energies are then combined with their flipped versions to get the actual and mirrored positions together in a 24x24 grid.

    Args:
        map_features (dict): The new map features observed by the agent. Contains the tile type and tile energy lists.

    Returns:
        tile_types, tile_energies : Seperate jnp.ndarrays for tile types and tile energies. Shape (24, 24).
    """
    tile_types = jnp.where(
        map_features["tile_type"] == -1,
        jnp.flip(map_features["tile_type"], axis=(0, 1)),
        map_features["tile_type"].mT,
    )
    tile_energies = jnp.where(
        map_features["energy"] == -1,
        jnp.flip(map_features["energy"], axis=(0, 1)),
        map_features["energy"].mT,
    )
    return tile_types, tile_energies


@jax.jit
def update_space(
    space,
    obs,
) -> Space:
    (
        new_tile_types,
        new_tile_energies,
        sensor_mask,
        relic_nodes,
        step,
    ) = _setup_for_update(obs)
    space = _move_obstacles(space)
    # Only if we haven't found the drift speed yet and we observed a change in tile type
    space = space.replace(step=step)
    space = _update_drift_if_needed(space, new_tile_types)
    space = space.replace(
        is_visible=sensor_mask,
        tile_type=jnp.where(new_tile_types == -1, space.tile_type, new_tile_types),
        energy=jnp.where(
            sensor_mask | jnp.flip(sensor_mask.mT), new_tile_energies, space.energy
        ),
    )
    space = _update_relic_mask_with_nodes(space, relic_nodes)
    space = space.replace(ALL_RELICS_FOUND=jnp.all(space.tile_type != -1))
    return space


@partial(jax.vmap, in_axes=(0, 0))
def update_space_batch(space, obs):
    return update_space(space, obs)


def get_tile_info(space: Space, x: int, y: int) -> dict:
    return Node.create(space, x, y)
