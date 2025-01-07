import json
import os
import unittest

import chex
import jax.numpy as jnp
from luxai_s3.state import EnvObs

from core.base import SPACE_SIZE
from core.space import (
    Space,
    _move_obstacles,
    _preprocess_map_features,
    get_valid_relic_nodes,
    parse_envobs_to_obs,
    parse_obs_to_envobs,
    update_relic_mask_with_nodes,
    update_unit_energies,
    update_unit_positions,
)
from tests.sample_input.sample_step_numpy import obs_step_0, obs_step_1


class TestObsParsers(unittest.TestCase):
    def setUp(self):
        # Get absolute path for test files
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sample_file_1 = os.path.join(
            test_dir, "sample_input", "sample_step_0_input.json"
        )
        sample_file_2 = os.path.join(test_dir, "sample_input", "sample_step_input.json")
        # Testing from the perspective of the obs that would be received by a single agent
        self.numpy_obs_1, self.numpy_obs_2 = obs_step_0, obs_step_1

        with open(sample_file_1, "r") as f:
            self.obs1 = json.load(f)
            self.obs1 = self.obs1["obs"]
        with open(sample_file_2, "r") as f:
            self.obs2 = json.load(f)
            self.obs2 = self.obs2["obs"]

        # Validate loaded data
        self.assertIsNotNone(self.obs1)
        self.assertIsNotNone(self.obs2)
        self.assertIsNotNone(self.numpy_obs_1)
        self.assertIsNotNone(self.numpy_obs_2)

    def test_parse_obs_to_envobs(self):
        for obs in [self.obs1, self.obs2, self.numpy_obs_1, self.numpy_obs_2]:
            envobs = parse_obs_to_envobs(obs)

            self.assertIsInstance(envobs, EnvObs)
            self.assertTrue(
                jnp.array_equiv(envobs.units.position, obs["units"]["position"])
            )
            self.assertTrue(
                jnp.array_equiv(envobs.units.energy, obs["units"]["energy"])
            )

            self.assertTrue(jnp.array_equiv(envobs.units_mask, obs["units_mask"]))
            self.assertTrue(jnp.array_equiv(envobs.sensor_mask, obs["sensor_mask"]))
            self.assertTrue(
                jnp.array_equiv(
                    envobs.map_features.energy, obs["map_features"]["energy"]
                )
            )
            self.assertTrue(
                jnp.array_equiv(
                    envobs.map_features.tile_type, obs["map_features"]["tile_type"]
                )
            )
            self.assertTrue(jnp.array_equiv(envobs.relic_nodes, obs["relic_nodes"]))
            self.assertTrue(
                jnp.array_equiv(envobs.relic_nodes_mask, obs["relic_nodes_mask"])
            )
            self.assertTrue(jnp.array_equiv(envobs.team_points, obs["team_points"]))
            self.assertTrue(jnp.array_equiv(envobs.team_wins, obs["team_wins"]))
            self.assertEqual(envobs.steps, obs["steps"])
            self.assertEqual(envobs.match_steps, obs["match_steps"])

    def test_parse_envobs_to_obs(self):
        for obs in [self.obs1, self.obs2, self.numpy_obs_1, self.numpy_obs_2]:
            envobs = parse_obs_to_envobs(obs)
            obs = parse_envobs_to_obs(envobs)

            self.assertIsInstance(obs, dict)
            self.assertTrue(
                jnp.array_equiv(envobs.units.position, obs["units"]["position"])
            )
            self.assertTrue(
                jnp.array_equiv(envobs.units.energy, obs["units"]["energy"])
            )

            self.assertTrue(jnp.array_equiv(envobs.units_mask, obs["units_mask"]))
            self.assertTrue(jnp.array_equiv(envobs.sensor_mask, obs["sensor_mask"]))
            self.assertTrue(
                jnp.array_equiv(
                    envobs.map_features.energy, obs["map_features"]["energy"]
                )
            )
            self.assertTrue(
                jnp.array_equiv(
                    envobs.map_features.tile_type, obs["map_features"]["tile_type"]
                )
            )
            self.assertTrue(jnp.array_equiv(envobs.relic_nodes, obs["relic_nodes"]))
            self.assertTrue(
                jnp.array_equiv(envobs.relic_nodes_mask, obs["relic_nodes_mask"])
            )
            self.assertTrue(jnp.array_equiv(envobs.team_points, obs["team_points"]))
            self.assertTrue(jnp.array_equiv(envobs.team_wins, obs["team_wins"]))
            self.assertEqual(envobs.steps, obs["steps"])
            self.assertEqual(envobs.match_steps, obs["match_steps"])


class TestSpace(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_move_obstacles(self):
        space = Space.create()
        space = space.replace(
            tile_type=jnp.array(
                [
                    [-1, -1, 0],
                    [-1, -1, -1],
                    [-1, -1, 2],
                ],
                dtype=jnp.int32,
            ),
            drift_speed=1.0,
            step=1,
        )
        updated_space = self.variant(_move_obstacles)(space)
        expected_tile_type = jnp.array(
            [
                [-1, 2, -1],
                [-1, 0, -1],
                [-1, -1, -1],
            ],
            dtype=jnp.int32,
        )
        chex.assert_trees_all_close(updated_space.tile_type, expected_tile_type)

    @chex.variants(with_jit=True, without_jit=True)
    def test_move_obstacles_no_drift(self):
        space = Space.create()
        space = space.replace(
            tile_type=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
            drift_speed=0.0,
            step=1,
        )
        updated_space = self.variant(_move_obstacles)(space)
        expected_tile_type = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        chex.assert_trees_all_close(updated_space.tile_type, expected_tile_type)

    @chex.variants(with_jit=True, without_jit=True)
    def test_move_obstacles_negative_drift(self):
        space = Space.create()
        space = space.replace(
            tile_type=jnp.array(
                [
                    [-1, -1, -1],
                    [2, -1, -1],
                    [0, -1, -1],
                ],
                dtype=jnp.int32,
            ),
            drift_speed=-1.0,
            step=1,
        )
        updated_space = self.variant(_move_obstacles)(space)
        expected_tile_type = jnp.array(
            [
                [-1, 2, -1],
                [-1, 0, -1],
                [-1, -1, -1],
            ],
            dtype=jnp.int32,
        )
        chex.assert_trees_all_close(updated_space.tile_type, expected_tile_type)

    @chex.variants(with_jit=True, without_jit=True)
    def test_preprocess_map_features(self):
        map_features = {
            "tile_type": jnp.array(
                [
                    [-1, 1, 2],
                    [3, -1, 5],
                    [6, 7, -1],
                ],
                dtype=jnp.int32,
            ),
            "energy": jnp.array(
                [
                    [-1, 10, 20],
                    [30, -1, 50],
                    [60, 70, -1],
                ],
                dtype=jnp.int32,
            ),
        }
        tile_types, tile_energies = self.variant(_preprocess_map_features)(map_features)
        expected_tile_types = jnp.array(
            [
                [-1, 3, 6],
                [1, -1, 7],
                [2, 5, -1],
            ],
            dtype=jnp.int32,
        )
        expected_tile_energies = jnp.array(
            [
                [-1, 30, 60],
                [10, -1, 70],
                [20, 50, -1],
            ],
            dtype=jnp.int32,
        )
        chex.assert_trees_all_close(tile_types, expected_tile_types)
        chex.assert_trees_all_close(tile_energies, expected_tile_energies)


class TestUpdateUnitPositions(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_update_unit_positions(self):
        space = Space.create()
        unit_positions = jnp.array(
            [
                [[0, 0], [1, 1], [-1, -1]],  # Team 0 positions
                [[2, 2], [3, 3], [-1, -1]],  # Team 1 positions
            ],
            dtype=jnp.int32,
        )
        team_id = 0
        updated_space = self.variant(update_unit_positions)(
            space, unit_positions, team_id
        )
        expected_friendly_positions = jnp.array(
            [[0, 0], [1, 1], [-1, -1]], dtype=jnp.int32
        )
        expected_enemy_positions = jnp.array(
            [[2, 2], [3, 3], [-1, -1]], dtype=jnp.int32
        )
        chex.assert_trees_all_close(
            updated_space.friendly_positions, expected_friendly_positions
        )
        chex.assert_trees_all_close(
            updated_space.enemy_positions, expected_enemy_positions
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_update_unit_positions_team_1(self):
        space = Space.create()
        unit_positions = jnp.array(
            [
                [[0, 0], [1, 1], [-1, -1]],  # Team 0 positions
                [[2, 2], [3, 3], [-1, -1]],  # Team 1 positions
            ],
            dtype=jnp.int32,
        )
        team_id = 1
        updated_space = self.variant(update_unit_positions)(
            space, unit_positions, team_id
        )
        expected_friendly_positions = jnp.array(
            [[2, 2], [3, 3], [-1, -1]], dtype=jnp.int32
        )
        expected_enemy_positions = jnp.array(
            [[0, 0], [1, 1], [-1, -1]], dtype=jnp.int32
        )
        chex.assert_trees_all_close(
            updated_space.friendly_positions, expected_friendly_positions
        )
        chex.assert_trees_all_close(
            updated_space.enemy_positions, expected_enemy_positions
        )

        chex.assert_trees_all_close(
            updated_space.enemy_positions, expected_enemy_positions
        )


class TestUpdateUnitEnergies(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_update_unit_energies(self):
        space = Space.create()
        unit_energies = jnp.array(
            [
                [100, 200, -1],  # Team 0 energies
                [300, 400, -1],  # Team 1 energies
            ],
            dtype=jnp.int32,
        )
        team_id = 0
        updated_space = self.variant(update_unit_energies)(
            space, unit_energies, team_id
        )
        expected_friendly_energy = jnp.array([100, 200, -1], dtype=jnp.int32)
        expected_enemy_energy = jnp.array([300, 400, -1], dtype=jnp.int32)
        chex.assert_trees_all_close(
            updated_space.friendly_energy, expected_friendly_energy
        )
        chex.assert_trees_all_close(updated_space.enemy_energy, expected_enemy_energy)

    @chex.variants(with_jit=True, without_jit=True)
    def test_update_unit_energies_team_1(self):
        space = Space.create()
        unit_energies = jnp.array(
            [
                [100, 200, -1],  # Team 0 energies
                [300, 400, -1],  # Team 1 energies
            ],
            dtype=jnp.int32,
        )
        team_id = 1
        updated_space = self.variant(update_unit_energies)(
            space, unit_energies, team_id
        )
        expected_friendly_energy = jnp.array([300, 400, -1], dtype=jnp.int32)
        expected_enemy_energy = jnp.array([100, 200, -1], dtype=jnp.int32)
        chex.assert_trees_all_close(
            updated_space.friendly_energy, expected_friendly_energy
        )
        chex.assert_trees_all_close(updated_space.enemy_energy, expected_enemy_energy)


class TestUpdateRelicMaskWithNodes(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_update_relic_mask_with_nodes(self):
        space = Space.create()
        relic_nodes = jnp.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
            ],
            dtype=jnp.int32,
        )
        updated_space = self.variant(update_relic_mask_with_nodes)(space, relic_nodes)
        expected_relic_mask = jnp.zeros(
            (SPACE_SIZE, SPACE_SIZE),
            dtype=jnp.bool,
        )
        expected_relic_mask = expected_relic_mask.at[0, 0].set(True)
        expected_relic_mask = expected_relic_mask.at[1, 1].set(True)
        expected_relic_mask = expected_relic_mask.at[2, 2].set(True)

        chex.assert_trees_all_close(updated_space.relic_mask, expected_relic_mask)

    @chex.variants(with_jit=True, without_jit=True)
    def test_update_relic_mask_with_nodes_existing_mask(self):
        space = Space.create()
        space = space.replace(
            relic_mask=space.relic_mask.at[0, 0]
            .set(True)
            .at[1, 1]
            .set(True)
            .at[1, 0]
            .set(True)
        )
        relic_nodes = jnp.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
            ],
            dtype=jnp.int32,
        )
        updated_space = self.variant(update_relic_mask_with_nodes)(space, relic_nodes)
        expected_relic_mask = jnp.zeros(
            (SPACE_SIZE, SPACE_SIZE),
            dtype=jnp.bool,
        )
        expected_relic_mask = (
            expected_relic_mask.at[0, 0]
            .set(True)
            .at[1, 1]
            .set(True)
            .at[2, 2]
            .set(True)
            .at[1, 0]
            .set(True)
            .at[23, 23]
            .set(True)
            .at[22, 22]
            .set(True)
            .at[21, 21]
            .set(True)
            .at[23, 22]
            .set(True)
        )
        chex.assert_trees_all_close(updated_space.relic_mask, expected_relic_mask)

    @chex.variants(with_jit=True, without_jit=True)
    def test_update_relic_mask_with_nodes_empty(self):
        space = Space.create()
        relic_nodes = jnp.array([], dtype=jnp.int32).reshape(0, 2)
        updated_space = self.variant(update_relic_mask_with_nodes)(space, relic_nodes)
        expected_relic_mask = jnp.zeros((SPACE_SIZE, SPACE_SIZE), dtype=jnp.bool)
        chex.assert_trees_all_close(updated_space.relic_mask, expected_relic_mask)
