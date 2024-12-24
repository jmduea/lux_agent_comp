import copy
import os
from sys import stderr

import numpy as np
import torch
from scipy.signal import convolve2d

from base import (
    SPACE_SIZE,
    ActionType,
    Global,
    NodeType,
    get_match_step,
    get_opposite,
    is_team_sector,
    warp_point,
)
from debug import show_energy_field, show_exploration_map, show_map
from pathfinding import (
    dstar,
    create_weights,
    estimate_energy_cost,
    find_closest_target,
    manhattan_distance,
    nearby_positions,
    path_to_actions,
)

from .dqn_model import DQNAgent


class Node:
    """
    Represents a node on a grid with coordinates (x, y). A node can have various types
    and properties such as energy, visibility, relics, and rewards. It supports operations
    for updating its relic and reward status and provides utilities for comparison and
    distance calculations.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False

        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = False

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y and self.type == other.type

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: bool):
        if self._explored_for_relic and self._relic != status:
            raise ValueError(
                f"Can't change the relic status {self._relic}->{status} for {self}"
                ", the tile has already been explored"
            )

        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: bool):
        if self._explored_for_reward and self._reward != status:
            raise ValueError(
                f"Can't change the reward status {self._reward}->{status} for {self}"
                ", the tile has already been explored"
            )

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Space:
    """
    Represents a 2D grid space where each cell is a node that can contain relics, rewards,
    and other properties. The space is symmetrical and supports operations for updating
    the status of nodes, shifting obstacles, and tracking relic and reward discoveries.

    Attributes
    ----------
        _nodes (list[list[Node]]):
            A list of lists representing the grid of nodes.
        _relic_nodes (set[Node]):
            A set of nodes that contain relics.
        _reward_nodes (set[Node]):
            A set of nodes that provide rewards.

    Methods
    -------
        __repr__:
            Returns a string representation of the space.
        __iter__:
            Allows iteration over all nodes in the space.
        relic_nodes:
            Returns the set of nodes with relics.
        reward_nodes:
            Returns the set of nodes with rewards.
        get_node:
            Retrieves the node at given coordinates.
        update:
            Updates the space based on observations and team data.
        _update_relic_map:
            Updates the relic map based on observations.
        _update_reward_status_from_reward_results:
            Updates reward status from results.
        _update_reward_results:
            Updates reward results from observations.
        _update_reward_status_from_relics_distribution:
            Updates reward status based on relic distribution.
        _update_relic_status:
            Updates the relic status of a node.
        _update_reward_status:
            Updates the reward status of a node.
        _update_map:
            Updates the map based on observations.
        _find_obstacle_movement_period:
            Finds the period of obstacle movement.
        _find_obstacle_movement_direction:
            Finds the direction of obstacle movement.
        clear:
            Clears visibility of all nodes.
        move_obstacles:
            Moves obstacles based on the current step.
        move:
            Moves the nodes in the space by a given offset.
    """

    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        # set of nodes with a relic
        self._relic_nodes: set[Node] = set()

        # set of nodes that provide points
        self._reward_nodes: set[Node] = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def update(self, step, obs, team_id, team_reward):
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(obs, team_id, team_reward)

    def _update_relic_map(self, obs, team_id, team_reward):
        for relic_id, (mask, xy) in enumerate(
            zip(obs["relic_nodes_mask"], obs["relic_nodes"])
        ):
            if mask:
                self._update_relic_status(*xy, status=True)

        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)

            if not node.explored_for_relic:
                all_relics_found = False

            if not node.explored_for_reward:
                all_rewards_found = False

        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found

        if not Global.ALL_RELICS_FOUND:
            if len(self._relic_nodes) == Global.MAX_RELIC_NODES:
                # all relics found, mark all nodes as explored for relics
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)

        if not Global.ALL_REWARDS_FOUND:
            self._update_reward_status_from_relics_distribution()
            self._update_reward_results(obs, team_id, team_reward)
            self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        """Update reward status based on Global.REWARD_RESULTS"""
        for result in Global.REWARD_RESULTS:
            if (
                not isinstance(result, dict)
                or "nodes" not in result
                or "reward" not in result
            ):
                continue

            nodes = result["nodes"]
            if not isinstance(nodes, (set, list)):
                continue

            # Get the reference nodes from our space
            space_nodes = set()
            for n in nodes:
                if not isinstance(n, Node):
                    continue
                space_node = self.get_node(n.x, n.y)
                if space_node:
                    space_nodes.add(space_node)

            if not space_nodes:
                continue  # No valid nodes found

            # Count known rewards
            known_reward = sum(1 for n in space_nodes if n.reward)

            # Get unknown nodes
            unknown_nodes = {
                n
                for n in space_nodes
                if not n.explored_for_reward or (n.explored_for_reward and not n.reward)
            }

            if not unknown_nodes:
                continue  # All nodes already explored

            try:
                reward = float(result["reward"]) - known_reward
            except (TypeError, ValueError):
                continue

            if reward < 0 or not isinstance(reward, (int, float)):
                continue

            # Update node statuses
            if reward == 0:
                # All nodes are empty
                for node in unknown_nodes:
                    self._update_reward_status(node.x, node.y, False)
            elif reward == len(unknown_nodes):
                # All nodes yield points
                for node in unknown_nodes:
                    self._update_reward_status(node.x, node.y, True)

    def _update_reward_results(self, obs, team_id, team_reward):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                ship_nodes.add(self.get_node(*position))

        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1

        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1

        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status=True):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)

        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # rewards are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)

        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        obstacles_shifted = False
        energy_nodes_shifted = False
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]

            if (
                is_visible
                and not node.is_unknown
                and node.type.value != obs_tile_type[x, y]
            ):
                obstacles_shifted = True

            if (
                is_visible
                and node.energy is not None
                and node.energy != obs_energy[x, y]
            ):
                energy_nodes_shifted = True

        Global.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)

        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = self._find_obstacle_movement_period(
                Global.OBSTACLES_MOVEMENT_STATUS
            )
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period

        if not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction

                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                # Can't find OBSTACLE_MOVEMENT_DIRECTION
                for node in self:
                    node.type = NodeType.unknown

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            node.is_visible = is_visible

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_node(*get_opposite(x, y)).type = node.type

            if is_visible:
                node.energy = int(obs_energy[x, y])

                # the energy field should be symmetrical
                self.get_node(*get_opposite(x, y)).energy = node.energy

            elif energy_nodes_shifted:
                # The energy field has changed
                # I cannot predict what the new energy field will be like.
                node.energy = None

    @staticmethod
    def _find_obstacle_movement_period(obstacles_movement_status):
        # Right now there are only two options for nebula_tile_drift_speed: 1 / 20 and 1 / 40
        if obstacles_movement_status and obstacles_movement_status[-1]:
            return 20 if len(obstacles_movement_status) - 21 % 40 < 20 else 40

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)

            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (
                    sensor_mask[x, y]
                    and not node.is_unknown
                    and obs_tile_type[x, y] != node.type.value
                ):
                    match = False
                    break

            if match:
                suitable_directions.append(direction)

        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def clear(self):
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step):
        if (
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and Global.OBSTACLE_MOVEMENT_PERIOD > 0
            and (step - 1) % Global.OBSTACLE_MOVEMENT_PERIOD == 0
        ):
            self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self


class Ship:
    """
    Represents a ship with a unique unit ID that can perform various tasks,
    such as moving, sapping, or targeting enemies. The ship maintains its
    energy level, position, and a list of potential sap targets within range.

    Attributes
    ----------
    unit_id (int):
        Unique identifier for the ship.
    starting_position (tuple[int, int] | None):
        The initial coordinates of the ship.
    energy (int):
        Current energy level of the ship.
    node (Node | None):
        Current node representing the ship's position.
    task (str | None):
        Current task assigned to the ship.
    target (Node | None):
        Current target node for the ship.
    action (ActionType | None):
        Current action the ship is performing.
    sap_targets (list[Ship]):
        List of enemy ships within sap range.
    """

    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.starting_position: tuple[int, int] | None = None
        self.energy = 0
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None
        self.sap_targets: list[Ship] = []

    def __repr__(self):
        return f"Ship({self.unit_id}, starting_position={self.starting_position}, node={self.node.coordinates}, energy={self.energy},)"

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    @property
    def lowest_energy_target(self):
        return min(self.sap_targets, key=lambda x: x.energy)

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None
        self.sap_targets.clear()

    def update_sap_targets(self, opp_fleet):
        """
        Update list of enemy ships within sap range.

        Args:
            opp_fleet (Fleet): Fleet of the opponent

        Returns:
            List of enemy ships within sap range
        """
        self.sap_targets.clear()

        if not self.node:
            return self.sap_targets

        for enemy_ship in opp_fleet:
            if not enemy_ship.node:
                continue

            distance = manhattan_distance(
                self.node.coordinates, enemy_ship.node.coordinates
            )

            if distance <= Global.UNIT_SAP_RANGE:
                self.sap_targets.append(enemy_ship)

        return self.sap_targets


class Fleet:
    """
    A fleet is a collection of ships on the board that are on the same team.
    """

    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0  # how many points have we scored in this match so far
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space, opp_fleet):
        """Update fleet state based on observations."""
        self.points = int(obs["team_points"][self.team_id])

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.energy = energy
                ship.node = space.get_node(*position)
            else:
                ship.clean()

            ship.update_sap_targets(opp_fleet)


class Agent:
    """
    The AI agent that controls a team of ships.

    The agent makes decisions based on the current state of the game, which is
    represented by the `Space` object. The agent uses the `Space` object to
    determine the positions of the ships, the energy levels of the nodes, and
    the positions of the obstacles.

    The agent makes decisions by calling the `act` method, which takes the
    current state of the game and returns an array of actions, where each action
    is represented as a triplet: (action_type, x_offset, y_offset).

    The agent also has methods for finding relics, finding rewards, and
    harvesting energy.

    The agent keeps track of the current state of the game, including the
    positions of the ships, the energy levels of the nodes, and the positions of
    the obstacles.

    The agent also has methods for showing the visible energy field, the
    explored energy field, the visible map, the explored map, and the exploration
    map.

    Parameters
    ----------
    player : str
        The player name.
    env_cfg : dict
        The environment configuration.

    Attributes
    ----------
    player : str
        The player name.
    team_id : int
        The team id.
    opp_team_id : int
        The opponent team id.
    env_cfg : dict
        The environment configuration.
    space : Space
        The game state.
    fleet : Fleet
        The fleet of ships.
    opp_fleet : Fleet
        The opponent fleet of ships.
    """

    def __init__(self, player: str, env_cfg) -> None:
        # Create models directory if it doesn't exist
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

        state_size = self._calculate_state_size()
        action_size = len(ActionType)  # Number of possible actions
        self.dqn_agent = DQNAgent(state_size, action_size, self.player, self.env_cfg)

    def _calculate_state_size(self):
        """
        Calculate the size of the state space with enhanced features.
        Features per ship:
        Global Features (shared across all ships):
        - Energy field map (24x24): 576
        - Nebula effect map (24x24): 576
        - Void field map (24x24): 576
        - Relic memory map (24x24): 576
        - Visibility mask (24x24): 576

        Per Ship Features:
        - Position (x, y): 2
        - Energy level: 1
        - Node type one-hot (empty, asteroid, nebula): 3
        - Local energy field (5x5): 25
        - Local nebula effect (5x5): 25
        - Local void field (5x5): 25
        - Local relic memory (5x5): 25
        - Local visibility (5x5): 25
        - Distance features:
            - To nearest relic: 1
            - To nearest reward: 1
            - To nearest enemy: 1
            - To nearest high energy: 1
        - Task encoding (one-hot): 3
        - Number of nearby enemies: 1
        - Current energy efficiency: 1

        Returns:
            int: Size of the state space
        """
        # Global features (5 maps of 24x24)
        global_features = 5 * 24 * 24  # 2880 features

        # Per ship features
        per_ship_features = (
            2  # position
            + 1  # energy
            + 3  # node type one-hot
            + 25  # local energy field
            + 25  # local nebula effect
            + 25  # local void field
            + 25  # local relic memory
            + 25  # local visibility
            + 4  # distance features
            + 3  # task encoding
            + 1  # nearby enemies
            + 1  # energy efficiency
        )  # 140 features

        total_size = global_features + per_ship_features
        print(
            f"Total state size: {total_size} (Global: {global_features}, Per ship: {per_ship_features})"
        )
        return total_size

    def _encode_state(self, ship: Ship, obs) -> dict:
        """
        Encode the state for a specific ship with enhanced features.

        Args:
            ship: The ship to encode state for
            obs: Current observation

        Returns:
            dict: Encoded state features
        """
        # Get ship position and create local view window
        x, y = ship.coordinates if ship.node else (-1, -1)
        local_window = 2  # Results in 5x5 window

        # Global feature maps
        energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE))
        nebula_map = np.zeros((SPACE_SIZE, SPACE_SIZE))
        void_map = np.zeros((SPACE_SIZE, SPACE_SIZE))
        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE))
        visibility_map = np.zeros((SPACE_SIZE, SPACE_SIZE))

        # Fill global maps
        for node in self.space:
            if node.is_visible:
                energy_map[node.y, node.x] = (
                    node.energy if node.energy is not None else 0
                )
                nebula_map[node.y, node.x] = 1 if node.type == NodeType.nebula else 0
                visibility_map[node.y, node.x] = 1
                if node.relic:
                    relic_map[node.y, node.x] = 1

        # Calculate void field from enemy ships
        for enemy_ship in self.opp_fleet:
            if enemy_ship.node and enemy_ship.node.is_visible:
                ex, ey = enemy_ship.coordinates
                void_strength = (
                    enemy_ship.energy
                    * self.env_cfg.get(
                        "unit_energy_void_factor", 0.25
                    )  # Use default value of 0.25 if not specified
                )
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < SPACE_SIZE and 0 <= ny < SPACE_SIZE:
                        void_map[ny, nx] = max(void_map[ny, nx], void_strength)

        # Extract local windows
        local_energy = self._extract_local_window(energy_map, x, y, local_window)
        local_nebula = self._extract_local_window(nebula_map, x, y, local_window)
        local_void = self._extract_local_window(void_map, x, y, local_window)
        local_relic = self._extract_local_window(relic_map, x, y, local_window)
        local_visibility = self._extract_local_window(
            visibility_map, x, y, local_window
        )

        # Calculate distance features
        nearest_relic = float("inf")
        nearest_reward = float("inf")
        nearest_enemy = float("inf")
        nearest_high_energy = float("inf")

        if ship.node:
            for node in self.space:
                if node.relic:
                    nearest_relic = min(
                        nearest_relic, ship.node.manhattan_distance(node)
                    )
                if node.reward:
                    nearest_reward = min(
                        nearest_reward, ship.node.manhattan_distance(node)
                    )
                if node.energy and node.energy > 50:  # Threshold for "high energy"
                    nearest_high_energy = min(
                        nearest_high_energy, ship.node.manhattan_distance(node)
                    )

            for enemy in self.opp_fleet:
                if enemy.node:
                    nearest_enemy = min(
                        nearest_enemy, ship.node.manhattan_distance(enemy.node)
                    )

        # Normalize distances
        max_distance = SPACE_SIZE * 2
        nearest_relic = min(nearest_relic, max_distance) / max_distance
        nearest_reward = min(nearest_reward, max_distance) / max_distance
        nearest_enemy = min(nearest_enemy, max_distance) / max_distance
        nearest_high_energy = min(nearest_high_energy, max_distance) / max_distance

        # Calculate energy efficiency
        energy_efficiency = ship.energy / 100 if ship.energy > 0 else 0

        # Count nearby enemies
        nearby_enemies = sum(1 for target in ship.sap_targets if target.energy > 0)

        # Encode current task
        task_encoding = [0, 0, 0]
        if ship.task == "find_relics":
            task_encoding[0] = 1
        elif ship.task == "find_rewards":
            task_encoding[1] = 1
        elif ship.task == "harvest":
            task_encoding[2] = 1

        # Node type one-hot encoding
        node_type = [0, 0, 0]  # [empty, asteroid, nebula]
        if ship.node:
            if ship.node.type == NodeType.empty:
                node_type[0] = 1
            elif ship.node.type == NodeType.asteroid:
                node_type[1] = 1
            elif ship.node.type == NodeType.nebula:
                node_type[2] = 1

        return {
            "global_features": {
                "energy_map": energy_map.flatten(),
                "nebula_map": nebula_map.flatten(),
                "void_map": void_map.flatten(),
                "relic_map": relic_map.flatten(),
                "visibility_map": visibility_map.flatten(),
            },
            "ship_features": {
                "position": [x / SPACE_SIZE, y / SPACE_SIZE],
                "energy": ship.energy / 100,
                "node_type": node_type,
                "local_energy": local_energy.flatten(),
                "local_nebula": local_nebula.flatten(),
                "local_void": local_void.flatten(),
                "local_relic": local_relic.flatten(),
                "local_visibility": local_visibility.flatten(),
                "distances": [
                    nearest_relic,
                    nearest_reward,
                    nearest_enemy,
                    nearest_high_energy,
                ],
                "task": task_encoding,
                "nearby_enemies": nearby_enemies / Global.MAX_UNITS,
                "energy_efficiency": energy_efficiency,
            },
        }

    def _extract_local_window(self, global_map, center_x, center_y, window_size):
        """Extract a local window from a global map with zero padding"""
        local = np.zeros((2 * window_size + 1, 2 * window_size + 1))
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                x, y = center_x + dx, center_y + dy
                if 0 <= x < SPACE_SIZE and 0 <= y < SPACE_SIZE:
                    local[dy + window_size, dx + window_size] = global_map[y, x]
        return local

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        match_step = get_match_step(step)

        if match_step == 0:
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)
            self.load_model()
            self.prev_state = {}
            self.prev_action = {}
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])
        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        # Update game state
        self.space.update(step, obs, self.team_id, self.fleet.points)
        self.fleet.update(obs, self.space, self.opp_fleet)
        self.opp_fleet.update(obs, self.space, self.fleet)

        # Create actions array
        actions = self.create_actions_array()

        # Get actions for each ship using DQN
        for ship in self.fleet:
            if ship.node is None:
                continue

            # Get state features for this ship
            state_dict = self._encode_state(ship, obs)

            # Make a copy of the state dict for replay buffer
            replay_state = copy.deepcopy(state_dict)

            # Convert state dict to tensor for forward pass
            state_tensor = {
                "global_features": {
                    k: torch.from_numpy(v).float().to(self.dqn_agent.device)
                    for k, v in state_dict["global_features"].items()
                },
                "ship_features": {
                    k: torch.from_numpy(v).float().to(self.dqn_agent.device)
                    if isinstance(v, np.ndarray)
                    else torch.tensor(v, dtype=torch.float32).to(self.dqn_agent.device)
                    for k, v in state_dict["ship_features"].items()
                },
            }

            # Get Q-values and select action
            with torch.no_grad():
                q_values = self.dqn_agent.forward(state_tensor)

                # Epsilon-greedy action selection
                if np.random.random() < self.dqn_agent.epsilon:
                    action_type = np.random.randint(0, 6)
                else:
                    action_type = q_values.argmax().item()

            # Convert action type to game action
            if action_type == 0:  # No action
                continue
            elif action_type <= 4:  # Movement actions
                actions[ship.unit_id] = [action_type, 0, 0]
            else:  # Sap action
                # Find best sap target
                best_target = None
                max_energy = 0
                for target in ship.sap_targets:
                    if target.energy > max_energy:
                        max_energy = target.energy
                        best_target = target

                if best_target:
                    tx, ty = best_target.coordinates
                    sx, sy = ship.coordinates
                    actions[ship.unit_id] = [5, tx - sx, ty - sy]

            # Store transition in replay buffer if we have previous state
            if ship.unit_id in self.prev_state and ship.unit_id in self.prev_action:
                reward = self._calculate_reward(ship, action_type, reward)
                self.dqn_agent.train(
                    self.prev_state[ship.unit_id],
                    self.prev_action[ship.unit_id],
                    reward,
                    replay_state,
                    False,  # done
                )

            # Store current state and action for next step
            self.prev_state[ship.unit_id] = replay_state
            self.prev_action[ship.unit_id] = action_type

        return actions

    def _calculate_reward(self, ship, action, reward=0):
        """Calculate reward for the given ship and action"""
        reward = reward / 16

        # Reward for gaining points
        if ship.node and ship.node.reward:
            reward += 10

        # Reward for finding new relic tiles
        if ship.node and ship.node.relic and not ship.node.explored_for_relic:
            reward += 5

        # Penalty for low energy
        if ship.energy < 20:
            reward -= 2

        # Reward for efficient energy management
        if (
            ship.energy > ship.energy and action <= 4
        ):  # Movement action with energy gain
            reward += 1

        # Penalty for invalid sap attempts
        if action == 5 and not ship.sap_targets:
            reward -= 1

        # Reward for successful sap
        if action == 5 and ship.sap_targets:
            reward += 3

        return reward

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is None:
                actions[i] = ActionType.center, 0, 0
                continue

            # For sap action we need coordinates
            if ship.action == ActionType.sap and len(ship.sap_targets) > 0:
                lowest_energy_target = ship.lowest_energy_target
                coordinates = lowest_energy_target.coordinates
                x, y = coordinates[0] - ship.node.x, coordinates[1] - ship.node.y
                actions[i] = ActionType.sap, x, y
            elif ship.action == ActionType.sap and len(ship.sap_targets) == 0:
                actions[i] = ActionType.center, 0, 0
            else:
                actions[i] = ship.action, 0, 0

        return actions

    def find_relics(self):
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                # We will only find relics in our part of the map
                # because relics are symmetrical.
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False

            path = dstar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_relics"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

                for x, y in path:
                    for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                        if xy in targets:
                            targets.remove(xy)

                return True

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self):
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics()

        relic_node_to_ship = {}
        for ship in self.fleet:
            if ship.task == "find_rewards":
                if ship.target is None:
                    ship.task = None
                    continue

                if (
                    ship.target in unexplored_relics
                    and ship.energy > Global.UNIT_MOVE_COST * 5
                ):
                    relic_node_to_ship[ship.target] = ship
                else:
                    ship.task = None
                    ship.target = None

        for relic in unexplored_relics:
            if relic not in relic_node_to_ship:
                # find the closest ship to the relic node
                min_distance, closes_ship = float("inf"), None
                for ship in self.fleet:
                    if ship.task and ship.task != "find_rewards":
                        continue

                    if ship.energy < Global.UNIT_MOVE_COST * 5:
                        continue

                    distance = manhattan_distance(ship.coordinates, relic.coordinates)
                    if distance < min_distance:
                        min_distance, closes_ship = distance, ship

                if closes_ship:
                    relic_node_to_ship[relic] = closes_ship

        def set_task(ship, relic_node, can_pause):
            targets = []
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))

            target, _ = find_closest_target(ship.coordinates, targets)

            if target == ship.coordinates and not can_pause:
                target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        n.coordinates
                        for n in self.space
                        if n.explored_for_reward and n.is_walkable
                    ],
                )

            if not target:
                return

            path = dstar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

        can_pause = True
        for n, s in sorted(
            list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id
        ):
            if set_task(s, n, can_pause):
                if s.target == s.node:
                    # If one ship is stationary, we will move all the other ships.
                    # This will help generate more useful data in Global.REWARD_RESULTS.
                    can_pause = False
            else:
                if s.task == "find_rewards":
                    s.task = None
                    s.target = None

    def get_unexplored_relics(self) -> list[Node]:
        relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)

        return relic_nodes

    def harvest(self):
        def set_task(ship, target_node):
            if ship.node == target_node:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = dstar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None

    def save_model(self):
        """Save the DQN model"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, "dqn_agent.pth")
            self.dqn_agent.save(model_path)
        except Exception as e:
            print(f"Error saving model: {e}", file=stderr)

    def load_model(self):
        """Load the DQN model if it exists"""
        if self.dqn_agent is None:
            self.dqn_agent = DQNAgent(expert=self)

        # For inference, move model to CPU if GPU is not available
        if not torch.cuda.is_available():
            self.dqn_agent.to_device(torch.device("cpu"))

        # Load model weights if they exist
        model_path = os.path.join(os.path.dirname(__file__), "dqn_model.pth")
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.dqn_agent.device)
                self.dqn_agent.load_state_dict(state_dict)
                print(f"Loaded DQN model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")

    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)
