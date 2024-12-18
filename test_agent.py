import numpy as np
from collections import defaultdict, deque
from typing import Tuple, Dict, Any


def direction_to(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    """
    Determine a direction action (0-5) from one position to another.
    Returns:
      0 = no move, 1 = up, 2 = right, 3 = down, 4 = left
    Heuristics: If horizontal distance is greater, move horizontally; otherwise move vertically.
    """
    fx, fy = from_pos
    tx, ty = to_pos
    dx = tx - fx
    dy = ty - fy
    # Vertical or horizontal preference based on which difference is larger
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 4  # move right if dx>0 else left
    elif abs(dy) > abs(dx):
        return 3 if dy > 0 else 1  # move down if dy>0 else up
    return 0  # already at target or no clear direction


class Agent2:
    def __init__(self, player: str, env_cfg: Dict[str, Any]) -> None:
        """
        Initialize the agent with environment configurations.
        Sets up map dimensions, parameters, and data structures for tracking map info, relic nodes, enemy positions, etc.
        """
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        np.random.seed(0)  # Fix seed for reproducibility

        # Extract environment parameters safely with defaults
        self.env_cfg = env_cfg
        self.map_width = env_cfg.get("map_width", 24)
        self.map_height = env_cfg.get("map_height", 24)
        self.max_units = env_cfg.get("max_units", 16)
        self.max_steps_in_match = env_cfg.get("max_steps_in_match", 100)

        # Costs and parameters extracted or defaulted
        self.move_cost = env_cfg.get("unit_move_cost", 2)
        self.sap_cost = env_cfg.get("unit_sap_cost", 30)
        self.sap_range = env_cfg.get("unit_sap_range", 4)
        self.sr = env_cfg.get("unit_sensor_range", 2)
        self.nebula_vision_red = env_cfg.get("nebula_tile_vision_reduction", 0)
        self.nebula_energy_red = env_cfg.get("nebula_tile_energy_reduction", 0)
        self.energy_void_factor = env_cfg.get("unit_energy_void_factor", 0.125)
        self.sap_dropoff_factor = env_cfg.get("unit_sap_dropoff_factor", 0.5)

        # Track discovered relic nodes and scoring patterns
        self.discovered_relic_nodes_ids = set()
        self.relic_node_positions = []

        # Map knowledge: store tile types, energy, whether visited, and if it's a relic tile
        self.tile_info = [
            [
                {"type": None, "energy": None, "visited": False, "relic": False}
                for _ in range(self.map_height)
            ]
            for _ in range(self.map_width)
        ]

        # Unit-specific exploration targets
        self.unit_explore_targets = dict()

        # Track enemy information using unit positions and a heatmap
        self.enemy_unit_positions = {uid: None for uid in range(self.max_units)}
        self.enemy_heatmap = np.zeros((self.map_width, self.map_height), dtype=float)

        # For each relic node, track discovered scoring offsets
        # (dx, dy) relative to the relic node center => bool whether it yields points
        self.relic_scoring_info = defaultdict(lambda: dict())

        # Track matches and scoring to adapt strategy
        self.match_count = 0
        self.last_match_points = [0, 0]
        self.global_step = 0

        # Divide the map into sectors for systematic exploration in early matches
        self.num_sectors = int(np.ceil(np.sqrt(self.max_units)))
        self.sector_width = self.map_width // self.num_sectors
        self.sector_height = self.map_height // self.num_sectors
        self.unit_sector_assignment = dict()

    def _manhattan_dist(self, p1, p2):
        """Compute Manhattan distance between two points p1 and p2."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _nearest_pos(self, pos, targets):
        """Given a position and a list of target positions, find the nearest one."""
        if not targets:
            return None
        dists = [self._manhattan_dist(pos, t) for t in targets]
        return targets[np.argmin(dists)]

    def _update_knowledge(self, obs):
        """
        Update the agent's known map information based on current observations.
        Mark tiles as visited, record their type and energy if visible.
        """
        tile_types = np.array(obs["map_features"]["tile_type"])
        energies = np.array(obs["map_features"]["energy"])
        sensor_mask = np.array(obs["sensor_mask"])
        for x in range(self.map_width):
            for y in range(self.map_height):
                if sensor_mask[x, y]:
                    t_type = tile_types[x, y]
                    t_energy = energies[x, y]
                    if t_type != -1:
                        self.tile_info[x][y]["type"] = t_type
                        self.tile_info[x][y]["energy"] = (
                            t_energy if t_energy != -1 else None
                        )
                        self.tile_info[x][y]["visited"] = True

    def _mark_relic_nodes(self, obs):
        """
        Identify visible relic nodes and record their positions.
        Update tile_info to mark relic positions.
        """
        observed_relic_node_positions = np.array(obs["relic_nodes"])
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])
        visible_relic_node_ids = np.where(observed_relic_nodes_mask)[0]
        for rid in visible_relic_node_ids:
            pos = observed_relic_node_positions[rid]
            if (
                pos[0] != -1
                and pos[1] != -1
                and rid not in self.discovered_relic_nodes_ids
            ):
                self.discovered_relic_nodes_ids.add(rid)
                self.relic_node_positions.append((pos[0], pos[1]))
                self.tile_info[pos[0]][pos[1]]["relic"] = True

    def _is_passable(self, x, y):
        """
        Check if a tile is passable (not an asteroid).
        Return False if coordinates out of bounds or tile is an asteroid.
        """
        if x < 0 or x >= self.map_width or y < 0 or y >= self.map_height:
            return False
        tile_type = self.tile_info[x][y]["type"]
        return tile_type != 2  # 2 is asteroid, impassable

    def _find_path(self, start, goal):
        """
        Find a path from start to goal using BFS.
        Return the direction of the first step towards the goal, or 0 if none found.
        """
        if start == goal:
            return 0
        directions = [
            (0, -1, 1),
            (1, 0, 2),
            (0, 1, 3),
            (-1, 0, 4),
        ]  # (dx, dy, action_code)
        visited = set([start])
        queue = deque([start])
        parents = {start: None}

        while queue:
            pos = queue.popleft()
            if pos == goal:
                # Reconstruct the first step of the path
                cur = pos
                while parents[cur] and parents[cur][0] != start:
                    cur = parents[cur][0]
                if parents[cur] is None:
                    return 0
                return parents[cur][1]

            for dx, dy, d_code in directions:
                nx, ny = pos[0] + dx, pos[1] + dy
                if self._is_passable(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parents[(nx, ny)] = (pos, d_code)
                    queue.append((nx, ny))
        return 0

    def _choose_sector_target(self, unit_id, unit_pos):
        """
        Assign a sector of the map to each unit and pick an unvisited tile in that sector to explore.
        If no unvisited in sector, pick any unvisited tile in the map.
        """
        if unit_id not in self.unit_sector_assignment:
            sector_id = unit_id % (self.num_sectors * self.num_sectors)
            sx = sector_id % self.num_sectors
            sy = sector_id // self.num_sectors
            self.unit_sector_assignment[unit_id] = (sx, sy)

        sx, sy = self.unit_sector_assignment[unit_id]
        sector_x_min = sx * self.sector_width
        sector_x_max = min((sx + 1) * self.sector_width, self.map_width)
        sector_y_min = sy * self.sector_height
        sector_y_max = min((sy + 1) * self.sector_height, self.map_height)

        candidates = [
            (x, y)
            for x in range(sector_x_min, sector_x_max)
            for y in range(sector_y_min, sector_y_max)
            if not self.tile_info[x][y]["visited"] and self._is_passable(x, y)
        ]
        if not candidates:
            # If no unvisited in sector, look anywhere unvisited
            candidates = [
                (x, y)
                for x in range(self.map_width)
                for y in range(self.map_height)
                if not self.tile_info[x][y]["visited"] and self._is_passable(x, y)
            ]
        if not candidates:
            # Entire map visited, pick random spot
            return (
                np.random.randint(0, self.map_width),
                np.random.randint(0, self.map_height),
            )
        dists = [self._manhattan_dist(unit_pos, c) for c in candidates]
        return candidates[np.argmin(dists)]

    def _update_enemy_positions(self, obs):
        """
        Update enemy positions from current observations.
        If we see enemy units, mark their positions.
        If not, guess where they might be (e.g., around last known positions, relic nodes, or corners).
        Enemy heatmap is gradually decayed and updated each turn.
        """
        units_mask = np.array(obs["units_mask"])
        enemy_mask = units_mask[self.opp_team_id]
        enemy_positions = np.array(obs["units"]["position"][self.opp_team_id])

        # Decay previous heatmap to reflect uncertainty over time
        self.enemy_heatmap *= 0.95

        for uid in range(self.max_units):
            if enemy_mask[uid]:
                pos = tuple(enemy_positions[uid])
                if pos[0] != -1 and pos[1] != -1:
                    self.enemy_unit_positions[uid] = pos
                    self.enemy_heatmap[pos[0], pos[1]] += 5.0
            else:
                # If we knew their last position, spread uncertainty
                last_pos = self.enemy_unit_positions.get(uid, None)
                if last_pos is not None:
                    x, y = last_pos
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                                self.enemy_heatmap[nx, ny] += 0.2
                else:
                    # Never seen: guess near relics or opposite corners
                    if self.relic_node_positions:
                        for rx, ry in self.relic_node_positions:
                            for dx in range(-3, 4):
                                for dy in range(-3, 4):
                                    nx, ny = rx + dx, ry + dy
                                    if (
                                        0 <= nx < self.map_width
                                        and 0 <= ny < self.map_height
                                    ):
                                        self.enemy_heatmap[nx, ny] += 0.1
                    else:
                        # Guess enemy near opposite corner spawn
                        corner = (
                            (self.map_width - 1, self.map_height - 1)
                            if self.team_id == 0
                            else (0, 0)
                        )
                        cx, cy = corner
                        for dx in range(-3, 4):
                            for dy in range(-3, 4):
                                nx, ny = cx + dx, cy + dy
                                if (
                                    0 <= nx < self.map_width
                                    and 0 <= ny < self.map_height
                                ):
                                    self.enemy_heatmap[nx, ny] += 0.05

    def _decide_sap(self, unit_pos, unit_energy, match_id):
        """
        Decide whether to sap enemies based on the enemy heatmap.
        The sap action hits a target tile and adjacent tiles.
        More aggressive sapping if match_id >= 2.
        """
        if unit_energy < self.sap_cost:
            return None
        threshold = 3 if match_id < 2 else 2
        x, y = unit_pos
        best_value = 0
        best_action = None
        for dx in range(-self.sap_range, self.sap_range + 1):
            for dy in range(-self.sap_range, self.sap_range + 1):
                tx, ty = x + dx, y + dy
                if 0 <= tx < self.map_width and 0 <= ty < self.map_height:
                    center_value = self.enemy_heatmap[tx, ty]
                    adj_value = 0
                    # Check adjacent tiles around target for dropoff sap effect
                    for ax in range(tx - 1, tx + 2):
                        for ay in range(ty - 1, ty + 2):
                            if (
                                (ax, ay) != (tx, ty)
                                and 0 <= ax < self.map_width
                                and 0 <= ay < self.map_height
                            ):
                                adj_value += (
                                    self.enemy_heatmap[ax, ay] * self.sap_dropoff_factor
                                )
                    total_value = center_value + adj_value
                    if total_value > best_value:
                        best_value = total_value
                        best_action = [5, dx, dy]  # action 5 = sap
        if best_value > threshold:
            return best_action
        return None

    def _around_relic_behavior(self, unit_pos, nearest_relic, match_id, unit_energy):
        """
        Behavior when near a relic node.
        If we know which tiles around the relic score points (discovered in early matches), go there.
        Else, move randomly or just probe around.
        """
        dist = self._manhattan_dist(unit_pos, nearest_relic)
        rx, ry = nearest_relic
        if dist <= 4:
            # Check for known scoring tiles
            known_scoring_tiles = [
                (rx + dx, ry + dy)
                for (dx, dy), val in self.relic_scoring_info[nearest_relic].items()
                if val
            ]
            if known_scoring_tiles and match_id >= 2:
                # After some matches, we know scoring patterns, pick a known scoring tile
                chosen = self._nearest_pos(unit_pos, known_scoring_tiles)
                d = self._find_path(unit_pos, chosen)
                return d
            else:
                # Random probing if we don't know the scoring pattern yet
                rand_dir = np.random.randint(0, 5)
                return rand_dir
        else:
            # Move closer to the relic node if not within range
            d = self._find_path(unit_pos, nearest_relic)
            return d if d is not None else 0

    def _avoid_bad_tiles(self, unit_pos, action):
        """
        If the chosen action leads onto a nebula tile that reduces energy,
        consider staying put (action 0) instead, unless we need to move.
        """
        if action == 0:
            return action
        x, y = unit_pos
        if action == 1:
            y -= 1
        elif action == 2:
            x += 1
        elif action == 3:
            y += 1
        elif action == 4:
            x -= 1

        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            tile_type = self.tile_info[x][y]["type"]
            if tile_type == 1 and self.nebula_energy_red > 0:
                # Nebula tile drains energy, maybe just don't move
                return 0
        return action

    def _track_scoring_events(self, obs):
        """
        Observe if team_points changed from the last step.
        If we gained points, guess that a unit near a relic might be on a scoring tile.
        This could help fill in self.relic_scoring_info if implemented fully.
        """
        team_points = np.array(obs["team_points"])
        if np.any(team_points != self.last_match_points):
            diff = team_points - self.last_match_points
            if diff[self.team_id] > 0:
                # We scored: could try to deduce which tile caused scoring
                pass
        self.last_match_points = team_points

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Main decision function called every turn.
        Reads current step and observations, updates internal knowledge,
        and decides actions for all units.
        """
        self.global_step = step
        match_id = step // self.max_steps_in_match
        # Increment match count when a new match starts
        if step % self.max_steps_in_match == 0 and step > 0:
            self.match_count += 1

        self._update_knowledge(obs)
        self._mark_relic_nodes(obs)
        self._update_enemy_positions(obs)
        self._track_scoring_events(obs)

        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])

        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.max_units, 3), dtype=int)
        known_relics = self.relic_node_positions

        for unit_id in available_unit_ids:
            ux, uy = unit_positions[unit_id]
            unit_pos = (ux, uy)
            unit_energy = unit_energys[unit_id]

            # Attempt a sap action if beneficial
            sap_action = self._decide_sap(unit_pos, unit_energy, match_id)
            if sap_action is not None:
                actions[unit_id] = sap_action
                continue

            # If unit is low on energy, don't move to save energy
            if unit_energy < self.move_cost:
                actions[unit_id] = [0, 0, 0]
                continue

            # Strategy differs by match_id:
            # In first two matches (0,1): focus on exploration and discovering relics
            # In later matches (>=2): use discovered relic info to score effectively
            if match_id < 2:
                # Early exploration phase
                if known_relics:
                    # If relic known, approach and try to discover scoring pattern
                    nearest_relic = self._nearest_pos(unit_pos, known_relics)
                    if nearest_relic is not None:
                        d = self._around_relic_behavior(
                            unit_pos, nearest_relic, match_id, unit_energy
                        )
                        d = self._avoid_bad_tiles(unit_pos, d)
                        actions[unit_id] = [d, 0, 0]
                    else:
                        # No relic known yet, explore assigned sector
                        if unit_id not in self.unit_explore_targets or step % 50 == 0:
                            self.unit_explore_targets[unit_id] = (
                                self._choose_sector_target(unit_id, unit_pos)
                            )
                        d = self._find_path(
                            unit_pos, self.unit_explore_targets[unit_id]
                        )
                        d = self._avoid_bad_tiles(unit_pos, d)
                        actions[unit_id] = [d, 0, 0]
                else:
                    # No relic discovered, just explore the map
                    if unit_id not in self.unit_explore_targets or step % 50 == 0:
                        self.unit_explore_targets[unit_id] = self._choose_sector_target(
                            unit_id, unit_pos
                        )
                    d = self._find_path(unit_pos, self.unit_explore_targets[unit_id])
                    d = self._avoid_bad_tiles(unit_pos, d)
                    actions[unit_id] = [d, 0, 0]
            else:
                # Later matches: exploit known relics and scoring tiles
                if known_relics:
                    nearest_relic = self._nearest_pos(unit_pos, known_relics)
                    if nearest_relic is not None:
                        d = self._around_relic_behavior(
                            unit_pos, nearest_relic, match_id, unit_energy
                        )
                        d = self._avoid_bad_tiles(unit_pos, d)
                        actions[unit_id] = [d, 0, 0]
                    else:
                        # No relic known by now is rare. Just idle or minimal movement.
                        actions[unit_id] = [0, 0, 0]
                else:
                    # If no relic known even after early matches, do nothing
                    actions[unit_id] = [0, 0, 0]

        return actions
