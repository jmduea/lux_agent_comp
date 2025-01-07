from enum import IntEnum


class Global:
    # Game related constants:

    SPACE_SIZE = 24
    MAX_UNITS = 16
    RELIC_REWARD_RANGE = 2
    MAX_STEPS_IN_MATCH = 100
    MAX_ENERGY_PER_TILE = 20
    MAX_RELIC_NODES = 6

    # We will find the exact value of these constants during the game
    UNIT_MOVE_COST = 1  # OPTIONS: list(range(1, 6))
    UNIT_SAP_COST = 30  # OPTIONS: list(range(30, 51))
    UNIT_SAP_RANGE = 3  # OPTIONS: list(range(3, 8))
    UNIT_SENSOR_RANGE = 2  # OPTIONS: list(range(2, 5))
    OBSTACLE_MOVEMENT_PERIOD = 20  # OPTIONS: 20, 40
    OBSTACLE_MOVEMENT_DIRECTION = (0, 0)  # OPTIONS: [(1, -1), (-1, 1)]

    # We will NOT find the exact value of these constants during the game
    NEBULA_ENERGY_REDUCTION = 10  # OPTIONS: [0, 10, 25]

    # Exploration flags:

    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = False
    OBSTACLE_MOVEMENT_PERIOD_FOUND = False
    OBSTACLE_MOVEMENT_DIRECTION_FOUND = False

    # Game logs:

    # REWARD_RESULTS: [{"nodes": Set[Node], "points": int}, ...]
    # A history of reward events, where each entry contains:
    # - "nodes": A set of nodes where our ships were located.
    # - "points": The number of points scored at that location.
    # This data will help identify which nodes yield points.
    REWARD_RESULTS = []

    # obstacles_movement_status: list of bool
    # A history log of obstacle (asteroids and nebulae) movement events.
    # - `True`: The ships' sensors detected a change in the obstacles' positions at this step.
    # - `False`: The sensors did not detect any changes.
    # This information will be used to determine the speed and direction of obstacle movement.
    OBSTACLES_MOVEMENT_STATUS = []

    # Others:

    # The energy on the unknown tiles will be used in the pathfinding
    HIDDEN_NODE_ENERGY = 0


SPACE_SIZE = Global.SPACE_SIZE
MAX_UNITS = Global.MAX_UNITS

_DIRECTIONS = [
    (0, 0),  # center
    (0, -1),  # up
    (1, 0),  # right
    (0, 1),  #  down
    (-1, 0),  # left
    (0, 0),  # sap
]


class ActionType(IntEnum):
    STAY = 0
    RIGHT = 1
    LEFT = 2
    DOWN = 3
    UP = 4
    SAP = 5

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @classmethod
    def from_coordinates(cls, current_position, next_position):
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]

        if dx < 0:
            return ActionType.LEFT
        elif dx > 0:
            return ActionType.RIGHT
        elif dy < 0:
            return ActionType.UP
        elif dy > 0:
            return ActionType.DOWN
        else:
            return ActionType.STAY

    def to_direction(self):
        return _DIRECTIONS[self]
