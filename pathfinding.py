import heapq
import numpy as np

from base import SPACE_SIZE, NodeType, Global, ActionType

CARDINAL_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class State:
    NEW = 0
    OPEN = 1
    CLOSED = 2
    RAISED = 3
    LOWER = 4


class DStarNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = State.NEW
        self.h = float("inf")  # Cost-to-goal estimate
        self.k = float("inf")  # Minimum of old h and new h
        self.parent = None


def dstar(weights, start, goal):
    # D* algorithm implementation
    # Returns the shortest path from start to goal

    min_weight = weights[np.where(weights >= 0)].min()
    nodes = {}
    open_list = []

    def get_node(pos):
        if pos not in nodes:
            nodes[pos] = DStarNode(pos[0], pos[1])
        return nodes[pos]

    def process_state():
        if not open_list:
            return -1

        k_old = get_k_min()
        node = get_node(heapq.heappop(open_list)[2])

        if node.state == State.CLOSED:
            return k_old

        if k_old < node.h:
            for neighbor_pos in get_neighbors(node.x, node.y):
                neighbor = get_node(neighbor_pos)
                if neighbor.h <= k_old and node.h > neighbor.h + get_cost(
                    neighbor_pos, (node.x, node.y)
                ):
                    node.parent = neighbor_pos
                    node.h = neighbor.h + get_cost(neighbor_pos, (node.x, node.y))

        if k_old == node.h:
            for neighbor_pos in get_neighbors(node.x, node.y):
                neighbor = get_node(neighbor_pos)
                if (
                    neighbor.state == State.NEW
                    or (
                        neighbor.parent == (node.x, node.y)
                        and neighbor.h
                        != node.h + get_cost((node.x, node.y), neighbor_pos)
                    )
                    or (
                        neighbor.parent != (node.x, node.y)
                        and neighbor.h
                        > node.h + get_cost((node.x, node.y), neighbor_pos)
                    )
                ):
                    neighbor.parent = (node.x, node.y)
                    insert(
                        neighbor_pos, node.h + get_cost((node.x, node.y), neighbor_pos)
                    )
        else:
            for neighbor_pos in get_neighbors(node.x, node.y):
                neighbor = get_node(neighbor_pos)
                if neighbor.state == State.NEW or (
                    neighbor.parent == (node.x, node.y)
                    and neighbor.h != node.h + get_cost((node.x, node.y), neighbor_pos)
                ):
                    neighbor.parent = (node.x, node.y)
                    insert(
                        neighbor_pos, node.h + get_cost((node.x, node.y), neighbor_pos)
                    )
                else:
                    if neighbor.parent != (
                        node.x,
                        node.y,
                    ) and neighbor.h > node.h + get_cost(
                        (node.x, node.y), neighbor_pos
                    ):
                        insert((node.x, node.y), node.h)
                    else:
                        if (
                            neighbor.parent != (node.x, node.y)
                            and node.h
                            > neighbor.h + get_cost(neighbor_pos, (node.x, node.y))
                            and neighbor.state == State.CLOSED
                            and neighbor.h > k_old
                        ):
                            insert(neighbor_pos, neighbor.h)
        return get_k_min()

    def get_k_min():
        if not open_list:
            return -1
        return open_list[0][0]

    def get_cost(pos1, pos2):
        if weights[pos2[1], pos2[0]] < 0:
            return float("inf")
        return weights[pos2[1], pos2[0]]

    def insert(pos, h_new):
        node = get_node(pos)
        if node.state == State.NEW:
            node.k = h_new
        elif node.state == State.OPEN:
            node.k = min(node.k, h_new)
        elif node.state == State.CLOSED:
            node.k = min(node.h, h_new)
        node.h = h_new
        node.state = State.OPEN
        heapq.heappush(open_list, (node.k, node.h, pos))

    # Initialize D*
    goal_node = get_node(goal)
    goal_node.h = 0
    insert(goal, 0)

    while True:
        k_min = process_state()
        if k_min == -1:
            break
        if get_node(start).state == State.CLOSED:
            break

    # Reconstruct path
    if get_node(start).h == float("inf"):
        return []

    path = []
    current = start
    while current != goal:
        if current is None:
            return []
        path.append(current)
        current_node = get_node(current)
        if current_node.parent is None:
            return []
        current = current_node.parent
    path.append(goal)
    return path


def manhattan_distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(x, y):
    for dx, dy in CARDINAL_DIRECTIONS:
        x_ = x + dx
        if x_ < 0 or x_ >= SPACE_SIZE:
            continue

        y_ = y + dy
        if y_ < 0 or y_ >= SPACE_SIZE:
            continue

        yield x_, y_


def reconstruct_path(nodes, start, goal):
    p = goal
    path = [p]
    while p != start:
        x = int(nodes[p[0], p[1], 0])
        y = int(nodes[p[0], p[1], 1])
        p = x, y
        path.append(p)
    return path[::-1]


def nearby_positions(x, y, distance):
    for x_ in range(max(0, x - distance), min(SPACE_SIZE, x + distance + 1)):
        for y_ in range(max(0, y - distance), min(SPACE_SIZE, y + distance + 1)):
            yield x_, y_


def create_weights(space):
    # create weights for D* algorithm

    weights = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
    for node in space:
        if not node.is_walkable:
            weight = -1
        else:
            node_energy = node.energy
            if node_energy is None:
                node_energy = Global.HIDDEN_NODE_ENERGY

            # pathfinding can't deal with negative weight
            weight = Global.MAX_ENERGY_PER_TILE + 1 - node_energy

        if node.type == NodeType.nebula:
            weight += Global.NEBULA_ENERGY_REDUCTION

        weights[node.y][node.x] = weight

    return weights


def find_closest_target(start, targets):
    target, min_distance = None, float("inf")
    for t in targets:
        d = manhattan_distance(start, t)
        if d < min_distance:
            target, min_distance = t, d

    return target, min_distance


def estimate_energy_cost(space, path):
    if len(path) <= 1:
        return 0

    energy = 0
    last_position = path[0]
    for x, y in path[1:]:
        node = space.get_node(x, y)
        if node.energy is not None:
            energy -= node.energy
        else:
            energy -= Global.HIDDEN_NODE_ENERGY

        if node.type == NodeType.nebula:
            energy += Global.NEBULA_ENERGY_REDUCTION

        if (x, y) != last_position:
            energy += Global.UNIT_MOVE_COST

    return energy


def path_to_actions(path):
    actions = []
    if not path:
        return actions

    last_position = path[0]
    for x, y in path[1:]:
        direction = ActionType.from_coordinates(last_position, (x, y))
        actions.append(direction)
        last_position = (x, y)

    return actions
