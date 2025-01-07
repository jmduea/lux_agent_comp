from enum import IntEnum

import jax
from flax import struct

from core.base import SPACE_SIZE


@struct.dataclass
class Node:
    x: int
    y: int
    node_type: int
    energy: int
    is_visible: bool
    is_unknown: bool
    is_walkable: bool
    explored_for_relic: bool

    @classmethod
    def create(cls, space, x, y):
        return cls(
            x=x,
            y=y,
            node_type=space.tile_type[y][x],
            energy=space.energy[y][x],
            is_visible=space.is_visible[y][x],
            is_unknown=space.tile_type[y][x] == -1,
            is_walkable=space.tile_type[y][x] != 2,
            explored_for_relic=space.explored_for_relic[y][x],
        )


def make_default_node(x, y) -> Node:
    return Node(x, y, -1, -1, False, True, True, False, False, False, False)


@jax.jit
def init_nodes(space_size=SPACE_SIZE):
    node_grid = []
    for y in range(space_size):
        row = []
        for x in range(space_size):
            row.append(make_default_node(x, y))
        node_grid.append(row)
    return node_grid


class NodeType(IntEnum):
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


# @register_pytree_node_class
# class Node:
#     """Node class representing a single tile on the map.

#     Params:
#         x, y (int): coordinates of the node.

#     Attributes:
#         x, y (int): coordinates of the node.
#         type (NodeType): type of the node.
#         energy (int): energy of the node.
#         is_visible (bool): whether the node is visible.
#         relic (bool): whether the node contains a relic.
#         reward (bool): whether the node provides points.
#         explored_for_relic (bool): whether the node has been explored for a relic.
#         explored_for_reward (bool): whether the node has been explored for reward.
#         is_unknown (bool): whether the node type is unknown.
#         is_walkable (bool): whether the node is walkable.
#         coordinates (tuple[int, int]): coordinates of the node.
#         is_upper_sector (bool): whether the node is in the upper sector.
#         is_lower_sector (bool): whether the node is in the lower sector.

#     Methods:
#         update_relic_status: update the relic status of the node.
#         update_reward_status: update the reward status of the node.
#         tree_flatten: flatten the node for JAX pytree.
#         tree_unflatten: unflatten the node for JAX pytree.
#     """

#     def __init__(self, x: int, y: int, type: NodeType = NodeType.unknown):
#         self.x = x
#         self.y = y
#         self.type: NodeType = type
#         self.energy: int = None
#         self.is_visible: bool = False

#         self._relic: bool = False
#         self._reward: bool = False
#         self._explored_for_relic: bool = False
#         self._explored_for_reward: bool = False

#     def __repr__(self) -> str:
#         return f"Node({self.x}, {self.y}, {self.type})"

#     def __hash__(self):
#         return self.coordinates.__hash__()

#     def __eq__(self, other) -> bool:
#         return self.x == other.x and self.y == other.y

#     @property
#     def relic(self) -> bool:
#         return self._relic

#     @property
#     def reward(self) -> bool:
#         return self._reward

#     @property
#     def explored_for_relic(self) -> bool:
#         return self._explored_for_relic

#     @property
#     def explored_for_reward(self) -> bool:
#         return self._explored_for_reward

#     def update_relic_status(self, status: bool):
#         if self._explored_for_relic and self._relic != status:
#             raise ValueError(
#                 f"Can't change the relic status {self._relic}->{status} for {self}"
#                 ", the tile has already been explored"
#             )

#         self._relic = status
#         self._explored_for_relic = True

#     def update_reward_status(self, status: bool):
#         if self._explored_for_reward and self._reward != status:
#             raise ValueError(
#                 f"Can't change the reward status {self._reward}->{status} for {self}"
#                 ", the tile has already been explored"
#             )

#         self._reward = status
#         self._explored_for_reward = True

#     @property
#     def is_unknown(self) -> bool:
#         return self.type == NodeType.unknown

#     @property
#     def is_walkable(self) -> bool:
#         return self.type != NodeType.asteroid

#     @property
#     def coordinates(self) -> tuple[int, int]:
#         return self.x, self.y

#     def manhattan_distance(self, other: "Node") -> int:
#         return abs(self.x - other.x) + abs(self.y - other.y)

#     def tree_flatten(
#         self,
#     ) -> tuple[
#         tuple[int, int, NodeType, int, bool, bool, bool, bool, bool], dict[str, int]
#     ]:
#         children: tuple[int, int, NodeType, int, bool, bool, bool, bool, bool] = (
#             self.x,
#             self.y,
#             self.type,
#             self.energy,
#             self.is_visible,
#             self._relic,
#             self._reward,
#             self._explored_for_relic,
#             self._explored_for_reward,
#         )
#         aux_data = None
#         return (children, aux_data)

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         (
#             x,
#             y,
#             node_type,
#             energy,
#             is_visible,
#             relic,
#             reward,
#             explored_for_relic,
#             explored_for_reward,
#         ) = children
#         node = cls(x, y, node_type)
#         node.energy = energy
#         node.is_visible = is_visible
#         node._relic = relic
#         node._reward = reward
#         node._explored_for_relic = explored_for_relic
#         node._explored_for_reward = explored_for_reward
#         return node
#         return node
#         return node
