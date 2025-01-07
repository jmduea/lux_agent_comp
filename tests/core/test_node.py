import unittest

import jax

from core.node import Node, init_nodes, make_default_node


class TestNode(unittest.TestCase):
    def test_default_node_creation(self):
        x, y = 0, 0
        node = make_default_node(x, y)
        self.assertIsInstance(node, Node)
        self.assertEqual(node.x, x)
        self.assertEqual(node.y, y)

    def test_node_flatten(self):
        node = make_default_node(0, 0)
        children, aux_data = jax.tree.flatten(node)
        (
            x,
            y,
            node_type,
            energy,
            is_visible,
            is_unknown,
            is_walkable,
            explored_for_relic,
            explored_for_reward,
            has_relic,
            has_reward,
        ) = children
        self.assertEqual(x, node.x)
        self.assertEqual(y, node.y)
        self.assertEqual(node_type, node.node_type)
        self.assertEqual(energy, node.energy)
        self.assertEqual(is_visible, node.is_visible)
        self.assertEqual(is_unknown, node.is_unknown)
        self.assertEqual(is_walkable, node.is_walkable)
        self.assertEqual(has_relic, node.has_relic)
        self.assertEqual(has_reward, node.has_reward)
        self.assertEqual(explored_for_relic, node.explored_for_relic)
        self.assertEqual(explored_for_reward, node.explored_for_reward)

    def test_node_unflatten(self):
        node = make_default_node(0, 0)
        children, aux_data = jax.tree.flatten(node)
        new_node = jax.tree.unflatten(aux_data, children)
        self.assertIsInstance(new_node, Node)
        self.assertEqual(new_node.x, node.x)
        self.assertEqual(new_node.y, node.y)
        self.assertEqual(new_node.node_type, node.node_type)
        self.assertEqual(new_node.energy, node.energy)
        self.assertEqual(new_node.is_visible, node.is_visible)
        self.assertEqual(new_node.is_unknown, node.is_unknown)
        self.assertEqual(new_node.is_walkable, node.is_walkable)
        self.assertEqual(new_node.has_relic, node.has_relic)
        self.assertEqual(new_node.has_reward, node.has_reward)
        self.assertEqual(new_node.explored_for_relic, node.explored_for_relic)
        self.assertEqual(new_node.explored_for_reward, node.explored_for_reward)

    def test_init_nodes(self):
        node_grid = init_nodes()
        self.assertIsInstance(node_grid, list)
        self.assertEqual(len(node_grid), 24)
        self.assertEqual(node_grid[23][23].x, 23)
        self.assertEqual(node_grid[23][23].y, 23)

    # class TestNode(unittest.TestCase):
    # def test_node_initialization(self):
    #     x, y = 5, 10
    #     node = Node(x, y)

    #     self.assertEqual(node.x, x)
    #     self.assertEqual(node.y, y)
    #     self.assertEqual(node.type, NodeType.unknown)
    #     self.assertIsNone(node.energy)
    #     self.assertFalse(node.is_visible)
    #     self.assertFalse(node.relic)
    #     self.assertFalse(node.reward)
    #     self.assertFalse(node.explored_for_relic)
    #     self.assertFalse(node.explored_for_reward)

    # def test_node_tree_flatten(self):
    #     x, y = 5, 10
    #     node = Node(x, y)

    #     children, aux_data = node.tree_flatten()

    #     self.assertIsInstance(children, tuple)
    #     self.assertIsNone(aux_data)
    #     self.assertEqual(len(children), len(node.__dict__))
    #     self.assertEqual(children[0], x)
    #     self.assertEqual(children[1], y)
    #     self.assertEqual(children[2], node.type)
    #     self.assertEqual(children[3], node.energy)
    #     self.assertEqual(children[4], node.is_visible)
    #     self.assertEqual(children[5], node.relic)
    #     self.assertEqual(children[6], node.reward)
    #     self.assertEqual(children[7], node.explored_for_relic)
    #     self.assertEqual(children[8], node.explored_for_reward)

    # def test_node_tree_unflatten(self):
    #     x, y = 5, 10
    #     node = Node(x, y)
    #     children, aux_data = node.tree_flatten()
    #     new_node = Node.tree_unflatten(aux_data, children)

    #     # Check if new_node is an instance of Node
    #     self.assertIsInstance(new_node, Node)

    #     # Check if new_node is actually just the old node object
    #     self.assertIsNot(new_node, node)

    #     # Check if the new node equivalent to the old node
    #     self.assertTrue(new_node.tree_flatten() == node.tree_flatten())

    # def test_update_relic_status_initial(self):
    #     node = Node(5, 10)
    #     self.assertFalse(node.relic)
    #     self.assertFalse(node.explored_for_relic)

    #     node.update_relic_status(True)
    #     self.assertTrue(node.relic)
    #     self.assertTrue(node.explored_for_relic)

    # def test_update_relic_status_change(self):
    #     node = Node(5, 10)
    #     node.update_relic_status(True)
    #     self.assertTrue(node.relic)
    #     self.assertTrue(node.explored_for_relic)

    #     with self.assertRaises(ValueError):
    #         node.update_relic_status(False)

    # def test_update_relic_status_no_change(self):
    #     node = Node(5, 10)
    #     node.update_relic_status(True)
    #     self.assertTrue(node.relic)
    #     self.assertTrue(node.explored_for_relic)

    #     node.update_relic_status(True)
    #     self.assertTrue(node.relic)
    #     self.assertTrue(node.explored_for_relic)

    # def test_update_relic_status_false_initial(self):
    #     node = Node(5, 10)
    #     node.update_relic_status(False)
    #     self.assertFalse(node.relic)
    #     self.assertTrue(node.explored_for_relic)

    # def test_update_relic_status_false_no_change(self):
    #     node = Node(5, 10)
    #     node.update_relic_status(False)
    #     self.assertFalse(node.relic)
    #     self.assertTrue(node.explored_for_relic)

    #     node.update_relic_status(False)
    #     self.assertFalse(node.relic)
    #     self.assertTrue(node.explored_for_relic)

    # def test_update_reward_status_initial(self):
    #     node = Node(5, 10)
    #     self.assertFalse(node.reward)
    #     self.assertFalse(node.explored_for_reward)

    #     node.update_reward_status(True)
    #     self.assertTrue(node.reward)
    #     self.assertTrue(node.explored_for_reward)

    # def test_update_reward_status_change(self):
    #     node = Node(5, 10)
    #     node.update_reward_status(True)
    #     self.assertTrue(node.reward)
    #     self.assertTrue(node.explored_for_reward)

    #     with self.assertRaises(ValueError):
    #         node.update_reward_status(False)

    # def test_update_reward_status_no_change(self):
    #     node = Node(5, 10)
    #     node.update_reward_status(True)
    #     self.assertTrue(node.reward)
    #     self.assertTrue(node.explored_for_reward)

    #     node.update_reward_status(True)
    #     self.assertTrue(node.reward)
    #     self.assertTrue(node.explored_for_reward)

    # def test_update_reward_status_false_initial(self):
    #     node = Node(5, 10)
    #     node.update_reward_status(False)
    #     self.assertFalse(node.reward)
    #     self.assertTrue(node.explored_for_reward)

    # def test_update_reward_status_false_no_change(self):
    #     node = Node(5, 10)
    #     node.update_reward_status(False)
    #     self.assertFalse(node.reward)
    #     self.assertTrue(node.explored_for_reward)

    #     node.update_reward_status(False)
    #     self.assertFalse(node.reward)
    #     self.assertTrue(node.explored_for_reward)
    #     self.assertFalse(node.reward)
    #     self.assertTrue(node.explored_for_reward)


if __name__ == "__main__":
    unittest.main()
