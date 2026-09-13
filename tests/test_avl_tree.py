# File: tests/test_avl_tree.py
"""Test suite for the AVL tree: BST correctness plus balance invariants."""

import math
import random

import pytest

from src.structures.avl_tree import AVLTree


def _assert_avl_invariants(tree):
    """Walk every node checking BST order, stored heights, and balance
    factors — the definition of a valid AVL tree."""
    def check(node):
        if node is None:
            return 0
        left_height = check(node.left)
        right_height = check(node.right)
        assert node.height == 1 + max(left_height, right_height), \
            f"stale height at key {node.key}"
        assert -1 <= node.balance_factor <= 1, \
            f"balance factor {node.balance_factor} at key {node.key}"
        if node.left:
            assert node.left.key < node.key
        if node.right:
            assert node.right.key > node.key
        return node.height

    check(tree._root)


class TestInsertAndTraversal:

    def test_in_order_is_sorted(self):
        tree = AVLTree()
        keys = [50, 30, 70, 20, 40, 60, 80]
        for k in keys:
            tree.insert(k)
        assert [k for k, _ in tree.in_order_traversal()] == sorted(keys)

    def test_empty_tree(self):
        tree = AVLTree()
        assert len(tree) == 0
        assert tree.is_empty()
        assert tree.height() == 0
        assert tree.in_order_traversal() == []

    def test_duplicate_insert_updates_value(self):
        tree = AVLTree()
        tree.insert("k", 1)
        tree.insert("k", 2)
        assert len(tree) == 1
        assert tree.search("k") == 2

    def test_values_travel_with_keys(self):
        tree = AVLTree()
        for k in [5, 3, 8]:
            tree.insert(k, k * 10)
        assert tree.in_order_traversal() == [(3, 30), (5, 50), (8, 80)]

    @pytest.mark.parametrize("n", [100, 1000])
    def test_random_inserts_match_sorted_oracle(self, n):
        random.seed(n)
        keys = random.sample(range(n * 10), n)
        tree = AVLTree()
        for k in keys:
            tree.insert(k)
        assert [k for k, _ in tree.in_order_traversal()] == sorted(keys)
        _assert_avl_invariants(tree)


class TestRotations:
    """Each of the four imbalance shapes must trigger its rotation.
    With 3 nodes the proof is simple: whatever the insert order, a
    balanced result must have height 2 with the middle key at root."""

    @pytest.mark.parametrize("order, case", [
        ([3, 2, 1], "LL: single right rotation"),
        ([1, 2, 3], "RR: single left rotation"),
        ([3, 1, 2], "LR: double rotation"),
        ([1, 3, 2], "RL: double rotation"),
    ], ids=["LL", "RR", "LR", "RL"])
    def test_three_node_rotations(self, order, case):
        tree = AVLTree()
        for k in order:
            tree.insert(k)
        assert tree.height() == 2, f"{case} did not rebalance"
        assert tree._root.key == 2, f"{case}: middle key not at root"
        _assert_avl_invariants(tree)

    def test_sorted_inserts_stay_logarithmic(self):
        """Sorted input is the killer of naive BSTs (height n). The
        AVL bound is height <= 1.44 * log2(n + 2)."""
        tree = AVLTree()
        n = 1024
        for k in range(n):
            tree.insert(k)
        assert tree.height() <= 1.44 * math.log2(n + 2)
        _assert_avl_invariants(tree)

    def test_reverse_sorted_inserts_stay_logarithmic(self):
        tree = AVLTree()
        n = 1024
        for k in range(n, 0, -1):
            tree.insert(k)
        assert tree.height() <= 1.44 * math.log2(n + 2)
        _assert_avl_invariants(tree)


class TestSearch:

    def test_search_finds_all_inserted(self):
        random.seed(3)
        keys = random.sample(range(10000), 500)
        tree = AVLTree()
        for k in keys:
            tree.insert(k, str(k))
        for k in keys:
            assert tree.search(k) == str(k)

    def test_search_missing_raises(self):
        tree = AVLTree()
        tree.insert(1)
        with pytest.raises(KeyError):
            tree.search(2)

    def test_search_empty_raises(self):
        with pytest.raises(KeyError):
            AVLTree().search(1)

    def test_contains(self):
        tree = AVLTree()
        tree.insert(5)
        assert 5 in tree
        assert 6 not in tree


class TestDeletion:

    def test_delete_leaf(self):
        tree = AVLTree()
        for k in [2, 1, 3]:
            tree.insert(k)
        tree.delete(1)
        assert [k for k, _ in tree.in_order_traversal()] == [2, 3]
        _assert_avl_invariants(tree)

    def test_delete_node_with_one_child(self):
        tree = AVLTree()
        for k in [2, 1, 4, 3]:
            tree.insert(k)
        tree.delete(4)  # has only left child 3
        assert [k for k, _ in tree.in_order_traversal()] == [1, 2, 3]
        _assert_avl_invariants(tree)

    def test_delete_node_with_two_children(self):
        tree = AVLTree()
        for k in [50, 30, 70, 20, 40, 60, 80]:
            tree.insert(k)
        tree.delete(50)  # root, two children -> successor 60 replaces it
        assert [k for k, _ in tree.in_order_traversal()] == \
            [20, 30, 40, 60, 70, 80]
        assert 50 not in tree
        _assert_avl_invariants(tree)

    def test_delete_missing_raises(self):
        tree = AVLTree()
        tree.insert(1)
        with pytest.raises(KeyError):
            tree.delete(99)
        assert len(tree) == 1  # failed delete must not change the tree

    def test_delete_rebalances(self):
        """Deleting from one side of a minimal unbalanced shape must
        trigger rotation, not just BST removal."""
        tree = AVLTree()
        for k in [2, 1, 3, 4]:
            tree.insert(k)
        tree.delete(1)  # right side now two levels deeper -> rotate
        assert tree.height() == 2
        _assert_avl_invariants(tree)

    def test_delete_everything(self):
        keys = list(range(64))
        random.seed(8)
        random.shuffle(keys)
        tree = AVLTree()
        for k in keys:
            tree.insert(k)
        random.shuffle(keys)
        for k in keys:
            tree.delete(k)
            _assert_avl_invariants(tree)
        assert tree.is_empty()
        assert tree.height() == 0

    def test_random_insert_delete_fuzz_against_dict(self):
        """1000 mixed operations mirrored against a dict oracle."""
        random.seed(17)
        tree = AVLTree()
        oracle = {}
        for _ in range(1000):
            key = random.randint(0, 200)
            if key in oracle and random.random() < 0.5:
                tree.delete(key)
                del oracle[key]
            else:
                tree.insert(key, key * 2)
                oracle[key] = key * 2
        assert len(tree) == len(oracle)
        assert tree.in_order_traversal() == sorted(oracle.items())
        _assert_avl_invariants(tree)


class TestHeightAndBalance:

    def test_height_single_node(self):
        tree = AVLTree()
        tree.insert(1)
        assert tree.height() == 1

    def test_height_grows_logarithmically(self):
        tree = AVLTree()
        for k in range(2 ** 10):
            tree.insert(k)
        # A perfectly balanced 1024-node tree has height 11 exactly;
        # AVL guarantees at most 1.44 * log2(n + 2) ~ 14.4
        assert 10 <= tree.height() <= 14

    def test_balance_factor_exposed_on_nodes(self):
        tree = AVLTree()
        for k in [2, 1, 3]:
            tree.insert(k)
        assert tree._root.balance_factor == 0
