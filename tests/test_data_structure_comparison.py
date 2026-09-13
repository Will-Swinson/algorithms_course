# File: tests/test_data_structure_comparison.py
"""Cross-structure comparison tests: heaps, AVL tree, and hash tables
loaded with the same dataset must agree with each other and with the
Python built-ins that serve as oracles (sorted, dict)."""

import random

import pytest

from src.structures import (
    MinHeap, MaxHeap, AVLTree, ChainingHashTable, LinearProbingHashTable,
)


@pytest.fixture
def dataset():
    """500 unique keys with values, plus a set of absent keys."""
    random.seed(42)
    keys = random.sample(range(100000), 500)
    absent = [k for k in range(100000, 100050)]
    return {k: f"value_{k}" for k in keys}, absent


def test_ordered_structures_agree(dataset):
    """MinHeap drain, MaxHeap drain (reversed), and AVL in-order all
    equal sorted(keys)."""
    items, _ = dataset
    keys = list(items)

    min_heap = MinHeap(keys)
    max_heap = MaxHeap(keys)
    tree = AVLTree()
    for k in keys:
        tree.insert(k)

    expected = sorted(keys)
    min_drained = [min_heap.extract_min() for _ in range(len(keys))]
    max_drained = [max_heap.extract_max() for _ in range(len(keys))]
    assert min_drained == expected
    assert max_drained[::-1] == expected
    assert [k for k, _ in tree.in_order_traversal()] == expected


def test_keyed_structures_agree_with_dict(dataset):
    """AVL tree and both hash tables must return exactly what a dict
    returns for every present key, and raise KeyError for absent ones."""
    items, absent = dataset

    tree = AVLTree()
    chaining = ChainingHashTable()
    probing = LinearProbingHashTable()
    for k, v in items.items():
        tree.insert(k, v)
        chaining.insert(k, v)
        probing.insert(k, v)

    for k, v in items.items():
        assert tree.search(k) == v
        assert chaining.get(k) == v
        assert probing.get(k) == v

    for structure_get in (tree.search, chaining.get, probing.get):
        for k in absent:
            with pytest.raises(KeyError):
                structure_get(k)


def test_deletions_stay_consistent(dataset):
    """Delete the same half of the keys everywhere; all structures must
    agree on what remains."""
    items, _ = dataset
    keys = list(items)
    to_delete = set(keys[::2])
    survivors = sorted(k for k in keys if k not in to_delete)

    tree = AVLTree()
    chaining = ChainingHashTable()
    probing = LinearProbingHashTable()
    for k, v in items.items():
        tree.insert(k, v)
        chaining.insert(k, v)
        probing.insert(k, v)
    for k in to_delete:
        tree.delete(k)
        chaining.delete(k)
        probing.delete(k)

    assert len(tree) == len(chaining) == len(probing) == len(survivors)
    assert [k for k, _ in tree.in_order_traversal()] == survivors
    assert sorted(k for k, _ in chaining.items()) == survivors
    assert sorted(k for k, _ in probing.items()) == survivors


def test_heap_sort_equals_avl_traversal():
    """Two independent O(n log n) sorting routes — drain a heap vs
    walk an AVL tree — must produce identical output."""
    random.seed(7)
    values = [random.randint(0, 999) for _ in range(1000)]  # duplicates likely

    heap = MinHeap(values)
    heap_sorted = [heap.extract_min() for _ in range(len(values))]

    tree = AVLTree()
    counts = {}
    for v in values:  # AVL keys are unique, so count duplicates in values
        counts[v] = counts.get(v, 0) + 1
        tree.insert(v)
    tree_sorted = []
    for k, _ in tree.in_order_traversal():
        tree_sorted.extend([k] * counts[k])

    assert heap_sorted == tree_sorted == sorted(values)
