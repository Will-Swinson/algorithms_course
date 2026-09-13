# File: tests/test_hash_table.py
"""Test suite for both hash tables: chaining and linear probing."""

import random

import pytest

from src.structures.hash_table import ChainingHashTable, LinearProbingHashTable

TABLES = [ChainingHashTable, LinearProbingHashTable]


class _Colliding:
    """Key whose hash is always 0 — forces every instance into the
    same bucket/probe chain while remaining distinguishable by ==."""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, _Colliding) and self.name == other.name


@pytest.mark.parametrize("table_cls", TABLES, ids=lambda c: c.__name__)
class TestBasicOperations:

    def test_insert_get_roundtrip(self, table_cls):
        table = table_cls()
        table.insert("a", 1)
        table.insert("b", 2)
        assert table.get("a") == 1
        assert table.get("b") == 2

    def test_insert_existing_key_updates(self, table_cls):
        table = table_cls()
        table.insert("k", "old")
        table.insert("k", "new")
        assert table.get("k") == "new"
        assert len(table) == 1

    def test_get_missing_raises(self, table_cls):
        with pytest.raises(KeyError):
            table_cls().get("ghost")

    def test_delete_removes_key(self, table_cls):
        table = table_cls()
        table.insert("k", 1)
        table.delete("k")
        assert len(table) == 0
        with pytest.raises(KeyError):
            table.get("k")

    def test_delete_missing_raises(self, table_cls):
        table = table_cls()
        table.insert("other", 1)
        with pytest.raises(KeyError):
            table.delete("ghost")
        assert len(table) == 1

    def test_contains(self, table_cls):
        table = table_cls()
        table.insert(5, "five")
        assert 5 in table
        assert 6 not in table

    def test_mixed_key_types(self, table_cls):
        table = table_cls()
        table.insert("str", 1)
        table.insert(42, 2)
        table.insert((1, 2), 3)
        assert table.get("str") == 1
        assert table.get(42) == 2
        assert table.get((1, 2)) == 3

    def test_unhashable_key_raises(self, table_cls):
        with pytest.raises(TypeError):
            table_cls().insert([1, 2], "lists are unhashable")

    def test_none_value_is_storable(self, table_cls):
        """A stored None must be returned, not confused with 'absent'."""
        table = table_cls()
        table.insert("k", None)
        assert table.get("k") is None
        assert "k" in table


@pytest.mark.parametrize("table_cls", TABLES, ids=lambda c: c.__name__)
class TestScaleAndRehashing:

    def test_thousand_items_roundtrip(self, table_cls):
        random.seed(42)
        items = {f"key_{i}": random.randint(0, 10**6) for i in range(1000)}
        table = table_cls()
        for k, v in items.items():
            table.insert(k, v)
        assert len(table) == 1000
        for k, v in items.items():
            assert table.get(k) == v

    def test_rehash_grows_capacity_and_preserves_items(self, table_cls):
        table = table_cls(initial_capacity=8)
        initial_capacity = table.capacity
        for i in range(100):
            table.insert(i, i * 2)
        assert table.capacity > initial_capacity  # rehash happened
        for i in range(100):
            assert table.get(i) == i * 2

    def test_load_factor_stays_below_threshold(self, table_cls):
        table = table_cls()
        for i in range(1000):
            table.insert(i, i)
            assert table.load_factor <= table._max_load_factor + 1e-9

    def test_load_factor_reflects_size_over_capacity(self, table_cls):
        table = table_cls(initial_capacity=100)
        for i in range(30):
            table.insert(i, i)
        assert table.load_factor == pytest.approx(30 / table.capacity)

    def test_items_yields_everything(self, table_cls):
        table = table_cls()
        expected = {i: str(i) for i in range(50)}
        for k, v in expected.items():
            table.insert(k, v)
        assert dict(table.items()) == expected


@pytest.mark.parametrize("table_cls", TABLES, ids=lambda c: c.__name__)
class TestCollisions:

    def test_all_colliding_keys_still_work(self, table_cls):
        """Every key hashes to bucket 0: the O(n) worst case must stay
        correct even though it is slow."""
        table = table_cls()
        keys = [_Colliding(f"k{i}") for i in range(50)]
        for i, key in enumerate(keys):
            table.insert(key, i)
        assert len(table) == 50
        for i, key in enumerate(keys):
            assert table.get(key) == i

    def test_delete_from_collision_chain(self, table_cls):
        """Deleting from the middle of a chain/cluster must not lose
        the keys that were inserted after the deleted one."""
        table = table_cls()
        keys = [_Colliding(f"k{i}") for i in range(10)]
        for i, key in enumerate(keys):
            table.insert(key, i)
        table.delete(keys[4])
        for i, key in enumerate(keys):
            if i == 4:
                assert key not in table
            else:
                assert table.get(key) == i


class TestLinearProbingTombstones:
    """Probing-specific behavior around deleted slots."""

    def test_reinsert_after_delete_reuses_slot(self):
        table = LinearProbingHashTable()
        table.insert("k", 1)
        table.delete("k")
        table.insert("k", 2)
        assert table.get("k") == 2
        assert len(table) == 1

    def test_probe_walks_through_tombstones(self):
        """k0..k2 collide into one cluster; deleting the FIRST must
        leave the ones probed past it reachable."""
        table = LinearProbingHashTable(initial_capacity=16)
        keys = [_Colliding(f"k{i}") for i in range(3)]
        for i, key in enumerate(keys):
            table.insert(key, i)
        table.delete(keys[0])
        assert table.get(keys[1]) == 1
        assert table.get(keys[2]) == 2

    def test_delete_reinsert_churn(self):
        """Heavy delete/reinsert cycles fill slots with tombstones;
        rehashing must clear them and keep the table usable."""
        table = LinearProbingHashTable(initial_capacity=8)
        for round_number in range(20):
            for i in range(50):
                table.insert(f"key_{i}", round_number)
            for i in range(0, 50, 2):
                table.delete(f"key_{i}")
        # Survivors: odd keys with the latest round's value
        for i in range(1, 50, 2):
            assert table.get(f"key_{i}") == 19
        for i in range(0, 50, 2):
            assert f"key_{i}" not in table


@pytest.mark.parametrize("table_cls", TABLES, ids=lambda c: c.__name__)
def test_fuzz_against_dict_oracle(table_cls):
    """1500 random insert/delete/get operations mirrored in a dict."""
    random.seed(31)
    table = table_cls()
    oracle = {}
    for _ in range(1500):
        key = random.randint(0, 100)
        op = random.random()
        if op < 0.5:
            value = random.randint(0, 10**6)
            table.insert(key, value)
            oracle[key] = value
        elif op < 0.75 and key in oracle:
            table.delete(key)
            del oracle[key]
        else:
            if key in oracle:
                assert table.get(key) == oracle[key]
            else:
                assert key not in table
    assert len(table) == len(oracle)
    assert dict(table.items()) == oracle
