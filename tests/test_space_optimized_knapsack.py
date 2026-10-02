"""Test suite for the O(W)-space 0/1 knapsack variants."""

import random

import pytest

from src.dp.knapsack import knapsack_tabulation
from src.dp_advanced.space_optimized_knapsack import (
    knapsack_two_row, knapsack_1d, knapsack_1d_with_items,
    unbounded_knapsack,
)
from src.utils.timer import CallStats, peak_memory

SOLVERS = [knapsack_two_row, knapsack_1d,
           lambda w, v, c: knapsack_1d_with_items(w, v, c)[0]]
SOLVER_IDS = ["two_row", "1d", "1d_with_items"]

WEIGHTS = [10, 20, 30]
VALUES = [60, 100, 120]
CAPACITY = 50


def brute_force(weights, values, capacity):
    best = 0
    n = len(weights)
    for mask in range(1 << n):
        chosen = [i for i in range(n) if mask >> i & 1]
        if sum(weights[i] for i in chosen) <= capacity:
            best = max(best, sum(values[i] for i in chosen))
    return best


def random_instance(seed, n=10, max_weight=15):
    rng = random.Random(seed)
    weights = [rng.randint(1, max_weight) for _ in range(n)]
    values = [rng.randint(1, 50) for _ in range(n)]
    return weights, values, rng.randint(0, sum(weights))


@pytest.mark.parametrize("solve", SOLVERS, ids=SOLVER_IDS)
class TestCorrectness:

    def test_textbook_instance(self, solve):
        assert solve(WEIGHTS, VALUES, CAPACITY) == 220

    def test_no_items(self, solve):
        assert solve([], [], 10) == 0

    def test_zero_capacity(self, solve):
        assert solve(WEIGHTS, VALUES, 0) == 0

    def test_nothing_fits(self, solve):
        assert solve([5, 6], [10, 20], 4) == 0

    def test_everything_fits(self, solve):
        assert solve(WEIGHTS, VALUES, 1000) == sum(VALUES)

    def test_each_item_at_most_once(self, solve):
        assert solve([1, 10], [5, 1], 10) == 5

    def test_zero_weight_item_taken(self, solve):
        assert solve([0, 5], [7, 3], 4) == 7

    def test_zero_weight_negative_value_skipped(self, solve):
        assert solve([0, 2], [-4, 3], 2) == 3

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_brute_force(self, solve, seed):
        w, v, c = random_instance(seed)
        assert solve(w, v, c) == brute_force(w, v, c)

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_week5_table(self, solve, seed):
        w, v, c = random_instance(seed + 100, n=60, max_weight=40)
        assert solve(w, v, c) == knapsack_tabulation(w, v, c)

    def test_mismatched_lengths_raise(self, solve):
        with pytest.raises(ValueError):
            solve([1, 2], [1], 5)

    def test_negative_capacity_raises(self, solve):
        with pytest.raises(ValueError):
            solve([1], [1], -1)

    def test_negative_weight_raises(self, solve):
        with pytest.raises(ValueError):
            solve([-1], [1], 5)


class TestIterationOrder:
    """The proof in the module docstring, checked on concrete inputs:
    reverse capacity order gives 0/1 knapsack, forward order lets an
    item be reused (unbounded knapsack)."""

    def test_forward_order_reuses_items(self):
        # One item of weight 1, value 5, capacity 10
        assert knapsack_1d([1], [5], 10) == 5
        assert unbounded_knapsack([1], [5], 10) == 50

    def test_orders_differ_on_textbook_instance(self):
        # Unbounded can take the weight-10 item five times: 300
        assert unbounded_knapsack(WEIGHTS, VALUES, CAPACITY) == 300
        assert knapsack_1d(WEIGHTS, VALUES, CAPACITY) == 220

    def test_unbounded_never_below_bounded(self):
        for seed in range(15):
            w, v, c = random_instance(seed)
            assert unbounded_knapsack(w, v, c) >= knapsack_1d(w, v, c)

    def test_unbounded_rejects_free_valuable_item(self):
        with pytest.raises(ValueError):
            unbounded_knapsack([0], [5], 10)


class TestItemReconstruction:

    def test_textbook_items(self):
        assert knapsack_1d_with_items(WEIGHTS, VALUES, CAPACITY) == \
            (220, [1, 2])

    def test_no_items_chosen_when_nothing_fits(self):
        assert knapsack_1d_with_items([5], [10], 4) == (0, [])

    @pytest.mark.parametrize("seed", range(25))
    def test_selection_feasible_and_optimal(self, seed):
        w, v, c = random_instance(seed, n=12)
        value, items = knapsack_1d_with_items(w, v, c)
        assert sum(w[i] for i in items) <= c
        assert sum(v[i] for i in items) == value
        assert value == brute_force(w, v, c)
        assert items == sorted(set(items))

    def test_zero_weight_item_reported(self):
        value, items = knapsack_1d_with_items([0, 3], [4, 5], 3)
        assert (value, items) == (9, [0, 1])


class TestSpace:

    def test_1d_uses_far_less_memory_than_week5_table(self):
        w, v, c = random_instance(7, n=200, max_weight=30)
        table, _ = peak_memory(lambda: knapsack_tabulation(w, v, c))
        row, _ = peak_memory(lambda: knapsack_1d(w, v, c))
        assert row * 20 < table

    def test_1d_memory_independent_of_item_count(self):
        rng = random.Random(3)
        few = ([rng.randint(1, 30) for _ in range(20)],
               [rng.randint(1, 99) for _ in range(20)])
        many = ([rng.randint(1, 30) for _ in range(400)],
                [rng.randint(1, 99) for _ in range(400)])
        small, _ = peak_memory(lambda: knapsack_1d(*few, 1000))
        large, _ = peak_memory(lambda: knapsack_1d(*many, 1000))
        assert large < 1.5 * small

    def test_two_row_counts_full_table_of_work(self):
        stats = CallStats()
        knapsack_two_row(WEIGHTS, VALUES, CAPACITY, stats=stats)
        assert stats.calls == len(WEIGHTS) * (CAPACITY + 1)

    def test_1d_skips_cells_below_item_weight(self):
        stats = CallStats()
        knapsack_1d(WEIGHTS, VALUES, CAPACITY, stats=stats)
        assert stats.calls == sum(CAPACITY - w + 1 for w in WEIGHTS)
        assert stats.max_depth == 0
