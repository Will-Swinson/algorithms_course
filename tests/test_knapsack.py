"""Test suite for the 0/1 knapsack solvers and trace_solution()."""

import itertools
import random

import pytest

from src.dp.knapsack import (
    knapsack_recursive, knapsack_memo, knapsack_tabulation,
    build_knapsack_table, trace_solution, knapsack_with_items,
)
from src.utils.timer import CallStats

ALL = [knapsack_recursive, knapsack_memo, knapsack_tabulation]
DP = [knapsack_memo, knapsack_tabulation]

# Classic textbook instance: best is items 1 + 2 (20 + 30 weight) = 220
WEIGHTS = [10, 20, 30]
VALUES = [60, 100, 120]
CAPACITY = 50


def brute_force(weights, values, capacity):
    """Independent oracle: try every subset explicitly."""
    best = 0
    n = len(weights)
    for mask in range(1 << n):
        w = sum(weights[i] for i in range(n) if mask >> i & 1)
        if w <= capacity:
            best = max(best, sum(values[i] for i in range(n)
                                 if mask >> i & 1))
    return best


def random_instance(seed, n=10):
    rng = random.Random(seed)
    weights = [rng.randint(1, 15) for _ in range(n)]
    values = [rng.randint(1, 50) for _ in range(n)]
    return weights, values, rng.randint(0, sum(weights))


@pytest.mark.parametrize("solve", ALL)
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

    def test_exact_fit(self, solve):
        assert solve([3, 4, 5], [30, 40, 50], 12) == 120

    def test_each_item_used_at_most_once(self, solve):
        """Unbounded knapsack would take the weight-1 item 10 times."""
        assert solve([1, 10], [5, 1], 10) == 5

    def test_greedy_by_ratio_is_wrong_here(self, solve):
        """Best ratio is item 0 (6/unit), but items 1 + 2 win."""
        assert solve([1, 2, 3], [6, 10, 12], 5) == 22

    def test_zero_weight_item_always_taken(self, solve):
        assert solve([0, 5], [7, 3], 4) == 7

    @pytest.mark.parametrize("seed", range(15))
    def test_matches_brute_force(self, solve, seed):
        w, v, c = random_instance(seed)
        assert solve(w, v, c) == brute_force(w, v, c)

    def test_mismatched_lengths_raise(self, solve):
        with pytest.raises(ValueError):
            solve([1, 2], [1], 5)

    def test_negative_capacity_raises(self, solve):
        with pytest.raises(ValueError):
            solve([1], [1], -1)

    def test_negative_weight_raises(self, solve):
        with pytest.raises(ValueError):
            solve([-1], [1], 5)


class TestDPAtScale:

    def test_memo_and_tabulation_agree_on_large_input(self):
        rng = random.Random(7)
        w = [rng.randint(1, 30) for _ in range(150)]
        v = [rng.randint(1, 100) for _ in range(150)]
        c = sum(w) // 2
        assert knapsack_memo(w, v, c) == knapsack_tabulation(w, v, c)

    def test_memo_handles_deep_recursion(self):
        """1,500 items -> 1,501-deep recursion, past the default limit."""
        n = 1500
        assert knapsack_memo([1] * n, [1] * n, n) == n


class TestTable:

    def test_dimensions(self):
        table = build_knapsack_table(WEIGHTS, VALUES, CAPACITY)
        assert len(table) == len(WEIGHTS) + 1
        assert all(len(row) == CAPACITY + 1 for row in table)

    def test_row_zero_is_all_zero(self):
        table = build_knapsack_table(WEIGHTS, VALUES, CAPACITY)
        assert set(table[0]) == {0}

    def test_rows_non_decreasing_in_capacity(self):
        table = build_knapsack_table(WEIGHTS, VALUES, CAPACITY)
        for row in table:
            assert row == sorted(row)

    def test_columns_non_decreasing_in_items(self):
        table = build_knapsack_table(WEIGHTS, VALUES, CAPACITY)
        for w in range(CAPACITY + 1):
            column = [table[i][w] for i in range(len(table))]
            assert column == sorted(column)


class TestTraceSolution:

    def test_textbook_items(self):
        table = build_knapsack_table(WEIGHTS, VALUES, CAPACITY)
        assert trace_solution(table, WEIGHTS, CAPACITY) == [1, 2]

    def test_empty_when_nothing_fits(self):
        table = build_knapsack_table([5], [10], 4)
        assert trace_solution(table, [5], 4) == []

    @pytest.mark.parametrize("seed", range(20))
    def test_selection_is_feasible_and_optimal(self, seed):
        w, v, c = random_instance(seed, n=12)
        value, items = knapsack_with_items(w, v, c)
        assert sum(w[i] for i in items) <= c
        assert sum(v[i] for i in items) == value
        assert value == brute_force(w, v, c)

    def test_indices_unique_and_sorted(self):
        _, items = knapsack_with_items([2, 3, 4, 5], [3, 4, 5, 6], 5)
        assert items == sorted(set(items))


class TestCallCounts:

    def test_recursive_bounded_by_full_tree(self):
        n = 10
        stats = CallStats()
        knapsack_recursive([1] * n, [1] * n, n, stats=stats)
        assert stats.calls == 2 ** (n + 1) - 1  # everything fits

    def test_memo_bounded_by_state_space(self):
        rng = random.Random(3)
        n = 20
        w = [rng.randint(1, 10) for _ in range(n)]
        c = sum(w) // 2
        stats = CallStats()
        knapsack_memo(w, [1] * n, c, stats=stats)
        # each distinct (i, w) state solved once, visited <= 2 times
        assert stats.calls <= 2 * (n + 1) * (c + 1)

    def test_memo_far_fewer_calls_than_recursive(self):
        n = 18
        rec, memo = CallStats(), CallStats()
        knapsack_recursive([2] * n, [1] * n, n, stats=rec)
        knapsack_memo([2] * n, [1] * n, n, stats=memo)
        assert memo.calls * 100 < rec.calls

    def test_tabulation_counts_every_cell(self):
        stats = CallStats()
        knapsack_tabulation(WEIGHTS, VALUES, CAPACITY, stats=stats)
        assert stats.calls == len(WEIGHTS) * (CAPACITY + 1)
        assert stats.max_depth == 0

    @pytest.mark.parametrize("solve", [knapsack_recursive, knapsack_memo])
    def test_recursion_depth_is_n_plus_1(self, solve):
        stats = CallStats()
        solve([1] * 8, [1] * 8, 4, stats=stats)
        assert stats.max_depth == 9
