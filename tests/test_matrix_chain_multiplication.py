"""Test suite for matrix chain multiplication (interval DP) and the
matrix utilities used to verify it."""

import inspect
import itertools
import re
import sys

import pytest

from src.dp_advanced.matrix_chain_multiplication import (
    mcm_recursive, mcm_memoized, mcm_bottom_up, build_mcm_tables,
    parenthesize, matrix_chain_order, split_table,
)
from src.utils.matrix_utils import (
    matrix_multiply, multiply_chain, random_chain_dimensions, random_matrix,
    left_to_right_cost,
)
from src.utils.timer import CallStats

SOLVERS = [mcm_recursive, mcm_memoized, mcm_bottom_up]
CLRS = [30, 35, 15, 5, 10, 20, 25]


def all_orders_min_cost(p):
    """Independent oracle: enumerate every full parenthesization."""
    def costs(i, j):
        if i == j:
            return [0]
        out = []
        for k in range(i, j):
            for left, right in itertools.product(costs(i, k),
                                                 costs(k + 1, j)):
                out.append(left + right + p[i - 1] * p[k] * p[j])
        return out
    return min(costs(1, len(p) - 1))


@pytest.mark.parametrize("solve", SOLVERS)
class TestCost:

    @pytest.mark.parametrize("p, expected", [
        (CLRS, 15125),
        ([10, 20, 30], 6000),
        ([40, 20, 30, 10, 30], 26000),
        ([10, 20, 30, 40, 30], 30000),
        ([1, 2, 3, 4, 3], 30),
        ([10, 100, 5, 50], 7500),
    ])
    def test_known_answers(self, solve, p, expected):
        assert solve(p) == expected

    def test_single_matrix_costs_nothing(self, solve):
        assert solve([7, 9]) == 0

    def test_two_matrices_single_product(self, solve):
        assert solve([3, 4, 5]) == 60

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_exhaustive_oracle(self, solve, seed):
        p = random_chain_dimensions(6, low=1, high=30, seed=seed)
        assert solve(p) == all_orders_min_cost(p)

    def test_never_worse_than_left_to_right(self, solve):
        for seed in range(10):
            p = random_chain_dimensions(7, seed=seed)
            assert solve(p) <= left_to_right_cost(p)

    @pytest.mark.parametrize("bad", [[], [5], [3, 0, 4], [3, -2, 4]])
    def test_invalid_dimensions_raise(self, solve, bad):
        with pytest.raises(ValueError):
            solve(bad)

    def test_non_int_dimension_raises(self, solve):
        with pytest.raises(TypeError):
            solve([3, 4.5, 6])


class TestDPAtScale:

    def test_memoized_and_bottom_up_agree(self):
        for n in [20, 60, 120]:
            p = random_chain_dimensions(n, seed=n)
            assert mcm_memoized(p) == mcm_bottom_up(p)

    def test_memoized_raises_recursion_limit_itself(self):
        """With only ~60 frames of headroom, a 200-deep recursion must
        still succeed because mcm_memoized raises the limit."""
        previous = sys.getrecursionlimit()
        p = random_chain_dimensions(200, seed=1)
        expected = mcm_bottom_up(p)
        sys.setrecursionlimit(len(inspect.stack()) + 60)
        try:
            assert mcm_memoized(p) == expected
        finally:
            sys.setrecursionlimit(previous)


class TestParenthesization:

    def test_clrs_order(self):
        assert matrix_chain_order(CLRS) == (15125, "((A1(A2A3))((A4A5)A6))")

    def test_memoized_method_same_answer(self):
        assert matrix_chain_order(CLRS, method="memoized") == \
            matrix_chain_order(CLRS)

    def test_single_matrix(self):
        assert matrix_chain_order([2, 3]) == (0, "A1")

    def test_two_matrices(self):
        assert matrix_chain_order([2, 3, 4]) == (24, "(A1A2)")

    def test_prefers_cheap_order(self):
        # (A1A2)A3 = 7,500; A1(A2A3) = 75,000
        assert matrix_chain_order([10, 100, 5, 50])[1] == "((A1A2)A3)"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            matrix_chain_order(CLRS, method="greedy")

    @pytest.mark.parametrize("seed", range(10))
    def test_string_lists_matrices_in_order_once(self, seed):
        n = 9
        _, order = matrix_chain_order(random_chain_dimensions(n, seed=seed))
        assert re.findall(r"A(\d+)", order) == \
            [str(i) for i in range(1, n + 1)]
        # one pair of parentheses per multiplication
        assert order.count("(") == order.count(")") == n - 1

    def test_parenthesize_subchain(self):
        _, split = build_mcm_tables(CLRS)
        assert parenthesize(split, 2, 3) == "(A2A3)"
        assert parenthesize(split, 4, 4) == "A4"


class TestAgainstRealArithmetic:
    """Multiply real matrices in the DP's order: the scalar
    multiplications actually performed must equal the DP cost, and the
    product must equal plain left-to-right multiplication."""

    @pytest.mark.parametrize("method", ["bottom_up", "memoized"])
    @pytest.mark.parametrize("seed", range(5))
    def test_cost_matches_performed_multiplications(self, method, seed):
        p = random_chain_dimensions(6, low=1, high=8, seed=seed)
        matrices = [random_matrix(p[i], p[i + 1], seed=seed * 10 + i)
                    for i in range(len(p) - 1)]
        product, performed = multiply_chain(matrices,
                                            split_table(p, method))
        assert performed == mcm_bottom_up(p)

        expected = matrices[0]
        for m in matrices[1:]:
            expected, _ = matrix_multiply(expected, m)
        assert product == expected

    def test_matrix_multiply_counts_pqr(self):
        product, count = matrix_multiply([[1, 2], [3, 4]], [[5], [6]])
        assert product == [[17], [39]]
        assert count == 2 * 2 * 1

    def test_matrix_multiply_dimension_mismatch(self):
        with pytest.raises(ValueError):
            matrix_multiply([[1, 2]], [[1, 2]])

    def test_left_to_right_cost(self):
        assert left_to_right_cost([10, 100, 5, 50]) == 5000 + 2500


class TestWork:

    @pytest.mark.parametrize("n", [1, 2, 4, 7])
    def test_recursive_makes_3_to_the_n_minus_1_calls(self, n):
        stats = CallStats()
        mcm_recursive(random_chain_dimensions(n, seed=1), stats=stats)
        assert stats.calls == 3 ** (n - 1)

    @pytest.mark.parametrize("n", [2, 5, 10, 30])
    def test_bottom_up_split_count(self, n):
        stats = CallStats()
        mcm_bottom_up(random_chain_dimensions(n, seed=2), stats=stats)
        assert stats.calls == n * (n * n - 1) // 6
        assert stats.max_depth == 0

    @pytest.mark.parametrize("n", [2, 5, 10, 30])
    def test_memoized_call_count(self, n):
        """Each interval is solved once and makes 2 calls per split."""
        stats = CallStats()
        mcm_memoized(random_chain_dimensions(n, seed=3), stats=stats)
        assert stats.calls == 1 + 2 * (n * (n * n - 1) // 6)

    def test_recursion_depth_is_n(self):
        stats = CallStats()
        mcm_memoized(random_chain_dimensions(12, seed=4), stats=stats)
        assert stats.max_depth == 12

    def test_utilities_seeded(self):
        assert random_chain_dimensions(5, seed=9) == \
            random_chain_dimensions(5, seed=9)
        assert len(random_chain_dimensions(5, seed=9)) == 6
        with pytest.raises(ValueError):
            random_chain_dimensions(0)
