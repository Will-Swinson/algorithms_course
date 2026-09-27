"""Test suite for naive, memoized, and tabulated Fibonacci."""

import pytest

from src.dp.fibonacci import (
    fib_naive, fib_memo, fib_tabulation, naive_call_count,
)
from src.utils.timer import CallStats

ALL = [fib_naive, fib_memo, fib_tabulation]
DP = [fib_memo, fib_tabulation]

# F(0) .. F(20)
KNOWN = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
         987, 1597, 2584, 4181, 6765]


@pytest.mark.parametrize("fib", ALL)
class TestCorrectness:

    def test_base_cases(self, fib):
        assert fib(0) == 0
        assert fib(1) == 1

    def test_known_values(self, fib):
        assert [fib(n) for n in range(len(KNOWN))] == KNOWN

    def test_recurrence_holds(self, fib):
        for n in range(2, 22):
            assert fib(n) == fib(n - 1) + fib(n - 2)

    def test_negative_raises(self, fib):
        with pytest.raises(ValueError):
            fib(-1)

    @pytest.mark.parametrize("bad", [3.0, "5", None, True])
    def test_non_int_raises(self, fib, bad):
        with pytest.raises(TypeError):
            fib(bad)


@pytest.mark.parametrize("fib", DP)
class TestDPScale:

    def test_large_value(self, fib):
        assert fib(90) == 2880067194370816120

    def test_matches_other_dp_method(self, fib):
        other = fib_tabulation if fib is fib_memo else fib_memo
        for n in [50, 100, 300]:
            assert fib(n) == other(n)

    def test_beyond_default_recursion_limit(self, fib):
        """fib_memo recurses n deep; it must raise the limit itself."""
        assert fib(5000) == fib_tabulation(5000)

    def test_arbitrary_precision(self, fib):
        assert len(str(fib(1000))) == 209


class TestCallCounts:
    """Call counts are the evidence for O(2ⁿ) vs O(n)."""

    @pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 20])
    def test_naive_matches_formula(self, n):
        stats = CallStats()
        fib_naive(n, stats=stats)
        assert stats.calls == naive_call_count(n)

    def test_naive_formula_known_value(self):
        assert naive_call_count(30) == 2_692_537

    @pytest.mark.parametrize("n", [1, 2, 10, 100])
    def test_memo_makes_2n_minus_1_calls(self, n):
        stats = CallStats()
        fib_memo(n, stats=stats)
        assert stats.calls == 2 * n - 1

    @pytest.mark.parametrize("n", [0, 1, 10, 100])
    def test_tabulation_fills_n_plus_1_cells(self, n):
        stats = CallStats()
        fib_tabulation(n, stats=stats)
        assert stats.calls == n + 1

    def test_naive_grows_exponentially(self):
        counts = [naive_call_count(n) for n in (20, 21, 22, 23)]
        ratios = [b / a for a, b in zip(counts, counts[1:])]
        for r in ratios:
            assert 1.6 < r < 1.65  # → golden ratio φ ≈ 1.618

    def test_memo_grows_linearly(self):
        a, b = CallStats(), CallStats()
        fib_memo(100, stats=a)
        fib_memo(200, stats=b)
        assert b.calls / a.calls == pytest.approx(2, rel=0.01)


class TestRecursionDepth:

    @pytest.mark.parametrize("fib", [fib_naive, fib_memo])
    def test_recursive_depth_is_n(self, fib):
        stats = CallStats()
        fib(15, stats=stats)
        assert stats.max_depth == 15

    def test_tabulation_has_no_recursion(self):
        stats = CallStats()
        fib_tabulation(100, stats=stats)
        assert stats.max_depth == 0

    def test_counter_balanced_after_call(self):
        stats = CallStats()
        fib_naive(10, stats=stats)
        assert stats._depth == 0

    def test_stats_optional(self):
        assert fib_naive(10) == fib_naive(10, stats=CallStats()) == 55
