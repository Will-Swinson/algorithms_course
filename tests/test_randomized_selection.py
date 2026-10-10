"""Test suite for randomized QuickSelect, median-of-medians, and the
sort-based baseline, plus Knuth's expected-comparisons formula."""

import random
from fractions import Fraction

import pytest

from src.randomized.randomized_quicksort import PartitionStats
from src.randomized.randomized_selection import (
    median_of_medians_select, randomized_select, sort_select,
)
from src.utils.probability_analysis import (
    expected_quickselect_comparisons, harmonic, summarize,
)
from src.utils.random_utils import random_array

SELECTS = [randomized_select, median_of_medians_select, sort_select]


@pytest.mark.parametrize("select", SELECTS, ids=lambda f: f.__name__)
class TestCorrectness:

    def test_every_rank_small(self, select):
        arr = [9, 2, 7, 4, 5, 1, 8, 3, 6]
        for k in range(1, 10):
            assert select(arr, k) == k

    @pytest.mark.parametrize("seed", range(40))
    def test_random_with_ties(self, select, seed):
        rng = random.Random(seed)
        arr = [rng.randint(0, rng.choice([3, 50, 10 ** 6]))
               for _ in range(rng.randint(1, 120))]
        k = rng.randint(1, len(arr))
        assert select(arr, k) == sorted(arr)[k - 1]

    def test_min_median_max(self, select):
        arr = random_array(1_001, seed=7)
        assert select(arr, 1) == 0
        assert select(arr, 501) == 500
        assert select(arr, 1_001) == 1_000

    def test_single_element(self, select):
        assert select([42], 1) == 42

    def test_all_equal(self, select):
        assert select([3] * 50, 25) == 3

    def test_does_not_modify_input(self, select):
        arr = [5, 1, 4, 2, 3]
        select(arr, 3)
        assert arr == [5, 1, 4, 2, 3]

    def test_strings(self, select):
        assert select(["d", "a", "c", "b"], 2) == "b"

    def test_empty_raises(self, select):
        with pytest.raises(ValueError):
            select([], 1)

    @pytest.mark.parametrize("k", [0, -1, 6])
    def test_k_out_of_range(self, select, k):
        with pytest.raises(ValueError):
            select([1, 2, 3, 4, 5], k)

    def test_k_wrong_type(self, select):
        with pytest.raises(TypeError):
            select([1, 2, 3], 1.0)
        with pytest.raises(TypeError):
            select([1, 2, 3], True)

    def test_non_list(self, select):
        with pytest.raises(TypeError):
            select((1, 2, 3), 1)


class TestWorkCounts:

    def test_randomized_linear_comparisons(self):
        n = 20_000
        arr = random_array(n, seed=1)
        for k in [1, n // 2, n]:
            s = PartitionStats()
            randomized_select(arr, k, s, random.Random(k))
            assert s.comparisons < 8 * n

    def test_randomized_mean_matches_knuth(self):
        n, k, trials = 2_000, 1_000, 300
        arr = random_array(n, seed=2)
        comps = []
        for t in range(trials):
            s = PartitionStats()
            randomized_select(arr, k, s, random.Random(t))
            comps.append(s.comparisons)
        summary = summarize(comps)
        expected = expected_quickselect_comparisons(n, k)
        assert abs(summary.mean - expected) < 4 * summary.std / trials ** 0.5

    def test_min_depth_near_harmonic(self):
        # finding the minimum takes H_n rounds in expectation
        n, trials = 5_000, 200
        arr = random_array(n, seed=3)
        depths = []
        for t in range(trials):
            s = PartitionStats()
            randomized_select(arr, 1, s, random.Random(t))
            depths.append(s.max_depth)
        assert abs(summarize(depths).mean - harmonic(n)) < 1.5

    def test_median_of_medians_linear_worst_case(self):
        # sorted and reverse inputs: no bad pivots for MoM
        for arr in [list(range(5_000)), list(range(5_000, 0, -1))]:
            s = PartitionStats()
            median_of_medians_select(arr, 2_500, s)
            assert s.comparisons < 20 * len(arr)

    def test_median_of_medians_deterministic(self):
        arr = random_array(3_000, seed=4)
        a, b = PartitionStats(), PartitionStats()
        median_of_medians_select(arr, 100, a)
        median_of_medians_select(arr, 100, b)
        assert a == b

    def test_median_of_medians_costs_more_than_quickselect(self):
        n = 10_000
        arr = random_array(n, seed=5)
        mom, qs = PartitionStats(), PartitionStats()
        median_of_medians_select(arr, n // 2, mom)
        randomized_select(arr, n // 2, qs, random.Random(0))
        assert mom.comparisons > qs.comparisons

    def test_seeded_reproducible(self):
        arr = random_array(1_000, seed=6)
        a, b = PartitionStats(), PartitionStats()
        randomized_select(arr, 300, a, random.Random(1))
        randomized_select(arr, 300, b, random.Random(1))
        assert a == b


class TestExpectedFormula:

    def exact_table(self, n_max):
        """C[n][k] = n − 1 + (1/n)·Σ_p (recurse into the side holding k)."""
        c = [[Fraction(0)] * (n_max + 2) for _ in range(n_max + 1)]
        for n in range(2, n_max + 1):
            for k in range(1, n + 1):
                total = Fraction(0)
                for p in range(1, n + 1):          # pivot rank
                    if k < p:
                        total += c[p - 1][k]
                    elif k > p:
                        total += c[n - p][k - p]
                c[n][k] = n - 1 + total / n
        return c

    def test_matches_recurrence(self):
        exact = self.exact_table(25)
        for n in range(1, 26):
            for k in range(1, n + 1):
                assert expected_quickselect_comparisons(n, k) == \
                    pytest.approx(float(exact[n][k]), abs=1e-9)

    def test_symmetric_in_k(self):
        for k in range(1, 51):
            assert expected_quickselect_comparisons(50, k) == pytest.approx(
                expected_quickselect_comparisons(50, 51 - k))

    def test_asymptotic_constants(self):
        n = 100_000
        assert expected_quickselect_comparisons(n, 1) / n == \
            pytest.approx(2.0, abs=0.01)
        assert expected_quickselect_comparisons(n, n // 2) / n == \
            pytest.approx(3.386, abs=0.01)

    def test_k_out_of_range(self):
        with pytest.raises(ValueError):
            expected_quickselect_comparisons(10, 11)
