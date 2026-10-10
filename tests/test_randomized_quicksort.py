"""Test suite for randomized and deterministic QuickSort, the shared
three-way partition, and the expected-comparisons formula."""

import random
from fractions import Fraction

import pytest

from src.randomized.randomized_quicksort import (
    PartitionStats, deterministic_quicksort, partition_three_way,
    randomized_quicksort,
)
from src.utils.probability_analysis import (
    expected_quicksort_comparisons, summarize,
)
from src.utils.random_utils import random_array
from tests.conftest import has_same_elements, is_sorted

SORTS = [randomized_quicksort, deterministic_quicksort]


@pytest.mark.parametrize("sort_func", SORTS, ids=lambda f: f.__name__)
class TestCorrectness:

    @pytest.mark.parametrize("case_name", [
        "empty", "single", "sorted", "reverse",
        "duplicates", "all_same", "negative", "mixed",
    ])
    def test_sample_arrays(self, sort_func, sample_arrays, case_name):
        original = sample_arrays[case_name]
        result = sort_func(original)
        assert result == sorted(original)

    def test_large_random(self, sort_func, large_random_array):
        result = sort_func(large_random_array)
        assert is_sorted(result)
        assert has_same_elements(result, large_random_array)

    @pytest.mark.parametrize("seed", range(30))
    def test_random_small_inputs_with_ties(self, sort_func, seed):
        rng = random.Random(seed)
        arr = [rng.randint(0, rng.choice([2, 10, 1000]))
               for _ in range(rng.randint(0, 60))]
        assert sort_func(arr) == sorted(arr)

    def test_does_not_modify_input(self, sort_func):
        arr = [5, 3, 1, 4, 2]
        sort_func(arr)
        assert arr == [5, 3, 1, 4, 2]

    def test_returns_new_list(self, sort_func):
        arr = [1, 2, 3]
        assert sort_func(arr) is not arr

    def test_strings_and_floats(self, sort_func):
        assert sort_func(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]
        assert sort_func([2.5, -1.0, 2.5, 0.0]) == [-1.0, 0.0, 2.5, 2.5]

    def test_rejects_non_list(self, sort_func):
        with pytest.raises(TypeError):
            sort_func((3, 1, 2))

    def test_rejects_incomparable(self, sort_func):
        with pytest.raises(TypeError):
            sort_func([1, "a", 2, "b"])

    def test_sorted_input_large_does_not_overflow_stack(self, sort_func):
        # first-pivot on sorted input is n levels deep; explicit stack
        arr = list(range(3_000))
        assert sort_func(arr) == arr


class TestPartition:

    @pytest.mark.parametrize("seed", range(40))
    def test_regions(self, seed):
        rng = random.Random(seed)
        arr = [rng.randint(0, 8) for _ in range(rng.randint(1, 40))]
        original = arr.copy()
        low = rng.randint(0, len(arr) - 1)
        high = rng.randint(low, len(arr) - 1)
        pivot_index = rng.randint(low, high)
        pivot = arr[pivot_index]
        lt, gt = partition_three_way(arr, low, high, pivot_index)
        assert low <= lt <= gt <= high
        assert all(x < pivot for x in arr[low:lt])
        assert all(x == pivot for x in arr[lt:gt + 1])
        assert all(x > pivot for x in arr[gt + 1:high + 1])
        assert arr[:low] == original[:low]
        assert arr[high + 1:] == original[high + 1:]
        assert sorted(arr[low:high + 1]) == sorted(original[low:high + 1])

    def test_counts_m_minus_one_comparisons(self):
        stats = PartitionStats()
        partition_three_way([4, 1, 7, 3, 9, 2], 0, 5, 2, stats)
        assert stats.comparisons == 5 and stats.partitions == 1

    def test_preserves_order_of_greater_elements(self):
        # the property that makes first-pivot quadratic on sorted input
        arr = [0, 1, 2, 3, 4, 5]
        lt, gt = partition_three_way(arr, 0, 5, 0)
        assert (lt, gt) == (0, 0)
        assert arr == [0, 1, 2, 3, 4, 5]

    def test_all_equal_single_block(self):
        arr = [7] * 10
        assert partition_three_way(arr, 0, 9, 4) == (0, 9)


class TestComparisonCounts:

    def test_deterministic_sorted_is_quadratic(self):
        for n in [10, 100, 500]:
            stats = PartitionStats()
            deterministic_quicksort(list(range(n)), stats)
            assert stats.comparisons == n * (n - 1) // 2
            assert stats.max_depth == n - 1

    def test_deterministic_reverse_is_quadratic(self):
        n = 300
        stats = PartitionStats()
        deterministic_quicksort(list(range(n, 0, -1)), stats)
        assert stats.comparisons == n * (n - 1) // 2

    def test_deterministic_counts_are_repeatable(self):
        arr = random_array(500, seed=3)
        counts = set()
        for _ in range(5):
            s = PartitionStats()
            deterministic_quicksort(arr, s)
            counts.add(s.comparisons)
        assert len(counts) == 1

    def test_all_equal_is_linear(self):
        for func in SORTS:
            stats = PartitionStats()
            func([1] * 1_000, stats)
            assert stats.comparisons == 999

    def test_randomized_sorted_input_near_expectation(self):
        n = 2_000
        stats = PartitionStats()
        randomized_quicksort(list(range(n)), stats, random.Random(1))
        assert stats.comparisons < 1.5 * expected_quicksort_comparisons(n)

    def test_randomized_mean_matches_formula(self):
        n, trials = 1_000, 200
        arr = random_array(n, seed=11)
        comps = []
        for t in range(trials):
            s = PartitionStats()
            randomized_quicksort(arr, s, random.Random(t))
            comps.append(s.comparisons)
        summary = summarize(comps)
        expected = expected_quicksort_comparisons(n)
        # 200 trials: std of mean ≈ 0.65n/√200 ≈ 46, so 4σ ≈ 185
        assert abs(summary.mean - expected) < 4 * summary.std / trials ** 0.5
        # theoretical std ≈ 0.648 n
        assert 0.4 * n < summary.std < 0.9 * n

    def test_randomized_depth_logarithmic(self):
        stats = PartitionStats()
        randomized_quicksort(random_array(10_000, seed=2), stats,
                             random.Random(5))
        assert stats.max_depth < 60      # ≈ 4.3 ln n ≈ 40 expected

    def test_seeded_rng_reproducible(self):
        arr = random_array(800, seed=4)
        a, b = PartitionStats(), PartitionStats()
        randomized_quicksort(arr, a, random.Random(9))
        randomized_quicksort(arr, b, random.Random(9))
        assert a == b

    def test_different_seeds_differ(self):
        arr = random_array(800, seed=4)
        counts = set()
        for seed in range(5):
            s = PartitionStats()
            randomized_quicksort(arr, s, random.Random(seed))
            counts.add(s.comparisons)
        assert len(counts) > 1


class TestExpectedFormula:

    def exact_recurrence(self, n_max):
        """E[C_n] = n − 1 + (2/n)·Σ_{i<n} E[C_i], exactly in fractions."""
        e = [Fraction(0), Fraction(0)]
        running = Fraction(0)
        for n in range(2, n_max + 1):
            running += e[n - 1]
            e.append(n - 1 + Fraction(2, n) * running)
        return e

    def test_matches_recurrence(self):
        exact = self.exact_recurrence(60)
        for n in range(0, 61):
            assert expected_quicksort_comparisons(n) == \
                pytest.approx(float(exact[n]), abs=1e-9)

    def test_small_values(self):
        assert expected_quicksort_comparisons(1) == 0
        assert expected_quicksort_comparisons(2) == pytest.approx(1)
        assert expected_quicksort_comparisons(3) == pytest.approx(8 / 3)

    def test_large_n_uses_asymptotic_harmonic(self):
        # continuity across the harmonic() switch at n = 1000
        a = expected_quicksort_comparisons(999)
        b = expected_quicksort_comparisons(1000)
        assert 0 < b - a < 20
