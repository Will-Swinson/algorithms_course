"""Test suite for Subset Sum: backtracking, the bitset DP, the greedy
1/2-approximation, and the instance generator."""

import itertools
import random

import pytest

from src.complexity.subset_sum_backtracking import (
    greedy_subset_sum, is_subset_sum_solution, random_subset_sum,
    subset_sum_backtracking, subset_sum_dp,
)
from src.utils.timer import CallStats


def best_by_enumeration(nums, target):
    return max(sum(c) for r in range(len(nums) + 1)
               for c in itertools.combinations(nums, r) if sum(c) <= target)


class TestBacktracking:

    def test_textbook_example(self):
        nums = [3, 34, 4, 12, 5, 2]
        found = subset_sum_backtracking(nums, 9)
        assert is_subset_sum_solution(nums, 9, found)
        assert subset_sum_backtracking(nums, 30) is None

    def test_edge_cases(self):
        assert subset_sum_backtracking([], 0) == []
        assert subset_sum_backtracking([], 5) is None
        assert subset_sum_backtracking([5, 1], 0) == []
        assert subset_sum_backtracking([0, 0, 7], 7) is not None

    def test_duplicates_use_distinct_indices(self):
        found = subset_sum_backtracking([4, 4, 4], 8)
        assert len(set(found)) == 2

    @pytest.mark.parametrize("seed", range(80))
    def test_matches_enumeration(self, seed):
        rng = random.Random(seed)
        nums = [rng.randint(0, 30) for _ in range(rng.randint(0, 10))]
        target = rng.randint(0, 80)
        found = subset_sum_backtracking(nums, target)
        exists = best_by_enumeration(nums, target) == target
        assert (found is not None) == exists
        assert found is None or is_subset_sum_solution(nums, target, found)

    def test_no_instance_explores_exponentially(self):
        counts = []
        for n in (8, 12, 16):
            nums, target = random_subset_sum(n, 10 ** 12, False, seed=n)
            stats = CallStats()
            assert subset_sum_backtracking(nums, target, stats) is None
            assert stats.calls < 2 ** (n + 1)
            counts.append(stats.calls)
        assert counts[1] > 4 * counts[0] and counts[2] > 4 * counts[1]

    def test_depth_at_most_n_plus_one(self):
        stats = CallStats()
        subset_sum_backtracking(list(range(1, 13)), 10 ** 6, stats)
        assert stats.max_depth <= 13

    @pytest.mark.parametrize("nums,target", [([1, -2], 1), ([1], -1)])
    def test_invalid(self, nums, target):
        with pytest.raises(ValueError):
            subset_sum_backtracking(nums, target)


class TestDP:

    @pytest.mark.parametrize("seed", range(60))
    def test_best_sum_matches_enumeration(self, seed):
        rng = random.Random(seed)
        nums = [rng.randint(1, 40) for _ in range(rng.randint(0, 10))]
        target = rng.randint(0, 120)
        assert subset_sum_dp(nums, target) == best_by_enumeration(nums,
                                                                  target)

    def test_decision_agrees_with_backtracking(self):
        for seed in range(30):
            nums, target = random_subset_sum(14, 200, seed % 2 == 0,
                                             seed=seed)
            assert (subset_sum_dp(nums, target) == target) == (
                subset_sum_backtracking(nums, target) is not None)

    def test_items_larger_than_target_ignored(self):
        assert subset_sum_dp([100, 3], 5) == 3


class TestGreedy:

    def test_never_exceeds_target(self):
        for seed in range(50):
            nums, target = random_subset_sum(30, 500, seed=seed)
            chosen = greedy_subset_sum(nums, target)
            assert sum(nums[i] for i in chosen) <= target
            assert len(set(chosen)) == len(chosen)

    @pytest.mark.parametrize("seed", range(100))
    def test_half_approximation(self, seed):
        rng = random.Random(seed)
        nums = [rng.randint(1, 100) for _ in range(rng.randint(1, 12))]
        target = rng.randint(1, sum(nums))
        got = sum(nums[i] for i in greedy_subset_sum(nums, target))
        assert 2 * got >= subset_sum_dp(nums, target)

    def test_ratio_can_approach_half(self):
        # greedy takes 51, then nothing fits; optimum 50 + 50 = 100
        nums, target = [51, 50, 50], 100
        got = sum(nums[i] for i in greedy_subset_sum(nums, target))
        assert got == 51 and subset_sum_dp(nums, target) == 100

    def test_takes_everything_that_fits(self):
        assert greedy_subset_sum([1, 2, 3], 100) == [0, 1, 2]


class TestGenerator:

    def test_planted_is_solvable(self):
        for seed in range(10):
            nums, target = random_subset_sum(20, 10 ** 9, True, seed=seed)
            assert subset_sum_backtracking(nums, target) is not None

    def test_seeded_and_bounded(self):
        a = random_subset_sum(15, 50, seed=3)
        assert a == random_subset_sum(15, 50, seed=3)
        assert all(1 <= x <= 50 for x in a[0])

    def test_invalid(self):
        with pytest.raises(ValueError):
            random_subset_sum(-1)
