# File: tests/test_quick_sort.py
"""Comprehensive test suite for quicksort and its partition helper."""

import random

import pytest

from src.sorting.quick_sort import (
    quick_sort,
    _partition_three_way,
    INSERTION_SORT_THRESHOLD,
)
from tests.conftest import is_sorted, has_same_elements

ALL_SORTS = [quick_sort]


@pytest.mark.parametrize("sort_func", ALL_SORTS, ids=lambda f: f.__name__)
class TestCorrectness:
    """Quicksort must sort every category of input correctly."""

    @pytest.mark.parametrize("case_name", [
        "empty", "single", "sorted", "reverse",
        "duplicates", "all_same", "negative", "mixed",
    ])
    def test_sorts_sample_arrays(self, sort_func, sample_arrays, case_name):
        original = sample_arrays[case_name]
        result = sort_func(original)
        assert is_sorted(result), f"{case_name}: output not sorted"
        assert has_same_elements(original, result), \
            f"{case_name}: output lost or gained elements"

    def test_matches_builtin_sorted(self, sort_func, sample_arrays):
        for original in sample_arrays.values():
            assert sort_func(original) == sorted(original)

    def test_large_random_array(self, sort_func, large_random_array):
        result = sort_func(large_random_array)
        assert result == sorted(large_random_array)

    def test_floats_and_ints_mix(self, sort_func):
        data = [3.5, 1, 2.2, -0.5, 2]
        assert sort_func(data) == sorted(data)

    def test_strings(self, sort_func):
        data = ["banana", "apple", "cherry", "apple"]
        assert sort_func(data) == sorted(data)


@pytest.mark.parametrize("sort_func", ALL_SORTS, ids=lambda f: f.__name__)
class TestSideEffects:
    """Quicksort must return a new list and leave the input untouched.

    Quicksort works in place internally, but on a COPY — so from the
    caller's point of view the contract matches the other sorts.
    """

    def test_original_not_modified(self, sort_func):
        original = [5, 3, 1, 4, 2]
        snapshot = original.copy()
        sort_func(original)
        assert original == snapshot

    def test_returns_new_list(self, sort_func):
        original = [2, 1]
        result = sort_func(original)
        assert result is not original


@pytest.mark.parametrize("sort_func", ALL_SORTS, ids=lambda f: f.__name__)
class TestErrorHandling:
    """Invalid input must raise TypeError, not fail silently."""

    @pytest.mark.parametrize("bad_input", [None, "not a list", 42, (1, 2, 3)])
    def test_non_list_input_raises(self, sort_func, bad_input):
        with pytest.raises(TypeError):
            sort_func(bad_input)

    def test_incomparable_elements_raise(self, sort_func):
        with pytest.raises(TypeError):
            sort_func([1, "two", 3])


class TestSmallArrays:
    """Every size at and around the insertion-sort cutoff must sort
    correctly — off-by-one bugs in the threshold hand-off live here."""

    @pytest.mark.parametrize("size", range(0, INSERTION_SORT_THRESHOLD + 6))
    def test_all_small_sizes(self, size):
        random.seed(size)  # deterministic but different data per size
        data = [random.randint(-50, 50) for _ in range(size)]
        assert quick_sort(data) == sorted(data)


class TestAdversarialInputs:
    """Inputs that make a NAIVE quicksort go quadratic (fixed first/last
    pivot, or two-way partitioning on duplicate-heavy data). With a
    randomized pivot and three-way partitioning these must stay fast —
    a quadratic regression would take ~30s+ here and time out review."""

    def test_already_sorted_large(self):
        data = list(range(10_000))
        assert quick_sort(data) == data

    def test_reverse_sorted_large(self):
        data = list(range(10_000, 0, -1))
        assert quick_sort(data) == sorted(data)

    def test_all_equal_large(self):
        data = [7] * 10_000
        assert quick_sort(data) == data

    def test_few_unique_large(self):
        random.seed(42)
        data = [random.choice([1, 2, 3]) for _ in range(10_000)]
        assert quick_sort(data) == sorted(data)


class TestPartitionHelper:
    """The three-way partition is tested in isolation on a raw list."""

    def test_partition_invariants(self):
        random.seed(7)
        arr = [random.randint(0, 9) for _ in range(200)]
        snapshot = arr.copy()

        lt, gt = _partition_three_way(arr, 0, len(arr) - 1)
        pivot = arr[lt]

        # Three regions: strictly less | all equal to pivot | strictly greater
        assert all(x < pivot for x in arr[:lt])
        assert all(x == pivot for x in arr[lt:gt + 1])
        assert all(x > pivot for x in arr[gt + 1:])
        # Partitioning rearranges, never adds or drops elements
        assert sorted(arr) == sorted(snapshot)

    def test_partition_respects_bounds(self):
        # Elements outside [low, high] must not be touched
        arr = [99, 5, 1, 4, 2, -99]
        _partition_three_way(arr, 1, 4)
        assert arr[0] == 99
        assert arr[5] == -99
        assert sorted(arr[1:5]) == [1, 2, 4, 5]
