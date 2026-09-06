# File: tests/test_merge_sort.py
"""Comprehensive test suite for merge sort and its merge helper."""

import pytest

from src.sorting.merge_sort import merge_sort, merge
from tests.conftest import is_sorted, has_same_elements

ALL_SORTS = [merge_sort]


@pytest.mark.parametrize("sort_func", ALL_SORTS, ids=lambda f: f.__name__)
class TestCorrectness:
    """Merge sort must sort every category of input correctly."""

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
    """Merge sort must return a new list and leave the input untouched."""

    def test_original_not_modified(self, sort_func):
        original = [5, 3, 1, 4, 2]
        snapshot = original.copy()
        sort_func(original)
        assert original == snapshot

    def test_returns_new_list(self, sort_func):
        original = [2, 1]
        result = sort_func(original)
        assert result is not original

    def test_returns_new_list_even_when_sorted(self, sort_func):
        # An implementation that returns the input unchanged when it is
        # already sorted would break the "never share the caller's list"
        # contract, so this case is pinned down explicitly.
        original = [1, 2, 3]
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


def test_stability():
    """
    Merge sort must be stable: elements that compare equal keep their
    original relative order. This is decided by a single comparison in
    merge() — on ties, the element from the LEFT half must win.
    """
    # Sort (key, tag) pairs by key only; tags record original order
    class Item:
        def __init__(self, key, tag):
            self.key, self.tag = key, tag

        def __lt__(self, other):
            return self.key < other.key

        def __gt__(self, other):
            return self.key > other.key

    data = [Item(1, "a"), Item(0, "b"), Item(1, "c"), Item(0, "d")]
    result = merge_sort(data)
    assert [(item.key, item.tag) for item in result] == \
        [(0, "b"), (0, "d"), (1, "a"), (1, "c")]


class TestMergeHelper:
    """The merge helper is tested in isolation: if these fail, the bug is
    in the merge step itself, not in the splitting or recursion."""

    @pytest.mark.parametrize("left, right", [
        ([], []),                      # both empty
        ([], [1, 2, 3]),               # left exhausted immediately
        ([1, 2, 3], []),               # right exhausted immediately
        ([5], [1, 2, 3, 4]),           # unequal lengths
        ([1, 3, 5], [2, 4, 6]),        # interleaved: merge must alternate
        ([1, 2], [10, 20]),            # non-overlapping: drain-the-rest path
        ([1, 3, 3], [2, 3, 7]),        # duplicates on both sides
    ])
    def test_merges_sorted_pairs(self, left, right):
        assert merge(left, right) == sorted(left + right)

    def test_does_not_mutate_inputs(self):
        left, right = [1, 3, 5], [2, 4]
        left_snapshot, right_snapshot = left.copy(), right.copy()
        merge(left, right)
        assert left == left_snapshot
        assert right == right_snapshot

    def test_linear_time_smoke(self):
        # Not a real complexity proof, just a smoke test: an accidentally
        # quadratic merge (e.g. list.insert or repeated pop(0)) would make
        # this noticeably slow under pytest.
        left = list(range(0, 20000, 2))
        right = list(range(1, 20000, 2))
        assert merge(left, right) == list(range(20000))
