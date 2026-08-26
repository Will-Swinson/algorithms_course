# File: tests/test_sorting.py
"""Comprehensive test suite for the basic sorting algorithms."""
import pytest

from src.sorting.basic_sorts import bubble_sort, selection_sort, insertion_sort
from tests.conftest import is_sorted, has_same_elements

ALL_SORTS = [bubble_sort, selection_sort, insertion_sort]
STABLE_SORTS = [bubble_sort, insertion_sort]


@pytest.mark.parametrize("sort_func", ALL_SORTS, ids=lambda f: f.__name__)
class TestCorrectness:
    """Every algorithm must sort every category of input correctly."""

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
    """Algorithms must return a new list and leave the input untouched."""

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


@pytest.mark.parametrize("sort_func", STABLE_SORTS, ids=lambda f: f.__name__)
def test_stability(sort_func):
    """
    Bubble and insertion sort are stable: elements that compare equal
    must keep their original relative order. (Selection sort is not
    stable, so it is deliberately excluded.)
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
    result = sort_func(data)
    assert [(item.key, item.tag) for item in result] == \
        [(0, "b"), (0, "d"), (1, "a"), (1, "c")]
