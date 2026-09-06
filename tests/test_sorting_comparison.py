# File: tests/test_sorting_comparison.py
"""Cross-algorithm comparison tests: all five sorts must agree with the
built-in sorted() (and therefore with each other) on every data shape
used by the Week 2 benchmarks."""

import random

import pytest

from src.sorting import (
    bubble_sort, selection_sort, insertion_sort, merge_sort, quick_sort,
)

ALL_SORTS = [bubble_sort, selection_sort, insertion_sort,
             merge_sort, quick_sort]

SIZE = 500
SEED = 42


def _make(data_type):
    """Small-scale versions of the six benchmark data types."""
    rng = random.Random(SEED)
    if data_type == "random_data":
        return [rng.randint(0, SIZE) for _ in range(SIZE)]
    if data_type == "sorted_data":
        return list(range(SIZE))
    if data_type == "reverse_data":
        return list(range(SIZE, 0, -1))
    if data_type == "nearly_sorted":
        data = list(range(SIZE))
        for _ in range(SIZE // 40):
            a, b = rng.randrange(SIZE), rng.randrange(SIZE)
            data[a], data[b] = data[b], data[a]
        return data
    if data_type == "many_duplicates":
        return [rng.randint(0, 9) for _ in range(SIZE)]
    if data_type == "few_unique":
        return [rng.choice([1, 2, 3]) for _ in range(SIZE)]
    raise ValueError(data_type)


DATA_TYPES = ["random_data", "sorted_data", "reverse_data",
              "nearly_sorted", "many_duplicates", "few_unique"]


@pytest.mark.parametrize("sort_func", ALL_SORTS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("data_type", DATA_TYPES)
def test_matches_oracle_on_benchmark_data(sort_func, data_type):
    data = _make(data_type)
    assert sort_func(data) == sorted(data)


@pytest.mark.parametrize("data_type", DATA_TYPES)
def test_all_algorithms_agree(data_type):
    """All five sorts, given identical input, produce identical output."""
    data = _make(data_type)
    outputs = [sort_func(data) for sort_func in ALL_SORTS]
    assert all(out == outputs[0] for out in outputs)
