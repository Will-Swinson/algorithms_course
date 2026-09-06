# File: src/sorting/__init__.py
"""Sorting algorithms package: re-exports all five sorts."""

from src.sorting.basic_sorts import bubble_sort, selection_sort, insertion_sort
from src.sorting.merge_sort import merge_sort, merge
from src.sorting.quick_sort import quick_sort

__all__ = [
    "bubble_sort",
    "selection_sort",
    "insertion_sort",
    "merge_sort",
    "merge",
    "quick_sort",
]
