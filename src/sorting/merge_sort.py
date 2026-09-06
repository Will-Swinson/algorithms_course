# File: src/sorting/merge_sort.py
"""
Merge sort — a divide and conquer sorting algorithm.

The list is recursively split in half until pieces of length 0 or 1
remain (which are sorted by definition), then pairs of sorted pieces
are combined with a linear-time merge.

Like the Week 1 sorts, merge_sort:
- Validates its input and raises TypeError for non-list input
- Never modifies the original list
- Returns a new sorted list in ascending order

Complexity summary:
    Algorithm       Best          Average       Worst         Space   Stable
    merge_sort      O(n log n)    O(n log n)    O(n log n)    O(n)    Yes
"""
from typing import List

from src.sorting.basic_sorts import _validate_input


def merge(left: List, right: List) -> List:
    """
    Merge two already-sorted lists into one sorted list.

    Walks both lists front-to-back with one index each, repeatedly
    taking the smaller head element. When one list runs out, the rest
    of the other is appended unchanged. Every element is looked at
    exactly once, so the merge runs in O(n + m) time for inputs of
    length n and m.

    Stability: on ties the element from `left` is taken first. Because
    merge_sort always passes the earlier half of the list as `left`,
    equal elements keep their original relative order.

    Args:
        left: A sorted list
        right: A sorted list

    Returns:
        A new sorted list containing every element of left and right

    Complexity:
        Time: O(n + m) — each element is copied exactly once
        Space: O(n + m) for the output list
    """
    merged = []
    i = j = 0

    # Take the smaller head element until one list is exhausted.
    # `right < left` (strict) means ties go to the left list — this
    # single comparison is what makes merge sort stable.
    while i < len(left) and j < len(right):
        if right[j] < left[i]:
            merged.append(right[j])
            j += 1
        else:
            merged.append(left[i])
            i += 1

    # One of these is empty; the other's remainder is already sorted
    # and every element in it is >= everything merged so far.
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def merge_sort(arr: List) -> List:
    """
    Sort a list using merge sort (divide and conquer).

    Divide: split the list into two halves at the midpoint.
    Conquer: recursively merge-sort each half.
    Combine: merge the two sorted halves in linear time.

    The recursion depth is log2(n) because the input halves at each
    level, and each level does O(n) total merge work, giving the
    O(n log n) bound in every case — unlike the Week 1 sorts, merge
    sort's running time does not depend on the input's initial order.

    Args:
        arr: List of mutually comparable elements

    Returns:
        A new list containing the elements of arr in ascending order

    Raises:
        TypeError: If arr is not a list, or its elements cannot be
            compared with one another (e.g. int vs str)

    Complexity:
        Time: O(n log n) best, average, and worst case
        Space: O(n) — the merge step builds new lists
    """
    _validate_input(arr, "merge_sort")

    # Base case: 0 or 1 elements are sorted by definition. Copy so the
    # caller never receives (a reference to) their own list back.
    if len(arr) <= 1:
        return arr.copy()

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
