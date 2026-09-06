# File: src/sorting/quick_sort.py
"""
QuickSort — divide and conquer with randomized pivots and optimizations.

Divide: pick a pivot and partition the list so everything smaller sits
to its left and everything larger to its right. Conquer: recursively
sort the two sides. Combine: nothing — after partitioning, the pivot
is already in its final place.

Optimizations implemented:
- Randomized pivot selection: no fixed input pattern (sorted, reverse,
  ...) can reliably force the O(n²) worst case; the expected running
  time is O(n log n) for EVERY input order.
- Three-way partitioning (Dutch national flag): elements equal to the
  pivot are grouped in the middle and never recursed into. On inputs
  with many duplicates this collapses the work to O(n).
- Insertion-sort cutoff: subarrays of <= INSERTION_SORT_THRESHOLD
  elements are finished with insertion sort, which beats quicksort on
  tiny inputs (no recursion overhead, great cache behavior).

Like the other sorts in this project, quick_sort:
- Validates its input and raises TypeError for non-list input
- Never modifies the original list (it sorts a copy in place)
- Returns a new sorted list in ascending order

Complexity summary:
    Algorithm       Best          Average       Worst    Space       Stable
    quick_sort      O(n)*         O(n log n)    O(n²)**  O(log n)    No

    *  all-equal input: three-way partitioning finishes in one pass
    ** requires astronomically unlucky random pivots; no input order
       can force it
"""
import random
from typing import List, Tuple

from src.sorting.basic_sorts import _validate_input

# Subarrays at or below this size are sorted with insertion sort
# instead of recursing further (assignment spec: threshold ~10).
INSERTION_SORT_THRESHOLD = 10


def quick_sort(arr: List) -> List:
    """
    Sort a list using randomized quicksort.

    The input is copied once up front, then all partitioning happens
    in place on that copy — combining the efficiency of in-place
    sorting with the project-wide "never modify the caller's list"
    contract.

    Args:
        arr: List of mutually comparable elements

    Returns:
        A new list containing the elements of arr in ascending order

    Raises:
        TypeError: If arr is not a list, or its elements cannot be
            compared with one another (e.g. int vs str)

    Complexity:
        Time: O(n log n) expected for every input order; O(n) on
            all-equal input thanks to three-way partitioning
        Space: O(log n) expected recursion depth (sorts in place on
            one O(n) copy of the input)
    """
    _validate_input(arr, "quick_sort")
    result = arr.copy()
    _quick_sort_range(result, 0, len(result) - 1)
    return result


def _quick_sort_range(arr: List, low: int, high: int) -> None:
    """
    Recursively sort arr[low..high] (inclusive bounds) in place.

    Small subarrays are handed to insertion sort; larger ones are
    partitioned three ways, and only the strictly-less and
    strictly-greater regions are recursed into — the middle block of
    pivot-equal elements is already in final position.
    """
    if high - low + 1 <= INSERTION_SORT_THRESHOLD:
        _insertion_sort_range(arr, low, high)
        return

    lt, gt = _partition_three_way(arr, low, high)
    _quick_sort_range(arr, low, lt - 1)
    _quick_sort_range(arr, gt + 1, high)


def _partition_three_way(arr: List, low: int, high: int) -> Tuple[int, int]:
    """
    Partition arr[low..high] around a randomly chosen pivot.

    Dutch national flag scheme: a single left-to-right pass maintains
    three growing regions —

        arr[low:lt]      strictly less than the pivot
        arr[lt:i]        equal to the pivot
        arr[i:gt+1]      not yet examined
        arr[gt+1:high+1] strictly greater than the pivot

    Each unexamined element is swapped into the region it belongs to.
    The pass ends when the unexamined region is empty.

    Args:
        arr: The list being sorted (modified in place)
        low: First index of the range to partition
        high: Last index of the range to partition (inclusive)

    Returns:
        (lt, gt): after the call, arr[lt..gt] holds every element equal
        to the pivot, in its final sorted position

    Complexity:
        Time: O(high - low) — one pass, each element examined once
        Space: O(1)
    """
    pivot = arr[random.randint(low, high)]

    lt = low        # next slot for a less-than element
    gt = high       # next slot for a greater-than element
    i = low         # current unexamined element

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
            # i is NOT advanced: the element swapped in from the right
            # is unexamined and must be classified on the next loop turn
        else:
            i += 1

    return lt, gt


def _insertion_sort_range(arr: List, low: int, high: int) -> None:
    """
    Sort arr[low..high] (inclusive bounds) in place with insertion sort.

    The Week 1 insertion_sort returns a new list; quicksort needs an
    in-place variant restricted to a subrange, so it is reimplemented
    here with the same shift-based inner loop.

    Complexity:
        Time: O(k²) worst case for a range of k elements — fine, since
            quicksort only calls this for k <= INSERTION_SORT_THRESHOLD
        Space: O(1)
    """
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
