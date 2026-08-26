# File: src/sorting/basic_sorts.py
"""
Basic comparison-based sorting algorithms.

Each algorithm:
- Validates its input and raises TypeError for non-list input
- Sorts a copy of the input (the original list is never modified)
- Returns the sorted list in ascending order

Complexity summary:
    Algorithm       Best        Average     Worst       Space   Stable
    bubble_sort     O(n)        O(n²)       O(n²)       O(1)    Yes
    selection_sort  O(n²)       O(n²)       O(n²)       O(1)    No
    insertion_sort  O(n)        O(n²)       O(n²)       O(1)    Yes
"""
from typing import List, Any


def _validate_input(arr: Any, algorithm_name: str) -> None:
    """
    Validate that the input is a list.

    Args:
        arr: The value to validate
        algorithm_name: Name used in the error message

    Raises:
        TypeError: If arr is not a list
    """
    if not isinstance(arr, list):
        raise TypeError(
            f"{algorithm_name} expects a list, got {type(arr).__name__}"
        )


def bubble_sort(arr: List) -> List:
    """
    Sort a list using optimized bubble sort.

    Repeatedly steps through the list, compares adjacent elements and
    swaps them if they are in the wrong order. After each pass the
    largest unsorted element has "bubbled up" to its final position,
    so each pass can stop one element earlier than the last.

    Optimizations over naive bubble sort:
    - Early exit: if a full pass makes no swaps, the list is already
      sorted and we stop immediately. This gives the O(n) best case
      on already-sorted input.
    - Shrinking boundary: pass i ignores the last i elements, which
      are already in their final positions.

    Args:
        arr: List of mutually comparable elements

    Returns:
        A new list containing the elements of arr in ascending order

    Raises:
        TypeError: If arr is not a list, or its elements cannot be
            compared with one another (e.g. int vs str)

    Complexity:
        Time:  O(n) best (already sorted), O(n²) average and worst
        Space: O(n) for the returned copy, O(1) auxiliary
    """
    _validate_input(arr, "bubble_sort")
    result = arr.copy()
    n = len(result)

    for i in range(n - 1):
        swapped = False
        # Last i elements are already in place after i passes
        for j in range(n - 1 - i):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        # No swaps means the list is sorted - stop early
        if not swapped:
            break

    return result


def selection_sort(arr: List) -> List:
    """
    Sort a list using selection sort.

    Divides the list into a sorted prefix and an unsorted suffix.
    On each pass, finds the minimum element of the unsorted suffix
    and swaps it into place at the boundary, growing the sorted
    prefix by one.

    Selection sort always scans the entire unsorted suffix, so it
    performs O(n²) comparisons even on sorted input. Its advantage
    is that it makes at most n - 1 swaps, the fewest of the three
    basic sorts, which matters when writes are expensive.

    Args:
        arr: List of mutually comparable elements

    Returns:
        A new list containing the elements of arr in ascending order

    Raises:
        TypeError: If arr is not a list, or its elements cannot be
            compared with one another

    Complexity:
        Time:  O(n²) best, average, and worst (comparisons don't
               depend on input order)
        Space: O(n) for the returned copy, O(1) auxiliary
    """
    _validate_input(arr, "selection_sort")
    result = arr.copy()
    n = len(result)

    for i in range(n - 1):
        # Find the index of the minimum element in result[i:]
        min_index = i
        for j in range(i + 1, n):
            if result[j] < result[min_index]:
                min_index = j
        # Swap it into position i (skip self-swaps)
        if min_index != i:
            result[i], result[min_index] = result[min_index], result[i]

    return result


def insertion_sort(arr: List) -> List:
    """
    Sort a list using insertion sort.

    Builds the sorted list one element at a time: each new element is
    shifted left past every larger element in the sorted prefix until
    it reaches its correct position, the way most people sort a hand
    of playing cards.

    On nearly-sorted input each element is already close to its final
    position, so very little shifting occurs - this makes insertion
    sort the fastest of the three basic sorts in practice and the
    standard choice for small or nearly-sorted inputs.

    Args:
        arr: List of mutually comparable elements

    Returns:
        A new list containing the elements of arr in ascending order

    Raises:
        TypeError: If arr is not a list, or its elements cannot be
            compared with one another

    Complexity:
        Time:  O(n) best (already sorted), O(n²) average and worst
        Space: O(n) for the returned copy, O(1) auxiliary
    """
    _validate_input(arr, "insertion_sort")
    result = arr.copy()

    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        # Shift elements greater than key one slot to the right
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key

    return result
