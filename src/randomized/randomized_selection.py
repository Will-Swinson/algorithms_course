"""
Randomized Selection (QuickSelect) — find the k-th smallest element.

QuickSelect is QuickSort that only recurses into ONE side: partition
around a random pivot, then the k-th smallest is either among the
smaller elements, among the pivot's equals (done), or among the larger
ones. Throwing away the other side each round turns the n log n of
sorting into a geometric series:

    n + (3/4)n + (3/4)²n + ...  =  O(n)   expected

(a random pivot lands in the middle half with probability 1/2, and
then shrinks the range to at most 3/4 of its size). The exact expected
comparison count is in probability_analysis: ~2n for the min or max,
~3.39n for the median. Like QuickSort it is Las Vegas — always right,
randomly fast. Its worst case is still Θ(n²) if every pivot is the
extreme element, which happens with probability ~ 2^n / n!.

median_of_medians_select is the deterministic counterpart (Blum,
Floyd, Pratt, Rivest, Tarjan 1973). It guarantees a good pivot by
taking the median of the medians of groups of five, which is always
larger than ~30% of the elements and smaller than ~30%. That buys a
WORST-case O(n) bound, at the cost of a much larger constant (it
recursively selects within the medians at every round).

sort_select is the trivial O(n log n) baseline: sorted(arr)[k − 1].

All three use 1-indexed k: k = 1 is the minimum, k = len(arr) the
maximum.

Recursion depth: QuickSelect is tail-recursive (one recursive call,
at the end), so it is written as a loop; `max_depth` counts the loop
rounds, which equals the depth the recursive version would reach.
Expected depth is O(log n).
"""
import random
from typing import List, Optional

from src.randomized.randomized_quicksort import (
    PartitionStats, _random_pivot, partition_three_way,
)
from src.sorting.basic_sorts import _validate_input

GROUP_SIZE = 5


def _validate_k(arr: List, k: int, name: str) -> None:
    _validate_input(arr, name)
    if not arr:
        raise ValueError(f"{name}: cannot select from an empty list")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"{name}: k must be an int, got {type(k).__name__}")
    if not 1 <= k <= len(arr):
        raise ValueError(f"{name}: k must be in [1, {len(arr)}], got {k}")


def randomized_select(arr: List, k: int,
                      stats: Optional[PartitionStats] = None,
                      rng: Optional[random.Random] = None):
    """
    The k-th smallest element of arr (1-indexed), by randomized
    QuickSelect.

    Args:
        arr: List of mutually comparable elements. Not modified.
        k: Rank to find, 1 <= k <= len(arr).
        stats: Optional counters (comparisons, partitions, max_depth).
        rng: Optional random.Random for reproducible pivots.

    Raises:
        TypeError: If arr is not a list or k is not an int.
        ValueError: If arr is empty or k is out of range.

    Complexity:
        Expected O(n) time (≤ ~3.39n comparisons for any k);
        O(n) space for the working copy.
    """
    _validate_k(arr, k, "randomized_select")
    work = arr.copy()
    rng = rng or random
    low, high, target = 0, len(work) - 1, k - 1
    depth = 0
    while True:
        depth += 1
        if stats is not None and depth > stats.max_depth:
            stats.max_depth = depth
        if low == high:
            return work[low]
        lt, gt = partition_three_way(work, low, high,
                                     _random_pivot(rng, low, high), stats)
        if target < lt:
            high = lt - 1
        elif target > gt:
            low = gt + 1
        else:
            return work[target]


def median_of_medians_select(arr: List, k: int,
                             stats: Optional[PartitionStats] = None):
    """
    The k-th smallest element of arr (1-indexed), deterministically, in
    worst-case linear time.

    Each round: split the range into groups of 5, find each group's
    median by insertion sort, recursively select the median of those
    medians, and partition around it. The pivot is guaranteed to have
    at least ~3n/10 elements on each side, so the recurrence is

        T(n) <= T(n/5) + T(7n/10) + O(n)  =  O(n)

    since 1/5 + 7/10 < 1.

    Complexity:
        Worst-case O(n) time, but with a constant several times larger
        than QuickSelect's expected one. O(n) space.
    """
    _validate_k(arr, k, "median_of_medians_select")
    return _mom_select(arr.copy(), 0, len(arr) - 1, k - 1, stats, 1)


def _mom_select(work: List, low: int, high: int, target: int,
                stats: Optional[PartitionStats], depth: int):
    """Select absolute index `target` within work[low..high]."""
    while True:
        if stats is not None and depth > stats.max_depth:
            stats.max_depth = depth
        if high - low < GROUP_SIZE:
            _insertion_sort(work, low, high, stats)
            return work[target]

        # Move each group's median to the front of the range
        num_medians = 0
        for start in range(low, high + 1, GROUP_SIZE):
            end = min(start + GROUP_SIZE - 1, high)
            _insertion_sort(work, start, end, stats)
            mid = (start + end) // 2
            dest = low + num_medians
            work[dest], work[mid] = work[mid], work[dest]
            num_medians += 1

        # Recursively find the median of the medians (it ends up at
        # index mom_index after that call has rearranged the front)
        mom_index = low + (num_medians - 1) // 2
        _mom_select(work, low, low + num_medians - 1, mom_index, stats,
                    depth + 1)

        lt, gt = partition_three_way(work, low, high, mom_index, stats)
        if target < lt:
            high = lt - 1
        elif target > gt:
            low = gt + 1
        else:
            return work[target]
        depth += 1


def _insertion_sort(work: List, low: int, high: int,
                    stats: Optional[PartitionStats]) -> None:
    """Insertion-sort work[low..high] in place, counting comparisons."""
    comparisons = 0
    for i in range(low + 1, high + 1):
        key = work[i]
        j = i - 1
        while j >= low:
            comparisons += 1
            if work[j] <= key:
                break
            work[j + 1] = work[j]
            j -= 1
        work[j + 1] = key
    if stats is not None:
        stats.comparisons += comparisons


def sort_select(arr: List, k: int, stats: Optional[PartitionStats] = None):
    """
    Baseline: the k-th smallest via a full sort (Python's Timsort).
    O(n log n). `stats` is accepted for a uniform signature but not
    filled — Timsort's comparisons happen in C.
    """
    _validate_k(arr, k, "sort_select")
    return sorted(arr)[k - 1]
