"""
Randomized QuickSort — a Las Vegas algorithm.

Las Vegas: the output is ALWAYS correct; only the running time is
random. The randomness lives entirely in the pivot choice. Picking the
pivot uniformly at random from the current subarray means no input
order is bad for it: sorted, reverse, or adversarially constructed
inputs all get the same expected cost,

    E[comparisons] = 2(n + 1)·H_n − 4n  ≈  1.386·n·log₂n

and the probability of a run noticeably above that falls off
exponentially. Contrast deterministic_quicksort below, which always
pivots on the FIRST element: on sorted input every partition peels off
a single element, giving n(n − 1)/2 comparisons — Θ(n²) every time,
and anyone who knows the rule can construct that input.

Both versions share one three-way partition, so the ONLY difference
between them is how the pivot index is picked. Three-way partitioning
keeps runs of equal keys from triggering the quadratic case. The
partition is Lomuto-style (one left-to-right scan) rather than Week 2's
Dutch national flag: the flag version swaps elements in from the right
end, which reverses and scrambles sorted input, so first-element
pivoting would quietly escape its own worst case. The Lomuto scan
leaves the greater-than elements in their original order, so sorted
input stays sorted and the Θ(n²) behavior shows up as the theory says.

Unlike Week 2's quick_sort there is no insertion-sort cutoff: with
every subarray partitioned, the comparison count is exactly the
quantity the expected-value formula describes.

The recursion is replaced by an explicit stack of (low, high, depth)
ranges. Deterministic QuickSort on sorted input recurses n levels
deep, which would overflow CPython's ~1,000-frame limit at n = 1,000;
the explicit stack lets the benchmark measure the Θ(n²) behavior
instead of crashing. `max_depth` still reports the depth the
recursive formulation would reach.

Complexity:
    randomized_quicksort      expected O(n log n) for every input,
                              worst O(n²) with vanishing probability
    deterministic_quicksort   O(n log n) on random input, Θ(n²) on
                              sorted / reverse input
    Space: O(n) for the copy; stack O(log n) expected (randomized)
"""
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.sorting.basic_sorts import _validate_input

# (rng, low, high) -> pivot index in [low, high]
PivotRule = Callable[[random.Random, int, int], int]


@dataclass
class PartitionStats:
    """
    Work counters filled in by the sort and select functions.

    comparisons  element-vs-pivot classifications (one per non-pivot
                 element per partition, i.e. m − 1 for a range of m)
    partitions   partition passes performed
    max_depth    deepest level the recursive formulation would reach
    """
    comparisons: int = 0
    partitions: int = 0
    max_depth: int = 0


def _random_pivot(rng: random.Random, low: int, high: int) -> int:
    # Same distribution as rng.randint(low, high), ~3× cheaper
    return low + int(rng.random() * (high - low + 1))


def _first_pivot(rng: random.Random, low: int, high: int) -> int:
    return low


def partition_three_way(arr: List, low: int, high: int, pivot_index: int,
                        stats: Optional[PartitionStats] = None
                        ) -> Tuple[int, int]:
    """
    Partition arr[low..high] in place around arr[pivot_index].

    The pivot is parked at arr[low], then one left-to-right scan grows
    three regions behind the scan position i:

        arr[low+1 : lt+1]   < pivot
        arr[lt+1 : eq+1]    == pivot
        arr[eq+1 : i]       > pivot
        arr[i : high+1]     not yet examined

    A smaller element is rotated to the end of the < region (two
    swaps), an equal one to the end of the == region (one swap), and a
    greater one stays where it is. Finally the pivot is swapped from
    arr[low] to the boundary at lt.

    Each of the m − 1 non-pivot elements is classified exactly once,
    which is what makes the comparison count match the m − 1 assumed
    by the expected-value formulas.

    Returns:
        (lo, hi) such that arr[lo..hi] holds every element equal to the
        pivot, already in final sorted position.
    """
    arr[low], arr[pivot_index] = arr[pivot_index], arr[low]
    pivot = arr[low]
    lt = eq = low
    for i in range(low + 1, high + 1):
        value = arr[i]
        if value < pivot:
            eq += 1
            arr[i] = arr[eq]            # first > element moves to the end
            lt += 1
            arr[eq] = arr[lt]           # first == element moves to the end
            arr[lt] = value
        elif value == pivot:
            eq += 1
            arr[i] = arr[eq]
            arr[eq] = value
    arr[low], arr[lt] = arr[lt], arr[low]
    if stats is not None:
        stats.comparisons += high - low
        stats.partitions += 1
    return lt, eq


def _quicksort(arr: List, pivot_rule: PivotRule, name: str,
               stats: Optional[PartitionStats],
               rng: Optional[random.Random]) -> List:
    _validate_input(arr, name)
    result = arr.copy()
    rng = rng or random                # module-level generator
    stack = [(0, len(result) - 1, 1)]
    while stack:
        low, high, depth = stack.pop()
        if low >= high:
            continue
        if stats is not None and depth > stats.max_depth:
            stats.max_depth = depth
        lt, gt = partition_three_way(result, low, high,
                                     pivot_rule(rng, low, high), stats)
        stack.append((low, lt - 1, depth + 1))
        stack.append((gt + 1, high, depth + 1))
    return result


def randomized_quicksort(arr: List, stats: Optional[PartitionStats] = None,
                         rng: Optional[random.Random] = None) -> List:
    """
    Sort with a uniformly random pivot at every partition.

    Args:
        arr: List of mutually comparable elements. Not modified.
        stats: Optional counters (comparisons, partitions, max_depth).
        rng: Optional random.Random for reproducible pivot choices;
            defaults to the module-level generator.

    Returns:
        A new list with the elements of arr in ascending order.

    Raises:
        TypeError: If arr is not a list or its elements don't compare.

    Complexity:
        Expected O(n log n) comparisons on EVERY input, exactly
        2(n+1)H_n − 4n in expectation for distinct keys.
    """
    return _quicksort(arr, _random_pivot, "randomized_quicksort", stats, rng)


def deterministic_quicksort(arr: List,
                            stats: Optional[PartitionStats] = None,
                            rng: Optional[random.Random] = None) -> List:
    """
    Sort with the FIRST element of each range as pivot (the classic
    textbook rule). Same partition and bookkeeping as
    randomized_quicksort; `rng` is accepted for a uniform signature but
    unused.

    Complexity:
        O(n log n) on random input; Θ(n²) comparisons — exactly
        n(n−1)/2 — on already-sorted or reverse-sorted input.
    """
    return _quicksort(arr, _first_pivot, "deterministic_quicksort", stats,
                      rng)
