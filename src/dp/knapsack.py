"""
0/1 Knapsack — choose items (each at most once) to maximize total
value without exceeding the knapsack's weight capacity W.

The decision for item i given remaining capacity w:

    K(i, w) = K(i+1, w)                                  skip item i
            | max(K(i+1, w), values[i] + K(i+1, w - weights[i]))
                                                  if item i fits
    K(n, w) = 0                                  no items left

OPTIMAL SUBSTRUCTURE: once item i is taken or skipped, the best use of
the remaining capacity on items i+1.. is itself a knapsack problem,
and its optimal answer is part of the overall optimum.
OVERLAPPING SUBPROBLEMS: different take/skip histories reach the same
(i, w) state — e.g. taking a weight-3 item then skipping a weight-5
item vs. the reverse — and naive recursion re-solves it each time.

1. Naive recursion explores the full take/skip tree: O(2ⁿ).
2. Top-down (memoization) caches K(i, w) in a dict. It only ever
   solves states actually reachable from (0, W), which can be far
   fewer than all n × (W+1) of them. O(n × W) worst case.
3. Bottom-up (tabulation) fills the full (n+1) × (W+1) table, where
   table[i][w] = best value using the FIRST i items with capacity w.
   O(n × W) time and space; trace_solution() walks the table backward
   to recover which items were chosen.

O(n × W) is PSEUDO-polynomial: polynomial in W's value, but W's value
is exponential in the number of bits used to write it down.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from src.dp._recursion import recursion_depth
from src.utils.timer import CallStats


def _validate(weights: Sequence[int], values: Sequence[int],
              capacity: int) -> None:
    if len(weights) != len(values):
        raise ValueError(f"weights and values differ in length "
                         f"({len(weights)} vs {len(values)})")
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")
    for w in weights:
        if w < 0:
            raise ValueError(f"weights must be non-negative, got {w}")


def knapsack_recursive(weights: Sequence[int], values: Sequence[int],
                       capacity: int,
                       stats: Optional[CallStats] = None) -> int:
    """
    Maximum value by trying every take/skip combination.

    Args:
        weights: Item weights (non-negative ints).
        values: Item values, same length as weights.
        capacity: Knapsack capacity W (non-negative int).
        stats: Optional counter for calls and recursion depth.

    Returns:
        The maximum total value that fits.

    Raises:
        ValueError: On mismatched lengths or negative weight/capacity.

    Complexity: O(2ⁿ) time, O(n) stack space.
    """
    _validate(weights, values, capacity)
    n = len(weights)

    def solve(i: int, remaining: int) -> int:
        if stats is not None:
            stats.enter()
        if i == n:
            best = 0
        else:
            best = solve(i + 1, remaining)                    # skip
            if weights[i] <= remaining:                       # take
                best = max(best, values[i]
                           + solve(i + 1, remaining - weights[i]))
        if stats is not None:
            stats.exit()
        return best

    with recursion_depth(n + 1):
        return solve(0, capacity)


def knapsack_memo(weights: Sequence[int], values: Sequence[int],
                  capacity: int,
                  stats: Optional[CallStats] = None) -> int:
    """
    Maximum value top-down: the naive recursion with each (i, w)
    answer cached the first time it is solved.

    A dict (not a full table) holds the cache, so memory tracks the
    number of states actually visited rather than n × (W+1).

    Args / Returns / Raises: as knapsack_recursive.

    Complexity: O(n × W) time and space in the worst case.
    """
    _validate(weights, values, capacity)
    n = len(weights)
    memo: Dict[Tuple[int, int], int] = {}

    def solve(i: int, remaining: int) -> int:
        if stats is not None:
            stats.enter()
        key = (i, remaining)
        if key not in memo:
            if i == n:
                best = 0
            else:
                best = solve(i + 1, remaining)
                if weights[i] <= remaining:
                    best = max(best, values[i]
                               + solve(i + 1, remaining - weights[i]))
            memo[key] = best
        if stats is not None:
            stats.exit()
        return memo[key]

    with recursion_depth(n + 1):
        return solve(0, capacity)


def build_knapsack_table(weights: Sequence[int], values: Sequence[int],
                         capacity: int,
                         stats: Optional[CallStats] = None
                         ) -> List[List[int]]:
    """
    Fill the bottom-up DP table.

    table[i][w] = best value achievable with the first i items and
    capacity w. Row 0 (no items) is all zeros; each later row depends
    only on the row above it.

    Args:
        weights, values, capacity: as knapsack_recursive.
        stats: Optional counter; receives one count per cell filled.

    Returns:
        The (n+1) × (capacity+1) table.

    Complexity: O(n × W) time and space.
    """
    _validate(weights, values, capacity)
    n = len(weights)
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        weight, value = weights[i - 1], values[i - 1]
        above = table[i - 1]
        row = table[i]
        for w in range(capacity + 1):
            if weight <= w:
                row[w] = max(above[w], value + above[w - weight])
            else:
                row[w] = above[w]
    if stats is not None:
        stats.count(n * (capacity + 1))
    return table


def knapsack_tabulation(weights: Sequence[int], values: Sequence[int],
                        capacity: int,
                        stats: Optional[CallStats] = None) -> int:
    """
    Maximum value bottom-up: the bottom-right cell of the DP table.

    Args / Returns / Raises: as knapsack_recursive.

    Complexity: O(n × W) time and space.
    """
    return build_knapsack_table(weights, values, capacity, stats)[-1][-1]


def trace_solution(table: List[List[int]], weights: Sequence[int],
                   capacity: int) -> List[int]:
    """
    Recover WHICH items produce the optimum from a filled table.

    Walk from table[n][W] upward: if table[i][w] differs from
    table[i-1][w], the first i-1 items alone could not reach this
    value, so item i-1 must have been taken — record it and spend its
    weight. Otherwise item i-1 was skipped.

    Args:
        table: Output of build_knapsack_table().
        weights: The same weights the table was built from.
        capacity: The same capacity the table was built from.

    Returns:
        Indices of the chosen items, in increasing order.

    Complexity: O(n).
    """
    chosen: List[int] = []
    w = capacity
    for i in range(len(table) - 1, 0, -1):
        if table[i][w] != table[i - 1][w]:
            chosen.append(i - 1)
            w -= weights[i - 1]
    chosen.reverse()
    return chosen


def knapsack_with_items(weights: Sequence[int], values: Sequence[int],
                        capacity: int) -> Tuple[int, List[int]]:
    """
    Convenience wrapper: (maximum value, indices of chosen items).
    """
    table = build_knapsack_table(weights, values, capacity)
    return table[-1][-1], trace_solution(table, weights, capacity)
