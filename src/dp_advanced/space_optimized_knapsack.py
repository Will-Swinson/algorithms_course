"""
0/1 Knapsack in O(W) space — the Week 5 table, compressed.

Week 5's tabulation fills an (n+1) × (W+1) table with the recurrence

    T[i][w] = max(T[i-1][w], v_i + T[i-1][w - w_i])     (if w_i <= w)
    T[i][w] = T[i-1][w]                                 (otherwise)

Row i reads ONLY row i-1. Every earlier row is dead weight once the
next one exists, so the table can shrink in two steps:

1. knapsack_two_row: keep just "previous" and "current" rows,
   swapping after each item. O(2W) space, obviously correct.
2. knapsack_1d: keep ONE row and overwrite it in place. O(W) space,
   correct only if the iteration order is right (proof below).

WHY THE CAPACITY LOOP MUST RUN FROM W DOWN TO w_i
-------------------------------------------------
Invariant: while processing item i in decreasing order of w, at the
moment cell dp[w] is updated,
    - dp[x] for x >  w already holds row i   (T[i][x]), and
    - dp[x] for x <= w still holds row i-1   (T[i-1][x]).
It holds before the loop (nothing rewritten yet), and the update at w
reads dp[w] and dp[w - w_i]. Since w_i >= 1, w - w_i < w, so both are
still row i-1 values — exactly the T[i-1][w] and T[i-1][w - w_i] the
recurrence demands. Writing dp[w] then extends the "row i" region down
by one, preserving the invariant. When the loop ends every dp[w] holds
T[i][w]; cells below w_i correctly keep T[i-1][w] because item i
cannot fit there. (Zero-weight items are handled separately: for them
the recurrence reads its own cell, so they are simply always taken.)

Iterating UPWARD breaks this: dp[w - w_i] may already include item i,
so item i can be taken again and again — that computes UNBOUNDED
knapsack instead. unbounded_knapsack() keeps that order on purpose,
and the tests use it to show the difference.

What O(W) gives up: the full table is what trace_solution() walks
backward to name the chosen items. knapsack_1d_with_items() recovers
them anyway by recording one bit per (item, capacity) decision in a
Python-int bitset — n×W BITS instead of n×W boxed ints.
"""
from typing import List, Optional, Sequence, Tuple

from src.dp.knapsack import _validate  # same input rules as Week 5
from src.utils.timer import CallStats


def knapsack_two_row(weights: Sequence[int], values: Sequence[int],
                     capacity: int,
                     stats: Optional[CallStats] = None) -> int:
    """
    Maximum value keeping only two rows of the DP table.

    Args:
        weights: Item weights (non-negative ints).
        values: Item values, same length as weights.
        capacity: Knapsack capacity W (non-negative int).
        stats: Optional counter; receives one count per cell computed.

    Returns:
        The maximum total value that fits.

    Raises:
        ValueError: On mismatched lengths or negative weight/capacity.

    Complexity: O(n × W) time, O(W) space (two rows of W + 1).
    """
    _validate(weights, values, capacity)
    previous = [0] * (capacity + 1)
    current = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for w in range(capacity + 1):
            if weight <= w:
                current[w] = max(previous[w], value + previous[w - weight])
            else:
                current[w] = previous[w]
        previous, current = current, previous
    if stats is not None:
        stats.count(len(weights) * (capacity + 1))
    return previous[capacity]


def knapsack_1d(weights: Sequence[int], values: Sequence[int],
                capacity: int,
                stats: Optional[CallStats] = None) -> int:
    """
    Maximum value with a single DP row updated in place.

    The capacity loop runs from W down to the item's weight, so each
    cell is overwritten only after every cell that still needs its
    old value has read it (see the module docstring for the proof).
    Cells below the item's weight are skipped entirely — they would
    just copy themselves.

    Args / Returns / Raises: as knapsack_two_row.

    Complexity: O(n × W) time, O(W) space.
    """
    _validate(weights, values, capacity)
    dp = [0] * (capacity + 1)
    cells = 0
    for weight, value in zip(weights, values):
        if weight == 0:
            # Free item: taking it never costs capacity, so take it
            # whenever it adds value
            if value > 0:
                for w in range(capacity + 1):
                    dp[w] += value
            cells += capacity + 1
            continue
        for w in range(capacity, weight - 1, -1):
            candidate = value + dp[w - weight]
            if candidate > dp[w]:
                dp[w] = candidate
        cells += max(capacity - weight + 1, 0)
    if stats is not None:
        stats.count(cells)
    return dp[capacity]


def knapsack_1d_with_items(weights: Sequence[int], values: Sequence[int],
                           capacity: int) -> Tuple[int, List[int]]:
    """
    O(W)-row knapsack that still recovers WHICH items are chosen.

    Alongside the single dp row, item i gets one Python int used as a
    bitset: bit w is 1 when taking item i improved dp[w]. Walking items
    backward from capacity W, a set bit means "item i was taken at this
    capacity" — the same question trace_solution() answers by
    comparing two table rows in Week 5.

    Returns:
        (maximum value, indices of chosen items in increasing order).

    Complexity: O(n × W) time; O(W) ints for the row plus n × W bits
    (about 1/64 the memory of a table of 8-byte references).
    """
    _validate(weights, values, capacity)
    dp = [0] * (capacity + 1)
    taken: List[int] = []
    for weight, value in zip(weights, values):
        # Mark decisions as ASCII '0'/'1' bytes, then pack the row into
        # one int in a single O(W) step. (Setting `bits |= 1 << w` per
        # cell copies the whole growing int each time: O(W²) per item.)
        flags = bytearray(b"0") * (capacity + 1)
        if weight == 0:
            if value > 0:
                for w in range(capacity + 1):
                    dp[w] += value
                flags = bytearray(b"1") * (capacity + 1)
        else:
            for w in range(capacity, weight - 1, -1):
                candidate = value + dp[w - weight]
                if candidate > dp[w]:
                    dp[w] = candidate
                    flags[w] = 49                  # ord("1")
        taken.append(int(flags[::-1], 2))          # bit w <- flags[w]

    chosen: List[int] = []
    w = capacity
    for i in range(len(weights) - 1, -1, -1):
        if (taken[i] >> w) & 1:
            chosen.append(i)
            w -= weights[i]
    chosen.reverse()
    return dp[capacity], chosen


def unbounded_knapsack(weights: Sequence[int], values: Sequence[int],
                       capacity: int) -> int:
    """
    The SAME single-row update as knapsack_1d, but iterating capacity
    UPWARD. dp[w - w_i] may then already include item i, so items can
    be reused without limit: this solves unbounded knapsack, not 0/1.
    Kept as the counterexample that demonstrates why knapsack_1d's
    loop direction matters.

    Raises:
        ValueError: As knapsack_two_row, or if any weight is 0 with a
            positive value (unbounded value).
    """
    _validate(weights, values, capacity)
    for weight, value in zip(weights, values):
        if weight == 0 and value > 0:
            raise ValueError("a zero-weight item with positive value "
                             "makes unbounded knapsack infinite")
    dp = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        if weight == 0:
            continue
        for w in range(weight, capacity + 1):
            candidate = value + dp[w - weight]
            if candidate > dp[w]:
                dp[w] = candidate
    return dp[capacity]
