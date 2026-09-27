"""
Fibonacci three ways — the smallest problem that shows why DP exists.

    F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)

1. Naive recursion transcribes the recurrence directly. F(n-1) and
   F(n-2) both recompute F(n-3), F(n-4), ... from scratch — the
   subproblems OVERLAP — so the call tree has 2·F(n+1) - 1 nodes:
   O(φⁿ) ≈ O(1.618ⁿ), bounded above by O(2ⁿ).
2. Top-down DP (memoization) keeps the same recursion but caches each
   F(k) the first time it is computed. Every k in 0..n is solved once;
   the second request for it is a dictionary hit. O(n) time, O(n)
   space (the cache plus an n-deep call stack).
3. Bottom-up DP (tabulation) drops the recursion: fill table[0..n] in
   increasing order so each entry's two dependencies already exist.
   O(n) time, O(n) space, no recursion at all.

Every function accepts an optional CallStats. Recursive versions
report each invocation (and the deepest nesting); tabulation reports
one "call" per table cell filled and a recursion depth of 0.
"""
from typing import Dict, Optional

from src.dp._recursion import recursion_depth
from src.utils.timer import CallStats


def _validate(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")


def fib_naive(n: int, stats: Optional[CallStats] = None) -> int:
    """
    F(n) by direct recursion, recomputing overlapping subproblems.

    Args:
        n: Index into the sequence (n >= 0).
        stats: Optional counter for calls and recursion depth.

    Returns:
        The n-th Fibonacci number.

    Raises:
        TypeError: If n is not an int.
        ValueError: If n is negative.

    Complexity: O(2ⁿ) time (precisely 2·F(n+1) - 1 calls),
    O(n) space for the call stack.
    """
    _validate(n)
    if stats is None:
        return _fib_naive(n)
    return _fib_naive_counted(n, stats)


def _fib_naive(n: int) -> int:
    if n < 2:
        return n
    return _fib_naive(n - 1) + _fib_naive(n - 2)


def _fib_naive_counted(n: int, stats: CallStats) -> int:
    # Separate from _fib_naive so the timed path pays no counting cost
    stats.enter()
    if n < 2:
        result = n
    else:
        result = (_fib_naive_counted(n - 1, stats)
                  + _fib_naive_counted(n - 2, stats))
    stats.exit()
    return result


def fib_memo(n: int, stats: Optional[CallStats] = None) -> int:
    """
    F(n) top-down: the naive recursion plus a cache of solved F(k).

    Only the leftmost path of the call tree does real work; every
    right-hand call F(k-2) finds its answer already cached. That
    collapses 2·F(n+1) - 1 calls to 2n - 1.

    Args:
        n: Index into the sequence (n >= 0).
        stats: Optional counter for calls and recursion depth.

    Returns:
        The n-th Fibonacci number.

    Raises:
        TypeError: If n is not an int.
        ValueError: If n is negative.

    Complexity: O(n) time, O(n) space (cache + n-deep call stack).
    """
    _validate(n)
    memo: Dict[int, int] = {0: 0, 1: 1}

    def solve(k: int) -> int:
        if stats is not None:
            stats.enter()
        if k not in memo:
            memo[k] = solve(k - 1) + solve(k - 2)
        if stats is not None:
            stats.exit()
        return memo[k]

    with recursion_depth(n):
        return solve(n)


def fib_tabulation(n: int, stats: Optional[CallStats] = None) -> int:
    """
    F(n) bottom-up: fill table[k] for k = 0..n in increasing order.

    Args:
        n: Index into the sequence (n >= 0).
        stats: Optional counter; receives one count per table cell.

    Returns:
        The n-th Fibonacci number.

    Raises:
        TypeError: If n is not an int.
        ValueError: If n is negative.

    Complexity: O(n) time, O(n) space, no recursion.
    """
    _validate(n)
    table = [0] * (n + 1)
    if n >= 1:
        table[1] = 1
    for k in range(2, n + 1):
        table[k] = table[k - 1] + table[k - 2]
    if stats is not None:
        stats.count(n + 1)
    return table[n]


def naive_call_count(n: int) -> int:
    """
    Exact number of calls fib_naive(n) makes: C(n) = 1 + C(n-1) + C(n-2)
    with C(0) = C(1) = 1, which solves to 2·F(n+1) - 1.

    Lets the benchmark report call counts for n = 40, 45 without
    running billions of instrumented calls. The test suite checks it
    against a real CallStats count.
    """
    _validate(n)
    return 2 * fib_tabulation(n + 1) - 1
