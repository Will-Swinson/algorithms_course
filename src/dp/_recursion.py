"""Temporarily raise Python's recursion limit for deep top-down DP."""
import sys
from contextlib import contextmanager
from typing import Iterator

# Headroom for the caller's own frames (pytest, benchmark harness, ...)
_HEADROOM = 200


@contextmanager
def recursion_depth(depth: int) -> Iterator[None]:
    """
    Guarantee at least `depth` nested calls are allowed inside the
    block, restoring the previous limit afterwards. Never lowers it.

    Memoized Fibonacci recurses n deep and memoized LCS up to
    len(x) + len(y) deep — past CPython's default limit of 1000 for
    the input sizes this project benchmarks.
    """
    previous = sys.getrecursionlimit()
    needed = depth + _HEADROOM
    if needed > previous:
        sys.setrecursionlimit(needed)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)
