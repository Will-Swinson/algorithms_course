"""
Measurement helpers for the Week 5 dynamic-programming benchmark.

Three things get measured, each in its own run so the instrumentation
for one never pollutes another:

- CallStats   — recursive calls and maximum recursion depth. The DP
                functions accept an optional `stats` argument and
                report into it; tabulation has no recursion, so it
                reports table cells filled (via count()) and leaves
                max_depth at 0.
- time_call   — wall-clock seconds via time.perf_counter(), re-running
                fast functions and averaging to beat timer resolution.
- peak_memory — peak bytes allocated during one call (tracemalloc).
                Note that tracemalloc sees heap objects (memo dicts,
                tables) but not interpreter stack frames, which is why
                recursion depth is tracked separately.
"""
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple


@dataclass
class CallStats:
    """
    Counter for recursive calls and recursion depth.

    Recursive code calls enter() at the top of each invocation and
    exit() before returning; iterative code calls count() once per
    subproblem it evaluates.
    """
    calls: int = 0
    max_depth: int = 0
    _depth: int = field(default=0, repr=False)

    def enter(self) -> None:
        self.calls += 1
        self._depth += 1
        if self._depth > self.max_depth:
            self.max_depth = self._depth

    def exit(self) -> None:
        self._depth -= 1

    def count(self, n: int = 1) -> None:
        """Record n subproblem evaluations with no recursion."""
        self.calls += n


class Timer:
    """
    Context manager around time.perf_counter().

        with Timer() as t:
            work()
        print(t.elapsed)
    """

    def __enter__(self) -> "Timer":
        self.elapsed = 0.0
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def time_call(func: Callable[[], Any], min_time: float = 0.05,
              max_runs: int = 200) -> float:
    """
    Seconds for one call of func().

    A call that finishes in under `min_time` is re-run (at least 3,
    at most `max_runs` times) and the average returned, so microsecond
    results aren't dominated by timer noise. Slow calls run once.
    """
    with Timer() as t:
        func()
    if t.elapsed >= min_time:
        return t.elapsed

    runs = min(max(3, int(min_time / max(t.elapsed, 1e-9))), max_runs)
    with Timer() as t:
        for _ in range(runs):
            func()
    return t.elapsed / runs


def peak_memory(func: Callable[[], Any]) -> Tuple[int, Any]:
    """
    Run func() once under tracemalloc.

    Returns:
        (peak bytes allocated during the call, func's return value)
    """
    tracemalloc.start()
    try:
        result = func()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, result


def count_calls(func: Callable[..., Any], *args) -> CallStats:
    """Run func(*args, stats=CallStats()) and return the filled stats."""
    stats = CallStats()
    func(*args, stats=stats)
    return stats
