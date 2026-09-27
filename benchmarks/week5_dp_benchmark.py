"""
Week 5 benchmark: naive recursion vs dynamic programming.

For each problem, every method is measured three separate ways so no
instrumentation leaks into another measurement:
  - time:      clean run, averaged over repeats when under 0.5 s
  - calls:     run with a CallStats counter — recursive invocations
               (and max recursion depth), or table cells for
               tabulation
  - memory:    peak heap bytes via tracemalloc

Experiments (results land in benchmarks/results/):

1. Fibonacci, n = 10, 20, 30, 35, 40, 45 — naive vs memo vs tabulation.
   Naive call counts past n = 35 come from the exact formula
   2·F(n+1) - 1 instead of billions of counted calls (calls_source
   column says which). -> fibonacci_comparison.png
2. 0/1 Knapsack — recursive on n <= 22 items, DP up to n = 500, plus
   a capacity sweep at fixed n showing O(n × W) growth in W.
   -> knapsack_performance.png
3. LCS on random DNA strings — recursive on lengths <= 14, DP on
   lengths 10 .. 1,000. -> lcs_performance.png

Raw measurements: dp_vs_recursive_table.csv

Usage:
    python benchmarks/week5_dp_benchmark.py          # full run (~4 min)
    python benchmarks/week5_dp_benchmark.py --quick  # naive fib <= 35
"""
import argparse
import csv
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dp import (
    fib_naive, fib_memo, fib_tabulation, naive_call_count,
    knapsack_recursive, knapsack_memo, knapsack_tabulation,
    lcs_recursive, lcs_memo, lcs_tabulation,
)
from src.utils.timer import count_calls, peak_memory, time_call
from src.utils.visualization import plot_dp_comparison

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_NAME = "dp_vs_recursive_table.csv"
SEED = 42
# Calls under this are repeated (>= 3 runs) and averaged; a single run
# of a 50-150 ms call proved too noisy (GC pauses, frequency scaling)
MIN_TIME = 0.5

FIELDS = ["problem", "method", "input_size", "parameter", "time_seconds",
          "calls", "calls_source", "max_depth", "peak_memory_bytes",
          "speedup_vs_recursive"]

FIB_SIZES = [10, 20, 30, 35, 40, 45]

KNAPSACK_RECURSIVE_SIZES = [4, 8, 12, 16, 20, 22]
KNAPSACK_DP_SIZES = KNAPSACK_RECURSIVE_SIZES + [50, 100, 200, 500]
KNAPSACK_MEMO_LIMIT = 200      # dict of (i, w) states grows too large past this
KNAPSACK_SWEEP_ITEMS = 100
KNAPSACK_SWEEP_CAPACITIES = [100, 250, 500, 1_000, 2_500, 5_000]

LCS_RECURSIVE_LENGTHS = [4, 6, 8, 10, 12, 14]
LCS_DP_LENGTHS = LCS_RECURSIVE_LENGTHS + [50, 100, 250, 500, 1_000]


def make_row(problem: str, method: str, size: int, parameter: str,
             seconds: float, calls: int, calls_source: str,
             max_depth: int, memory: Optional[int]) -> Dict:
    row = {
        "problem": problem,
        "method": method,
        "input_size": size,
        "parameter": parameter,
        "time_seconds": seconds,
        "calls": calls,
        "calls_source": calls_source,
        "max_depth": max_depth,
        "peak_memory_bytes": memory,
        "speedup_vs_recursive": None,
    }
    mem = f"{memory / 1024:10,.1f} KiB" if memory is not None else " " * 14
    print(f"  {problem:9s} {method:12s} n={size:<6d} {parameter:14s} "
          f"{seconds:12.6f}s  calls={calls:<14,d} depth={max_depth:<5d} "
          f"{mem}", flush=True)
    return row


def measure(problem: str, method: str, size: int, parameter: str,
            func: Callable, args: Tuple, count: bool = True,
            memory: bool = True) -> Dict:
    """Time, count, and memory-profile func(*args) in separate runs."""
    seconds = time_call(lambda: func(*args), min_time=MIN_TIME)
    calls, depth, source = 0, 0, "not_measured"
    if count:
        stats = count_calls(func, *args)
        calls, depth, source = stats.calls, stats.max_depth, "measured"
    peak = peak_memory(lambda: func(*args))[0] if memory else None
    return make_row(problem, method, size, parameter, seconds, calls,
                    source, depth, peak)


def add_speedups(rows: List[Dict]) -> None:
    """Fill speedup_vs_recursive where a recursive time exists."""
    baseline = {(r["problem"], r["input_size"], r["parameter"]):
                r["time_seconds"] for r in rows
                if r["method"] == "recursive"}
    for r in rows:
        key = (r["problem"], r["input_size"], r["parameter"])
        if r["method"] != "recursive" and key in baseline:
            r["speedup_vs_recursive"] = baseline[key] / r["time_seconds"]


# ---------------------------------------------------------------------------
# 1. Fibonacci
# ---------------------------------------------------------------------------

def bench_fibonacci(sizes: Sequence[int] = FIB_SIZES, count_limit: int = 35,
                    memory_limit: int = 30,
                    naive_limit: int = 45) -> List[Dict]:
    """
    Args:
        sizes: Values of n to test.
        count_limit: Largest n whose naive calls are actually counted;
            above it the exact formula 2·F(n+1) - 1 is used.
        memory_limit: Largest n to run naive under tracemalloc (which
            slows every allocation).
        naive_limit: Largest n to run naive at all (--quick lowers it).
    """
    print("\n=== Fibonacci: naive vs memoization vs tabulation ===")
    rows = []
    for n in sizes:
        if n <= naive_limit:
            if n <= count_limit:
                rows.append(measure("fibonacci", "recursive", n, "",
                                    fib_naive, (n,),
                                    memory=n <= memory_limit))
            else:
                seconds = time_call(lambda: fib_naive(n),
                                    min_time=MIN_TIME)
                rows.append(make_row("fibonacci", "recursive", n, "",
                                     seconds, naive_call_count(n),
                                     "formula", n, None))
        rows.append(measure("fibonacci", "memoization", n, "",
                            fib_memo, (n,)))
        rows.append(measure("fibonacci", "tabulation", n, "",
                            fib_tabulation, (n,)))
    return rows


# ---------------------------------------------------------------------------
# 2. 0/1 Knapsack
# ---------------------------------------------------------------------------

def make_items(n: int, seed: int = SEED) -> Tuple[List[int], List[int], int]:
    """
    n seeded random items: weights 1-30, values 1-100, capacity half
    the total weight — so roughly half the items fit and the take/skip
    tree branches as much as possible.
    """
    rng = random.Random(seed + n)
    weights = [rng.randint(1, 30) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    return weights, values, sum(weights) // 2


def bench_knapsack(recursive_sizes: Sequence[int] = KNAPSACK_RECURSIVE_SIZES,
                   dp_sizes: Sequence[int] = KNAPSACK_DP_SIZES,
                   memo_limit: int = KNAPSACK_MEMO_LIMIT) -> List[Dict]:
    print("\n=== 0/1 Knapsack: recursive vs memoization vs tabulation ===")
    rows = []
    for n in sorted(set(recursive_sizes) | set(dp_sizes)):
        weights, values, capacity = make_items(n)
        args = (weights, values, capacity)
        param = f"W={capacity}"
        if n in recursive_sizes:
            rows.append(measure("knapsack", "recursive", n, param,
                                knapsack_recursive, args))
        if n in dp_sizes:
            if n <= memo_limit:
                rows.append(measure("knapsack", "memoization", n, param,
                                    knapsack_memo, args))
            rows.append(measure("knapsack", "tabulation", n, param,
                                knapsack_tabulation, args))
    return rows


def bench_knapsack_capacity(
        n: int = KNAPSACK_SWEEP_ITEMS,
        capacities: Sequence[int] = KNAPSACK_SWEEP_CAPACITIES) -> List[Dict]:
    """Fixed item count, growing W: exposes the pseudo-polynomial W term."""
    print(f"\n=== 0/1 Knapsack: capacity sweep at n = {n} ===")
    weights, values, _ = make_items(n)
    rows = []
    for capacity in capacities:
        args = (weights, values, capacity)
        param = f"W={capacity}"
        rows.append(measure("knapsack_capacity", "memoization", n, param,
                            knapsack_memo, args, memory=False))
        rows.append(measure("knapsack_capacity", "tabulation", n, param,
                            knapsack_tabulation, args, memory=False))
    return rows


# ---------------------------------------------------------------------------
# 3. Longest Common Subsequence
# ---------------------------------------------------------------------------

def make_strings(length: int, seed: int = SEED) -> Tuple[str, str]:
    """Two independent random DNA strings (alphabet ACGT)."""
    rng = random.Random(seed + length)
    return ("".join(rng.choice("ACGT") for _ in range(length)),
            "".join(rng.choice("ACGT") for _ in range(length)))


def bench_lcs(recursive_lengths: Sequence[int] = LCS_RECURSIVE_LENGTHS,
              dp_lengths: Sequence[int] = LCS_DP_LENGTHS) -> List[Dict]:
    print("\n=== LCS: recursive vs memoization vs tabulation ===")
    rows = []
    for length in sorted(set(recursive_lengths) | set(dp_lengths)):
        x, y = make_strings(length)
        param = f"{length}x{length}"
        if length in recursive_lengths:
            rows.append(measure("lcs", "recursive", length, param,
                                lcs_recursive, (x, y)))
        if length in dp_lengths:
            rows.append(measure("lcs", "memoization", length, param,
                                lcs_memo, (x, y)))
            rows.append(measure("lcs", "tabulation", length, param,
                                lcs_tabulation, (x, y)))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def series_for(rows: List[Dict], problem: str) -> Dict[str, Dict[str, list]]:
    """Reshape rows into plot_dp_comparison()'s {method: {...}} input."""
    series: Dict[str, Dict[str, list]] = {}
    for method in ["recursive", "memoization", "tabulation"]:
        points = sorted((r for r in rows if r["problem"] == problem
                         and r["method"] == method),
                        key=lambda r: r["input_size"])
        if points:
            series[method] = {
                "sizes": [r["input_size"] for r in points],
                "time": [r["time_seconds"] for r in points],
                "calls": [r["calls"] for r in points],
                "memory": [r["peak_memory_bytes"] for r in points],
            }
    return series


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["time_seconds"] = f"{r['time_seconds']:.9f}"
            if r["speedup_vs_recursive"] is not None:
                out["speedup_vs_recursive"] = \
                    f"{r['speedup_vs_recursive']:.1f}"
            writer.writerow(out)


def print_summary(rows: List[Dict], problem: str) -> None:
    """Console table: one line per size, methods side by side."""
    print(f"\n--- {problem} summary ---")
    print(f"{'n':>6} | {'recursive':>12} {'calls':>14} | "
          f"{'memo':>10} {'calls':>8} | {'tabulation':>10} "
          f"{'cells':>9} | {'speedup (tab)':>13}")
    by_size: Dict[int, Dict[str, Dict]] = {}
    for r in rows:
        if r["problem"] == problem:
            by_size.setdefault(r["input_size"], {})[r["method"]] = r
    for n in sorted(by_size):
        m = by_size[n]
        rec, memo, tab = (m.get("recursive"), m.get("memoization"),
                          m.get("tabulation"))
        cell = lambda r, k, fmt: format(r[k], fmt) if r else "—"
        ratio = tab["speedup_vs_recursive"] if tab else None
        speed = ("—" if ratio is None
                 else f"{ratio:.1f}×" if ratio < 10 else f"{ratio:,.0f}×")
        print(f"{n:>6} | {cell(rec, 'time_seconds', '12.6f'):>12} "
              f"{cell(rec, 'calls', ','):>14} | "
              f"{cell(memo, 'time_seconds', '10.6f'):>10} "
              f"{cell(memo, 'calls', ','):>8} | "
              f"{cell(tab, 'time_seconds', '10.6f'):>10} "
              f"{cell(tab, 'calls', ','):>9} | {speed:>13}")


def make_plots(rows: List[Dict], results_dir: str) -> None:
    plot_dp_comparison(
        series_for(rows, "fibonacci"),
        os.path.join(results_dir, "fibonacci_comparison.png"),
        title="Fibonacci: exponential naive recursion vs linear DP",
        x_label="n")
    plot_dp_comparison(
        series_for(rows, "knapsack"),
        os.path.join(results_dir, "knapsack_performance.png"),
        title="0/1 Knapsack: O(2ⁿ) recursion vs O(n × W) DP "
              "(W = ½ total weight)",
        x_label="Number of items (n)", log_x=True)
    plot_dp_comparison(
        series_for(rows, "lcs"),
        os.path.join(results_dir, "lcs_performance.png"),
        title="LCS on random DNA strings: recursion vs O(m × n) DP",
        x_label="String length (m = n)", log_x=True)


def run(results_dir: str = RESULTS_DIR, quick: bool = False) -> List[Dict]:
    os.makedirs(results_dir, exist_ok=True)
    rows: List[Dict] = []
    rows += bench_fibonacci(naive_limit=35 if quick else 45)
    rows += bench_knapsack()
    rows += bench_knapsack_capacity()
    rows += bench_lcs()
    add_speedups(rows)

    csv_path = os.path.join(results_dir, CSV_NAME)
    write_csv(rows, csv_path)
    print(f"\nSaved {csv_path} ({len(rows)} measurements)")

    for problem in ["fibonacci", "knapsack", "lcs"]:
        print_summary(rows, problem)

    make_plots(rows, results_dir)
    print(f"\nSaved plots to {results_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true",
                        help="skip naive Fibonacci for n > 35 (~90 s saved)")
    args = parser.parse_args()
    start = time.perf_counter()
    run(quick=args.quick)
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
