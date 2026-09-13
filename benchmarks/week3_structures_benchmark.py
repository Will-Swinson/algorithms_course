# File: benchmarks/week3_structures_benchmark.py
"""
Week 3 benchmark: heaps, AVL tree, and hash tables vs Python built-ins.

Measures per-operation insert / search / delete times for input sizes
10^3 .. 10^6 and produces, under benchmarks/results/:

    heap_performance.png     MinHeap vs Python heapq (insert + extract)
    tree_performance.png     AVL tree vs built-in dict (insert + search)
    hash_performance.png     chaining vs probing, plus lookup time as a
                             function of load factor
    complexity_shapes.png    one search plot showing the three shapes:
                             logarithmic (AVL/heap), constant (hash),
                             linear (list)
    comparison_table.csv     every raw measurement

Also measures the engineered O(n) worst case: a fixed-capacity table
where every key collides.
"""
import csv
import heapq
import os
import random
import sys
import time
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.structures import (
    MinHeap, AVLTree, ChainingHashTable, LinearProbingHashTable,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "comparison_table.csv")

SEED = 42
SIZES = [1_000, 10_000, 100_000, 1_000_000]
SEARCHES = 10_000          # lookups per timing sample
LIST_SEARCHES = 100        # linear scans are expensive; fewer samples
DELETES = 10_000

ROWS: List[Dict] = []      # accumulated measurements


def record(structure: str, operation: str, size: int, total: float,
           ops: int) -> None:
    """Append one measurement (per-operation seconds) and print it."""
    per_op = total / ops
    ROWS.append({
        "structure": structure,
        "operation": operation,
        "size": size,
        "ops_timed": ops,
        "total_seconds": f"{total:.6f}",
        "seconds_per_op": f"{per_op:.9f}",
    })
    print(f"  {structure:22s} {operation:8s} n={size:<9d} "
          f"{per_op * 1e6:10.3f} us/op", flush=True)


def timed(func: Callable) -> float:
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


def make_keys(n: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    """n unique keys in insertion order, plus a lookup sample."""
    keys = rng.sample(range(n * 10), n)
    lookups = rng.choices(keys, k=SEARCHES)
    return keys, lookups


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_heaps() -> None:
    print("\n=== Heap: MinHeap vs heapq ===")
    rng = random.Random(SEED)
    for n in SIZES:
        values = [rng.randint(0, n * 10) for _ in range(n)]

        heap = MinHeap()
        record("MinHeap", "insert", n,
               timed(lambda: [heap.insert(v) for v in values]), n)
        record("MinHeap", "extract", n,
               timed(lambda: [heap.extract_min() for _ in range(n)]), n)

        hq: List[int] = []
        record("heapq", "insert", n,
               timed(lambda: [heapq.heappush(hq, v) for v in values]), n)
        record("heapq", "extract", n,
               timed(lambda: [heapq.heappop(hq) for _ in range(n)]), n)


def bench_tree_vs_dict() -> None:
    print("\n=== AVL tree vs built-in dict ===")
    rng = random.Random(SEED)
    for n in SIZES:
        keys, lookups = make_keys(n, rng)
        deletions = rng.sample(keys, min(DELETES, n))

        tree = AVLTree()
        record("AVLTree", "insert", n,
               timed(lambda: [tree.insert(k, k) for k in keys]), n)
        record("AVLTree", "search", n,
               timed(lambda: [tree.search(k) for k in lookups]),
               len(lookups))
        record("AVLTree", "delete", n,
               timed(lambda: [tree.delete(k) for k in deletions]),
               len(deletions))

        d: Dict[int, int] = {}
        record("dict", "insert", n,
               timed(lambda: [d.__setitem__(k, k) for k in keys]), n)
        record("dict", "search", n,
               timed(lambda: [d[k] for k in lookups]), len(lookups))
        record("dict", "delete", n,
               timed(lambda: [d.__delitem__(k) for k in deletions]),
               len(deletions))


def bench_hash_tables() -> None:
    print("\n=== Hash tables: chaining vs linear probing ===")
    rng = random.Random(SEED)
    for n in SIZES:
        keys, lookups = make_keys(n, rng)
        deletions = rng.sample(keys, min(DELETES, n))

        for name, table in [("ChainingHashTable", ChainingHashTable()),
                            ("LinearProbingHashTable",
                             LinearProbingHashTable())]:
            record(name, "insert", n,
                   timed(lambda: [table.insert(k, k) for k in keys]), n)
            record(name, "search", n,
                   timed(lambda: [table.get(k) for k in lookups]),
                   len(lookups))
            record(name, "delete", n,
                   timed(lambda: [table.delete(k) for k in deletions]),
                   len(deletions))


class _AlwaysCollides:
    """Key with a constant hash: every instance lands in bucket 0."""

    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = n

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, _AlwaysCollides) and self.n == other.n


def bench_worst_case() -> None:
    """Every key collides and rehashing is disabled: lookups must walk
    the whole chain — the O(n) degenerate case."""
    print("\n=== Hash worst case: all keys colliding ===")
    for n in [1_000, 5_000, 10_000]:
        table = ChainingHashTable(initial_capacity=8,
                                  max_load_factor=float("inf"))
        keys = [_AlwaysCollides(i) for i in range(n)]
        for key in keys:
            table.insert(key, key.n)
        rng = random.Random(SEED)
        sample = rng.choices(keys, k=100)
        record("ChainingHashTable", "worst_get", n,
               timed(lambda: [table.get(k) for k in sample]), len(sample))


def bench_list_linear() -> None:
    """Unsorted Python list: the linear-search baseline."""
    print("\n=== List: linear search baseline ===")
    rng = random.Random(SEED)
    for n in SIZES:
        keys, _ = make_keys(n, rng)
        lookups = rng.choices(keys, k=LIST_SEARCHES)
        key_list = list(keys)
        record("list", "search", n,
               timed(lambda: [key_list.index(k) for k in lookups]),
               len(lookups))


def bench_load_factor() -> None:
    """Lookup time as the load factor climbs, at fixed capacity (no
    rehashing) — the data behind 'load factor vs lookup time'."""
    print("\n=== Lookup time vs load factor (fixed capacity 2^17) ===")
    capacity = 2 ** 17
    rng = random.Random(SEED)
    for name, table in [
        ("ChainingHashTable",
         ChainingHashTable(initial_capacity=capacity,
                           max_load_factor=float("inf"))),
        ("LinearProbingHashTable",
         LinearProbingHashTable(initial_capacity=capacity,
                                max_load_factor=0.95)),
    ]:
        inserted: List[int] = []
        for target in [0.1, 0.3, 0.5, 0.7, 0.9]:
            goal = int(capacity * target)
            while len(inserted) < goal:
                k = rng.randint(0, 10 ** 9)
                table.insert(k, k)
                inserted.append(k)
            lookups = rng.choices(inserted, k=SEARCHES)
            total = timed(lambda: [table.get(k) for k in lookups])
            ROWS.append({
                "structure": name,
                "operation": f"get@load={target:.1f}",
                "size": goal,
                "ops_timed": SEARCHES,
                "total_seconds": f"{total:.6f}",
                "seconds_per_op": f"{total / SEARCHES:.9f}",
            })
            print(f"  {name:22s} load={target:.1f} "
                  f"{total / SEARCHES * 1e6:8.3f} us/op", flush=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _series(structure: str, operation: str) -> Tuple[List[int], List[float]]:
    points = sorted(
        (row["size"], float(row["seconds_per_op"]))
        for row in ROWS
        if row["structure"] == structure and row["operation"] == operation
    )
    return [p[0] for p in points], [p[1] for p in points]


def _loglog_axis(ax, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of stored items (n)")
    ax.set_ylabel("Time per operation (s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def plot_heap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for op, ax in zip(["insert", "extract"], axes):
        for structure, color in [("MinHeap", "tab:blue"),
                                 ("heapq", "tab:green")]:
            sizes, times = _series(structure, op)
            ax.plot(sizes, times, marker="o", label=structure, color=color)
        _loglog_axis(ax, f"Heap {op}: per-operation time")
    fig.suptitle("Binary heap performance: MinHeap vs Python heapq")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "heap_performance.png"), dpi=150)
    plt.close(fig)


def plot_tree() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for op, ax in zip(["insert", "search"], axes):
        for structure, color in [("AVLTree", "tab:blue"),
                                 ("dict", "tab:orange")]:
            sizes, times = _series(structure, op)
            ax.plot(sizes, times, marker="o", label=structure, color=color)
        _loglog_axis(ax, f"{op}: per-operation time")
    fig.suptitle("AVL tree vs built-in dict")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "tree_performance.png"), dpi=150)
    plt.close(fig)


def plot_hash() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for structure, color in [("ChainingHashTable", "tab:blue"),
                             ("LinearProbingHashTable", "tab:red")]:
        for op, style in [("insert", "-"), ("search", "--")]:
            sizes, times = _series(structure, op)
            ax.plot(sizes, times, style, marker="o", color=color,
                    label=f"{structure.replace('HashTable', '')} {op}")
    _loglog_axis(ax, "Chaining vs linear probing")

    ax = axes[1]
    loads = [0.1, 0.3, 0.5, 0.7, 0.9]
    for structure, color in [("ChainingHashTable", "tab:blue"),
                             ("LinearProbingHashTable", "tab:red")]:
        times = []
        for load in loads:
            for row in ROWS:
                if (row["structure"] == structure
                        and row["operation"] == f"get@load={load:.1f}"):
                    times.append(float(row["seconds_per_op"]))
        ax.plot(loads, times, marker="o", color=color,
                label=structure.replace("HashTable", ""))
    ax.set_xlabel("Load factor")
    ax.set_ylabel("Lookup time per operation (s)")
    ax.set_title("Lookup cost vs load factor (fixed capacity)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Hash table performance")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "hash_performance.png"), dpi=150)
    plt.close(fig)


def plot_complexity_shapes() -> None:
    """The assignment's three growth shapes on one search chart."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for structure, op, label, color in [
        ("list", "search", "list linear search — O(n)", "tab:red"),
        ("AVLTree", "search", "AVL search — O(log n)", "tab:blue"),
        ("ChainingHashTable", "search", "hash search — O(1)", "tab:green"),
    ]:
        sizes, times = _series(structure, op)
        ax.plot(sizes, times, marker="o", label=label, color=color)
    _loglog_axis(ax, "Search cost by structure: linear vs logarithmic "
                     "vs constant")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "complexity_shapes.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start = time.perf_counter()

    bench_heaps()
    bench_tree_vs_dict()
    bench_hash_tables()
    bench_worst_case()
    bench_list_linear()
    bench_load_factor()

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"\nSaved {CSV_PATH} ({len(ROWS)} measurements)")

    plot_heap()
    plot_tree()
    plot_hash()
    plot_complexity_shapes()
    print(f"Saved 4 plots to {RESULTS_DIR}")
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
