"""
Week 6 benchmark: advanced dynamic programming.

Every measurement runs three separate times so instrumentation never
leaks into another number (same approach as Week 5):
  - time:    clean run, averaged over repeats when under 0.5 s
  - work:    run with a CallStats counter (cells, calls, relaxations,
             transitions, or tours — the `work_unit` column says which)
  - memory:  peak heap bytes via tracemalloc

Experiments (results land in benchmarks/results/):

1. Knapsack space: Week 5's 2D table vs a two-row table vs the 1D
   in-place row, with and without item reconstruction. Varies n at
   fixed W, then W at fixed n.          -> knapsack_space_comparison.png
2. Matrix chain multiplication: naive recursion vs memoized vs
   bottom-up, plus the optimal-vs-left-to-right cost gap.
                                         -> mcm_performance.png,
                                            mcm_dp_table.png
3. Floyd–Warshall at V = 50, 100, 200, 500 on sparse and dense
   directed graphs, against all-pairs Dijkstra (Week 4).
                                         -> floyd_warshall_scaling.png
4. TSP: brute force (n <= 12) vs Held–Karp bitmask DP (n <= 18).
                                         -> tsp_bitmask_runtime.png

Raw measurements: comparison_table.csv

Usage:
    python benchmarks/week6_dp_advanced_benchmark.py          # ~8 min
    python benchmarks/week6_dp_advanced_benchmark.py --quick  # skips slowest cases
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dp import knapsack_tabulation, knapsack_with_items
from src.dp_advanced import (
    knapsack_two_row, knapsack_1d, knapsack_1d_with_items,
    mcm_recursive, mcm_memoized, mcm_bottom_up, build_mcm_tables,
    floyd_warshall, dijkstra_all_pairs,
    tsp_bitmask, tsp_brute_force,
)
from src.utils.graph_generator import generate_graph
from src.utils.matrix_utils import (
    euclidean_distance_matrix, left_to_right_cost, random_chain_dimensions,
    random_points,
)
from src.utils.timer import CallStats, peak_memory, time_call
from src.utils.visualization import (
    plot_dp_comparison, plot_metric_panels, plot_table_heatmap,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_NAME = "comparison_table.csv"
SEED = 42
MIN_TIME = 0.5

FIELDS = ["experiment", "method", "input_size", "parameter",
          "time_seconds", "work", "work_unit", "max_depth",
          "peak_memory_bytes", "speedup_vs_baseline"]

# Which method each method's speedup is measured against
BASELINES = {
    "two_row": "standard_2d",
    "space_optimized_1d": "standard_2d",
    "1d_with_items": "2d_with_items",
    "memoization": "recursive",
    "bottom_up": "recursive",
    "floyd_warshall": "dijkstra_all_pairs",
    "bitmask": "brute_force",
}

KNAPSACK_FIXED_W = 1_000
KNAPSACK_N_SIZES = [50, 100, 200, 400, 800]
KNAPSACK_FIXED_N = 100
KNAPSACK_W_SIZES = [500, 1_000, 2_500, 5_000, 10_000]

MCM_RECURSIVE_SIZES = [4, 6, 8, 10, 12, 14, 16]
MCM_DP_SIZES = MCM_RECURSIVE_SIZES + [25, 50, 100, 200, 300]
CLRS_CHAIN = [30, 35, 15, 5, 10, 20, 25]

FW_SIZES = [50, 100, 200, 500]

TSP_BRUTE_SIZES = [4, 5, 6, 7, 8, 9, 10, 11, 12]
TSP_BITMASK_SIZES = TSP_BRUTE_SIZES + [13, 14, 15, 16, 17, 18]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def make_row(experiment: str, method: str, size: int, parameter: str,
             seconds: Optional[float], work: int, work_unit: str,
             max_depth: int = 0, memory: Optional[int] = None) -> Dict:
    row = {
        "experiment": experiment, "method": method, "input_size": size,
        "parameter": parameter, "time_seconds": seconds, "work": work,
        "work_unit": work_unit, "max_depth": max_depth,
        "peak_memory_bytes": memory, "speedup_vs_baseline": None,
    }
    secs = f"{seconds:11.6f}s" if seconds is not None else " " * 12
    mem = f"{memory / 1024:11,.1f} KiB" if memory is not None else ""
    print(f"  {experiment:12s} {method:20s} n={size:<5d} {parameter:10s} "
          f"{secs}  {work:>13,} {work_unit:12s} {mem}", flush=True)
    return row


def measure(experiment: str, method: str, size: int, parameter: str,
            func: Callable, args: Tuple, work_unit: str,
            count: bool = True, memory: bool = True) -> Dict:
    """Time, count, and memory-profile func(*args) in separate runs."""
    seconds = time_call(lambda: func(*args), min_time=MIN_TIME)
    work = depth = 0
    if count:
        stats = CallStats()
        func(*args, stats=stats)
        work, depth = stats.calls, stats.max_depth
    peak = peak_memory(lambda: func(*args))[0] if memory else None
    return make_row(experiment, method, size, parameter, seconds, work,
                    work_unit, depth, peak)


def add_speedups(rows: List[Dict]) -> None:
    """speedup = baseline time / method time, at the same experiment,
    size, and parameter."""
    times = {(r["experiment"], r["method"], r["input_size"],
              r["parameter"]): r["time_seconds"] for r in rows}
    for r in rows:
        base = BASELINES.get(r["method"])
        key = (r["experiment"], base, r["input_size"], r["parameter"])
        if base and times.get(key) and r["time_seconds"]:
            r["speedup_vs_baseline"] = times[key] / r["time_seconds"]


# ---------------------------------------------------------------------------
# 1. Knapsack: standard 2D vs space-optimized
# ---------------------------------------------------------------------------

def make_items(n: int, seed: int = SEED) -> Tuple[List[int], List[int]]:
    """n seeded random items: weights 1-50, values 1-100."""
    rng = random.Random(seed + n)
    return ([rng.randint(1, 50) for _ in range(n)],
            [rng.randint(1, 100) for _ in range(n)])


def _with_items_2d(weights, values, capacity, stats=None):
    """Week 5 table + trace_solution, counted like the 1D version."""
    if stats is not None:
        stats.count(len(weights) * (capacity + 1))
    return knapsack_with_items(weights, values, capacity)


def _with_items_1d(weights, values, capacity, stats=None):
    if stats is not None:
        stats.count(len(weights) * (capacity + 1))
    return knapsack_1d_with_items(weights, values, capacity)


KNAPSACK_METHODS = [
    ("standard_2d", knapsack_tabulation),
    ("two_row", knapsack_two_row),
    ("space_optimized_1d", knapsack_1d),
    ("2d_with_items", _with_items_2d),
    ("1d_with_items", _with_items_1d),
]


def bench_knapsack(n_sizes: Sequence[int] = KNAPSACK_N_SIZES,
                   fixed_w: int = KNAPSACK_FIXED_W,
                   w_sizes: Sequence[int] = KNAPSACK_W_SIZES,
                   fixed_n: int = KNAPSACK_FIXED_N) -> List[Dict]:
    rows = []
    sweeps = [("knapsack_n", [(n, fixed_w) for n in n_sizes],
               f"vary n, W = {fixed_w:,}"),
              ("knapsack_w", [(fixed_n, w) for w in w_sizes],
               f"vary W, n = {fixed_n}")]
    for experiment, cases, label in sweeps:
        print(f"\n=== Knapsack space: {label} ===")
        for n, capacity in cases:
            weights, values = make_items(n)
            args = (weights, values, capacity)
            # input_size is the swept variable: n, or W for knapsack_w
            if experiment == "knapsack_n":
                size, param = n, f"W={capacity}"
            else:
                size, param = capacity, f"n={n}"
            answers = set()
            for method, func in KNAPSACK_METHODS:
                rows.append(measure(experiment, method, size, param,
                                    func, args, "cells"))
                result = func(*args)
                answers.add(result[0] if isinstance(result, tuple)
                            else result)
            assert len(answers) == 1, f"knapsack methods disagree: {answers}"
    return rows


# ---------------------------------------------------------------------------
# 2. Matrix chain multiplication
# ---------------------------------------------------------------------------

def bench_mcm(recursive_sizes: Sequence[int] = MCM_RECURSIVE_SIZES,
              dp_sizes: Sequence[int] = MCM_DP_SIZES) -> List[Dict]:
    print("\n=== Matrix chain: recursive vs memoized vs bottom-up ===")
    rows = []
    for n in sorted(set(recursive_sizes) | set(dp_sizes)):
        p = random_chain_dimensions(n, seed=SEED + n)
        answers = set()
        if n in recursive_sizes:
            rows.append(measure("mcm", "recursive", n, "", mcm_recursive,
                                (p,), "calls"))
            answers.add(mcm_recursive(p))
        if n in dp_sizes:
            rows.append(measure("mcm", "memoization", n, "", mcm_memoized,
                                (p,), "calls"))
            rows.append(measure("mcm", "bottom_up", n, "", mcm_bottom_up,
                                (p,), "splits"))
            answers.update({mcm_memoized(p), mcm_bottom_up(p)})
            optimal = mcm_bottom_up(p)
            rows.append(make_row("mcm_cost", "optimal", n, "", None,
                                 optimal, "scalar_mults"))
            rows.append(make_row("mcm_cost", "left_to_right", n, "", None,
                                 left_to_right_cost(p), "scalar_mults"))
        assert len(answers) == 1, f"MCM methods disagree: {answers}"
    return rows


# ---------------------------------------------------------------------------
# 3. Floyd–Warshall vs all-pairs Dijkstra
# ---------------------------------------------------------------------------

def make_graph(v: int, density: str, seed: int = SEED):
    """Directed, weighted (1-100), reachable from 0. sparse: E = 4V;
    dense: E = 25% of the V(V-1) possible edges."""
    edges = 4 * v if density == "sparse" else v * (v - 1) // 4
    return generate_graph(v, edges, directed=True, weighted=True,
                          seed=seed + v)


def _dijkstra_all_pairs(graph, stats=None):
    return dijkstra_all_pairs(graph)


def bench_floyd_warshall(sizes: Sequence[int] = FW_SIZES) -> List[Dict]:
    print("\n=== Floyd–Warshall vs all-pairs Dijkstra ===")
    rows = []
    for density in ["sparse", "dense"]:
        for v in sizes:
            g = make_graph(v, density)
            param = f"{density} E={g.num_edges()}"
            rows.append(measure(f"fw_{density}", "floyd_warshall", v,
                                param, floyd_warshall, (g,),
                                "relaxations"))
            rows.append(measure(f"fw_{density}", "dijkstra_all_pairs", v,
                                param, _dijkstra_all_pairs, (g,), "",
                                count=False))
            fw, dj = floyd_warshall(g), dijkstra_all_pairs(g)
            for row_fw, row_dj in zip(fw.dist, dj.dist):
                for a, b in zip(row_fw, row_dj):
                    assert a == b or math.isclose(a, b, rel_tol=1e-9), \
                        "Floyd–Warshall and Dijkstra disagree"
    return rows


# ---------------------------------------------------------------------------
# 4. TSP: brute force vs bitmask DP
# ---------------------------------------------------------------------------

def make_cities(n: int, seed: int = SEED) -> List[List[float]]:
    return euclidean_distance_matrix(random_points(n, seed=seed + n))


def bench_tsp(brute_sizes: Sequence[int] = TSP_BRUTE_SIZES,
              bitmask_sizes: Sequence[int] = TSP_BITMASK_SIZES
              ) -> List[Dict]:
    print("\n=== TSP: brute force vs bitmask DP ===")
    rows = []
    for n in sorted(set(brute_sizes) | set(bitmask_sizes)):
        dist = make_cities(n)
        costs = []
        if n in brute_sizes:
            rows.append(measure("tsp", "brute_force", n, "",
                                tsp_brute_force, (dist,), "tours"))
            costs.append(tsp_brute_force(dist)[0])
        if n in bitmask_sizes:
            rows.append(measure("tsp", "bitmask", n, "", tsp_bitmask,
                                (dist,), "transitions"))
            costs.append(tsp_bitmask(dist)[0])
        assert all(math.isclose(c, costs[0]) for c in costs), \
            f"TSP methods disagree: {costs}"
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _select(rows, experiment, method, field="time_seconds"):
    pts = sorted((r["input_size"], r[field]) for r in rows
                 if r["experiment"] == experiment and r["method"] == method)
    return [p[0] for p in pts], [p[1] for p in pts]


def _ratio(rows, experiment, numerator, denominator, field):
    xs, top = _select(rows, experiment, numerator, field)
    xs2, bottom = _select(rows, experiment, denominator, field)
    lookup = dict(zip(xs2, bottom))
    pts = [(x, t / lookup[x]) for x, t in zip(xs, top)
           if lookup.get(x) and t is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def _kib(series):
    xs, ys = series
    return xs, [y / 1024 if y is not None else None for y in ys]


def _scaled_ref(xs, f, anchor_x, anchor_y):
    """Reference curve c·f(x), scaled to pass through one measured point."""
    c = anchor_y / f(anchor_x)
    return xs, [c * f(x) for x in xs]


def plot_knapsack(rows: List[Dict], results_dir: str) -> None:
    methods = [m for m, _ in KNAPSACK_METHODS]
    w_param = next((r["parameter"] for r in rows
                    if r["experiment"] == "knapsack_n"), "")
    panels = [
        {"title": f"Peak memory vs items ({w_param})",
         "xlabel": "Items (n)", "ylabel": "Peak memory (KiB)",
         "series": {m: _kib(_select(rows, "knapsack_n", m,
                                    "peak_memory_bytes")) for m in methods},
         "logx": True, "logy": True},
        {"title": "Peak memory vs capacity (n fixed)",
         "xlabel": "Capacity (W)", "ylabel": "Peak memory (KiB)",
         "series": {m: _kib(_select(rows, "knapsack_w", m,
                                    "peak_memory_bytes")) for m in methods},
         "logx": True, "logy": True},
        {"title": "Runtime vs items (same O(n × W) work)",
         "xlabel": "Items (n)", "ylabel": "Time (s)",
         "series": {m: _select(rows, "knapsack_n", m) for m in methods},
         "logx": True, "logy": True},
        {"title": "Standard 2D ÷ space-optimized 1D",
         "xlabel": "Items (n)", "ylabel": "Ratio (×)",
         "series": {
             "memory saved (×)": _ratio(rows, "knapsack_n", "standard_2d",
                                        "space_optimized_1d",
                                        "peak_memory_bytes"),
             "speedup (×)": _ratio(rows, "knapsack_n", "standard_2d",
                                   "space_optimized_1d", "time_seconds"),
             "memory saved, with items (×)": _ratio(
                 rows, "knapsack_n", "2d_with_items", "1d_with_items",
                 "peak_memory_bytes"),
         },
         "logx": True, "logy": True},
    ]
    plot_metric_panels(panels,
                       os.path.join(results_dir,
                                    "knapsack_space_comparison.png"),
                       "0/1 Knapsack: O(n × W) table vs O(W) row")


def plot_mcm(rows: List[Dict], results_dir: str) -> None:
    series = {}
    for method in ["recursive", "memoization", "bottom_up"]:
        sizes, times = _select(rows, "mcm", method)
        if sizes:
            series[method] = {
                "sizes": sizes, "time": times,
                "calls": _select(rows, "mcm", method, "work")[1],
                "memory": _select(rows, "mcm", method,
                                  "peak_memory_bytes")[1],
            }
    plot_dp_comparison(series,
                       os.path.join(results_dir, "mcm_performance.png"),
                       title="Matrix chain multiplication: Θ(3ⁿ) recursion "
                             "vs O(n³) interval DP",
                       x_label="Matrices in chain (n)", log_x=True)

    cost, split = build_mcm_tables(CLRS_CHAIN)
    n = len(CLRS_CHAIN) - 1
    table = [[cost[i][j] if j >= i else None for j in range(1, n + 1)]
             for i in range(1, n + 1)]
    notes = [[f"{cost[i][j]:,}\nk={split[i][j]}" if j > i
              else ("0" if j == i else "")
              for j in range(1, n + 1)] for i in range(1, n + 1)]
    plot_table_heatmap(table, os.path.join(results_dir, "mcm_dp_table.png"),
                       "MCM cost table m[i][j] (k = best split), "
                       f"p = {CLRS_CHAIN}",
                       row_labels=[f"i={i}" for i in range(1, n + 1)],
                       col_labels=[f"j={j}" for j in range(1, n + 1)],
                       annotations=notes)


def plot_floyd_warshall(rows: List[Dict], results_dir: str) -> None:
    fw_sparse = _select(rows, "fw_sparse", "floyd_warshall")
    refs = {}
    if fw_sparse[0]:
        refs["c·V³"] = _scaled_ref(fw_sparse[0], lambda v: v ** 3,
                                   fw_sparse[0][-1], fw_sparse[1][-1])
    mem = _select(rows, "fw_dense", "floyd_warshall", "peak_memory_bytes")
    mem_refs = {}
    if mem[0]:
        mem_refs["c·V²"] = _kib(_scaled_ref(mem[0], lambda v: v ** 2,
                                            mem[0][-1], mem[1][-1]))
    relax = _select(rows, "fw_dense", "floyd_warshall", "work")
    colors = {"FW sparse": "tab:blue", "FW dense": "tab:purple",
              "Dijkstra sparse": "tab:orange",
              "Dijkstra dense": "tab:red"}
    panels = [
        {"title": "Runtime vs vertices", "xlabel": "Vertices (V)",
         "ylabel": "Time (s)", "colors": colors,
         "series": {
             "FW sparse": fw_sparse,
             "FW dense": _select(rows, "fw_dense", "floyd_warshall"),
             "Dijkstra sparse": _select(rows, "fw_sparse",
                                        "dijkstra_all_pairs"),
             "Dijkstra dense": _select(rows, "fw_dense",
                                       "dijkstra_all_pairs"),
         }, "refs": refs, "logx": True, "logy": True},
        {"title": "Floyd–Warshall speedup over all-pairs Dijkstra",
         "xlabel": "Vertices (V)", "ylabel": "Dijkstra time ÷ FW time",
         "series": {
             "sparse (E = 4V)": _ratio(rows, "fw_sparse",
                                       "dijkstra_all_pairs",
                                       "floyd_warshall", "time_seconds"),
             "dense (E = ¼V²)": _ratio(rows, "fw_dense",
                                       "dijkstra_all_pairs",
                                       "floyd_warshall", "time_seconds"),
         }, "refs": {"break-even": ([min(FW_SIZES), max(FW_SIZES)],
                                    [1, 1])},
         "logx": True, "logy": True},
        {"title": "Peak memory", "xlabel": "Vertices (V)",
         "ylabel": "Peak memory (KiB)", "colors": colors,
         "series": {
             "FW dense": _kib(mem),
             "Dijkstra dense": _kib(_select(rows, "fw_dense",
                                            "dijkstra_all_pairs",
                                            "peak_memory_bytes")),
         }, "refs": mem_refs, "logx": True, "logy": True},
        {"title": "Relaxations attempted (dense)", "xlabel": "Vertices (V)",
         "ylabel": "Relaxations", "series": {"Floyd–Warshall": relax},
         "refs": {"V³": (relax[0], [v ** 3 for v in relax[0]])},
         "logx": True, "logy": True},
    ]
    plot_metric_panels(panels,
                       os.path.join(results_dir, "floyd_warshall_scaling.png"),
                       "Floyd–Warshall: Θ(V³) all-pairs DP vs "
                       "V × Dijkstra")


def plot_tsp(rows: List[Dict], results_dir: str) -> None:
    brute_work = _select(rows, "tsp", "brute_force", "work")
    dp_work = _select(rows, "tsp", "bitmask", "work")
    ns = sorted(set(brute_work[0]) | set(dp_work[0]))
    panels = [
        {"title": "Runtime", "xlabel": "Cities (n)", "ylabel": "Time (s)",
         "series": {"brute force": _select(rows, "tsp", "brute_force"),
                    "bitmask DP": _select(rows, "tsp", "bitmask")},
         "colors": {"brute force": "tab:red", "bitmask DP": "tab:green"},
         "logy": True},
        {"title": "Work: tours tried vs DP transitions",
         "xlabel": "Cities (n)", "ylabel": "Count",
         "series": {"brute force (tours)": brute_work,
                    "bitmask DP (transitions)": dp_work},
         "refs": {"(n-1)!": (ns, [math.factorial(n - 1) for n in ns]),
                  "n²·2ⁿ": (ns, [n * n * 2 ** n for n in ns])},
         "colors": {"brute force (tours)": "tab:red",
                    "bitmask DP (transitions)": "tab:green",
                    "(n-1)!": "tab:red", "n²·2ⁿ": "tab:green"},
         "logy": True},
        {"title": "Bitmask DP speedup over brute force",
         "xlabel": "Cities (n)", "ylabel": "Speedup (×)",
         "series": {"bitmask vs brute force": _ratio(
             rows, "tsp", "brute_force", "bitmask", "time_seconds")},
         "colors": {"bitmask vs brute force": "tab:green"},
         "logy": True},
        {"title": "Peak memory: O(n) vs O(n·2ⁿ)", "xlabel": "Cities (n)",
         "ylabel": "Peak memory (KiB)",
         "series": {"brute force": _kib(_select(rows, "tsp", "brute_force",
                                                "peak_memory_bytes")),
                    "bitmask DP": _kib(_select(rows, "tsp", "bitmask",
                                               "peak_memory_bytes"))},
         "colors": {"brute force": "tab:red", "bitmask DP": "tab:green"},
         "logy": True},
    ]
    plot_metric_panels(panels,
                       os.path.join(results_dir, "tsp_bitmask_runtime.png"),
                       "TSP: (n-1)! brute force vs O(n²·2ⁿ) bitmask DP")


def make_plots(rows: List[Dict], results_dir: str) -> None:
    plot_knapsack(rows, results_dir)
    plot_mcm(rows, results_dir)
    plot_floyd_warshall(rows, results_dir)
    plot_tsp(rows, results_dir)


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            if r["time_seconds"] is not None:
                out["time_seconds"] = f"{r['time_seconds']:.9f}"
            if r["speedup_vs_baseline"] is not None:
                out["speedup_vs_baseline"] = \
                    f"{r['speedup_vs_baseline']:.2f}"
            writer.writerow(out)


def print_summary(rows: List[Dict]) -> None:
    """One console table per experiment, methods side by side."""
    experiments: Dict[str, List[str]] = {}
    for r in rows:
        methods = experiments.setdefault(r["experiment"], [])
        if r["method"] not in methods:
            methods.append(r["method"])
    for experiment, methods in experiments.items():
        cost_table = experiment == "mcm_cost"
        print(f"\n--- {experiment} "
              f"({'scalar multiplications' if cost_table else 'time | peak memory'}) ---")
        print(f"{'size':>6} | " + " | ".join(f"{m:>24}" for m in methods))
        sizes = sorted({r["input_size"] for r in rows
                        if r["experiment"] == experiment})
        for n in sizes:
            cells = []
            for m in methods:
                r = next((r for r in rows if r["experiment"] == experiment
                          and r["method"] == m and r["input_size"] == n),
                         None)
                if r is None:
                    cells.append(f"{'—':>24}")
                elif cost_table:
                    cells.append(f"{r['work']:>24,}")
                else:
                    mem = (f"{r['peak_memory_bytes'] / 1024:,.0f} KiB"
                           if r["peak_memory_bytes"] is not None else "—")
                    cells.append(f"{r['time_seconds']:>10.5f}s {mem:>12}")
            print(f"{n:>6} | " + " | ".join(cells))


def run(results_dir: str = RESULTS_DIR, quick: bool = False) -> List[Dict]:
    os.makedirs(results_dir, exist_ok=True)
    rows: List[Dict] = []
    rows += bench_knapsack()
    rows += bench_mcm(recursive_sizes=[n for n in MCM_RECURSIVE_SIZES
                                       if not quick or n <= 14])
    rows += bench_floyd_warshall([v for v in FW_SIZES
                                  if not quick or v <= 200])
    rows += bench_tsp(brute_sizes=[n for n in TSP_BRUTE_SIZES
                                   if not quick or n <= 10],
                      bitmask_sizes=[n for n in TSP_BITMASK_SIZES
                                     if not quick or n <= 16])
    add_speedups(rows)

    csv_path = os.path.join(results_dir, CSV_NAME)
    write_csv(rows, csv_path)
    print(f"\nSaved {csv_path} ({len(rows)} rows)")
    print_summary(rows)
    make_plots(rows, results_dir)
    print(f"\nSaved plots to {results_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true",
                        help="smaller sizes: brute-force TSP <= 10, "
                             "Floyd–Warshall <= 200, MCM recursion <= 14")
    args = parser.parse_args()
    start = time.perf_counter()
    run(quick=args.quick)
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
