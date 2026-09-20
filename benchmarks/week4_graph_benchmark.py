# File: benchmarks/week4_graph_benchmark.py
"""
Week 4 benchmark: graph representations, traversals, and Dijkstra.

Experiments (results land in benchmarks/results/):

1. Representation: adjacency list vs matrix — build time, memory
   footprint, and neighbor-lookup cost at n = 100 / 1,000 / 10,000.
2. BFS vs DFS: traversal time and peak traversal memory on sparse
   (E = 2V) and dense (E = ¼V²) graphs.
   -> bfs_vs_dfs_sparse.png, bfs_vs_dfs_dense.png
3. Dijkstra: heap-based vs list-based priority queue scaling, and the
   effect of graph density at fixed V.
   -> dijkstra_performance.png
4. Traversal-order visualizations on a small graph (networkx layout).
   -> traversal_bfs.png, traversal_dfs.png, shortest_path.png

Raw measurements: comparison_table.csv
"""
import csv
import os
import sys
import time
import tracemalloc
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.graphs import (
    Graph, ADJACENCY_LIST, ADJACENCY_MATRIX,
    bfs, dfs_iterative, dijkstra, dijkstra_list_based, reconstruct_path,
)
from src.utils.graph_generator import generate_graph, generate_sparse_graph
from src.utils.visualization import draw_traversal

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "comparison_table.csv")
SEED = 42

ROWS: List[Dict] = []


def record(experiment: str, subject: str, nodes: int, edges: int,
           metric: str, value: float, unit: str) -> None:
    ROWS.append({
        "experiment": experiment,
        "subject": subject,
        "nodes": nodes,
        "edges": edges,
        "metric": metric,
        "value": f"{value:.9f}" if unit == "seconds" else f"{value:.0f}",
        "unit": unit,
    })
    pretty = f"{value:.6f}s" if unit == "seconds" else f"{value:,.0f} {unit}"
    print(f"  {experiment:14s} {subject:22s} V={nodes:<7d} E={edges:<9d} "
          f"{metric:12s} {pretty}", flush=True)


def timed(func: Callable, min_time: float = 0.05) -> float:
    """Time func(); re-run and average when it's too fast to trust."""
    start = time.perf_counter()
    func()
    elapsed = time.perf_counter() - start
    if elapsed < min_time:
        runs = max(3, int(min_time / max(elapsed, 1e-9)))
        runs = min(runs, 200)
        start = time.perf_counter()
        for _ in range(runs):
            func()
        elapsed = (time.perf_counter() - start) / runs
    return elapsed


def graph_memory_bytes(graph: Graph) -> int:
    """Nominal container memory of the representation itself."""
    if graph.representation == ADJACENCY_LIST:
        total = sys.getsizeof(graph._adj)
        total += sum(sys.getsizeof(nbrs) for nbrs in graph._adj.values())
    else:
        total = sys.getsizeof(graph._matrix)
        total += sum(sys.getsizeof(row) for row in graph._matrix)
        total += sys.getsizeof(graph._index)
    return total


# ---------------------------------------------------------------------------
# 1. Adjacency list vs adjacency matrix
# ---------------------------------------------------------------------------

def bench_representations() -> None:
    print("\n=== Representation: adjacency list vs matrix (sparse E=2V) ===")
    import random
    for n in [100, 1_000, 10_000]:
        for rep in [ADJACENCY_LIST, ADJACENCY_MATRIX]:
            start = time.perf_counter()
            g = generate_sparse_graph(n, representation=rep, seed=SEED)
            build = time.perf_counter() - start
            e = g.num_edges()

            record("representation", rep, n, e, "build", build, "seconds")
            record("representation", rep, n, e, "memory",
                   graph_memory_bytes(g), "bytes")

            rng = random.Random(SEED)
            sample = [rng.randrange(n) for _ in range(1_000)]
            lookup = timed(lambda: [g.get_neighbors(v) for v in sample])
            record("representation", rep, n, e, "1k_lookups",
                   lookup, "seconds")

            record("representation", rep, n, e, "bfs_full",
                   timed(lambda: bfs(g, 0)), "seconds")


# ---------------------------------------------------------------------------
# 2. BFS vs DFS on sparse and dense graphs
# ---------------------------------------------------------------------------

def _traversal_peak_memory(func: Callable) -> int:
    tracemalloc.start()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def bench_traversals() -> None:
    scenarios = {
        "sparse": [100, 1_000, 10_000, 50_000],   # E = 2V
        "dense": [100, 500, 1_000, 2_000],        # E = 1/4 V^2
    }
    for density, sizes in scenarios.items():
        print(f"\n=== BFS vs DFS on {density} graphs ===")
        for n in sizes:
            if density == "sparse":
                g = generate_sparse_graph(n, seed=SEED)
            else:
                g = generate_graph(n, num_edges=(n * (n - 1) // 2) // 2,
                                   seed=SEED)
            e = g.num_edges()
            for name, func in [("bfs", bfs), ("dfs", dfs_iterative)]:
                record(f"traversal_{density}", name, n, e, "time",
                       timed(lambda: func(g, 0)), "seconds")
                record(f"traversal_{density}", name, n, e, "peak_memory",
                       _traversal_peak_memory(lambda: func(g, 0)), "bytes")


# ---------------------------------------------------------------------------
# 3. Dijkstra: heap vs list PQ, and density effects
# ---------------------------------------------------------------------------

def bench_dijkstra() -> None:
    print("\n=== Dijkstra: heap vs list priority queue (sparse) ===")
    for n in [100, 500, 1_000, 3_000, 10_000]:
        g = generate_sparse_graph(n, weighted=True, seed=SEED)
        e = g.num_edges()
        record("dijkstra_pq", "heap", n, e, "time",
               timed(lambda: dijkstra(g, 0)), "seconds")
        if n <= 3_000:  # list PQ is O(V^2): too slow beyond this
            record("dijkstra_pq", "list", n, e, "time",
                   timed(lambda: dijkstra_list_based(g, 0)), "seconds")

    print("\n=== Dijkstra: density effect at V=1,000 ===")
    v = 1_000
    max_edges = v * (v - 1) // 2
    for label, edges in [("sparse_2V", 2 * v),
                         ("medium_10%", max_edges // 10),
                         ("dense_50%", max_edges // 2)]:
        g = generate_graph(v, edges, weighted=True, seed=SEED)
        e = g.num_edges()
        record("dijkstra_density", f"heap_{label}", v, e, "time",
               timed(lambda: dijkstra(g, 0)), "seconds")
        record("dijkstra_density", f"list_{label}", v, e, "time",
               timed(lambda: dijkstra_list_based(g, 0)), "seconds")


# ---------------------------------------------------------------------------
# 4. Traversal-order visualizations
# ---------------------------------------------------------------------------

def make_visualizations() -> None:
    print("\n=== Traversal visualizations (n=20) ===")
    g = generate_sparse_graph(20, seed=7)
    draw_traversal(g, bfs(g, 0),
                   os.path.join(RESULTS_DIR, "traversal_bfs.png"),
                   title="BFS from node 0 — level-by-level rings")
    draw_traversal(g, dfs_iterative(g, 0),
                   os.path.join(RESULTS_DIR, "traversal_dfs.png"),
                   title="DFS from node 0 — deep probing with backtracking")

    weighted = generate_sparse_graph(20, weighted=True,
                                     weight_range=(1, 9), seed=7)
    distances, preds = dijkstra(weighted, 0)
    reachable = [v for v in weighted.nodes()
                 if v != 0 and distances[v] != float("inf")]
    target = max(reachable, key=lambda v: distances[v])
    path = reconstruct_path(preds, 0, target)
    path_edges = list(zip(path, path[1:]))
    draw_traversal(weighted, path,
                   os.path.join(RESULTS_DIR, "shortest_path.png"),
                   title=f"Dijkstra: shortest path 0 -> {target} "
                         f"(cost {distances[target]:g})",
                   path_edges=path_edges)
    print("  saved traversal_bfs.png, traversal_dfs.png, shortest_path.png")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _series(experiment: str, subject: str, metric: str):
    pts = sorted((r["nodes"], float(r["value"])) for r in ROWS
                 if r["experiment"] == experiment
                 and r["subject"] == subject and r["metric"] == metric)
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_traversals(density: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric, ylabel, ax in [("time", "Traversal time (s)", axes[0]),
                               ("peak_memory", "Peak memory (bytes)",
                                axes[1])]:
        for name, color in [("bfs", "tab:blue"), ("dfs", "tab:orange")]:
            sizes, values = _series(f"traversal_{density}", name, metric)
            ax.plot(sizes, values, marker="o", label=name.upper(),
                    color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Nodes (V)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.replace('_', ' ')} — {density} graphs")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"BFS vs DFS on {density} graphs "
                 f"({'E = 2V' if density == 'sparse' else 'E = ¼V²'})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"bfs_vs_dfs_{density}.png"),
                dpi=150)
    plt.close(fig)


def plot_dijkstra() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for subject, color in [("heap", "tab:blue"), ("list", "tab:red")]:
        sizes, times = _series("dijkstra_pq", subject, "time")
        ax.plot(sizes, times, marker="o", color=color,
                label=f"{subject}-based PQ")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes (V), sparse graphs (E = 2V)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Heap PQ: O((V+E) log V) vs list PQ: O(V²)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    labels = ["sparse_2V", "medium_10%", "dense_50%"]
    x = range(len(labels))
    for pq, color, offset in [("heap", "tab:blue", -0.18),
                              ("list", "tab:red", 0.18)]:
        values = []
        for label in labels:
            for row in ROWS:
                if (row["experiment"] == "dijkstra_density"
                        and row["subject"] == f"{pq}_{label}"):
                    values.append(float(row["value"]))
        ax.bar([i + offset for i in x], values, width=0.36,
               color=color, label=f"{pq}-based PQ")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["sparse\nE=2V", "medium\nE=10% max",
                        "dense\nE=50% max"])
    ax.set_ylabel("Time (s)")
    ax.set_title("Density effect at V = 1,000")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.suptitle("Dijkstra's algorithm performance")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "dijkstra_performance.png"),
                dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start = time.perf_counter()

    bench_representations()
    bench_traversals()
    bench_dijkstra()
    make_visualizations()

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"\nSaved {CSV_PATH} ({len(ROWS)} measurements)")

    plot_traversals("sparse")
    plot_traversals("dense")
    plot_dijkstra()
    print(f"Saved plots to {RESULTS_DIR}")
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
