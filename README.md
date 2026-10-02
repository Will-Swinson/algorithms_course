# Advanced Algorithms Course

## Description

My implementation of algorithms studied in Advanced Algorithms course,
following Chapter 1.6 ("Setting Up Your Algorithm Laboratory") of
_Advanced Algorithms: A Journey Through Computational Problem Solving_
by Dr. Moody Amakobe.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify the environment:

```bash
python scripts/check_environment.py
```

## Project Structure

```
src/sorting/basic_sorts.py    Bubble, selection, and insertion sort (Week 1)
src/sorting/merge_sort.py     Merge sort with merge helper (Week 2)
src/sorting/quick_sort.py     Randomized quicksort: three-way partition,
                              insertion-sort cutoff (Week 2)
src/structures/heap.py        MinHeap, MaxHeap, PriorityQueue (Week 3)
src/structures/avl_tree.py    Self-balancing AVL tree (Week 3)
src/structures/hash_table.py  Chaining + linear-probing hash tables (Week 3)
src/graphs/graph.py           Graph: adjacency list + matrix (Week 4)
src/graphs/bfs.py             Breadth-first search (Week 4)
src/graphs/dfs.py             Depth-first search, iterative + recursive (Week 4)
src/graphs/dijkstra.py        Dijkstra via Week 3 MinHeap (Week 4)
src/dp/fibonacci.py           Fibonacci: naive, memoized, tabulated (Week 5)
src/dp/knapsack.py            0/1 knapsack: recursive, memoized, tabulated,
                              trace_solution() item recovery (Week 5)
src/dp/lcs.py                 Longest common subsequence: recursive,
                              memoized, tabulated + reconstruction (Week 5)
src/dp_advanced/space_optimized_knapsack.py
                              O(W)-space 0/1 knapsack: two-row and in-place
                              1D versions, bitset item recovery (Week 6)
src/dp_advanced/matrix_chain_multiplication.py
                              Interval DP: recursive, memoized, bottom-up,
                              optimal parenthesization (Week 6)
src/dp_advanced/floyd_warshall.py
                              All-pairs shortest paths, negative edges,
                              negative-cycle detection (Week 6)
src/dp_advanced/bitmask_traveling_salesman.py
                              Held-Karp bitmask DP vs brute-force TSP (Week 6)
src/utils/benchmark.py        Reusable benchmarking framework
src/utils/graph_generator.py  Seeded random graph generators (Week 4)
src/utils/timer.py            Call/recursion-depth counter, timing, and
                              tracemalloc memory helpers (Week 5)
src/utils/matrix_utils.py     Matrix chains, counted matrix multiplication,
                              graph -> weight matrix, city distances (Week 6)
src/utils/bitmask_utils.py    Bit-manipulation helpers for set states (Week 6)
src/utils/visualization.py    networkx traversal-order drawings (Week 4),
                              recursive-vs-DP comparison plots (Week 5),
                              metric grids and DP-table heatmaps (Week 6)
tests/                        Pytest test suite
benchmarks/                   Benchmark runners + raw results (CSV, PNG)
analysis/                     Weekly technical reports (Weeks 2-6)
examples/                     Runnable demos (week2_demo.py .. week6_demo.py)
docs/performance_analysis.md  Week 1 performance report with charts
docs/images/                  Week 1 benchmark visualizations
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

Week 1 (basic sorts):

```bash
python benchmarks/run_sorting_benchmarks.py
```

Regenerates the charts in `docs/images/` and the raw timing data in
`benchmarks/results/sorting_benchmarks.csv`.

Week 2 (all five algorithms, 6 data types, n up to 50,000):

```bash
python benchmarks/week2_performance.py
```

Writes per-data-type plots, `comparison_table.csv`, and
`complexity_summary.csv` to `benchmarks/results/`. The full run takes
roughly an hour (the O(n²) sorts at n = 50,000 dominate); results are
appended incrementally, so an interrupted run resumes where it left off.

Week 3 (heaps, AVL tree, hash tables vs Python built-ins, n up to 10⁶):

```bash
python benchmarks/week3_structures_benchmark.py
```

Writes `heap_performance.png`, `tree_performance.png`,
`hash_performance.png`, `complexity_shapes.png`, and
`comparison_table.csv` to `benchmarks/results/` (runs in under a
minute).

Week 4 (graph representations, BFS/DFS, Dijkstra):

```bash
python benchmarks/week4_graph_benchmark.py
```

Writes `bfs_vs_dfs_sparse.png`, `bfs_vs_dfs_dense.png`,
`dijkstra_performance.png`, traversal-order drawings
(`traversal_bfs.png`, `traversal_dfs.png`, `shortest_path.png`), and
`comparison_table.csv` to `benchmarks/results/` (runs in under a
minute).

Week 5 (dynamic programming: Fibonacci, knapsack, LCS):

```bash
python benchmarks/week5_dp_benchmark.py          # ~2 min
python benchmarks/week5_dp_benchmark.py --quick  # skips naive fib(40), fib(45)
```

Compares naive recursion against memoization and tabulation on time,
calls, recursion depth, and peak memory. Writes
`fibonacci_comparison.png`, `knapsack_performance.png`,
`lcs_performance.png`, and `dp_vs_recursive_table.csv` to
`benchmarks/results/`. Most of the runtime is naive `fib(45)` alone
(~80 s, 3.7 billion calls).

Week 6 (advanced DP: space optimization, interval DP, Floyd-Warshall,
bitmask TSP):

```bash
python benchmarks/week6_dp_advanced_benchmark.py          # ~8 min
python benchmarks/week6_dp_advanced_benchmark.py --quick  # skips slowest cases
```

Writes `knapsack_space_comparison.png`, `mcm_performance.png`,
`mcm_dp_table.png`, `floyd_warshall_scaling.png`,
`tsp_bitmask_runtime.png`, and `comparison_table.csv` to
`benchmarks/results/`. The CSV shares its name with Week 4's
(as the assignment specifies), so it now holds the Week 6 data; Week
4's version remains in git history.

## Demos

```bash
python examples/week2_demo.py   # the five sorting algorithms
python examples/week3_demo.py   # priority queue, AVL balance, hashing
python examples/week4_demo.py   # graph representations, BFS/DFS, Dijkstra
python examples/week5_demo.py   # DP: call explosion, project picking, DNA LCS
python examples/week6_demo.py   # 1D knapsack, matrix chains, Floyd-Warshall, TSP
```

## Current Progress

- [x] Week 1: Environment setup, basic sorting algorithms, benchmarking
      framework, test suite, and performance analysis
- [x] Week 2: Divide and conquer — merge sort, randomized quicksort,
      full 5-algorithm benchmark comparison, and technical report
- [x] Week 3: Core data structures — binary heaps + priority queue,
      AVL tree, hash tables (chaining and linear probing), benchmark
      comparison against Python built-ins, and technical report
- [x] Week 4: Graphs — adjacency list/matrix representations, BFS and
      DFS traversals, Dijkstra's shortest paths (heap vs list PQ),
      traversal visualizations, and technical report
- [x] Week 5: Dynamic programming — Fibonacci, 0/1 knapsack, and LCS
      as naive recursion, memoization, and tabulation; call/depth/
      memory benchmarking, speedup visualizations, and technical report
- [x] Week 6: Advanced DP — O(W)-space knapsack, matrix chain
      multiplication (interval DP), Floyd-Warshall vs all-pairs
      Dijkstra, bitmask TSP vs brute force, benchmarks, and report

## Author

Will Swinson
