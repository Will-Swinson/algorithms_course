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
src/utils/benchmark.py        Reusable benchmarking framework
src/utils/graph_generator.py  Seeded random graph generators (Week 4)
src/utils/visualization.py    networkx traversal-order drawings (Week 4)
tests/                        Pytest test suite
benchmarks/                   Benchmark runners + raw results (CSV, PNG)
analysis/                     Week 2 and Week 3 technical reports
examples/                     Runnable demos (week2_demo.py, week3_demo.py)
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

## Demos

```bash
python examples/week2_demo.py   # the five sorting algorithms
python examples/week3_demo.py   # priority queue, AVL balance, hashing
python examples/week4_demo.py   # graph representations, BFS/DFS, Dijkstra
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

## Author

Will Swinson
