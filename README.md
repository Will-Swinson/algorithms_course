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
src/randomized/randomized_quicksort.py
                              Random vs first-element pivot QuickSort with
                              comparison/depth counters (Week 7)
src/randomized/randomized_selection.py
                              QuickSelect, median-of-medians, sort baseline
                              (Week 7)
src/randomized/montecarlo_primality.py
                              Fermat and Miller-Rabin tests, trial division,
                              exact liar fractions (Week 7)
src/randomized/bloom_filter.py
                              Bloom filter: salted double hashing, optimal
                              sizing, FP-rate estimates (Week 7)
src/randomized/minhash_similarity.py
                              MinHash signatures, Jaccard, text shingles
                              (Week 7)
src/complexity/reduction_demo.py
                              Polynomial-time reductions with logged
                              steps: 3-SAT -> Vertex Cover -> Independent
                              Set -> Clique, 3-SAT -> Subset Sum;
                              verifiers, validation, P/NP catalog (Week 8)
src/complexity/satisfiability_solver.py
                              SAT: brute force, backtracking/DPLL,
                              linear-time 2-SAT, WalkSAT (Week 8)
src/complexity/subset_sum_backtracking.py
                              Subset Sum: backtracking, bitset DP, greedy
                              1/2-approximation (Week 8)
src/complexity/vertex_cover_approx.py
                              Vertex Cover: matching 2-approximation,
                              max-degree greedy, branch and bound (Week 8)
src/complexity/traveling_salesman_approx.py
                              TSP: nearest neighbor, MST double-tree,
                              2-opt, MST lower bound (Week 8)
src/approximation/vertex_cover_2approx.py
                              Vertex Cover: greedy edge picking with its
                              matching certificate, LP rounding (Week 9)
src/approximation/set_cover_lpround.py
                              Set Cover: LP relaxation, deterministic and
                              randomized rounding, greedy, exact (Week 9)
src/approximation/facility_location_greedy.py
                              Facility location: star greedy, local search,
                              exact enumeration (Week 9)
src/approximation/tsp_metric_approx.py
                              Metric TSP: MST doubling, Christofides,
                              Euler circuits (Week 9)
src/approximation/maxcut_randomized.py
                              Max-Cut: random cut, derandomized cut, local
                              search, exact enumeration (Week 9)
src/flow/core/network.py      FlowNetwork: residual graph in edge arrays,
                              automatic twin edges, min cut (Week 10)
src/flow/core/edmonds_karp.py Edmonds-Karp (BFS augmenting paths) and
                              capacity scaling (Week 10)
src/flow/core/push_relabel.py Push-relabel: FIFO and highest-label, gap
                              and global-relabel heuristics (Week 10)
src/flow/core/min_cost_flow.py
                              Min-cost max-flow: successive shortest
                              paths with Johnson potentials (Week 10)
src/flow/applications/        Bipartite matching + assignment, graph-cut
                              image segmentation, scheduling (deadlines,
                              staffing, tournament elimination) (Week 10)
src/flow/visualization/       Network/matching/segmentation/Gantt plots,
                              EK and push-relabel animations (Week 10)
src/flow/utils/               Generators, validators, DIMACS/edge-list/
                              image readers (Week 10)
src/flow/main.py              Flow solver CLI (Week 10)
src/strings/core/kmp.py       KMP: prefix function, linear-time search,
                              fallback trace; brute-force baseline (Week 11)
src/strings/core/rabin_karp.py
                              Rabin-Karp: verified / Monte Carlo search,
                              multi-pattern batches (Week 11)
src/strings/core/suffix_array.py
                              Suffix array (numpy prefix doubling), Kasai
                              LCP, O(m log n) search, LCS (Week 11)
src/strings/core/suffix_tree.py
                              Ukkonen suffix tree with active-point trace,
                              Graphviz DOT export (Week 11)
src/strings/core/search_engine.py
                              StringSearchEngine: one API over all four,
                              timing stats, method recommendation (Week 11)
src/strings/applications/     Plagiarism detection, DNA search (FASTA,
                              mutations), top-k autocomplete (Week 11)
src/strings/visualization/    SA/LCP plots, KMP traces, suffix tree and
                              Ukkonen phase drawings (Week 11)
src/strings/utils/            Rolling hashes, text loading/streaming,
                              synthetic text/DNA generators (Week 11)
src/utils/benchmark.py        Reusable benchmarking framework
src/utils/graph_generator.py  Seeded random graph generators (Week 4)
src/utils/timer.py            Call/recursion-depth counter, timing, and
                              tracemalloc memory helpers (Week 5)
src/utils/matrix_utils.py     Matrix chains, counted matrix multiplication,
                              graph -> weight matrix, city distances (Week 6)
src/utils/bitmask_utils.py    Bit-manipulation helpers for set states (Week 6)
src/utils/random_utils.py     Seeded arrays, strings, and Jaccard-controlled
                              set pairs (Week 7)
src/utils/probability_analysis.py
                              Expected comparisons, error bounds, FP rates,
                              mean/variance, Wilson intervals (Week 7)
src/utils/graph_utils.py      Edge-list graphs: random G(n,p), K(a,b),
                              complements, Week 4 Graph conversion (Week 8),
                              weighted edges (Week 9)
src/utils/lp_solver.py        Two-phase simplex LP solver on numpy (Week 9)
src/utils/visualization.py    networkx traversal-order drawings (Week 4),
                              recursive-vs-DP comparison plots (Week 5),
                              metric grids and DP-table heatmaps (Week 6),
                              histogram/scatter/band panels (Week 7),
                              highlighted gadget graphs (Week 8),
                              TSP tour drawings (Week 9)
tests/                        Pytest test suite
benchmarks/                   Benchmark runners + raw results (CSV, PNG)
analysis/                     Weekly technical reports (Weeks 2-11)
examples/                     Runnable demos (week2_demo.py .. week11_demo.py)
examples/flow_inputs/         Sample inputs for the flow CLI (DIMACS,
                              edge list, matching, image, schedules)
examples/string_inputs/       Sample FASTA genome for the DNA search demo
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
`benchmarks/results/`. The CSV shares its name with earlier weeks'
(as the assignment specifies); each run overwrites it, and earlier
versions remain in git history.

Week 7 (randomized algorithms vs deterministic counterparts):

```bash
python benchmarks/week7_randomized_benchmark.py          # ~1 min
python benchmarks/week7_randomized_benchmark.py --quick  # ~20 s
```

Runs every randomized algorithm many times with fresh seeds on a fixed
input, recording mean/variance of runtime, comparison counts against
their exact expected values, Fermat/Miller-Rabin error rates, Bloom
filter false-positive rates, and MinHash estimation error. Writes
`quicksort_comparison.png`, `selection_runtime.png`,
`primality_accuracy.png`, `bloom_filter_fp_rate.png`,
`minhash_similarity_plot.png`, and `summary_table.csv` to
`benchmarks/results/`.

Week 8 (complexity: exact exponential solvers vs polynomial heuristics
and approximations):

```bash
python benchmarks/week8_complexity_benchmark.py          # ~40 s
python benchmarks/week8_complexity_benchmark.py --quick  # ~10 s
```

Measures exponential growth of backtracking Subset Sum, brute-force/
DPLL SAT, branch-and-bound Vertex Cover, and brute-force/Held-Karp TSP
against greedy, WalkSAT, 2-approximation, nearest-neighbor, MST, and
2-opt heuristics; approximation ratios against exact optima; and
validates every reduction on 300 random formulas. Writes
`runtime_growth.png`, `approximation_quality.png`,
`tradeoff_curves.png`, `reduction_3sat_vertex_cover.png`, and
`comparison_table.csv` to `benchmarks/results/`. Weeks 2-4 and 6 use
the same CSV name (as each assignment specifies); each run overwrites
it, and earlier versions remain in git history.

Week 9 (approximation algorithms vs exact optima and their guarantees):

```bash
python benchmarks/week9_approx_benchmark.py          # ~1 min
python benchmarks/week9_approx_benchmark.py --quick  # ~10 s
```

Compares Vertex Cover (edge picking, LP rounding), Set Cover (greedy,
deterministic and randomized LP rounding), facility location (greedy,
local search), metric TSP (MST doubling, Christofides), and Max-Cut
(random, derandomized, local search) against exact solvers on small
instances and against LP/MST/total-weight bounds on large ones, with
30-1,000 randomized trials for means and variances. Writes
`approx_ratios.png`, `cost_vs_optimal.png`,
`randomized_maxcut_results.png`, `set_cover_rounding.png`,
`approx_runtime.png`, `tsp_tours.png`, and `comparison_table.csv` (now
holding the Week 9 data) to `benchmarks/results/`.

Week 10 (network flow: max-flow, min-cost flow, applications):

```bash
python benchmarks/week10_flow_benchmark.py          # ~5 min
python benchmarks/week10_flow_benchmark.py --quick  # ~45 s
```

Times Edmonds-Karp, capacity scaling, push-relabel (FIFO and
highest-label), min-cost max-flow, and networkx's preflow_push on
sparse (1k-50k nodes), dense (up to ~290k arcs) and segmentation-grid
networks; measures peak memory, augmenting-path counts, push-relabel
convergence and a gap/global-relabel ablation; benchmarks matching,
segmentation and scheduling. Every flow is checked by the validators.
Writes `flow_runtime.png`, `flow_memory.png`, `augmentations.png`,
`pr_convergence.png`, `flow_applications.png`, example figures
(`network_example.png`, `matching_example.png`,
`segmentation_example.png`, `flow_storyboard.png`), the animations
`ek_animation.gif` and `pr_animation.gif`, and
`flow_comparison_table.csv` to `benchmarks/results/`.

Week 11 (string search: KMP, Rabin-Karp, suffix array, suffix tree):

```bash
python benchmarks/week11_string_benchmark.py          # ~4 min
python benchmarks/week11_string_benchmark.py --quick  # ~40 s
```

Compares brute force, KMP, Rabin-Karp, suffix array, suffix tree, and
Python's `str.find` (C reference) on English-like text and DNA from
10 KB to 5 MB: preprocessing time, search latency, index memory, the
query count at which an index pays off, pattern-length effects, and
the a^n worst case. Measures Rabin-Karp collision rates against
(n - m + 1)/q, bad base/modulus choices, and batch matching; SA/LCP
construction and LCP statistics; and the applications. Every answer
is checked against `str.find`. Writes `runtime_comparison.png`,
`collision_analysis.png`, `sa_lcp_stats.png`,
`heatmap_dataset_sizes.png`, `string_applications.png`, example
figures (`suffix_array_example.png`, `suffix_tree_example.png` and
`.dot`, `ukkonen_steps.png`, `kmp_trace.png`), and
`string_comparison_table.csv` to `benchmarks/results/`.

## Flow Solver CLI

```bash
python -m src.flow.main --ek -i examples/flow_inputs/clrs.max --validate
python -m src.flow.main --mcf -i examples/flow_inputs/shipping.min
python -m src.flow.main --pr --random 50000 --validate
python -m src.flow.main --match examples/flow_inputs/matching.txt --plot match.png
python -m src.flow.main --segment examples/flow_inputs/shapes.png --plot seg.png
python -m src.flow.main --schedule examples/flow_inputs/jobs.json --plot gantt.png
python -m src.flow.main --pr -i examples/flow_inputs/clrs.max --animate pr.gif
```

Solvers: `--ek`, `--pr` (`--variant fifo|highest`), `--mcf`,
`--scaling`. Applications: `--match`, `--segment`, `--schedule`
(job deadlines, job-machine hours, staffing, tournament elimination;
JSON formats in `python -m src.flow.main --help`).

## Demos

```bash
python examples/week2_demo.py   # the five sorting algorithms
python examples/week3_demo.py   # priority queue, AVL balance, hashing
python examples/week4_demo.py   # graph representations, BFS/DFS, Dijkstra
python examples/week5_demo.py   # DP: call explosion, project picking, DNA LCS
python examples/week6_demo.py   # 1D knapsack, matrix chains, Floyd-Warshall, TSP
python examples/week7_demo.py   # random pivots, Carmichael numbers, Bloom, MinHash
python examples/week8_demo.py   # P vs NP catalog, reductions, blow-up, approximations
python examples/week9_demo.py   # certified ratios, LP rounding, Christofides, Max-Cut
python examples/week10_demo.py  # max-flow solvers, min cut, matching, segmentation
python examples/week11_demo.py  # KMP, Rabin-Karp, suffix array/tree, search engine
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
- [x] Week 7: Randomized algorithms — randomized QuickSort and
      QuickSelect (Las Vegas), Fermat/Miller-Rabin primality, Bloom
      filters, and MinHash (Monte Carlo); error-rate and variance
      benchmarks against theory, and technical report
- [x] Week 8: Computational complexity — logged polynomial-time
      reductions (3-SAT -> Vertex Cover/Clique/Subset Sum) validated
      against exact solvers, backtracking Subset Sum and DPLL SAT vs
      greedy/WalkSAT, Vertex Cover and TSP approximations, runtime and
      approximation-ratio benchmarks, and technical report
- [x] Week 9: Approximation algorithms — Vertex Cover 2-approximation
      with self-certifying matching, Set Cover LP relaxation (own
      simplex solver) with deterministic and randomized rounding,
      greedy and local-search facility location, MST doubling and
      Christofides for metric TSP, randomized Max-Cut with
      derandomization and local search; benchmarks and report
- [x] Week 10: Network flow — modular flow solver (residual-graph
      network, Edmonds-Karp, capacity scaling, push-relabel FIFO and
      highest-label, min-cost max-flow), matching, graph-cut
      segmentation, scheduling, validators, animations, CLI,
      benchmarks to 50k nodes, and technical report
- [x] Week 11: String algorithms — KMP, Rabin-Karp (rolling hash,
      collisions, batches), suffix array + Kasai LCP, Ukkonen suffix
      tree, unified StringSearchEngine with recommendations,
      plagiarism detection, DNA search, autocomplete; benchmarks to
      5 MB and technical report

## Author

Will Swinson
