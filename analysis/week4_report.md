# Week 4 Technical Report: Graph Representations and Traversal Algorithms

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** Graphs — Representations, BFS/DFS, and Dijkstra's Algorithm

## Executive Summary

Representation choice dominated every other factor measured this week:
on a sparse 10,000-node graph, the adjacency matrix consumed **850 MB
against the adjacency list's 2.8 MB** (304×) and ran a full BFS 322×
slower, because its `get_neighbors` must scan an entire V-length row
to find a handful of edges. On top of a sane representation, algorithm
engineering still mattered: swapping Dijkstra's priority queue from a
linear-scan list to the Week 3 binary min-heap made it **18× faster at
V = 3,000** on sparse graphs — an advantage that shrank to 1.2× on
dense graphs, exactly as the O(V²)-vs-O((V+E) log V) analysis
predicts.

## Methodology

**Hardware & environment:** Apple Silicon Mac (arm64), macOS 15.7.3,
CPython 3.13.5 in the project venv, single-threaded. Libraries:
matplotlib for plots, networkx for graph layout only (all graph
algorithms are this project's own implementations), pytest for the
418-test suite.

**Graph generation** (`src/utils/graph_generator.py`): seeded RNG
(seed 42) for full reproducibility. Density vocabulary: *sparse* means
E = 2V (average degree 4, typical of road networks), *dense* means
E = ¼V² to ½V² (an appreciable fraction of all possible edges). Every
generated graph is guaranteed connected by first linking all nodes
with a random spanning tree (V−1 edges, rooted at node 0) and then
adding random extra edges — so every traversal reaches all V nodes and
comparisons across algorithms are fair. Weighted graphs draw uniform
weights from [1, 100].

**Measurement:** `time.perf_counter()` around whole operations;
anything faster than 50 ms is re-run and averaged (up to 200
repetitions) to defeat timer resolution. Traversal memory is the peak
reported by `tracemalloc` around a single traversal, measured in a
separate run from timing so instrumentation cannot pollute the
clock. Representation memory sums `sys.getsizeof` over the actual
storage containers. Raw data: `benchmarks/results/comparison_table.csv`
(71 measurements).

## Results

### Adjacency list vs adjacency matrix (sparse graphs, E = 2V)

| V | List memory | Matrix memory | List: 1k lookups | Matrix: 1k lookups | List: full BFS | Matrix: full BFS |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 29.8 KB | 97.8 KB | 0.37 ms | 1.15 ms | 0.031 ms | 0.119 ms |
| 1,000 | 287 KB | 8.61 MB | 0.34 ms | 12.3 ms | 0.36 ms | 12.1 ms |
| 10,000 | 2.80 MB | **850 MB** | 0.44 ms | 158.6 ms | 4.07 ms | 1,310 ms |

The list's lookup cost is **flat** (~0.4 µs per `get_neighbors`
regardless of V — it returns just the node's ~4 stored edges), while
the matrix's grows **linearly with V** (1.2 µs → 12 µs → 159 µs per
lookup) because it must scan a whole row of mostly-`None` cells. Since
BFS/DFS call `get_neighbors` once per node, matrix-backed traversal is
O(V²) on any graph, and the 850 MB footprint is O(V²) storage made
tangible: 10⁸ cells to represent 20,000 edges. The matrix's virtues —
O(1) `has_edge` and O(1) edge updates — never outweigh that on sparse
data.

### BFS vs DFS: time and memory

| Scenario | V | E | BFS time | DFS time | BFS peak mem | DFS peak mem |
|---|---:|---:|---:|---:|---:|---:|
| sparse | 10,000 | 20,000 | 3.94 ms | 5.44 ms | 698 KB | 783 KB |
| sparse | 50,000 | 100,000 | 29.8 ms | 33.5 ms | 2.79 MB | 3.19 MB |
| dense | 1,000 | 249,750 | 18.1 ms | 34.0 ms | 54.9 KB | 2.11 MB |
| dense | 2,000 | 999,500 | 108 ms | 226 ms | 183 KB | **8.62 MB** |

Both algorithms scaled linearly in V + E on sparse graphs (5× the
nodes from 10k → 50k cost 7.6× the time — linear growth plus cache
effects), confirming the O(V + E) analysis. The surprise is on dense
graphs, where DFS ran ~2× slower and used up to **47× more peak
memory** than BFS — the opposite of the textbook heuristic that DFS is
the memory-light option. The cause is implementation-level: this
iterative DFS pushes every not-yet-visited neighbor onto the stack and
filters stale entries at pop time, so on a dense graph the stack can
briefly hold O(E) entries. BFS marks nodes visited at *enqueue* time,
which bounds its queue at O(V) no matter the density. The lesson:
asymptotic class comes from the algorithm, but constants — and even
which structure grows — come from implementation decisions.

### Dijkstra: heap vs list priority queue (sparse, E = 2V)

| V | Heap PQ | List PQ | Speedup |
|---:|---:|---:|---:|
| 100 | 0.20 ms | 0.25 ms | 1.2× |
| 1,000 | 2.83 ms | 18.4 ms | 6.5× |
| 3,000 | 9.00 ms | 162.9 ms | **18.1×** |
| 10,000 | 46.1 ms | (not run — O(V²)) | — |

Growth rates match theory cleanly: tripling V from 1,000 → 3,000 grew
the list version 8.9× (≈ 3², quadratic) but the heap version only
3.2× (≈ V log V). The list PQ's cost is dominated by V linear scans
for the minimum; the heap replaces each scan with an O(log V)
extract — the same Week 3 MinHeap, now doing production work inside a
graph algorithm.

### Dijkstra: the density effect (V = 1,000)

| Density | E | Heap PQ | List PQ | Heap advantage |
|---|---:|---:|---:|---:|
| sparse (E = 2V) | 2,000 | 2.75 ms | 18.1 ms | 6.6× |
| medium (10% of max) | 49,950 | 15.8 ms | 25.0 ms | 1.6× |
| dense (50% of max) | 249,750 | 44.8 ms | 53.8 ms | 1.2× |

The heap's advantage **shrinks as density rises**, converging near
parity on dense graphs. The reason: the list version's O(V²) already
touches every node pair, and once E approaches V², the graph itself
has ~V² edges to relax — so both algorithms are doing Θ(V²) edge work
and the heap merely adds a log factor to each of its E pushes. The
priority-queue upgrade is a *sparse-graph* optimization, which is
fortunate, because real-world graphs (roads, social networks, the
web) are overwhelmingly sparse.

## Discussion

**When BFS beats DFS and vice versa.** On raw traversal time they are
the same complexity class, and our sparse-graph times differ by only
~13%. The choice is really about *what the order guarantees*: BFS
visits nodes in increasing hop distance, so it is the right tool for
shortest unweighted paths, "degrees of separation," and level-based
processing; DFS's plunge-then-backtrack order is what cycle
detection, topological sorting, and connectivity decompositions need.
Our dense-graph measurements add an engineering criterion: with this
stack discipline, prefer BFS when density is high and memory is tight,
and prefer iterative DFS over recursive whenever paths can exceed
Python's ~1,000-frame recursion limit (the test suite demonstrates a
5,000-node path that recursion cannot survive).

**How density shapes everything.** Density decided the representation
verdict (the matrix's O(V²) memory is unusable at 10k nodes for 20k
edges), inverted DFS's memory profile, and erased the heap PQ's
advantage. A single number — E relative to V² — predicts more about
observed performance than any other property measured this week.

**Practical implications.** These effects are why large-scale systems
are built the way they are: social networks (billions of nodes,
average degree in the hundreds — extremely sparse) store adjacency
lists and run BFS variants for friend suggestions; GPS routing runs
heap-based Dijkstra (and its A* descendant) over sparse road graphs,
where our 18× PQ speedup is the difference between interactive and
sluggish; AI planners searching state spaces choose BFS or DFS by
memory budget, precisely the trade-off our dense-graph memory table
quantifies.

## Visualization Summary

- `traversal_bfs.png` / `traversal_dfs.png` — the same 20-node graph,
  nodes numbered and shaded by visit order (light = early). BFS's
  shading radiates outward from node 0 in rings; DFS's follows long
  chains into the periphery before backtracking. Seeing the two
  orders on an identical layout is the fastest way to internalize
  queue-vs-stack behavior.
- `shortest_path.png` — Dijkstra's predecessor map reconstructed into
  an actual route (highlighted in red) to the most expensive-to-reach
  node, over the weighted version of the same graph.
- `bfs_vs_dfs_sparse.png` / `bfs_vs_dfs_dense.png` — time and peak
  memory vs V; the dense memory panel shows DFS's O(E) stack growth
  diverging from BFS's O(V) queue.
- `dijkstra_performance.png` — left: heap vs list PQ scaling on
  log-log axes (the widening gap is the log-factor difference);
  right: the density bars showing the two implementations converging.

## Conclusion

Week 4 closes the loop the course has been building: Week 1–2's
algorithms and Week 3's structures are not separate topics but one
discipline. Dijkstra's algorithm *is* a graph loop wrapped around a
priority queue — its complexity class changed the moment the Week 3
MinHeap was substituted for a list. The graph itself is just a
dictionary of dictionaries (or a grid), and which one you pick sets a
floor under every algorithm that runs on it: no traversal can beat its
representation's `get_neighbors` cost. The unifying insight is that
structure and connectivity — sparse vs dense, list vs matrix, queue vs
stack — determine computational efficiency before a single line of
algorithm code runs.

## References

1. Amakobe, M. *Advanced Algorithms: A Journey Through Computational
   Problem Solving* — graph representations, traversals, and shortest
   paths chapters; §1.6 (Algorithm Laboratory).
2. Python Software Foundation: `collections.deque`, `tracemalloc`,
   `time.perf_counter()` documentation, Python 3.13.
3. networkx documentation (v3.6) — used for layout/drawing only.
4. Project reports: `analysis/week2_report.md`,
   `analysis/week3_report.md`.
