# Week 6 Technical Report: Advanced Dynamic Programming

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** Space Optimization, Interval DP, Floyd–Warshall, and Bitmask TSP

## Executive Summary

Week 5 showed that DP makes exponential problems polynomial. This week
looked at what limits DP after that: memory, scale, and states with
more than one dimension. Each technique addressed one of these limits.

- **Space optimization:** compressing the knapsack table to a single
  row cut peak memory **304×** (11.6 MiB → 39 KiB at 800 items). The
  1D version also ran 1.5× faster.
- **Interval DP:** matrix chain multiplication went from 1.2 s by
  recursion to 60 µs bottom-up at 16 matrices, a **20,000×** speedup.
  The orders it found needed up to 15× fewer multiplications than
  simple left-to-right evaluation.
- **Graph DP:** Floyd–Warshall handled negative edge weights, which
  Week 4's Dijkstra rejects. It also beat repeated Dijkstra on small
  and dense graphs, but was 3.6× slower on a sparse 500-vertex graph.
- **State compression:** bitmask DP solved 12-city TSP **1,145×**
  faster than brute force and reached 18 cities in 1.3 s, at the cost
  of 130 MiB of DP table.

## Methodology

**Hardware & environment:** Apple M3 Pro (12 cores, 18 GB), macOS
15.7.3, CPython 3.13.5, single-threaded. matplotlib for plots, pytest
for the 1,129-test suite (423 tests new this week).

**Inputs** (all seeded):
- *Knapsack:* weights 1–50 and values 1–100. One sweep varies n from
  50 to 800 at W = 1,000; another varies W from 500 to 10,000 at
  n = 100.
- *Matrix chains:* random dimensions 5–100. Recursion ran up to 16
  matrices and DP up to 300.
- *Floyd–Warshall:* directed graphs with weights 1–100 at V = 50, 100,
  200 and 500. Sparse graphs have E = 4V; dense graphs have 25% of all
  possible edges.
- *TSP:* random points in a 100 × 100 square with Euclidean distances.
  Brute force ran up to 12 cities and the bitmask DP up to 18.

**Measurement:** this reuses the Week 5 harness (`src/utils/timer.py`).
Time, work counts and peak memory each come from a separate run, and
anything under 0.5 s is repeated and averaged. Memory is `tracemalloc`
peak heap usage. Every run also checks correctness: all methods must
return the same answer, and Floyd–Warshall's full distance matrix must
match all-pairs Dijkstra. Raw data is in
`benchmarks/results/comparison_table.csv` (145 rows). The full run
takes 7.6 minutes.

## Results

### Standard vs space-optimized knapsack (`knapsack_space_comparison.png`)

| Configuration | 2D table (Week 5) | 1D row | Memory saved | Speedup |
|---|---:|---:|---:|---:|
| n = 50, W = 1,000 | 2.6 ms, 1.3 MiB | 2.1 ms, 39 KiB | 34× | 1.3× |
| n = 800, W = 1,000 | 43.7 ms, 11.6 MiB | 28.9 ms, **39 KiB** | **304×** | 1.5× |
| n = 100, W = 10,000 | 55.7 ms, 35.9 MiB | 41.1 ms, 391 KiB | 94× | 1.4× |
| with item recovery, n = 800 | 41.3 ms, 11.6 MiB | 30.6 ms, 97 KiB | 122× | 1.35× |

The 1D row's memory stayed at **39 KiB for every n from 50 to 800**,
while the table grew linearly with n. That is the O(n × W) → O(W)
change measured directly. Along the capacity axis both versions grow
with W, and the savings ratio stays near n. The two-row version used
exactly twice the 1D row's memory and the same time as the full
table. The 1D version is faster as well as smaller: it skips
capacities below the item's weight and writes a cell only when the
value improves, whereas the table copies every cell down from the row
above.

### Matrix chain multiplication (`mcm_performance.png`, `mcm_dp_table.png`)

| n | Recursive | Calls | Memoized | Bottom-up | Speedup (bottom-up) |
|---:|---:|---:|---:|---:|---:|
| 12 | 14.6 ms | 177,147 | 54 µs | 29 µs | 501× |
| 14 | 127 ms | 1,594,323 | 81 µs | 45 µs | 2,813× |
| 16 | **1.20 s** | 14,348,907 | 114 µs | 60 µs | **20,124×** |
| 100 | — | — | 25.4 ms | 11.4 ms | — |
| 300 | — | — | 830 ms | 304 ms | — |

The recursive call count is exactly 3^(n−1), and each +2 matrices
multiplied recursive time by 8.7–9.4×, close to 3² = 9. Bottom-up DP grew
3.41× from n = 200 to 300, against the (1.5)³ = 3.375 that O(n³)
predicts. Memory went up 2.34× against the 2.25 that O(n²) predicts.
The optimal orders matter in practice too. For 100 random matrices the
DP's order needs 1.23 million scalar multiplications, against 18.6
million for left-to-right evaluation (15×). In one case (n = 12) the
left-to-right order happened to be optimal already.

### Floyd–Warshall vs all-pairs Dijkstra (`floyd_warshall_scaling.png`)

| V | FW sparse | Dijkstra sparse | FW dense | Dijkstra dense |
|---:|---:|---:|---:|---:|
| 50 | **2.6 ms** | 3.8 ms | **3.8 ms** | 7.6 ms |
| 100 | 19.2 ms | 18.6 ms | **31.0 ms** | 47.0 ms |
| 200 | 137 ms | **79 ms** | **242 ms** | 302 ms |
| 500 | 2.05 s | **0.57 s** | 4.70 s | 4.56 s |

On dense graphs, relaxations attempted followed V³ almost exactly
(124.0 million at V = 500, against 125 million for V³). On sparse
graphs only 51% of the V³ relaxations ran, because a row is skipped
whenever vertex i cannot yet reach k. Time grew a little faster than
V³: 19.4× from V = 200 to 500, against 15.6× predicted. The extra
likely comes from memory effects as the matrix grows to 11.5 MiB.

### MCM vs TSP scalability (`tsp_bitmask_runtime.png`)

| Cities | Brute force | Tours | Bitmask DP | Transitions | DP memory | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 20 µs | 120 | 31 µs | 165 | 10 KiB | 0.6× |
| 10 | 91 ms | 362,880 | 1.5 ms | 9,225 | 337 KiB | 60× |
| 12 | **11.2 s** | 39,916,800 | **9.8 ms** | 56,331 | 1.5 MiB | **1,145×** |
| 18 | — | — | 1.32 s | 8,912,913 | 130 MiB | — |

Each extra city multiplied brute-force time by about n − 1 (12× going
from 11 to 12 cities). The bitmask DP's time grew only 2.1–2.4× per
city, but its memory doubled with every city, as the O(n·2ⁿ) table
predicts.
Below 7 cities brute force was faster, because building the table
costs more than it saves. Comparing the two DP problems shows the gap
between polynomial and exponential DP. MCM handled 300 matrices in
0.3 s and would grow about 3.4× with 50% more matrices. Bitmask TSP
doubles in time and memory with every single city added.

## Discussion

**When to prioritize space optimization.** Time costs at most a slower
answer. Memory has a hard limit, and a program that exceeds it gets no
answer at all. With n = 10⁴ items and W = 10⁶, the knapsack table has
10¹⁰ cells and cannot be allocated, while the 1D row has only 10⁶.
Optimizing for space makes sense when every DP state depends only on a
fixed window of earlier states, such as the previous row. It also
makes sense when only the final value is needed. When the actual
solution is needed, compression costs something: the backward walk
that recovers it needs information the single row discards. Here, item
recovery cost one bit per decision instead of a full table cell, so it
was still 122× smaller. Compression needs a known evaluation order,
which tabulation has and memoization does not. That makes tabulation
the version to start from.

**Readability vs efficiency.** The optimized code depends on reasoning
that the code itself does not show. Reversing the 1D knapsack's loop
direction silently produces the unbounded knapsack (the demo returns
14 instead of 9 that way). Floyd–Warshall's in-place update is correct
only because row k and column k cannot change during stage k. Both
arguments are written in the docstrings and locked in by tests.
Performance problems can also hide in code that looks clean. The first
version of the bitset item recovery used `bits |= 1 << w`, which looks
constant-time. Because every update copies the whole growing integer,
it was actually quadratic in W and 2.5× slower than the table it
replaced. Packing each row once fixed it.

**Insights from state compression and interval DP.** Both depend on
choosing a state that keeps only what the rest of the computation
needs. For TSP, the cheapest way to finish a tour depends on *which*
cities have been visited and where the salesman is now, but not on the
order of the visits. Recognizing that collapses 39.9 million tours
into 56,331 transitions over (set, city) states. Interval DP makes a
similar observation: every parenthesization ends with one final
multiplication, so the subproblems are contiguous ranges, solved in
order of increasing length. For Floyd–Warshall, the deciding factor is
density. Repeated Dijkstra is asymptotically better on sparse graphs,
and it won by 3.6× at V = 500. On dense graphs Floyd–Warshall's simple
inner loop kept up with Dijkstra's heap operations despite the
theoretical gap. Floyd–Warshall is also the only one of the two that
handles negative weights.

**Real-world applications.** *Compilers and numerical libraries*
choose the order of chained matrix or tensor products using the same
cost model as MCM. Database query optimizers have long used DP over
subsets of tables to choose join orders, which is the same
subset-state idea as bitmask TSP. *Route optimization* uses
Held–Karp-style DP for exact solutions over small sets of stops, and
uses precomputed all-pairs distance tables like Floyd–Warshall's
output. *Genome alignment* depends on the knapsack row trick:
Hirschberg's linear-space alignment keeps two rows of the alignment
table, which is what makes aligning long sequences possible.

## Conclusion

Week 5 started from recursion and showed that remembering answers
removes exponential blowup. This week showed that what gets remembered,
and in what shape, decides how far DP can scale. A table can shrink to
one row, a set of visited cities can become a single integer, and the
order of a computation can make an in-place update safe. In each case
the gain came from looking closely at the recurrence to see what it
actually reads, and storing only that.
