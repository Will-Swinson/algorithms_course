# Week 8 Technical Report: Computational Complexity and Approximation

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** Reductions, NP-complete solvers, and approximation algorithms

## Executive Summary

Every exact algorithm here for an NP-complete problem grew
exponentially, and every heuristic stayed polynomial and close to
optimal.

- **The explosion is real:** Subset Sum backtracking on a "no" instance
  took 32 µs at n = 10 and 1.38 s at n = 28 (≈ 1.8ⁿ), while greedy took
  2.9 µs. Brute-force SAT grew about 2× per added variable.
- **Reductions are cheap and correct:** 3-SAT → Vertex Cover built a
  29,000-element graph from 3,000 clauses in 8 ms. On 300 random
  formulas (187 satisfiable, 113 not), all three reductions gave the
  same yes/no answer as the original formula, and mapped every
  solution back correctly.
- **Approximations beat their guarantees:** the matching cover averaged
  1.48–1.58× optimal against its proven bound of 2, and reached exactly
  2 only on the complete bipartite graphs K(n, n). Nearest neighbor +
  2-opt was within 1% of the optimal tour on average for n ≤ 13.
- **The boundary is sharp:** 2-SAT with 100,000 variables was decided
  in 262 ms. 3-SAT needed 377 ms of DPLL search at only 70 variables.

## Methodology

**Environment:** Apple M3 Pro, macOS 15.7.3, CPython 3.13.5,
single-threaded. The test suite has 2,553 tests, 985 of them new this
week.

**Problems and datasets.** All inputs come from fixed seeds:

- *SAT:* random 3-SAT at the phase transition m = 4.26n, where about
  half the formulas are satisfiable and solvers struggle most.
- *Subset Sum:* "no" instances of 12-digit values aimed at half the
  total (the worst case for search), planted "yes" instances, and
  values ≤ 1,000 for the dynamic-programming (DP) experiments.
- *Vertex Cover:* random graphs G(n, p) with average degree 4, plus
  K(n, n).
- *TSP:* random cities in a 100 × 100 square.

**Metrics:**
- *Cost:* runtime, plus search-tree nodes, which don't depend on the
  machine.
- *Approximation ratio:* the heuristic's result divided by the exact
  optimum. Above n = 13, TSP tours are divided by the minimum spanning
  tree (MST) weight instead; MST ≤ OPT, so this overstates the true
  ratio.

**Output:** `comparison_table.csv` has 244 rows from a 36-second run.

## Results

### Part 1: Reductions (`reduction_3sat_vertex_cover.png`)

`reduction_demo.py` implements 3-SAT → Vertex Cover → Independent
Set → Clique (individually and as a chain) and 3-SAT → Subset Sum.
Each reduction logs every construction step and maps solutions in both
directions (`forward` and `recover`).

In 3-SAT → Vertex Cover, each variable becomes an edge x_i — ¬x_i and
each clause a triangle whose vertices link to their literals, with
k = n + 2m. A cover of size k must take exactly one end of every
variable edge (the truth value) and two vertices of every triangle.
The vertex left out must have its literal covered, so that literal is
true. The figure shows a satisfying assignment becoming a 9-vertex
cover.

| Clauses m | VC: elements | VC: time | Subset Sum: digits | Subset Sum: time |
|---:|---:|---:|---:|---:|
| 100 | 969 | 0.26 ms | 30,258 | 0.40 ms |
| 1,000 | 9,705 | 2.9 ms | 3.05M | 30 ms |
| 3,000 | 29,112 | 8.3 ms | 27.4M | 361 ms |

The Vertex Cover reduction is linear. The Subset Sum reduction is
quadratic, because each of its 2(n + m) numbers has n + m digits.
Both are polynomial, which is all a reduction requires.

### Part 2: Exact search vs heuristics (`runtime_growth.png`)

| Problem | Exact method | Largest n | Time | Nodes | Polynomial alternative |
|---|---|---:|---:|---:|---|
| Subset Sum | backtracking | 28 | 1.38 s | 7.1M | greedy: 2.9 µs |
| 3-SAT | brute force | 16 | 78 ms | 36k | WalkSAT |
| 3-SAT | backtracking | 30 | 850 ms | 92k | — |
| 3-SAT | DPLL | 70 | 377 ms | 2.3k | WalkSAT: 9.7 ms |
| Vertex Cover | branch and bound | 260 | 234 ms | 2.2k | 2-approx: 13 µs |
| TSP | brute force / Held–Karp | 10 / 13 | 86 / 23 ms | 9! tours | NN: 5.7 µs |

**Subset Sum.** Backtracking explored 7.1 million of the 537 million
possible nodes at n = 28. Pruning on the sum still left to collect cut
the growth base from 2 to about 1.8, but the curve is still a straight
line on a log scale.

The bitset DP solved n = 3,000 values ≤ 1,000 in 66 ms. It cannot touch
the 12-digit instances, though: its table would need about 7·10¹² bits
(roughly 0.9 TB). Its O(n·T) cost is polynomial in the target T, but T
takes only log T digits to write down. That makes the DP exponential
in the length of the input: pseudo-polynomial.

**SAT.**
- *Pruning helped hugely, but not enough.* Pruning a branch as soon as
  a clause was falsified let plain backtracking reach n = 30.
  Propagating forced literals (DPLL) cut the n = 30 search from 92,184
  nodes to 96.
- *DPLL still grew exponentially:* from 11 nodes at n = 10 to 2,344 at
  n = 70.
- *2-SAT is different in kind:* 100× more variables cost about 200×
  more time (1.3 ms at 1,000, 262 ms at 100,000). Clauses of two
  literals become implications, so satisfiability reduces to finding
  strongly connected components. Three-literal clauses have no such
  structure.

**TSP.** Brute force grew roughly ×n per added city (×9.5 from 9 to
10 cities), against ×2.4–2.7 for Held–Karp. Held–Karp was 61× faster at n = 10 and is still exponential.

### Part 3: Approximation quality (`approximation_quality.png`)

| Algorithm | Guarantee | Mean ratio | Worst observed |
|---|---|---:|---:|
| VC matching 2-approx | ≤ 2 | 1.48–1.58 | 1.68 (2.00 on K(n,n)) |
| VC greedy max-degree | O(log n) | 1.03–1.06 | 1.10 |
| TSP nearest neighbor, n ≤ 13 | none | 1.06–1.17 | 1.31 |
| TSP repeated NN | none | 1.00–1.05 | 1.19 |
| TSP MST double-tree | ≤ 2 | 1.04–1.21 | 1.33 |
| TSP NN + 2-opt | none | 1.000–1.010 | 1.044 |
| Subset Sum greedy | ≥ 1/2 | 0.969–1.000 | 0.645 |
| Random assignment, 3-SAT | 7/8 expected | 0.870–0.886 | 0.767 |

**The guarantees held:** no cover or MST tour exceeded 2× optimal, no
greedy subset fell below half the best sum, and random assignments
averaged 7/8 of clauses satisfied.

**Practice beat the guarantees,** and the algorithms with the
strongest guarantees were often not the best in practice:
- Greedy max-degree has only a logarithmic guarantee but beat the
  2-approximation by about 50 points.
- The MST tour, the only TSP method with a constant bound, was the
  worst TSP heuristic: its tour was 1.55× the MST bound, against
  1.38× for nearest neighbor and 1.20× for 2-opt.

**Size effects:**
- Subset Sum greedy improved with n. With 1,000 values, small items
  could fill the last gap, so it hit the target exactly 90% of the
  time, against 0% for n ≤ 10.
- WalkSAT satisfied every satisfiable formula up to n = 60 within
  20,000 flips, and 9 of 10 at n = 70. Across all formulas, including
  unsatisfiable ones, it averaged over 99% of clauses satisfied.

### Part 4: Trade-off curves (`tradeoff_curves.png`)

**WalkSAT** at n = 50 shows the clearest trade-off. Its success rate
rose with the flip budget:

| Flips | 300 | 1,000 | 3,000 | 10,000 | 30,000 |
|---|---:|---:|---:|---:|---:|
| Solved | 40% | 62% | 84% | 96% | 100% |

Even 30,000 flips took 3.0 ms per run, against 18 ms for DPLL.

**2-opt** at n = 200 dropped the tour from 1.283× to 1.229× the MST
bound within 3 passes. It converged after 5 passes, at 6.6 ms in
total.

**Vertex Cover at n = 260:**

| Method | Time | Ratio to optimal |
|---|---:|---:|
| Matching 2-approx | 13 µs | 1.57 |
| Greedy max-degree | 0.25 ms | 1.05 |
| Branch and bound (exact) | 234 ms | 1.00 |

Each step toward the exact answer cost one to three orders of
magnitude more time.

## Discussion

**P vs NP.** P is the set of problems solvable in polynomial time. NP
is the set whose "yes" answers can be checked in polynomial time. The
verifiers here do that checking: `is_satisfied` runs in O(m·k) and
`is_vertex_cover` in O(E). Finding a solution is another matter. Cook
and Levin showed that every NP problem reduces to SAT, and the
reductions here carry 3-SAT's hardness to Vertex Cover, Independent
Set, Clique and Subset Sum. A polynomial algorithm for any one of them
would therefore solve all of NP.

Decades of effort have produced none: the best known 3-SAT algorithms
are still about O(1.3ⁿ). The widely held belief that P ≠ NP rests on
that history of failure, not on a proof.

**Why exponential curves explode.**
- Each extra Subset Sum item doubles the possible subsets, and pruning
  only shrinks the base (to 1.8 here).
- At that rate, each 10 more items multiply the cost by about 360.
  Running 1,000× faster buys only about 12 more items.
- Polynomial curves behave the opposite way: nearest neighbor went
  from 1,000 to 2,000 cities at a 3.9× cost, as n² predicts.

Cost also depends on the instance, not just on n. A search can stop
at its first solution, but proving "no" means exhausting the tree:
planted "yes" Subset Sum instances finished up to 4.6× sooner than
"no" instances of the same size.

**Why reductions and approximations matter in practice.** One
reduction from a known NP-complete problem tells you to stop looking
for an exact fast algorithm. Reductions also let one solver serve many
problems: industrial SAT solvers, descendants of the DPLL here,
handle hardware verification and package-dependency resolution once
those are encoded as SAT. Approximation algorithms then supply
answers with guarantees, such as the matching cover's 2× bound, and
unguaranteed heuristics like 2-opt and WalkSAT often come closer
still. Routing and scheduling software runs on exactly these methods.

## Conclusion

Knowing a problem's complexity class changes the design question.
- For problems in P, such as 2-SAT, sorting and MST, the goal is the
  fastest exact algorithm.
- For NP-complete problems, there are three useful directions:
  - Exploit structure that makes the instance easier: small values for
    Subset Sum's DP, clauses of two literals for 2-SAT, small n for
    Held–Karp.
  - Prune hard so the exponential search covers usable sizes (DPLL,
    branch and bound).
  - Accept a bounded or measured loss of optimality.

This week's data shows the stakes. Exact search on 28 Subset Sum
items took 1.4 seconds; greedy took 3 microseconds, at the price of
never being able to answer "no." The best heuristics lost only a few
percent, and the tight K(n, n) case shows exactly how badly a
guarantee can be met. Understanding these
computational limits tells a designer where to spend effort: on the
structure of real inputs and on approximation ratios, not on searching
for an efficient exact algorithm that almost certainly does not exist.
