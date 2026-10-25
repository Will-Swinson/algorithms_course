# Week 9 Technical Report: Approximation Algorithms

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** Designing efficient solutions when optimality is impossible

## Executive Summary

Every algorithm stayed within its proven guarantee on every instance,
and most finished far inside it. They ran in polynomial time on inputs
where the exact solvers took seconds or would not finish.

- **Guarantees held:**
  - Vertex Cover edge picking: worst 1.78× optimal (bound 2).
  - Christofides: worst 1.18× (bound 1.5).
  - MST doubling: worst 1.34× (bound 2).
  - Derandomized Max-Cut: never below 0.73 of optimal (bound ½).
- **Simple greedy often beat LP rounding.** At n = 100, greedy Set
  Cover was 1.15× optimal, against 1.49× for randomized rounding and
  1.56× for deterministic rounding. But on a constructed family
  greedy degraded to 4× while LP rounding stayed optimal.
- **Randomized Max-Cut matched theory.** Over 70 graphs, the mean cut
  was within 1.2% of W/2 (half the total edge weight) and the variance
  within 12.5% of Σw²/4. Local search lifted cuts from about 0.62 to
  0.92–0.96 of optimal.
- **Local search was nearly exact for facility location:** it found
  the optimum on all 30 test instances.

## Approximation Theory Overview

An algorithm is a **ρ-approximation** if, for every input, its cost is
at most ρ·OPT (for minimization) or its value at least OPT/ρ (for
maximization). Since OPT is unknown, every proof compares against a
**lower bound** that is computable:

| Problem | Algorithm | Lower bound used | Guarantee |
|---|---|---|---|
| Vertex Cover | greedy edge picking | maximal matching M | 2 |
| Vertex Cover (weighted) | LP, keep x ≥ ½ | LP* | 2 |
| Set Cover | greedy | dual fitting | H_n ≈ ln n |
| Set Cover | LP, keep x ≥ 1/f | LP* | f (max element frequency) |
| Set Cover | randomized rounding, t = ⌈ln n⌉+1 rounds | LP* | O(log n) expected |
| Facility Location (metric) | star greedy | dual fitting | 1.861 |
| Facility Location (metric) | add/drop/swap local search | local optimality | 3 |
| Metric TSP | MST doubling | MST ≤ OPT | 2 |
| Metric TSP | Christofides | MST + matching ≤ OPT/2 | 3/2 |
| Max-Cut | random cut / conditional expectations | W ≥ OPT | ½ |

**Edge picking.** The edges it acts on share no endpoints, so OPT
needs one vertex per edge: OPT ≥ |M|, while the cover is 2|M|. Each
run returns M, so it **certifies its own ratio** without knowing OPT.

**LP relaxation.** Write the problem as an integer program and allow
fractional values. The LP optimum is ≤ OPT, solvable in polynomial
time, and its values say how much each set "wants" to be chosen.

**Limits.** Some guarantees cannot be improved unless P = NP:
- Set Cover: (1 − ε)·ln n.
- Metric facility location: 1.463.
- Vertex Cover: 2 − ε, assuming the Unique Games Conjecture.
- Max-Cut: Goemans–Williamson's 0.878, also assuming it.
- General TSP: no constant factor at all; the triangle inequality is
  what makes 2 and 3/2 possible.

Some problems admit a **PTAS**: any 1 + ε in time polynomial in n. If
the time is also polynomial in 1/ε, that is an **FPTAS**. Knapsack has
an FPTAS; Euclidean TSP has a PTAS (Arora 1998). None of the
problems above can have a PTAS unless P = NP.

## Methodology

**Environment:** Apple M3 Pro, macOS 15.7.3, CPython 3.13.5,
single-threaded. 3,261 tests pass, 708 of them new this week.

**Datasets.** All synthetic, from fixed seeds:
- *Graphs:* random G(n, p) with average degree 4, weighted 1–10 for
  Max-Cut, plus K(n, n), the case where the 2× bound is exactly met.
- *Set Cover:* random instances (n elements, 0.6n sets, each element
  in each set with probability 0.1, costs 1–10), plus the constructed
  family that fools greedy.
- *Facility Location and TSP:* random points in a 100 × 100 square,
  so distances are metric. Facility opening costs are 20–60.

**Exact baselines:**
- Vertex Cover branch and bound (Week 8).
- Set Cover branch and bound over bitmasks.
- Facility Location: all 2^m subsets of facilities.
- TSP: Held–Karp (Week 6).
- Max-Cut: all 2^(n−1) cuts, computed with numpy.

Larger instances are measured against a computable lower bound (LP*,
MST, or W), which overstates the true ratio.

**LP solver.** No scipy is installed, so `src/utils/lp_solver.py` is a
two-phase simplex on numpy tableaus:
- Phase 1 finds a feasible vertex by driving artificial variables to
  zero; Phase 2 then optimizes the real objective.
- It uses Dantzig's pivot rule, switching to Bland's rule after a run
  of degenerate pivots so it cannot cycle.
- It matched brute-force vertex enumeration on 120 random LPs.
- A 100-element Set Cover LP solves in 25 ms.

**Randomization.** Every random run gets its own seeded
`random.Random`:
- Randomized rounding: 30 roundings per instance.
- Random Max-Cut: 1,000 cuts per graph, with the measured mean and
  variance compared to W/2 and Σw²/4.
- Local search: starts from 100 random cuts per graph.

**Output:** `comparison_table.csv` has 259 rows from a 1-minute run.

## Results

### Approximation ratios (`approx_ratios.png`, `cost_vs_optimal.png`)

Ratios are cost / OPT, the mean over 5 instances at the largest size
with a known optimum; brackets give the worst instance at any size.

| Problem (n) | Algorithm | Mean ratio [worst] | Guarantee |
|---|---|---:|---:|
| Vertex Cover (180) | edge picking | 1.58 [1.78] | 2 |
| | LP rounding | 1.86 [2.00] | 2 |
| Set Cover (100) | greedy | 1.15 [1.23] | 5.19 |
| | deterministic rounding | 1.56 [1.95] | 15 |
| | randomized rounding | 1.49 [1.95] | ~6 |
| Facility (80 clients) | star greedy | 1.03 [1.06] | 1.861 |
| | local search | 1.000 [1.000] | 3 |
| Metric TSP (13) | MST doubling | 1.16 [1.34] | 2 |
| | Christofides | 1.08 [1.18] | 1.5 |
| | Christofides, greedy matching | 1.10 [1.31] | — |
| Max-Cut (20, weighted) | random cut (mean) | 0.59 | ≥ 0.5 |
| | derandomized | 0.89 [0.73] | ≥ 0.5 |
| | random + local search | 0.94 [0.69] | ≥ 0.5 |

**Vertex Cover.** On K(n, n), edge picking gave exactly 2.00 at every
size, confirming that the bound is tight.

LP rounding was the weaker method here, reaching exactly 2 on one
instance. On these graphs the LP optimum typically puts ½ on every
vertex, so threshold rounding keeps almost all of them.

LP rounding does have one real advantage: vertex weights. On a star
with a 100-cost center, it chose the five leaves, at total weight 5.

**Set Cover.** The LP was tight: LP*/OPT stayed between 0.97 and 1.00,
and for n ≤ 60 most LPs were already integral. Rounding is free then,
but on fractional instances both roundings overpaid.

At n = 400, measured against LP*, greedy reached 1.39× while
deterministic rounding reached 5.4×. Its guarantee f grew to 39,
because each element appears in more sets as n grows.

On the constructed family, greedy grew as k/2, from 1.0 up to 4.0 at
510 elements, while deterministic rounding stayed optimal.

**TSP.** Christofides' matching cost was at most OPT/2 on every
instance, the key step of its proof.

For n = 20–500, measured against the MST bound:
- Christofides: 1.43 down to 1.29.
- Greedy matching: 1.38–1.47.
- MST doubling: steady near 1.57.

`tsp_tours.png` shows why: shortcutting the doubled tree leaves long
diagonals that Christofides avoids.

### Randomized algorithms (`randomized_maxcut_results.png`, `set_cover_rounding.png`)

**Random cut.**
- Over 70 graphs, the measured mean of 1,000 cuts was within 1.2% of
  W/2 (0.4% on average). The variance was within 12.5% of Σw²/4.
- Variances add here because the cut indicators of any two edges are
  independent.
- The guarantee holds only in expectation. On average 12% of single
  unweighted cuts and 22% of weighted cuts fell below OPT/2.
- *Derandomization* places vertices greedily by conditional
  expectation. That removes the bad draws entirely, at 0.88–0.94 of
  optimal.
- *Local search* from random starts reached 0.92–0.95 on average.

On one 20-vertex graph (OPT 31, W 37), 5,000 random cuts centred on
W/2 = 18.5, while 5,000 local searches averaged 28.4 and hit OPT 5.6%
of the time.

**Randomized rounding** has a trade-off between the number of rounds
(t) and cost (n = 80):

| Rounds t | 1 | 2 | 3 | 5 | 8 |
|---|---:|---:|---:|---:|---:|
| Mean cost / OPT | 1.10 | 1.16 | 1.23 | 1.32 | 1.37 |
| Repairs per run | 1.93 | 0.53 | 0.07 | 0 | 0 |

Each extra round buys coverage the cheap patching step would have
supplied anyway, so the theoretically safe t = ⌈ln n⌉ + 1 overpays. A
single round plus repair was cheapest.

The variance grew with instance size. The standard deviation of the
ratio across 30 roundings was 0 for n ≤ 60, where LPs were integral,
then 0.06 at n = 80–100 and 0.30 at n = 400.

### Runtime and scalability (`approx_runtime.png`)

| Algorithm | Largest n | Time | Exact method at its limit |
|---|---:|---:|---|
| VC edge picking | 100,000 | 67 ms | branch and bound, n = 180: 44 ms |
| VC LP rounding | 200 | 575 ms | — |
| Set Cover greedy | 400 | 0.9 ms | branch and bound, n = 100: 1.26 s |
| Set Cover LP | 100 | 25 ms | — |
| Facility greedy + LS | 400 (m = 100) | 400 ms | 2^m, m = 16: 269 ms |
| TSP MST doubling | 1,000 | 59 ms | Held–Karp, n = 13: 28 ms |
| TSP Christofides | 500 | 2.96 s | — |
| Max-Cut local search | 10,000 | 72 ms | 2^(n−1) cuts, n = 20: 70 ms |

Each exact solver doubled or quadrupled in cost per small step in n.
The approximations grew polynomially.

Christofides' blossom matching is O(n³). That makes it 200× slower
than MST doubling at n = 500, for about 18% shorter tours.

## Discussion

**When approximation shines.**
- *Problems where a one-line argument gives a constant factor:*
  Vertex Cover, metric TSP, Max-Cut. These algorithms are fast,
  simple, and carry a proof.
- *Problems with structure for local moves to exploit:* local search
  found every facility-location optimum.
- *As a certificate:* the matching, LP value and MST bound each bound
  the error on the instance in hand.

**When exact optimization is infeasible.**
- Every exact baseline grew exponentially. Set Cover branch and bound
  went from 5 ms at n = 60 to 1.26 s at n = 100.
- Facility enumeration roughly quadrupled with every 2 extra
  facilities.
- Max-Cut enumeration grew 9× from n = 18 to 20.
- Beyond these sizes only bounds remain: the LP was within 3% of
  OPT, the MST about 25% below it.

**Trade-offs between accuracy and runtime.**
- *Guarantees and typical behaviour diverge.* Greedy Set Cover had
  the weaker guarantee yet won on random inputs; Christofides won on
  quality but cost 200× the time of MST doubling.
- *Randomization is simple but single runs vary.* Repetition,
  derandomization or local search make the expectation dependable.

## Conclusion

Approximation turns "NP-hard" from a dead end into an engineering
problem:
1. Find a lower bound you can compute.
2. Design an algorithm whose cost you can charge against it.
3. Check the real ratio on the instances you actually have.

The bounds were never violated, and most measured ratios sat well
inside them: 0–60% above optimal (86% for LP-rounded Vertex Cover)
against proven factors of 1.5–15. In practice the best approach was
layered:
1. Start with a guaranteed algorithm for safety (edge picking, MST
   doubling, the derandomized cut).
2. Improve the result with local search.
3. Use LP or MST bounds to report how close to optimal the answer is.

Approximation algorithms give up the optimum in exchange for a
provable, measurable answer to how close is close enough.
