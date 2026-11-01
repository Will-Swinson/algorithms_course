# Week 10 Technical Report: Network Flow Algorithms

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** A modular flow network solver

## Executive Summary

I built a flow network solver in `src/flow/`, laid out as the
assignment specifies: `core/`, `applications/`, `visualization/`,
`utils/` and a CLI in `main.py`. It contains Edmonds–Karp, capacity
scaling, push–relabel (FIFO and highest-label) and min-cost max-flow,
and applies them to bipartite matching, graph-cut segmentation and
four scheduling models. Every flow the benchmark produced, 155
configurations on networks of up to 50,178 nodes, passed independent
checks of capacity, conservation, residual consistency and the
min-cut optimality certificate.

- **Push–relabel dominated.** FIFO push–relabel was 46× faster than
  Edmonds–Karp on 20,000-node sparse networks (0.12 s vs 5.5 s), 31×
  on dense ones and 96× on segmentation grids. It solved 50,000-node
  sparse networks in 0.32 s and grids in 1.2 s. It was also 3.3–4.3×
  faster than networkx's `preflow_push`.
- **The heuristics matter more than the textbook bound.** Without
  gap and global relabeling, push–relabel did not finish a 4,096-pixel
  grid within 10 s; with them, it took 0.08 s.
- **Applications work at scale.** Graph cuts segmented a 200×200 image
  in 0.62 s, and at noise σ = 0.3 reached IoU 0.86 against 0.47 for
  thresholding. Matching 20,000 + 20,000 vertices took 0.71 s, close
  to networkx's Hopcroft–Karp (0.62 s).

## Theory Overview

A flow network is a directed graph with capacities c(u, v), a source
s and a sink t. A flow obeys capacities and conserves flow at every
node except s and t. The **residual graph** G_f holds the unused
capacity c − f on each arc and, on a reverse arc, the flow f that can
still be cancelled. An **augmenting path** is any s → t path in G_f.

**Max-flow min-cut.** For any cut (S, T) with s ∈ S and t ∈ T, every
flow is at most the capacity of the arcs from S to T. When G_f has no
augmenting path, let S be the nodes reachable from s. Every arc
leaving S is saturated and every arc entering S is empty, so
|f| = cap(S, T). That proves both that the flow is maximum and that
the cut is minimum. My validators check exactly this: t is not
reachable in G_f, and the cut capacity equals the flow value.

**Min-cost optimality.** A flow of value k is cheapest exactly when
G_f has no negative-cost cycle. Equivalently, there are potentials π
with reduced cost c(u, v) + π(u) − π(v) ≥ 0 on every residual arc.
Min-cost flow returns its potentials, so that check takes O(E).

## Algorithm Comparison

The network stores arcs in parallel arrays. Edge 2k is u → v and
edge 2k + 1 is its residual twin, so the reverse of edge e is e ^ 1.
`add_edge` creates both arcs. Every solver changes residual capacities
in place, with no hashing in the inner loops.

| Algorithm | Bound | Idea |
|---|---|---|
| Edmonds–Karp | O(VE²) | BFS shortest augmenting path |
| Capacity scaling | O(E² log U) | only paths with residual ≥ Δ, halving Δ |
| Push–relabel FIFO | O(V³) | local pushes guided by height labels |
| Push–relabel highest-label | O(V²√E) | always discharge the highest active node |
| Min-cost max-flow | O(F·E log V) | Dijkstra on reduced costs (Johnson potentials) |

**Edmonds–Karp: simple but slower.** The O(VE²) bound follows from
one lemma: BFS distances never decrease. `augmentations.png` shows
this directly: on 5,000 nodes the augmenting path length rises in
steps from 3 to 11 arcs and never drops. The weakness is that each
augmentation repeats a BFS over the whole graph. On sparse networks
the number of augmentations grew slowly (109 at n = 1,000, 631 at
n = 20,000), but each BFS grew with E, so runtime rose roughly as n².

**Capacity scaling** halved the number of augmentations on uniform
capacities (149 vs 271 at n = 5,000) and on exponential ones (180 vs
377). On unit capacities the counts were identical (69), as theory
predicts: only the Δ = 1 phase exists. It ran 1.8–3.1× faster than
Edmonds–Karp at the largest sizes but kept the same growth rate.

**Push–relabel: asymptotically faster.** It never searches for a
whole path. Each operation is a push along one arc or a relabel of
one node. Two heuristics make this practical:
- **Global relabeling** recomputes exact distances to t by reverse BFS
  roughly every n relabels.
- **The gap heuristic** lifts at once every node that can no longer
  reach t.

FIFO and highest-label variants give the same flows. My highest-label
version was slower, though: 1.35× on 50,000-node sparse networks and
4.6× on the largest grid. Its better bound concerns worst-case
non-saturating pushes, but on these inputs it did *more* pushes
(348,519 vs 229,481 on the 20,000-node grid) and twice the relabels.

**Min-cost flow: the cost/flow trade-off.** Successive shortest paths
send the cheapest unit first. Per-unit path costs never decrease, so
the cost curve is convex. In the demo the first 3 units cost 2 each
and the remaining 4 cost 5 each. Because it runs one Dijkstra per
augmentation and its bound depends on the flow value F, it was the
slowest solver: 2.7 s at n = 5,000, against 0.44 s for Edmonds–Karp
and 0.011 s for push–relabel. The answer also carries more
information: the cheapest of all maximum flows, with a certificate.

## Applications

- **Matching.** s → left → right → t with unit capacities. Because
  the flow is integral, the saturated middle arcs form the matching.
  All four solvers matched networkx's Hopcroft–Karp sizes on every
  test graph. Using min-cost flow on the same network solves the
  assignment problem, checked against brute force over every
  permutation.
- **Segmentation.** Each pixel becomes a node:
  - Terminal arcs carry the Gaussian data cost (I − μ)².
  - Neighbour arcs carry exp(−ΔI²/2σ²), so cutting between similar
    pixels is expensive.

  Each s–t cut is a labeling with that energy, so the minimum cut is
  the globally optimal segmentation. On 3×3 images it matched
  exhaustive search over all 512 labelings. Two refinements:
  - Reparametrization keeps one terminal arc per pixel.
  - Seed pixels get arcs larger than any possible cut, which fixes
    their label.
- **Scheduling.** Four models share the same flow construction:
  - Job → machine hours, and project staffing (cheapest assignment by
    min-cost flow).
  - Preemptive deadline scheduling: job → time slot → sink, where
    slot capacity is the number of machines. It agreed with brute
    force on 15 random instances.
  - Tournament elimination: teams on the source side of the min cut
    prove a team cannot finish first. It agreed with enumerating
    every game outcome.

## Performance Findings

| Family (largest n) | EK | Scaling | PR FIFO | PR HL | networkx |
|---|---|---|---|---|---|
| Sparse, 20,000 | 5.48 s | 1.74 s | 0.118 s | 0.130 s | 0.509 s |
| Sparse, 50,000 | — | — | 0.319 s | 0.432 s | — |
| Dense, 1,200 (288k arcs) | 2.74 s | 1.53 s | 0.088 s | 0.096 s | — |
| Grid, 4,098 | 7.08 s | 3.50 s | 0.074 s | 0.130 s | 0.248 s |
| Grid, 50,178 | — | — | 1.21 s | 5.62 s | — |

Empty cells are sizes where the method was not run: Edmonds–Karp and
scaling would take tens of seconds there.

- **Scaling.** Push–relabel FIFO grew almost linearly on sparse
  networks: 2.7× more time from 20k to 50k nodes, which is 2.5× more
  nodes. Edmonds–Karp grew 3.8× when n doubled. Its speed-up over
  Edmonds–Karp widened with size (`flow_runtime.png`, bottom right),
  most of all on grids.
- **Sparse vs dense.** Dense graphs did not change the ranking.
  Push–relabel's advantage grew from 3× at 100 nodes to 31× at 1,200,
  because Edmonds–Karp's BFS cost grows with the arc count.
- **Memory.** On 20,000 nodes, Edmonds–Karp peaked at 3.7 MB above
  the network itself and push–relabel at 10–13 MB
  (`flow_memory.png`). About a third of FIFO's 10.2 MB is the gap
  heuristic's one Python set per height: with gaps off it peaked at
  6.8 MB. Generating the network peaked at about 320 bytes per arc.
- **Convergence** (`pr_convergence.png`). After a quarter of its
  operations, FIFO already had 98% of the final flow at t on the
  50,000-node sparse network and 84% on the 20,000-node grid. The
  rest of the run settles the last units and drains excess back to s.
- **Ablation:**
  - Dropping global relabeling made runs 3.8× slower on sparse
    networks and 2.7× slower on grids.
  - Dropping the gap heuristic alone barely mattered.
  - Dropping both pushed all three grid runs past the 10 s cap.

  The O(V³) bound is one thing; how many relabels actually happen is
  what decides runtime.

## Conclusion

Engineering lessons:
1. **Representation decides speed.** Arc arrays with twin-by-XOR
   indexing let one data structure serve five algorithms with no
   conversion step.
2. **Theory picks the family; heuristics decide the speed.**
   Push–relabel's better bound showed up as 30–100× speed-ups, but
   only with global relabeling. Without the heuristics it was the
   slowest algorithm of all.
3. **Certify, don't trust.** A max-flow solver can prove its own
   answer through the min cut, and min-cost flow through its
   potentials. Cross-checks caught two of my own bugs during
   development:
   - Comparison against networkx showed the gap heuristic firing
     while the sink still sat at height 0.
   - The min-cost validator showed that potentials of unreachable
     nodes were left stale, which broke the certificate (though not
     the cost).

Scalability: in pure Python, push–relabel handled 50,000-node networks
in under 1.5 s. Edmonds–Karp, capacity scaling and min-cost flow
stopped being practical at about 10⁴ nodes. Matching, segmentation
and scheduling became fast once each was reduced to a single max-flow
call.
