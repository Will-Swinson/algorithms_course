"""
Week 9 demo: approximation algorithms and the proofs behind them.

Run from the project root:

    python examples/week9_demo.py

Shows (1) the Vertex Cover 2-approximation certifying its own ratio,
and where the factor 2 is tight, (2) Set Cover's LP relaxation and
both roundings, plus a family that fools greedy, (3) greedy and local
search for facility location against the exact optimum, (4) MST
doubling vs Christofides on a small metric TSP, and (5) the random
Max-Cut's expectation and variance against theory, then derandomized
and improved by local search.
"""
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.approximation import (
    christofides_tour, derandomized_cut, deterministic_rounding,
    exact_facility_location, exact_maxcut, exact_set_cover,
    expected_random_cut, greedy_bad_instance, greedy_facility_location,
    greedy_set_cover, local_search_cut, local_search_facility_location,
    lp_relaxation, lp_rounding_vertex_cover, matching_vertex_cover,
    mst_doubling_tour, optimal_tour, random_cut, random_metric_instance,
    random_set_cover, randomized_rounding, total_weight,
)
from src.complexity.vertex_cover_approx import exact_vertex_cover
from src.utils.graph_utils import (
    complete_bipartite_edges, random_graph_edges, random_weighted_edges,
)
from src.utils.matrix_utils import euclidean_distance_matrix, random_points


def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_vertex_cover():
    header("1. Vertex Cover: greedy edge picking certifies its own ratio")
    edges = random_graph_edges(60, 4, seed=1)
    res = matching_vertex_cover(edges)
    opt = len(exact_vertex_cover(edges))
    print(f"  Random graph: 60 vertices, {len(edges)} edges")
    print(f"  Picked {len(res.matching)} disjoint edges → cover of "
          f"{res.cost} vertices")
    print(f"  Any cover needs ≥ {len(res.matching)} vertices (one per "
          f"disjoint edge), so ratio ≤ {res.cost}/{len(res.matching)} = "
          f"{res.certified_ratio:.0f}")
    print(f"  Actual optimum: {opt} → true ratio {res.cost / opt:.2f}")
    tight = complete_bipartite_edges(8, 8)
    print(f"  Tight case K(8,8): cover {matching_vertex_cover(tight).cost}, "
          f"optimum {len(exact_vertex_cover(tight))} → exactly 2")
    star = [(0, i) for i in range(1, 6)]
    weights = {0: 100, **{i: 1 for i in range(1, 6)}}
    lp = lp_rounding_vertex_cover(star, weights)
    print(f"  Weighted star (center costs 100, leaves 1): LP rounding "
          f"takes {sorted(lp.cover)}, weight {lp.cost:g}")


def demo_set_cover():
    header("2. Set Cover: LP relaxation, then rounding")
    inst = random_set_cover(70, 42, 0.1, seed=8)
    x, lp = lp_relaxation(inst)
    fractional = sum(1 for v in x if 1e-9 < v < 1 - 1e-9)
    opt = exact_set_cover(inst).cost
    print(f"  70 elements, 42 sets, max frequency f = {inst.max_frequency}")
    print(f"  LP* = {lp:.2f} ≤ OPT = {opt:.2f}   "
          f"({fractional} sets with fractional x)")
    det = deterministic_rounding(inst, x)
    print(f"  Deterministic (x ≥ 1/f): {det.cost:7.2f}  ratio "
          f"{det.cost / opt:.3f}  (bound f·LP* = "
          f"{inst.max_frequency * lp:.1f})")
    runs = [randomized_rounding(inst, x, rng=random.Random(s)).cost
            for s in range(200)]
    print(f"  Randomized (200 runs):    mean {statistics.mean(runs):.2f}, "
          f"std {statistics.pstdev(runs):.2f}, best {min(runs):.2f}, "
          f"worst {max(runs):.2f}")
    print(f"  Greedy:                   {greedy_set_cover(inst).cost:7.2f}  "
          f"ratio {greedy_set_cover(inst).cost / opt:.3f}")
    bad = greedy_bad_instance(6)
    print(f"  Greedy's trap ({bad.num_elements} elements): greedy "
          f"{greedy_set_cover(bad).cost:g} sets, OPT "
          f"{exact_set_cover(bad).cost:g}, LP rounding "
          f"{deterministic_rounding(bad).cost:g}")


def demo_facility():
    header("3. Facility Location: 12 sites, 50 clients")
    inst = random_metric_instance(12, 50, seed=5)
    opt = exact_facility_location(inst)
    for name, sol in (("star greedy", greedy_facility_location(inst)),
                      ("local search", local_search_facility_location(inst)),
                      ("exact (4,095 subsets)", opt)):
        print(f"  {name:22s} open {sorted(sol.open)}  cost {sol.cost:8.1f}"
              f"  ratio {sol.cost / opt.cost:.3f}")


def demo_tsp():
    header("4. Metric TSP on 12 random cities")
    dist = euclidean_distance_matrix(random_points(12, seed=3))
    opt, _ = optimal_tour(dist)
    doubling = mst_doubling_tour(dist)
    chris = christofides_tour(dist)
    print(f"  MST weight {doubling.mst_cost:.1f} ≤ OPT {opt:.1f}")
    print(f"  MST doubling:  {doubling.cost:6.1f}  ratio "
          f"{doubling.cost / opt:.3f}  (≤ 2)")
    print(f"  Christofides:  {chris.cost:6.1f}  ratio "
          f"{chris.cost / opt:.3f}  (≤ 1.5)")
    print(f"  Christofides matched {2 * len(chris.matching)} odd-degree "
          f"cities for {chris.matching_cost:.1f} ≤ OPT/2 = {opt / 2:.1f}")


def demo_maxcut():
    header("5. Max-Cut: flip a coin per vertex")
    edges = random_weighted_edges(20, 4, seed=8)
    w = total_weight(edges)
    opt = exact_maxcut(edges, range(20)).value
    mean, var = expected_random_cut(edges)
    vals = [random_cut(edges, range(20), random.Random(s)).value
            for s in range(10_000)]
    print(f"  20 vertices, total weight W = {w:g}, OPT = {opt:g}")
    print(f"  Theory:   E = W/2 = {mean:g},  Var = Σw²/4 = {var:g}")
    print(f"  Measured: mean {statistics.mean(vals):.1f},  variance "
          f"{statistics.pvariance(vals):.1f}  (10,000 cuts)")
    print(f"  Mean / OPT = {statistics.mean(vals) / opt:.3f} ≥ 0.5; "
          f"{sum(v < opt / 2 for v in vals) / len(vals):.0%} of single "
          "cuts fall below OPT/2")
    d = derandomized_cut(edges, range(20)).value
    ls = local_search_cut(edges, vertices=range(20), rng=random.Random(1))
    print(f"  Derandomized: {d:g} ({d / opt:.3f} of OPT, ≥ W/2 guaranteed)")
    print(f"  Random + local search: {ls.value:g} ({ls.value / opt:.3f} of "
          f"OPT) after {ls.flips} flips")


if __name__ == "__main__":
    demo_vertex_cover()
    demo_set_cover()
    demo_facility()
    demo_tsp()
    demo_maxcut()
