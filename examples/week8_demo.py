"""
Week 8 demo: computational complexity on small, readable inputs.

Run from the project root:

    python examples/week8_demo.py

Shows (1) where familiar problems sit among P, NP, NP-complete and
NP-hard, (2) a 3-SAT formula reduced step by step to Vertex Cover and
to Subset Sum, with solutions mapped back, (3) exponential search
blowing up while greedy stays flat, (4) 2-SAT in linear time next to
3-SAT, and (5) approximation algorithms for Vertex Cover and TSP with
their ratios against the exact optimum.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.complexity import (
    PHASE_TRANSITION_RATIO, IndependentSetToClique, ReductionChain, ThreeSAT,
    ThreeSatToSubsetSum, ThreeSatToVertexCover, VertexCoverToIndependentSet,
    backtracking_sat, exact_vertex_cover, format_cnf, greedy_subset_sum,
    greedy_vertex_cover, mst_tour, mst_weight, nearest_neighbor_tour,
    optimal_tour, random_ksat, random_subset_sum, repeated_nearest_neighbor,
    subset_sum_backtracking, two_opt, two_sat, vertex_cover_2approx, walksat,
)
from src.complexity.reduction_demo import (
    format_catalog, solve_subset_sum, solve_vertex_cover, verify_3sat,
)
from src.utils.graph_utils import complete_bipartite_edges, random_graph_edges
from src.utils.matrix_utils import euclidean_distance_matrix, random_points
from src.utils.timer import CallStats

FORMULA = ThreeSAT([(1, -2, 3), (-1, 2, 3), (-1, -2, -3)], 3)


def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_classes():
    header("1. Complexity classes")
    print(format_catalog())


def demo_reductions():
    header("2a. 3-SAT → Vertex Cover, step by step")
    r = ThreeSatToVertexCover(verbose=True)
    vc = r.reduce(FORMULA)
    cover = solve_vertex_cover(vc)
    assignment = r.recover(cover)
    print(f"\n  Exact solver finds a cover of size {len(cover)} ≤ k = {vc.k}:")
    print(f"    {sorted(cover)}")
    print(f"  Mapped back: {assignment} — satisfies the formula: "
          f"{verify_3sat(FORMULA, assignment)}")

    header("2b. 3-SAT → Subset Sum")
    r = ThreeSatToSubsetSum(verbose=True)
    ss = r.reduce(FORMULA)
    chosen = solve_subset_sum(ss)
    print(f"\n  Subset found: {[r.labels[i] for i in chosen]} "
          f"(sum {sum(ss.numbers[i] for i in chosen)} = target)")
    print(f"  Mapped back: {r.recover(chosen)}")

    header("2c. Chaining: 3-SAT → Vertex Cover → Independent Set → Clique")
    chain = ReductionChain([ThreeSatToVertexCover(),
                            VertexCoverToIndependentSet(),
                            IndependentSetToClique()])
    clique = chain.reduce(FORMULA)
    print(f"  Clique instance: {len(clique.vertices)} vertices, "
          f"{len(clique.edges)} edges, k = {clique.k}")
    assignment = backtracking_sat(FORMULA.clauses, FORMULA.num_vars)
    big_clique = chain.forward(assignment)
    print(f"  Satisfying assignment ↦ clique of size {len(big_clique)}; "
          f"mapped back through 3 reductions: {chain.recover(big_clique)}")


def demo_explosion():
    header("3. Exponential search vs a greedy heuristic (Subset Sum, "
           "no solution exists)")
    print(f"  {'n':>3} {'nodes explored':>15} {'backtracking':>13} "
          f"{'greedy':>9}")
    for n in (8, 12, 16, 20, 24):
        nums, target = random_subset_sum(n, 10 ** 12, solvable=False, seed=n)
        stats = CallStats()
        start = time.perf_counter()
        subset_sum_backtracking(nums, target, stats)
        slow = time.perf_counter() - start
        start = time.perf_counter()
        greedy_subset_sum(nums, target)
        fast = time.perf_counter() - start
        print(f"  {n:>3} {stats.calls:>15,} {slow * 1e3:>10.2f} ms "
              f"{fast * 1e6:>6.1f} µs")
    print("  Every +4 items multiplies the search 4–20×; greedy barely "
          "moves (but can't prove 'no').")


def demo_two_vs_three_sat():
    header("4. The P / NP-complete boundary: 2-SAT vs 3-SAT")
    n = 100_000
    cnf2 = random_ksat(n, n // 2, k=2, seed=1)
    start = time.perf_counter()
    result = two_sat(cnf2, n)
    print(f"  2-SAT, {n:,} variables: "
          f"{'satisfiable' if result else 'unsatisfiable'} in "
          f"{(time.perf_counter() - start) * 1e3:.0f} ms (linear time)")
    n = 60
    cnf3 = random_ksat(n, round(PHASE_TRANSITION_RATIO * n), seed=4)
    stats = CallStats()
    start = time.perf_counter()
    result = backtracking_sat(cnf3, n, stats=stats)
    print(f"  3-SAT, {n} variables, {len(cnf3)} clauses: "
          f"{'satisfiable' if result else 'unsatisfiable'} after "
          f"{stats.calls:,} DPLL nodes, "
          f"{(time.perf_counter() - start) * 1e3:.0f} ms")
    ws = walksat(cnf3, n, rng=random.Random(1))
    print(f"  WalkSAT on the same formula: {ws.satisfied}/{ws.num_clauses} "
          f"clauses after {ws.flips:,} flips "
          f"({'solved' if ws.solved else 'not solved — proves nothing'})")


def demo_approximation():
    header("5a. Vertex Cover: 2-approximation vs optimum")
    edges = random_graph_edges(150, 4, seed=2)
    opt = len(exact_vertex_cover(edges))
    for name, func in (("matching 2-approx", vertex_cover_2approx),
                       ("greedy max-degree", greedy_vertex_cover)):
        size = len(func(edges))
        print(f"  {name:18s} {size:4d} vertices  ratio {size / opt:.2f}")
    print(f"  exact (B&B)        {opt:4d} vertices")
    tight = complete_bipartite_edges(10, 10)
    print(f"  On K(10,10) the matching cover takes "
          f"{len(vertex_cover_2approx(tight))} vs optimum "
          f"{len(exact_vertex_cover(tight))}: the factor 2 is tight.")

    header("5b. TSP on 12 random cities")
    dist = euclidean_distance_matrix(random_points(12, seed=5))
    opt, _ = optimal_tour(dist)
    nn_cost, nn_tour = nearest_neighbor_tour(dist)
    results = [("nearest neighbor", nn_cost),
               ("repeated NN", repeated_nearest_neighbor(dist)[0]),
               ("MST double-tree", mst_tour(dist)[0]),
               ("NN + 2-opt", two_opt(dist, nn_tour)[0])]
    for name, cost in results:
        print(f"  {name:18s} {cost:7.1f}  ratio {cost / opt:.3f}")
    print(f"  Held–Karp optimum  {opt:7.1f}   (MST lower bound "
          f"{mst_weight(dist):.1f})")


if __name__ == "__main__":
    print("Formula used in the reductions:", format_cnf(FORMULA.clauses))
    demo_classes()
    demo_reductions()
    demo_explosion()
    demo_two_vs_three_sat()
    demo_approximation()
