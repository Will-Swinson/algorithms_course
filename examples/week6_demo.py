"""
Week 6 demo: advanced DP patterns on small, readable inputs.

Run from the project root:

    python examples/week6_demo.py

Shows (1) the 1D knapsack row being overwritten item by item, and
what goes wrong if the loop runs the other way, (2) matrix-chain
ordering for a chain where the obvious order costs 5× more,
(3) Floyd–Warshall on a network with negative edges that Dijkstra
refuses, and (4) bitmask TSP on 10 delivery stops, with the work
compared against trying every route.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dp_advanced import (
    knapsack_1d, knapsack_1d_with_items, unbounded_knapsack,
    matrix_chain_order, floyd_warshall, dijkstra_all_pairs,
    tsp_bitmask, tsp_brute_force,
)
from src.graphs import Graph
from src.utils.matrix_utils import (
    euclidean_distance_matrix, format_matrix, left_to_right_cost,
    random_points,
)
from src.utils.timer import CallStats


def demo_knapsack():
    print("=" * 66)
    print("1. One DP row instead of a table: 0/1 knapsack, W = 7")
    print("=" * 66)
    weights, values, capacity = [1, 3, 4, 5], [2, 4, 5, 7], 7
    dp = [0] * (capacity + 1)
    row = lambda cells: "  ".join(f"{c:>2}" for c in cells)
    print(f"  {'capacity w':20s}{row(range(capacity + 1))}")
    print(f"  {'start':20s}{row(dp)}")
    for weight, value in zip(weights, values):
        for w in range(capacity, weight - 1, -1):     # high to low
            dp[w] = max(dp[w], value + dp[w - weight])
        print(f"  {f'+ item (w={weight}, v={value})':20s}{row(dp)}")
    best, items = knapsack_1d_with_items(weights, values, capacity)
    assert best == dp[capacity] == knapsack_1d(weights, values, capacity)
    print(f"\n  Best value {best} using items {items} — from 8 cells of "
          f"memory instead of 5 × 8.")
    print(f"  Same update looping LOW to HIGH reuses items: "
          f"{unbounded_knapsack(weights, values, capacity)} "
          f"(that's unbounded knapsack, not 0/1).")


def demo_matrix_chain():
    print()
    print("=" * 66)
    print("2. Matrix chain: where to put the parentheses")
    print("=" * 66)
    p = [40, 20, 30, 10, 30, 5]
    shapes = ", ".join(f"A{i}: {p[i - 1]}×{p[i]}" for i in range(1, len(p)))
    print(f"  {shapes}")
    cost, order = matrix_chain_order(p)
    naive = left_to_right_cost(p)
    print(f"  {'Left to right ((((A1A2)A3)A4)A5)':36s} {naive:>7,} "
          "multiplications")
    print(f"  {'Optimal ' + order:36s} {cost:>7,} multiplications")
    print(f"  Same product, {naive / cost:.1f}× less arithmetic.")


def demo_floyd_warshall():
    print()
    print("=" * 66)
    print("3. Floyd–Warshall with negative edges (rebates on some routes)")
    print("=" * 66)
    g = Graph(directed=True)
    for u, v, w in [("Depot", "A", 4), ("Depot", "B", 7), ("A", "B", -2),
                    ("B", "C", 3), ("A", "C", 6), ("C", "D", -1),
                    ("D", "A", 5)]:
        g.add_edge(u, v, w)
    result = floyd_warshall(g)
    print(format_matrix(result.dist, labels=result.nodes, width=7))
    print(f"\n  Depot -> D: cost {result.distance('Depot', 'D'):g} via "
          f"{' -> '.join(result.path('Depot', 'D'))}")
    try:
        dijkstra_all_pairs(g)
    except ValueError as err:
        print(f"  Week 4 Dijkstra on the same graph: ValueError — "
              f"{str(err).split(':')[0]}")


def demo_tsp():
    print()
    print("=" * 66)
    print("4. Bitmask TSP: shortest loop through 10 delivery stops")
    print("=" * 66)
    dist = euclidean_distance_matrix(random_points(10, seed=7))
    dp_stats, bf_stats = CallStats(), CallStats()
    cost, tour = tsp_bitmask(dist, stats=dp_stats)
    bf_cost, _ = tsp_brute_force(dist, stats=bf_stats)
    assert abs(cost - bf_cost) < 1e-9
    print(f"  Route: {' -> '.join(map(str, tour))}   length {cost:.1f}")
    print(f"  Brute force checked {bf_stats.calls:,} routes; the bitmask DP "
          f"made {dp_stats.calls:,} transitions")
    print(f"  over states like dp[{1 | 1 << 3 | 1 << 5:#012b}][5] = 'visited "
          f"{{0, 3, 5}}, standing at 5'.")


if __name__ == "__main__":
    demo_knapsack()
    demo_matrix_chain()
    demo_floyd_warshall()
    demo_tsp()
