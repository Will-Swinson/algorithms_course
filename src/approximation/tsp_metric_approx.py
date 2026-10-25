"""
Metric TSP: MST doubling (2-approximation) and Christofides (3/2).

Both need the triangle inequality d(a, c) ≤ d(a, b) + d(b, c), which
lets a walk that repeats cities be "shortcut" — skip any city already
visited — without getting longer. Without it, TSP cannot be
approximated within any constant factor unless P = NP.

Shared building blocks:

    MST            Prim's algorithm, O(n²)                 (Week 8)
    Euler circuit  Hierholzer's algorithm on a multigraph whose
                   vertices all have even degree           O(E)
    shortcut       keep each city's first appearance       O(E)

MST doubling (2-approximation). Double every MST edge; every degree is
now even, so an Euler circuit exists. It costs 2·MST, and shortcutting
only shortens it. Deleting one edge from the optimal tour leaves a
spanning tree, so MST ≤ OPT and the tour is ≤ 2·OPT.

Christofides (3/2-approximation, 1976). Instead of doubling, fix only
the vertices of odd MST degree (there is always an even number of
them). Add a minimum-weight perfect matching M on those vertices; now
every degree is even. Euler circuit + shortcut gives a tour of cost
≤ MST + M. Shortcutting the optimal tour down to the odd vertices
gives a cycle of cost ≤ OPT, and that cycle splits into two perfect
matchings on them, the cheaper costing ≤ OPT/2. So M ≤ OPT/2 and the
tour is ≤ OPT + OPT/2 = 1.5·OPT.

The matching step uses networkx's blossom algorithm (O(n³)), which is
the cost driver: Christofides is O(n³) overall vs O(n²) for doubling.
`matching="greedy"` swaps in a greedy matching (closest pair first),
which keeps the structure but loses the 3/2 proof.
"""
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import networkx as nx

from src.complexity.traveling_salesman_approx import mst_parents, mst_weight
from src.dp_advanced.bitmask_traveling_salesman import tour_cost, tsp_bitmask

Matrix = Sequence[Sequence[float]]

# The exact solver for small instances is Week 6's Held–Karp DP
optimal_tour = tsp_bitmask

__all__ = ["TourResult", "mst_doubling_tour", "christofides_tour",
           "eulerian_circuit", "shortcut", "odd_degree_vertices",
           "min_weight_perfect_matching", "greedy_perfect_matching",
           "mst_edges", "mst_weight", "optimal_tour", "tour_cost"]


@dataclass
class TourResult:
    cost: float
    tour: List[int]                          # [0, ..., 0]
    mst_cost: float                          # lower bound: MST ≤ OPT
    euler_walk: List[int] = field(default_factory=list)
    matching: List[Tuple[int, int]] = field(default_factory=list)
    matching_cost: float = 0.0

    @property
    def certified_ratio(self) -> float:
        """cost / MST — an upper bound on cost / OPT."""
        return self.cost / self.mst_cost if self.mst_cost else 1.0


def _validate(dist: Matrix) -> int:
    n = len(dist)
    if n == 0:
        raise ValueError("need at least one city")
    if any(len(row) != n for row in dist):
        raise ValueError("distance matrix must be square")
    return n


def mst_edges(dist: Matrix) -> List[Tuple[int, int]]:
    parent = mst_parents(dist)
    return [(parent[v], v) for v in range(1, len(dist))]


def odd_degree_vertices(n: int, edges: Sequence[Tuple[int, int]]
                        ) -> List[int]:
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return [v for v in range(n) if degree[v] % 2]


def eulerian_circuit(n: int, edges: Sequence[Tuple[int, int]],
                     start: int = 0) -> List[int]:
    """
    Hierholzer's algorithm for a connected multigraph with all degrees
    even. Walk until stuck (back at a vertex with no unused edges),
    then back up and splice in sub-circuits. Iterative, O(E).

    Returns:
        Vertex sequence starting and ending at `start`.
    """
    adj: Dict[int, List[Tuple[int, int]]] = {v: [] for v in range(n)}
    for k, (u, v) in enumerate(edges):
        adj[u].append((v, k))
        adj[v].append((u, k))
    used = [False] * len(edges)
    stack, circuit = [start], []
    while stack:
        v = stack[-1]
        while adj[v] and used[adj[v][-1][1]]:
            adj[v].pop()                       # discard used edges
        if adj[v]:
            u, k = adj[v].pop()
            used[k] = True
            stack.append(u)
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    return circuit


def shortcut(walk: Sequence[int]) -> List[int]:
    """Keep each city's first visit, then return to the start."""
    seen = set()
    tour = []
    for v in walk:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    return tour + [tour[0]]


def min_weight_perfect_matching(dist: Matrix, vertices: Sequence[int]
                                ) -> List[Tuple[int, int]]:
    """
    Exact minimum-weight perfect matching on `vertices` (even count)
    in the complete graph, via networkx's blossom algorithm (O(k³)).
    """
    if len(vertices) % 2:
        raise ValueError("perfect matching needs an even number of vertices")
    g = nx.Graph()
    for u, v in itertools.combinations(vertices, 2):
        g.add_edge(u, v, weight=dist[u][v])
    matching = nx.min_weight_matching(g)
    return sorted(tuple(sorted(e)) for e in matching)


def greedy_perfect_matching(dist: Matrix, vertices: Sequence[int]
                            ) -> List[Tuple[int, int]]:
    """Closest unmatched pair first. O(k² log k); no 1/2 guarantee."""
    if len(vertices) % 2:
        raise ValueError("perfect matching needs an even number of vertices")
    pairs = sorted(itertools.combinations(vertices, 2),
                   key=lambda e: dist[e[0]][e[1]])
    matched = set()
    out = []
    for u, v in pairs:
        if u not in matched and v not in matched:
            matched.update((u, v))
            out.append((min(u, v), max(u, v)))
    return out


def mst_doubling_tour(dist: Matrix) -> TourResult:
    """
    2-approximation: MST, doubled, Euler circuit, shortcut.

    Complexity: O(n²).
    """
    n = _validate(dist)
    if n == 1:
        return TourResult(0.0, [0, 0], 0.0, [0])
    tree = mst_edges(dist)
    walk = eulerian_circuit(n, tree + tree)
    tour = shortcut(walk)
    return TourResult(tour_cost(dist, tour), tour,
                      sum(dist[u][v] for u, v in tree), walk)


def christofides_tour(dist: Matrix, matching: str = "optimal") -> TourResult:
    """
    Christofides' 3/2-approximation for metric TSP.

    Args:
        dist: Symmetric metric distance matrix.
        matching: "optimal" (blossom, keeps the 3/2 guarantee) or
            "greedy" (faster, no guarantee).

    Complexity: O(n²) MST + O(k³) matching on k ≤ n odd vertices.
    """
    n = _validate(dist)
    if n == 1:
        return TourResult(0.0, [0, 0], 0.0, [0])
    if matching not in ("optimal", "greedy"):
        raise ValueError("matching must be 'optimal' or 'greedy'")
    tree = mst_edges(dist)
    odd = odd_degree_vertices(n, tree)
    match = (min_weight_perfect_matching if matching == "optimal"
             else greedy_perfect_matching)(dist, odd)
    walk = eulerian_circuit(n, tree + match)
    tour = shortcut(walk)
    return TourResult(tour_cost(dist, tour), tour,
                      sum(dist[u][v] for u, v in tree), walk, match,
                      sum(dist[u][v] for u, v in match))
