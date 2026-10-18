"""
Traveling Salesman: the shortest tour visiting every city once.

TSP is NP-hard. Week 6 solved it exactly with Held–Karp in
O(n²·2ⁿ), which tops out around 15–20 cities. This module trades
optimality for polynomial time:

    nearest_neighbor_tour        always go to the closest unvisited city   O(n²)
    repeated_nearest_neighbor    nearest neighbor from every start city    O(n³)
    mst_tour                     walk a minimum spanning tree in preorder,
                                 skipping repeats; ≤ 2 × optimal on
                                 metric inputs                             O(n²)
    two_opt                      improve any tour by reversing segments
                                 while that shortens it                    O(n²) per pass

Guarantees differ. For distances obeying the triangle inequality:

  * mst_tour is a 2-approximation. Deleting one edge of the optimal
    tour leaves a spanning tree, so MST ≤ OPT. Walking around the tree
    uses each edge twice (2·MST), and skipping already-visited cities
    only shortens the walk (triangle inequality): tour ≤ 2·MST ≤ 2·OPT.
  * nearest neighbor has no constant guarantee: on adversarial inputs
    it can be Θ(log n) × optimal, though on random points it is
    typically ~25% above.
  * 2-opt ends at a local optimum with no crossing edges; it has no
    useful worst-case bound but is usually within ~5% in practice.

mst_weight() doubles as a lower bound (MST ≤ OPT), which lets the
benchmark bound the error of a heuristic tour even when the optimum is
too expensive to compute.

Tours use the Week 6 format: [start, c1, ..., c_{n-1}, start].
"""
from typing import List, Optional, Sequence, Tuple

from src.dp_advanced.bitmask_traveling_salesman import tour_cost, tsp_bitmask
from src.utils.timer import CallStats

INF = float("inf")
Matrix = Sequence[Sequence[float]]
Tour = Tuple[float, List[int]]

# The exact solver is Week 6's Held–Karp DP
optimal_tour = tsp_bitmask


def _validate(dist: Matrix) -> int:
    n = len(dist)
    if n == 0:
        raise ValueError("need at least one city")
    for row in dist:
        if len(row) != n:
            raise ValueError("distance matrix must be square")
    return n


def nearest_neighbor_tour(dist: Matrix, start: int = 0) -> Tour:
    """
    Greedy construction: from the current city, move to the nearest
    city not yet visited; return to start at the end.

    Args:
        dist: n×n distance matrix.
        start: Starting city.

    Returns:
        (tour length, tour)

    Complexity: O(n²) — n steps, each scanning up to n candidates.
    """
    n = _validate(dist)
    if not 0 <= start < n:
        raise ValueError(f"start must be in [0, {n}), got {start}")
    visited = [False] * n
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        row = dist[current]
        best, nearest = INF, -1
        for city in range(n):
            if not visited[city] and row[city] < best:
                best, nearest = row[city], city
        if nearest == -1:                    # only INF roads remain
            nearest = next(c for c in range(n) if not visited[c])
        visited[nearest] = True
        tour.append(nearest)
        current = nearest
    tour.append(start)
    return tour_cost(dist, tour), tour


def repeated_nearest_neighbor(dist: Matrix) -> Tour:
    """
    Nearest neighbor from every possible start city; keep the shortest,
    rotated to begin at city 0. O(n³).
    """
    n = _validate(dist)
    best_cost, best_tour = INF, None
    for start in range(n):
        cost, tour = nearest_neighbor_tour(dist, start)
        if cost < best_cost:
            best_cost, best_tour = cost, tour
    cycle = best_tour[:-1]
    i = cycle.index(0)
    rotated = cycle[i:] + cycle[:i] + [0]
    return tour_cost(dist, rotated), rotated


def mst_parents(dist: Matrix) -> List[int]:
    """
    Prim's algorithm on the complete graph, O(n²) with an array (no
    heap: every pair is an edge, so a heap would only add log n).

    Returns:
        parent[v] for each city (parent[0] = −1, the root).
    """
    n = _validate(dist)
    in_tree = [False] * n
    key = [INF] * n
    parent = [-1] * n
    key[0] = 0
    for _ in range(n):
        u = min((v for v in range(n) if not in_tree[v]), key=key.__getitem__)
        in_tree[u] = True
        row = dist[u]
        for v in range(n):
            if not in_tree[v] and row[v] < key[v]:
                key[v] = row[v]
                parent[v] = u
    return parent


def mst_weight(dist: Matrix) -> float:
    """Total MST weight — a lower bound on the optimal tour."""
    parent = mst_parents(dist)
    return sum(dist[parent[v]][v] for v in range(1, len(dist)))


def mst_tour(dist: Matrix) -> Tour:
    """
    Double-tree 2-approximation: preorder walk of the MST from city 0.

    The preorder visits each city the first time the full walk around
    the tree reaches it — exactly the shortcut version of that walk.

    Complexity: O(n²) for Prim, O(n) for the walk.
    """
    n = _validate(dist)
    parent = mst_parents(dist)
    children: List[List[int]] = [[] for _ in range(n)]
    for v in range(1, n):
        children[parent[v]].append(v)
    tour, stack = [], [0]
    while stack:
        u = stack.pop()
        tour.append(u)
        # reversed so the closest-indexed child is visited first
        stack.extend(reversed(children[u]))
    tour.append(0)
    return tour_cost(dist, tour), tour


def two_opt(dist: Matrix, tour: Sequence[int],
            max_passes: Optional[int] = None,
            stats: Optional[CallStats] = None) -> Tour:
    """
    2-opt local search for symmetric distances.

    Remove two edges (a, b) and (c, d) of the tour and reconnect as
    (a, c) and (b, d), which reverses the path b..c. Apply any such
    move that shortens the tour; stop after a full pass finds none
    (a 2-optimal tour: no two edges can be uncrossed profitably).

    Args:
        dist: Symmetric n×n distance matrix.
        tour: Starting tour [s, ..., s].
        max_passes: Optional cap on improvement passes.
        stats: Optional counter; receives one count per move evaluated.

    Returns:
        (length, improved tour)

    Complexity: O(n²) per pass. The number of passes has no useful
    worst-case bound, so the total cost is measured, not promised.
    """
    _validate(dist)
    route = list(tour[:-1])
    n = len(route)
    evaluated = 0
    passes = 0
    improved = n >= 4
    while improved and (max_passes is None or passes < max_passes):
        improved = False
        passes += 1
        for i in range(n - 1):
            a, b = route[i], route[i + 1]
            d_ab = dist[a][b]
            # j stops at n-1 when i = 0 so the two edges never share a city
            for j in range(i + 2, n if i else n - 1):
                c, d = route[j], route[(j + 1) % n]
                evaluated += 1
                delta = dist[a][c] + dist[b][d] - d_ab - dist[c][d]
                if delta < -1e-12:
                    route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                    improved = True
                    a, b = route[i], route[i + 1]
                    d_ab = dist[a][b]
    if stats is not None:
        stats.count(evaluated)
    result = route + [route[0]]
    return tour_cost(dist, result), result
