"""
Floyd–Warshall — all-pairs shortest paths as dynamic programming.

Number the vertices 0..V-1 and define

    d_k[i][j] = length of the shortest i -> j path whose INTERMEDIATE
                vertices all come from {0, ..., k-1}

d_0 is just the edge weights. Allowing vertex k as an intermediate
either doesn't help, or the best path goes through k exactly once and
splits into two paths that use only {0..k-1}:

    d_{k+1}[i][j] = min(d_k[i][j],  d_k[i][k] + d_k[k][j])

After all V vertices are allowed, d_V is the answer. That is the
triple nested loop: the OUTER loop over the intermediate vertex k is
the DP stage, and the order matters — k must be outermost.

Space: the recurrence describes V+1 matrices (O(V³)), but stage k+1
reads only stage k, and it is safe to update ONE matrix in place:
during stage k, row k and column k cannot change, because
d[k][k] >= 0 (no negative cycles) gives
    d[i][k] + d[k][k] >= d[i][k]   and   d[k][k] + d[k][j] >= d[k][j].
So every value read in stage k is the same whether it was written this
stage or not. O(V²) space — the same compression idea as the 1D
knapsack.

Negative weights: allowed, unlike Dijkstra (whose greedy "settle the
closest node" step assumes no later edge can make a path shorter).
Negative CYCLES make "shortest" undefined; they show up as a negative
diagonal entry d[i][i] < 0 and raise NegativeCycleError. Note that in
an UNDIRECTED graph any negative edge u—v is itself a negative cycle
u -> v -> u.

Complexity: Θ(V³) time regardless of edge count, Θ(V²) space.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from src.graphs.dijkstra import dijkstra
from src.graphs.graph import Graph
from src.utils.matrix_utils import INF, Matrix, graph_to_matrix
from src.utils.timer import CallStats

PredMatrix = List[List[Optional[int]]]


class NegativeCycleError(ValueError):
    """Raised when the graph contains a cycle of negative total weight."""


@dataclass
class AllPairsShortestPaths:
    """
    All-pairs result: dist[i][j] and pred[i][j] are indexed by the
    positions of nodes in `nodes`.

    pred[i][j] is the vertex just before j on a shortest i -> j path,
    or None when j is unreachable from i (or j == i).
    """
    nodes: List[Any]
    dist: Matrix
    pred: PredMatrix

    def __post_init__(self) -> None:
        self._index = {node: i for i, node in enumerate(self.nodes)}

    def distance(self, u: Any, v: Any) -> float:
        """Shortest-path distance u -> v (INF if unreachable)."""
        return self.dist[self._index[u]][self._index[v]]

    def path(self, u: Any, v: Any) -> Optional[List[Any]]:
        """Shortest path [u, ..., v] as node labels, or None."""
        indices = reconstruct_path(self.pred, self._index[u],
                                   self._index[v])
        if indices is None:
            return None
        return [self.nodes[i] for i in indices]


def floyd_warshall_matrix(weights: Matrix,
                          stats: Optional[CallStats] = None
                          ) -> Tuple[Matrix, PredMatrix]:
    """
    Floyd–Warshall on a V×V weight matrix.

    Args:
        weights: weights[i][j] = edge weight i -> j, INF for no edge.
            The diagonal is normally 0. Not modified.
        stats: Optional counter; receives one count per (k, i, j)
            relaxation actually attempted (rows with d[i][k] = INF are
            skipped, since nothing can route through k).

    Returns:
        (dist, pred): distance and predecessor matrices.

    Raises:
        ValueError: If the matrix is not square.
        NegativeCycleError: If any vertex can reach a negative cycle
            that leads back to it (some dist[i][i] < 0).

    Complexity: Θ(V³) time, Θ(V²) space.
    """
    n = len(weights)
    for row in weights:
        if len(row) != n:
            raise ValueError("weight matrix must be square")

    dist = [list(row) for row in weights]
    pred: PredMatrix = [
        [i if (i != j and weights[i][j] != INF) else None for j in range(n)]
        for i in range(n)
    ]

    relaxations = 0
    for k in range(n):                     # DP stage: allow vertex k
        dist_k = dist[k]
        pred_k = pred[k]
        for i in range(n):
            dist_i = dist[i]
            through_k = dist_i[k]
            if through_k == INF:
                continue                   # i can't reach k: no change
            pred_i = pred[i]
            for j in range(n):
                candidate = through_k + dist_k[j]
                if candidate < dist_i[j]:
                    dist_i[j] = candidate
                    pred_i[j] = pred_k[j]  # last hop of the k -> j leg
            relaxations += n

    if stats is not None:
        stats.count(relaxations)
    for i in range(n):
        if dist[i][i] < 0:
            raise NegativeCycleError(
                f"negative cycle through vertex index {i}")
    return dist, pred


def floyd_warshall(graph: Graph,
                   stats: Optional[CallStats] = None
                   ) -> AllPairsShortestPaths:
    """
    All-pairs shortest paths for a Week 4 Graph.

    Args:
        graph: Weighted graph (directed or undirected, list or matrix
            representation). Negative edge weights are allowed.
        stats: Optional relaxation counter (see floyd_warshall_matrix).

    Returns:
        AllPairsShortestPaths with node order = graph.nodes().

    Raises:
        NegativeCycleError: If the graph has a negative cycle.

    Complexity: Θ(V³) time, Θ(V²) space.
    """
    nodes, weights = graph_to_matrix(graph)
    dist, pred = floyd_warshall_matrix(weights, stats)
    return AllPairsShortestPaths(nodes, dist, pred)


def reconstruct_path(pred: PredMatrix, i: int,
                     j: int) -> Optional[List[int]]:
    """
    Rebuild the shortest i -> j path (as vertex indices) by following
    predecessors backward from j.

    Returns:
        [i, ..., j], [i] when i == j, or None if j is unreachable.

    Complexity: O(path length).
    """
    if i == j:
        return [i]
    if pred[i][j] is None:
        return None
    path = [j]
    for _ in range(len(pred)):
        j = pred[i][j]
        path.append(j)
        if j == i:
            path.reverse()
            return path
    raise ValueError("predecessor matrix contains a cycle")


def dijkstra_all_pairs(graph: Graph) -> AllPairsShortestPaths:
    """
    All-pairs shortest paths by running Week 4's heap-based Dijkstra
    from every vertex: O(V · (V + E) log V). Faster than Floyd–Warshall
    on sparse graphs, but cannot handle negative weights.

    Returns:
        AllPairsShortestPaths in the same format as floyd_warshall().

    Raises:
        ValueError: On any negative edge weight (from dijkstra()).
    """
    nodes = graph.nodes()
    index = {node: i for i, node in enumerate(nodes)}
    dist: Matrix = []
    pred: PredMatrix = []
    for source in nodes:
        distances, predecessors = dijkstra(graph, source)
        dist.append([distances[v] for v in nodes])
        pred.append([None if predecessors[v] is None
                     else index[predecessors[v]] for v in nodes])
    return AllPairsShortestPaths(nodes, dist, pred)
