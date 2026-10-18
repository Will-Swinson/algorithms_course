"""
Vertex Cover: choose as few vertices as possible so that every edge
has at least one endpoint chosen.

The decision version ("is there a cover of size ≤ k?") is NP-complete
(3-SAT reduces to it; see reduction_demo.py), so no polynomial exact
algorithm is known. Three approaches:

    vertex_cover_2approx   maximal matching; never worse than 2 × optimal   O(V + E)
    greedy_vertex_cover    repeatedly take the max-degree vertex; often
                           better in practice, but only an O(log n)
                           guarantee                                 O(E log V)
    exact_vertex_cover     branch and bound                          O(2^n) worst

Why the matching algorithm is a 2-approximation. It scans the edges
and, whenever an edge has neither endpoint covered yet, takes BOTH
endpoints. The edges it acted on share no endpoints (they form a
matching M), so any cover, including the optimal one, needs a separate
vertex for each of them: OPT ≥ |M|. The algorithm took exactly 2|M|
vertices, so its cover is at most 2·OPT. The bound is tight: on the
complete bipartite graph K_{n,n} it takes all 2n vertices while one
side (n vertices) suffices.
"""
import heapq
from typing import Dict, Hashable, Iterable, List, Optional, Set

from src.utils.graph_utils import Edge, adjacency_sets
from src.utils.timer import CallStats


def is_vertex_cover(edges: Iterable[Edge], cover: Iterable[Hashable]) -> bool:
    """Certificate check: every edge has an endpoint in cover. O(V + E)."""
    chosen = set(cover)
    return all(u in chosen or v in chosen for u, v in edges)


def vertex_cover_2approx(edges: Iterable[Edge]) -> Set[Hashable]:
    """
    Maximal-matching 2-approximation (greedy edge picking).

    One pass over the edges: if neither endpoint is covered, take both.

    Returns:
        A vertex cover of size ≤ 2 × minimum.

    Complexity: O(E) time (plus O(V) for the set), one pass.
    """
    cover: Set[Hashable] = set()
    for u, v in edges:
        if u != v and u not in cover and v not in cover:
            cover.add(u)
            cover.add(v)
    return cover


def greedy_vertex_cover(edges: Iterable[Edge]) -> Set[Hashable]:
    """
    Max-degree greedy: take the vertex touching the most uncovered
    edges, delete those edges, repeat.

    Usually smaller than the matching cover on random graphs, but its
    worst case is a Θ(log n) factor above optimal (there are bipartite
    graphs that lure it into picking the wrong side over and over).

    A lazy max-heap of (−degree, vertex) avoids rescanning: stale
    entries are skipped when popped.

    Complexity: O(E log V).
    """
    adj = adjacency_sets(edges)
    degree = {v: len(nbrs) for v, nbrs in adj.items()}
    heap = [(-d, i, v) for i, (v, d) in enumerate(degree.items()) if d]
    heapq.heapify(heap)
    index = {v: i for _, i, v in heap}       # tie-breaker; vertices may
    cover: Set[Hashable] = set()             # not be mutually comparable
    while heap:
        neg_d, _, v = heapq.heappop(heap)
        if v in cover or -neg_d != degree[v]:
            continue                         # stale entry
        cover.add(v)
        for u in adj[v]:
            if u not in cover:
                degree[u] -= 1
                if degree[u]:
                    heapq.heappush(heap, (-degree[u], index[u], u))
        degree[v] = 0
    return cover


def exact_vertex_cover(edges: Iterable[Edge],
                       stats: Optional[CallStats] = None) -> Set[Hashable]:
    """
    Minimum vertex cover by branch and bound.

    Pick the remaining vertex v of highest degree. Any cover either
    contains v, or does not — and then must contain every neighbor of
    v (otherwise an edge at v is uncovered). So branch:

        take v            delete v and its edges
        take N(v)         delete v, its neighbors, and their edges

    Prune a branch when (vertices taken) + (a lower bound for the rest)
    can't beat the best cover found so far. The lower bound is the size
    of a greedy maximal matching of the remaining graph: each matched
    edge needs its own cover vertex. The search starts from the
    2-approximation as its first "best", and a degree-1 vertex never
    needs branching — taking its neighbor is always at least as good.

    Args:
        edges: Undirected edges.
        stats: Optional counter; stats.calls = branch nodes visited.

    Returns:
        A minimum vertex cover.

    Complexity: O(2^n · E) worst case; the high-degree-first branching
    and the matching bound keep sparse graphs of ~60 vertices fast.
    """
    stats = stats if stats is not None else CallStats()
    adj = adjacency_sets(edges)
    best: List[Set[Hashable]] = [vertex_cover_2approx(
        [(u, v) for u in adj for v in adj[u]])]

    def remove(graph: Dict[Hashable, Set[Hashable]],
               gone: Set[Hashable]) -> Dict[Hashable, Set[Hashable]]:
        out = {}
        for u, nbrs in graph.items():
            if u not in gone:
                rest = nbrs - gone
                if rest:
                    out[u] = rest
        return out

    def matching_bound(graph: Dict[Hashable, Set[Hashable]]) -> int:
        matched: Set[Hashable] = set()
        size = 0
        for u, nbrs in graph.items():
            if u in matched:
                continue
            for v in nbrs:
                if v not in matched:
                    matched.add(u)
                    matched.add(v)
                    size += 1
                    break
        return size

    def branch(graph: Dict[Hashable, Set[Hashable]],
               taken: Set[Hashable]) -> None:
        stats.enter()
        try:
            # Degree-1 rule: cover a pendant edge by its non-leaf end
            forced: Set[Hashable] = set()
            while True:
                leaf = next((u for u, n in graph.items() if len(n) == 1),
                            None)
                if leaf is None:
                    break
                forced.add(next(iter(graph[leaf])))
                graph = remove(graph, forced)
            taken = taken | forced
            if not graph:
                if len(taken) < len(best[0]):
                    best[0] = taken
                return
            if len(taken) + matching_bound(graph) >= len(best[0]):
                return
            v = max(graph, key=lambda u: len(graph[u]))
            branch(remove(graph, {v}), taken | {v})
            nbrs = graph[v]
            branch(remove(graph, nbrs | {v}), taken | nbrs)
        finally:
            stats.exit()

    branch(adj, set())
    return best[0]
