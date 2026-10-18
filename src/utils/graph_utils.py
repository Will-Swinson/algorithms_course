"""
Edge-list graph helpers for Week 8.

The complexity code works on plain undirected edge lists [(u, v), ...]
instead of the Week 4 Graph class: reductions build graphs out of
gadgets, take complements, and hand them between problems, and a list
of pairs is the simplest thing to build, print, and check.
edges_from_graph() converts a Week 4 Graph when needed.
"""
import itertools
import random
from typing import (Dict, Hashable, Iterable, List, Optional, Sequence,
                    Set, Tuple)

from src.graphs.graph import Graph

Edge = Tuple[Hashable, Hashable]


def normalize_edges(edges: Iterable[Edge]) -> List[Edge]:
    """Drop self-loops and duplicate edges ((u, v) and (v, u) are one)."""
    seen: Set[frozenset] = set()
    out = []
    for u, v in edges:
        key = frozenset((u, v))
        if u != v and key not in seen:
            seen.add(key)
            out.append((u, v))
    return out


def vertices_of(edges: Iterable[Edge]) -> Set[Hashable]:
    """Every endpoint that appears in an edge."""
    return {x for e in edges for x in e}


def adjacency_sets(edges: Iterable[Edge],
                   vertices: Optional[Iterable[Hashable]] = None
                   ) -> Dict[Hashable, Set[Hashable]]:
    """Neighbor sets; `vertices` adds isolated vertices too."""
    adj: Dict[Hashable, Set[Hashable]] = {v: set() for v in vertices or ()}
    for u, v in edges:
        if u == v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def complement_edges(vertices: Sequence[Hashable],
                     edges: Iterable[Edge]) -> List[Edge]:
    """Edges of the complement graph: every non-adjacent pair. O(V²)."""
    present = {frozenset(e) for e in edges}
    return [(u, v) for u, v in itertools.combinations(vertices, 2)
            if frozenset((u, v)) not in present]


def edges_from_graph(graph: Graph) -> List[Edge]:
    """Undirected edge list of a Week 4 Graph (weights dropped)."""
    edges = []
    for u in graph.nodes():
        for v, _ in graph.get_neighbors(u):
            edges.append((u, v))
    return normalize_edges(edges)


def random_graph_edges(n: int, avg_degree: float = 3.0,
                       seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Erdős–Rényi G(n, p) on vertices 0..n-1 with p chosen so the
    expected degree is avg_degree. Each of the n(n−1)/2 pairs is kept
    independently with probability p.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n < 2:
        return []
    p = min(1.0, avg_degree / (n - 1))
    rng = random.Random(seed)
    if n <= 2_000:
        return [(u, v) for u, v in itertools.combinations(range(n), 2)
                if rng.random() < p]
    # Large sparse graphs: sample the expected edge count directly
    # instead of flipping n²/2 coins.
    target = round(p * n * (n - 1) / 2)
    edges: Set[Tuple[int, int]] = set()
    while len(edges) < target:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def complete_bipartite_edges(a: int, b: int) -> List[Tuple[int, int]]:
    """
    K_{a,b}: left vertices 0..a-1, right a..a+b-1, every left-right
    pair joined. Its minimum vertex cover is the smaller side, but any
    maximal matching has min(a, b) edges whose 2·min(a, b) endpoints a
    matching-based cover takes — the tight case for the 2-approximation.
    """
    return [(u, a + v) for u in range(a) for v in range(b)]

