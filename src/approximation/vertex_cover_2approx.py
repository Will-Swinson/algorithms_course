"""
Vertex Cover: two 2-approximations, each carrying its own proof.

1. Greedy edge picking (maximal matching). Scan the edges; whenever an
   edge has neither endpoint covered, take both endpoints.

   Proof that |C| ≤ 2·OPT. The edges that triggered a pick share no
   endpoints, so they form a matching M, and |C| = 2|M|. Any vertex
   cover, the optimal one included, must contain an endpoint of every
   edge of M, and since those edges are disjoint it needs |M| distinct
   vertices: OPT ≥ |M|. Therefore |C| = 2|M| ≤ 2·OPT.

   The matching is returned too, so every run certifies its own
   quality: |C| / |M| ≤ 2 bounds the true ratio |C| / OPT without ever
   computing OPT. Tight on K(n, n): C = all 2n vertices, OPT = n.

2. LP relaxation + threshold rounding (also handles vertex weights).

       minimize Σ w_v x_v   subject to   x_u + x_v ≥ 1 for each edge,
                                         x_v ≥ 0

   Take every vertex with x_v ≥ 1/2. Each edge has x_u + x_v ≥ 1, so
   one endpoint is ≥ 1/2: a cover. Each chosen vertex has 1 ≤ 2x_v, so
   w(C) ≤ 2·Σ w_v x_v = 2·LP* ≤ 2·OPT. (This is set-cover threshold
   rounding with frequency f = 2.)

The factor 2 is believed to be the limit: beating 2 − ε for every ε
would refute the Unique Games Conjecture (Khot–Regev 2008).
"""
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Set

from src.complexity.vertex_cover_approx import (
    exact_vertex_cover, is_vertex_cover,
)
from src.utils.graph_utils import Edge, normalize_edges, vertices_of
from src.utils.lp_solver import solve_lp

__all__ = ["CoverResult", "matching_vertex_cover", "lp_vertex_cover",
           "lp_rounding_vertex_cover", "exact_vertex_cover",
           "is_vertex_cover", "cover_weight"]


@dataclass
class CoverResult:
    cover: Set[Hashable]
    cost: float                 # size, or total weight if weighted
    lower_bound: float          # |M| for matching, LP* for LP rounding
    matching: Optional[List[Edge]] = None

    @property
    def certified_ratio(self) -> float:
        """Upper bound on cost / OPT, from this run's own lower bound."""
        return self.cost / self.lower_bound if self.lower_bound else 1.0


def cover_weight(cover: Iterable[Hashable],
                 weights: Optional[Dict[Hashable, float]] = None) -> float:
    return sum(weights[v] for v in cover) if weights else len(set(cover))


def matching_vertex_cover(edges: Iterable[Edge]) -> CoverResult:
    """
    Greedy edge picking: take both ends of every edge that is not yet
    covered.

    Returns:
        CoverResult with the cover, the matching M it was built from,
        and lower_bound = |M| ≤ OPT.

    Complexity: O(V + E), one pass.
    """
    cover: Set[Hashable] = set()
    matching: List[Edge] = []
    for u, v in edges:
        if u != v and u not in cover and v not in cover:
            cover.update((u, v))
            matching.append((u, v))
    return CoverResult(cover, len(cover), len(matching), matching)


def lp_vertex_cover(edges: Iterable[Edge],
                    weights: Optional[Dict[Hashable, float]] = None
                    ) -> tuple:
    """
    Solve the LP relaxation.

    Returns:
        (x, LP*) with x a dict vertex -> fractional value. Its optimal
        vertices are half-integral: every x_v is 0, 1/2 or 1.
    """
    edges = normalize_edges(edges)
    vertices = sorted(vertices_of(edges), key=repr)
    if not edges:
        return {}, 0.0
    index = {v: i for i, v in enumerate(vertices)}
    c = [weights[v] if weights else 1.0 for v in vertices]
    a = []
    for u, v in edges:
        row = [0.0] * len(vertices)
        row[index[u]] = row[index[v]] = 1.0
        a.append(row)
    result = solve_lp(c, A_ge=a, b_ge=[1.0] * len(edges))
    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.status}")
    return ({v: float(result.x[index[v]]) for v in vertices},
            result.objective)


def lp_rounding_vertex_cover(edges: Iterable[Edge],
                             weights: Optional[Dict[Hashable, float]] = None
                             ) -> CoverResult:
    """
    Take every vertex with LP value ≥ 1/2. Weight ≤ 2·LP* ≤ 2·OPT.

    Complexity: one LP with V variables and E constraints (simplex;
    polynomial-time LP algorithms exist, but simplex is faster here).
    """
    x, lp_value = lp_vertex_cover(edges, weights)
    cover = {v for v, val in x.items() if val >= 0.5 - 1e-9}
    return CoverResult(cover, cover_weight(cover, weights), lp_value)
