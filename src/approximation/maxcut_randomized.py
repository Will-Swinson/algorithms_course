"""
Max-Cut: split the vertices into two sides to maximize the total
weight of edges that cross. NP-hard (one of Karp's 21 problems).

Randomized ½-approximation. Put each vertex on a side by a fair coin
flip. An edge crosses with probability exactly 1/2, so by linearity

    E[cut] = W / 2 ≥ OPT / 2,        W = total edge weight.

The crossing indicators of two different edges are pairwise
independent (even when the edges share a vertex), so the variances
add:

    Var[cut] = Σ_e w_e² / 4.

Both formulas are checked empirically by the benchmark.

Turning the expectation into a guarantee:

  derandomized_cut — the method of conditional expectations. Place
    vertices one at a time on the side that cuts more of the weight to
    already-placed neighbors. Each decision keeps the conditional
    expectation ≥ W/2, so the result is ≥ W/2 deterministically.
    O(V + E).

  local_search_cut — flip any vertex that has more incident weight on
    its own side than across; each flip strictly increases the cut, so
    it terminates. At a local optimum every vertex has ≥ half its
    incident weight crossing, and summing over vertices gives cut ≥ W/2.
    Starting from a random cut, it often finishes well above W/2.

  exact_maxcut — all 2^(n−1) partitions, vectorized with numpy; n ≤ 22.

Goemans–Williamson's semidefinite-programming algorithm reaches 0.878,
and under the Unique Games Conjecture nothing polynomial does better
(Khot et al. 2007). Unweighted input: pass (u, v) pairs; weighted:
(u, v, w) triples.
"""
import random
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple

import numpy as np

MAX_EXACT_VERTICES = 22

WeightedEdge = Tuple[Hashable, Hashable, float]


@dataclass
class CutResult:
    side: Dict[Hashable, int]                # vertex -> 0 or 1
    value: float
    flips: int = 0                           # local-search moves

    @property
    def left(self) -> Set[Hashable]:
        return {v for v, s in self.side.items() if s == 0}


def as_weighted(edges: Iterable) -> List[WeightedEdge]:
    """(u, v) -> (u, v, 1.0); self-loops dropped (they never cross)."""
    out = []
    for e in edges:
        u, v = e[0], e[1]
        w = float(e[2]) if len(e) > 2 else 1.0
        if u != v:
            out.append((u, v, w))
    return out


def vertex_list(edges: Iterable[WeightedEdge],
                vertices: Optional[Iterable[Hashable]] = None) -> List:
    vs = dict.fromkeys(vertices or ())
    for u, v, _ in edges:
        vs.setdefault(u)
        vs.setdefault(v)
    return list(vs)


def total_weight(edges: Iterable) -> float:
    return sum(w for _, _, w in as_weighted(edges))


def cut_value(edges: Iterable, side: Dict[Hashable, int]) -> float:
    return sum(w for u, v, w in as_weighted(edges) if side[u] != side[v])


def expected_random_cut(edges: Iterable) -> Tuple[float, float]:
    """(mean, variance) of the random cut's value: (W/2, Σw²/4)."""
    ws = [w for _, _, w in as_weighted(edges)]
    return sum(ws) / 2, sum(w * w for w in ws) / 4


def random_cut(edges: Iterable, vertices: Optional[Iterable] = None,
               rng: Optional[random.Random] = None) -> CutResult:
    """Each vertex picks a side by a fair coin. E[value] = W/2. O(V + E)."""
    edges = as_weighted(edges)
    rng = rng or random
    side = {v: rng.randrange(2) for v in vertex_list(edges, vertices)}
    return CutResult(side, cut_value(edges, side))


def derandomized_cut(edges: Iterable,
                     vertices: Optional[Iterable] = None) -> CutResult:
    """Conditional expectations: value ≥ W/2 guaranteed. O(V + E)."""
    edges = as_weighted(edges)
    adj: Dict[Hashable, List[Tuple[Hashable, float]]] = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    side: Dict[Hashable, int] = {}
    for v in vertex_list(edges, vertices):
        weight_to = [0.0, 0.0]
        for u, w in adj.get(v, ()):
            if u in side:
                weight_to[side[u]] += w
        # join the side opposite the heavier neighbor weight
        side[v] = 0 if weight_to[1] >= weight_to[0] else 1
    return CutResult(side, cut_value(edges, side))


def local_search_cut(edges: Iterable, start: Optional[Dict] = None,
                     vertices: Optional[Iterable] = None,
                     rng: Optional[random.Random] = None,
                     max_flips: int = 1_000_000) -> CutResult:
    """
    Flip any vertex whose flip increases the cut until none does.

    gain[v] = (weight to v's own side) − (weight across) is maintained
    incrementally, so a flip costs O(deg v). Vertices with positive
    gain wait in a work list.

    Args:
        start: Initial sides (default: a random cut from `rng`).

    Returns:
        A locally optimal cut, value ≥ W/2; .flips = moves made.
    """
    edges = as_weighted(edges)
    if start is None:
        start = random_cut(edges, vertices, rng).side
    side = dict(start)
    for v in vertex_list(edges, vertices):
        side.setdefault(v, 0)
    adj: Dict[Hashable, List[Tuple[Hashable, float]]] = {v: [] for v in side}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    gain = {v: sum(w if side[u] == side[v] else -w for u, w in adj[v])
            for v in side}
    work = [v for v in side if gain[v] > 1e-12]
    flips = 0
    while work and flips < max_flips:
        v = work.pop()
        if gain[v] <= 1e-12:
            continue
        side[v] ^= 1
        gain[v] = -gain[v]
        flips += 1
        for u, w in adj[v]:
            # v moved: edges to u's side flip between "same" and "across"
            gain[u] += 2 * w if side[u] == side[v] else -2 * w
            if gain[u] > 1e-12:
                work.append(u)
    return CutResult(side, cut_value(edges, side), flips)


def exact_maxcut(edges: Iterable, vertices: Optional[Iterable] = None
                 ) -> CutResult:
    """
    Optimal cut by evaluating all 2^(n−1) partitions at once (vertex 0
    fixed on side 0 to skip mirror images). numpy computes, for every
    partition bitmask, Σ_e w_e · [bit u ≠ bit v].

    Complexity: O(2^n · E) arithmetic, vectorized.
    """
    edges = as_weighted(edges)
    vs = vertex_list(edges, vertices)
    n = len(vs)
    if n > MAX_EXACT_VERTICES:
        raise ValueError(f"{n} vertices exceeds the exact limit of "
                         f"{MAX_EXACT_VERTICES}")
    if n <= 1:
        return CutResult({v: 0 for v in vs}, 0.0)
    index = {v: i for i, v in enumerate(vs)}
    masks = np.arange(1 << (n - 1), dtype=np.int64) << 1   # bit 0 = 0
    values = np.zeros(masks.shape[0])
    for u, v, w in edges:
        values += w * (((masks >> index[u]) ^ (masks >> index[v])) & 1)
    best = int(masks[int(np.argmax(values))])
    side = {v: (best >> index[v]) & 1 for v in vs}
    return CutResult(side, float(values.max()))
