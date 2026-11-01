"""
Bipartite matching as maximum flow.

    s → every left vertex      capacity 1
    left → right (each edge)   capacity 1
    every right vertex → t     capacity 1

An integral flow of value k picks k left→right edges with no shared
endpoint (each left vertex receives one unit, each right vertex
forwards one), and every matching gives such a flow, so the maximum
flow equals the maximum matching size. All the solvers here return
integral flows on integral capacities, so the matching is read off
the saturated left→right edges.

min_cost_assignment adds a cost to each left→right edge and runs
min-cost max-flow: the cheapest among all maximum matchings — the
assignment problem (what the Hungarian algorithm solves) as a
special case of min-cost flow.

On unit-capacity networks Edmonds–Karp's augmenting paths are exactly
the alternating paths of the classic matching algorithms; Hopcroft–
Karp's O(E√V) bound comes from augmenting along many shortest paths
per phase, which push–relabel also approximates in practice.
"""
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from src.flow.core import get_solver, min_cost_max_flow
from src.flow.core.network import FlowNetwork, FlowResult


@dataclass
class MatchingResult:
    pairs: List[Tuple[Hashable, Hashable]]
    flow: FlowResult
    cost: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self.pairs)

    def as_dict(self) -> Dict[Hashable, Hashable]:
        return dict(self.pairs)


class _BipartiteNetwork:
    """Label ↔ node-id bookkeeping for the s/left/right/t network."""

    def __init__(self, left: Sequence[Hashable], right: Sequence[Hashable],
                 edges: Iterable[Tuple], costs: bool = False):
        self.left = list(dict.fromkeys(left))
        self.right = list(dict.fromkeys(right))
        self.left_id = {v: 1 + i for i, v in enumerate(self.left)}
        self.right_id = {v: 1 + len(self.left) + i
                         for i, v in enumerate(self.right)}
        n = 2 + len(self.left) + len(self.right)
        self.s, self.t = 0, n - 1
        self.net = FlowNetwork(n)
        for v in self.left:
            self.net.add_edge(self.s, self.left_id[v], 1)
        for v in self.right:
            self.net.add_edge(self.right_id[v], self.t, 1)
        self.edge_pairs: Dict[int, Tuple[Hashable, Hashable]] = {}
        seen = set()
        for edge in edges:
            a, b = edge[0], edge[1]
            if a not in self.left_id or b not in self.right_id:
                raise ValueError(f"edge {edge[:2]} joins unknown vertices")
            if (a, b) in seen:
                continue
            seen.add((a, b))
            cost = edge[2] if costs and len(edge) > 2 else 0
            e = self.net.add_edge(self.left_id[a], self.right_id[b], 1, cost)
            self.edge_pairs[e] = (a, b)

    def matched_pairs(self) -> List[Tuple[Hashable, Hashable]]:
        return [pair for e, pair in self.edge_pairs.items()
                if self.net.flow_on(e) > 0.5]


def bipartite_matching(left: Sequence[Hashable], right: Sequence[Hashable],
                       edges: Iterable[Tuple[Hashable, Hashable]],
                       solver: str = "pr") -> MatchingResult:
    """
    Maximum-cardinality matching of the bipartite graph (left, right,
    edges) via max-flow with the named solver ("ek", "pr", ...).
    Duplicate edges are ignored.
    """
    model = _BipartiteNetwork(left, right, edges)
    result = get_solver(solver)(model.net, model.s, model.t)
    return MatchingResult(model.matched_pairs(), result)


def min_cost_assignment(left: Sequence[Hashable], right: Sequence[Hashable],
                        weighted_edges: Iterable[Tuple[Hashable, Hashable, float]]
                        ) -> MatchingResult:
    """
    Cheapest maximum matching: weighted_edges are (left, right, cost).
    Negative costs are allowed (to maximize weight, pass −weight).
    """
    model = _BipartiteNetwork(left, right, weighted_edges, costs=True)
    result = min_cost_max_flow(model.net, model.s, model.t)
    return MatchingResult(model.matched_pairs(), result, cost=result.cost)


def assignment_from_matrix(costs: Sequence[Sequence[float]]) -> MatchingResult:
    """Square or rectangular cost matrix → cheapest maximum assignment of
    rows (left, 0..r−1) to columns (right, 0..c−1)."""
    rows = range(len(costs))
    cols = range(len(costs[0]) if costs else 0)
    return min_cost_assignment(list(rows), list(cols),
                               [(i, j, costs[i][j]) for i in rows for j in cols])


def is_matching(pairs: Iterable[Tuple[Hashable, Hashable]],
                edges: Optional[Iterable[Tuple]] = None) -> bool:
    """No vertex used twice (and, if edges given, every pair is an edge)."""
    pairs = list(pairs)
    lefts = [a for a, _ in pairs]
    rights = [b for _, b in pairs]
    if len(set(lefts)) != len(lefts) or len(set(rights)) != len(rights):
        return False
    if edges is not None:
        allowed = {(e[0], e[1]) for e in edges}
        return all(p in allowed for p in pairs)
    return True
