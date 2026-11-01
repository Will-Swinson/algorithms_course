"""
Flow network: a directed graph with edge capacities (and optional
per-unit costs), stored as its own residual graph.

Representation. Edges live in parallel arrays indexed by edge id.
Every add_edge(u, v, c) appends a forward edge 2k (u → v, capacity c)
and its residual twin 2k + 1 (v → u, capacity 0), so the twin of edge
e is always e ^ 1. `cap[e]` holds the *residual* capacity; sending f
units along e is

    cap[e] -= f;   cap[e ^ 1] += f

and the flow on forward edge e is base[e] − cap[e]. Costs are
antisymmetric (cost[e ^ 1] = −cost[e]), so cancelling flow refunds its
cost. adj[u] lists every edge id leaving u in the residual graph,
forward and backward alike.

This is the layout used by most production max-flow codes: O(1)
access to an edge's reverse, no hashing in the inner loops, and the
residual graph never has to be rebuilt. Parallel and antiparallel
edges (u → v and v → u) are allowed and each keeps its own twin;
self-loops are rejected because they can never carry useful flow.

The solvers in this package work *in place*: they call reset(), then
leave the network holding the final flow, so the validators and the
min-cut helpers can inspect it afterwards.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

Number = float


@dataclass(frozen=True)
class Edge:
    """A forward edge as seen from outside: capacity, cost, current flow."""
    id: int
    u: int
    v: int
    capacity: Number
    cost: Number
    flow: Number


@dataclass(frozen=True)
class ResidualArc:
    """One arc of the residual graph with positive residual capacity."""
    v: int
    capacity: Number
    cost: Number
    edge: int
    forward: bool          # True: unused forward capacity; False: cancellable flow


@dataclass
class FlowResult:
    """
    What every solver returns.

    value       total flow from s to t
    flow        {(u, v): flow} on forward edges carrying flow (parallel
                edges summed); flow_matrix() gives the dense form
    min_cut     source side of a minimum s–t cut (max-flow certificate)
    cost        total cost (min-cost flow only, else None)
    iterations  augmentations (path algorithms) or discharges (push–relabel)
    stats       algorithm-specific counters, documented by each solver
    trace       per-step snapshots when the solver ran with record=True
    """
    value: Number
    flow: Dict[Tuple[int, int], Number]
    min_cut: FrozenSet[int]
    n: int
    algorithm: str
    iterations: int = 0
    cost: Optional[Number] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def flow_matrix(self) -> np.ndarray:
        """Dense n × n matrix F with F[u, v] = flow on u → v."""
        return flow_matrix_from(self.flow, self.n)


def flow_matrix_from(flow: Dict[Tuple[int, int], Number], n: int) -> np.ndarray:
    matrix = np.zeros((n, n))
    for (u, v), f in flow.items():
        matrix[u, v] += f
    return matrix


class FlowNetwork:
    """Directed capacitated network with automatic residual edges."""

    def __init__(self, n: int = 0):
        if n < 0:
            raise ValueError("number of nodes must be non-negative")
        self.adj: List[List[int]] = [[] for _ in range(n)]
        self.to: List[int] = []
        self.cap: List[Number] = []      # residual capacity
        self.base: List[Number] = []     # original capacity (0 on twins)
        self.cost: List[Number] = []

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.adj)

    @property
    def num_edges(self) -> int:
        """Forward edges (each also has a residual twin)."""
        return len(self.to) // 2

    def add_node(self) -> int:
        self.adj.append([])
        return len(self.adj) - 1

    def add_nodes(self, k: int) -> range:
        start = self.n
        self.adj.extend([] for _ in range(k))
        return range(start, start + k)

    def add_edge(self, u: int, v: int, capacity: Number,
                 cost: Number = 0) -> int:
        """
        Add u → v and its residual twin v → u; return the forward id.

        Capacity must be finite and non-negative (use a large finite
        number for "infinite" edges, so flows stay well defined).
        """
        n = len(self.adj)
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) has an endpoint outside 0..{n - 1}")
        if u == v:
            raise ValueError(f"self-loop at node {u}")
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError(f"capacity must be finite and >= 0, got {capacity}")
        if not math.isfinite(cost):
            raise ValueError(f"cost must be finite, got {cost}")
        e = len(self.to)
        self.to += (v, u)
        self.cap += (capacity, 0)
        self.base += (capacity, 0)
        self.cost += (cost, -cost)
        self.adj[u].append(e)
        self.adj[v].append(e + 1)
        return e

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Tuple]) -> "FlowNetwork":
        """Build from (u, v, capacity) or (u, v, capacity, cost) tuples."""
        net = cls(n)
        for edge in edges:
            net.add_edge(*edge)
        return net

    def copy(self) -> "FlowNetwork":
        other = FlowNetwork()
        other.adj = [list(a) for a in self.adj]
        other.to = list(self.to)
        other.cap = list(self.cap)
        other.base = list(self.base)
        other.cost = list(self.cost)
        return other

    # ------------------------------------------------------------------
    # residual operations
    # ------------------------------------------------------------------
    def tail(self, e: int) -> int:
        return self.to[e ^ 1]

    def reset(self) -> None:
        """Zero flow: every residual capacity back to its original value."""
        self.cap = list(self.base)

    def push(self, e: int, amount: Number) -> None:
        """Send `amount` along residual edge e (forward or twin)."""
        if amount < 0 or amount > self.cap[e] + 1e-12:
            raise ValueError(f"cannot push {amount} on edge {e} "
                             f"(residual {self.cap[e]})")
        self.cap[e] -= amount
        self.cap[e ^ 1] += amount

    def flow_on(self, e: int) -> Number:
        """Flow on forward edge e."""
        if e & 1:
            raise ValueError("flow_on() takes a forward edge id (even)")
        return self.base[e] - self.cap[e]

    def edges(self) -> Iterator[Edge]:
        """Every forward edge with its capacity, cost, and current flow."""
        to, cap, base, cost = self.to, self.cap, self.base, self.cost
        for e in range(0, len(to), 2):
            yield Edge(e, to[e + 1], to[e], base[e], cost[e], base[e] - cap[e])

    def get_residual_graph(self, tol: float = 0.0) -> Dict[int, List[ResidualArc]]:
        """
        The residual graph G_f as adjacency lists: for each node, the
        arcs with residual capacity > tol. Forward arcs carry unused
        capacity; backward arcs carry flow that can be cancelled.
        """
        to, cap, cost = self.to, self.cap, self.cost
        return {u: [ResidualArc(to[e], cap[e], cost[e], e, not e & 1)
                    for e in self.adj[u] if cap[e] > tol]
                for u in range(self.n)}

    # ------------------------------------------------------------------
    # flow inspection
    # ------------------------------------------------------------------
    def excess(self) -> List[Number]:
        """inflow − outflow at every node (0 everywhere but s, t in a flow)."""
        exc = [0] * self.n
        to, cap, base = self.to, self.cap, self.base
        for e in range(0, len(to), 2):
            f = base[e] - cap[e]
            if f:
                exc[to[e]] += f
                exc[to[e + 1]] -= f
        return exc

    def flow_value(self, s: int) -> Number:
        """Net flow leaving s."""
        return -self.excess()[s]

    def total_cost(self) -> Number:
        to, cap, base, cost = self.to, self.cap, self.base, self.cost
        return sum((base[e] - cap[e]) * cost[e] for e in range(0, len(to), 2))

    def flow_dict(self, tol: float = 0.0) -> Dict[Tuple[int, int], Number]:
        """{(u, v): flow} for forward edges carrying more than tol."""
        out: Dict[Tuple[int, int], Number] = {}
        to, cap, base = self.to, self.cap, self.base
        for e in range(0, len(to), 2):
            f = base[e] - cap[e]
            if f > tol:
                key = (to[e + 1], to[e])
                out[key] = out.get(key, 0) + f
        return out

    def flow_matrix(self) -> np.ndarray:
        """Dense n × n flow assignment matrix (small networks only)."""
        return flow_matrix_from(self.flow_dict(), self.n)

    def reachable(self, s: int, tol: float = 0.0) -> Set[int]:
        """Nodes reachable from s through arcs with residual capacity > tol."""
        seen = [False] * self.n
        seen[s] = True
        queue = deque([s])
        to, cap, adj = self.to, self.cap, self.adj
        while queue:
            u = queue.popleft()
            for e in adj[u]:
                v = to[e]
                if not seen[v] and cap[e] > tol:
                    seen[v] = True
                    queue.append(v)
        return {v for v in range(self.n) if seen[v]}

    def min_cut(self, s: int, tol: float = 0.0) -> FrozenSet[int]:
        """
        Source side S of the cut induced by the current flow: everything
        reachable from s in the residual graph. When the flow is maximum,
        t ∉ S and cap(S, V∖S) equals the flow value (max-flow min-cut).
        """
        return frozenset(self.reachable(s, tol))

    def cut_capacity(self, source_side: Iterable[int]) -> Number:
        """Total original capacity of forward edges leaving source_side."""
        inside = [False] * self.n
        for v in source_side:
            inside[v] = True
        to, base = self.to, self.base
        return sum(base[e] for e in range(0, len(to), 2)
                   if inside[to[e + 1]] and not inside[to[e]])

    def cut_edges(self, source_side: Iterable[int]) -> List[int]:
        inside = set(source_side)
        return [e for e in range(0, len(self.to), 2)
                if self.to[e + 1] in inside and self.to[e] not in inside]

    def validate(self, s: int, t: int, **checks) -> "Any":
        """Run utils.validators.validate_flow on the current flow."""
        from src.flow.utils.validators import validate_flow
        return validate_flow(self, s, t, **checks)

    def result(self, s: int, algorithm: str, **kwargs) -> FlowResult:
        """Package the current flow as a FlowResult."""
        return FlowResult(value=self.flow_value(s), flow=self.flow_dict(),
                          min_cut=self.min_cut(s), n=self.n,
                          algorithm=algorithm, **kwargs)

    def __repr__(self) -> str:
        return f"FlowNetwork(n={self.n}, edges={self.num_edges})"


def check_terminals(net: FlowNetwork, s: int, t: int) -> None:
    if not (0 <= s < net.n and 0 <= t < net.n):
        raise ValueError(f"source {s} / sink {t} outside 0..{net.n - 1}")
    if s == t:
        raise ValueError("source and sink must differ")
