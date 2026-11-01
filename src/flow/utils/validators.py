"""
Flow validation: independent checks that a FlowNetwork's current state
is a feasible — and, optionally, maximum or minimum-cost — flow.

Every check returns a list of human-readable violations (empty means
it passed), so tests can assert on them and the CLI can print them.

    check_capacity      0 ≤ f(e) ≤ c(e) on every forward edge
    check_conservation  inflow = outflow at every node except s and t,
                        and what leaves s arrives at t
    check_residual      residual capacities are consistent with the
                        flow: r(e) ≥ 0 and r(e) + r(e^1) = c(e) + c(e^1)
    check_max_flow      max-flow min-cut certificate: t unreachable in
                        G_f, and the cut (reachable set) has capacity
                        exactly |f| — together they prove optimality
    check_min_cost      min-cost optimality: no negative-cost cycle in
                        G_f. With potentials π (from min_cost_max_flow)
                        it is an O(E) check of reduced costs ≥ 0;
                        without, Bellman–Ford looks for a cycle

validate_flow() runs the requested checks and returns a
ValidationReport (truthy when everything passed).
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from src.flow.core.network import FlowNetwork


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if self.ok:
            return f"valid ({', '.join(self.checks)})"
        return "INVALID:\n  " + "\n  ".join(self.errors)


def _limit(errors: List[str], cap: int = 20) -> List[str]:
    if len(errors) > cap:
        return errors[:cap] + [f"... and {len(errors) - cap} more"]
    return errors


def check_capacity(net: FlowNetwork, tol: float = 1e-9) -> List[str]:
    errors = []
    for edge in net.edges():
        if edge.flow < -tol:
            errors.append(f"edge {edge.u}->{edge.v} (id {edge.id}) has "
                          f"negative flow {edge.flow}")
        elif edge.flow > edge.capacity + tol:
            errors.append(f"edge {edge.u}->{edge.v} (id {edge.id}) carries "
                          f"{edge.flow} > capacity {edge.capacity}")
    return _limit(errors)


def check_conservation(net: FlowNetwork, s: int, t: int,
                       tol: float = 1e-9) -> List[str]:
    excess = net.excess()
    errors = [f"node {v} violates conservation (in − out = {x})"
              for v, x in enumerate(excess)
              if v != s and v != t and abs(x) > tol]
    if abs(excess[s] + excess[t]) > tol:
        errors.append(f"flow out of s ({-excess[s]}) != flow into t "
                      f"({excess[t]})")
    return _limit(errors)


def check_residual(net: FlowNetwork, tol: float = 1e-9) -> List[str]:
    errors = []
    to, cap, base = net.to, net.cap, net.base
    for e in range(0, len(to), 2):
        if cap[e] < -tol or cap[e + 1] < -tol:
            errors.append(f"edge id {e} has negative residual capacity")
        if abs(cap[e] + cap[e + 1] - base[e] - base[e + 1]) > tol:
            errors.append(f"edge id {e}: residual pair sums to "
                          f"{cap[e] + cap[e + 1]}, expected {base[e]}")
    for u, arcs in enumerate(net.adj):
        for e in arcs:
            if to[e ^ 1] != u:
                errors.append(f"adjacency of {u} lists edge {e} that "
                              f"starts at {to[e ^ 1]}")
    return _limit(errors)


def check_max_flow(net: FlowNetwork, s: int, t: int,
                   value: Optional[float] = None,
                   tol: float = 1e-9) -> List[str]:
    errors = []
    side = net.min_cut(s, tol)
    flow = net.flow_value(s)
    if value is not None and abs(value - flow) > tol * max(1, abs(flow)):
        errors.append(f"reported value {value} != flow in network {flow}")
    if t in side:
        errors.append("an augmenting path s -> t remains: flow is not maximum")
    else:
        cut = net.cut_capacity(side)
        if abs(cut - flow) > tol * max(1, abs(cut)):
            errors.append(f"cut capacity {cut} != flow value {flow}")
    return errors


def check_min_cost(net: FlowNetwork,
                   potentials: Optional[Sequence[float]] = None,
                   tol: float = 1e-9) -> List[str]:
    to, cap, cost = net.to, net.cap, net.cost
    arcs = [e for e in range(len(to)) if cap[e] > tol]
    if potentials is not None:
        bad = [e for e in arcs
               if cost[e] + potentials[to[e ^ 1]] - potentials[to[e]] < -tol]
        return _limit([f"residual arc {to[e ^ 1]}->{to[e]} (id {e}) has "
                       f"negative reduced cost" for e in bad])
    # Bellman–Ford from a virtual root joined to every node at cost 0
    dist = [0.0] * net.n
    for _ in range(net.n):
        changed = False
        for e in arcs:
            nd = dist[to[e ^ 1]] + cost[e]
            if nd < dist[to[e]] - tol:
                dist[to[e]] = nd
                changed = True
        if not changed:
            return []
    return ["residual graph has a negative-cost cycle: flow is not min-cost"]


def validate_flow(net: FlowNetwork, s: int, t: int,
                  value: Optional[float] = None, max_flow: bool = False,
                  min_cost: bool = False,
                  potentials: Optional[Sequence[float]] = None,
                  tol: Optional[float] = None) -> ValidationReport:
    """
    Check the network's current flow. Capacity, conservation and
    residual consistency are always checked; pass max_flow=True for the
    min-cut certificate and min_cost=True for cost optimality.
    """
    if tol is None:
        biggest = max((abs(c) for c in net.base), default=1)
        tol = 1e-9 * max(1.0, biggest, math.sqrt(len(net.base)))
    report = ValidationReport()
    report.errors += check_capacity(net, tol)
    report.errors += check_conservation(net, s, t, tol * max(1, net.n))
    report.errors += check_residual(net, tol)
    report.checks += ["capacity", "conservation", "residual"]
    if max_flow:
        report.errors += check_max_flow(net, s, t, value, tol * max(1, net.n))
        report.checks.append("max-flow certificate")
    if min_cost:
        report.errors += check_min_cost(net, potentials, tol * max(1, net.n))
        report.checks.append("min-cost optimality")
    return report


def assert_valid(net: FlowNetwork, s: int, t: int, **checks) -> None:
    """validate_flow, raising AssertionError on any violation."""
    report = validate_flow(net, s, t, **checks)
    if not report:
        raise AssertionError(str(report))
