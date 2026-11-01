"""
Minimum-cost maximum flow by successive shortest augmenting paths.

Each edge has a per-unit cost. Among all maximum flows (or all flows
of a required value), find the cheapest:

    while the residual graph has an s → t path and more flow is wanted:
        P ← the *cheapest* s → t path in G_f (costs, not hop counts)
        push the bottleneck along P

Augmenting along a cheapest path keeps the flow optimal for its value
— there is never a negative-cost cycle in G_f — so the flow after k
units is a min-cost flow of value k, and the per-unit path cost is
non-decreasing (the min-cost curve is convex: the cost/flow trade-off
the benchmark plots).

Residual arcs have negative costs (cancelling flow refunds it), which
rules out plain Dijkstra. Johnson's trick fixes that: keep a node
potential π and search on reduced costs

    c_π(u, v) = c(u, v) + π(u) − π(v) ≥ 0,

which change every path's cost by the same π(s) − π(t), so shortest
paths are unchanged. After each Dijkstra, π(v) += dist(v) keeps every
residual reduced cost non-negative, including the new reverse arcs
(they lie on a shortest path, so their reduced cost is exactly 0).
Nodes the search did not reach (or reached beyond t) get
π(v) += dist(t) instead — truncating at dist(t) keeps *every* residual
reduced cost non-negative, not only those on reachable arcs, so the
final π is a certificate for the whole graph. Initial potentials come
from Bellman–Ford (from a virtual root joined to every node) when the
input has negative costs, zero otherwise; a negative-cost cycle is
reported as an error, because flow could circulate around it forever
and the min-cost problem would be unbounded.

Running time O(F · E log V) for flow value F with integer capacities
— pseudo-polynomial: fine when F is moderate, which is why the
benchmark runs it on smaller instances than the max-flow solvers.

The final potentials are an optimality certificate: every residual
arc has reduced cost ≥ 0 (validators.check_min_cost verifies it).

FlowResult extras: cost, iterations (augmentations), and
stats["path_costs"] (cost per unit of each augmenting path),
stats["bottlenecks"], stats["potentials"], stats["dijkstra_pops"],
stats["feasible"] (requested flow achieved).
"""
import heapq
import math
from typing import List, Optional

from src.flow.core.network import FlowNetwork, FlowResult, check_terminals


def _bellman_ford(net: FlowNetwork) -> List[float]:
    """Shortest residual distances from a virtual root with a 0-cost arc
    to every node (valid potentials); raises on a negative cycle."""
    n = net.n
    dist = [0] * n
    to, cap, cost = net.to, net.cap, net.cost
    edges = [e for e in range(len(to)) if cap[e] > 0]
    for _ in range(n):
        changed = False
        for e in edges:
            nd = dist[to[e ^ 1]] + cost[e]
            v = to[e]
            if nd < dist[v]:
                dist[v] = nd
                changed = True
        if not changed:
            return dist
    raise ValueError("negative-cost cycle in the network: "
                     "min-cost flow is unbounded")


def min_cost_max_flow(net: FlowNetwork, s: int, t: int,
                      flow_limit: Optional[float] = None,
                      record: bool = False) -> FlowResult:
    """
    Cheapest flow of value min(flow_limit, max flow) from s to t.

    flow_limit=None sends the full maximum flow.
    """
    check_terminals(net, s, t)
    if flow_limit is not None and flow_limit < 0:
        raise ValueError("flow_limit must be non-negative")
    net.reset()
    n = net.n
    adj, to, cap, cost = net.adj, net.to, net.cap, net.cost
    if any(cost[e] < 0 and cap[e] > 0 for e in range(0, len(to), 2)):
        pi = _bellman_ford(net)
    else:
        pi = [0] * n

    value = 0
    total = 0
    path_costs, bottlenecks, trace = [], [], []
    pops = 0
    inf = math.inf
    while flow_limit is None or value < flow_limit:
        dist = [inf] * n
        parent = [-1] * n
        dist[s] = 0
        heap = [(0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            pops += 1
            pu = pi[u]
            for e in adj[u]:
                if cap[e] > 0:
                    v = to[e]
                    nd = d + cost[e] + pu - pi[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        parent[v] = e
                        heapq.heappush(heap, (nd, v))
        if dist[t] == inf:
            break
        dt = dist[t]
        for v in range(n):
            pi[v] += dist[v] if dist[v] < dt else dt

        path = []
        v = t
        while v != s:
            e = parent[v]
            path.append(e)
            v = to[e ^ 1]
        b = min(cap[e] for e in path)
        if flow_limit is not None:
            b = min(b, flow_limit - value)
        unit = sum(cost[e] for e in path)
        for e in path:
            cap[e] -= b
            cap[e ^ 1] += b
        value += b
        total += b * unit
        path_costs.append(unit)
        bottlenecks.append(b)
        if record:
            path.reverse()
            trace.append({"path": path, "bottleneck": b, "value": value,
                          "unit_cost": unit, "cost": total,
                          "flows": [net.base[e] - cap[e]
                                    for e in range(0, len(cap), 2)]})

    stats = {"path_costs": path_costs, "bottlenecks": bottlenecks,
             "potentials": pi, "dijkstra_pops": pops,
             "feasible": flow_limit is None or value >= flow_limit}
    return net.result(s, "min_cost_flow", iterations=len(path_costs),
                      cost=total, stats=stats, trace=trace)


def min_cost_flow(net: FlowNetwork, s: int, t: int, amount: float,
                  record: bool = False) -> FlowResult:
    """
    Cheapest way to send exactly `amount` units; stats["feasible"] is
    False (and the result holds the max flow instead) if the network
    cannot carry that much.
    """
    return min_cost_max_flow(net, s, t, flow_limit=amount, record=record)
