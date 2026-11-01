"""
Edmonds–Karp: Ford–Fulkerson with breadth-first augmenting paths.

    while the residual graph G_f has an s → t path:
        P ← a *shortest* such path (BFS, fewest edges)
        b ← min residual capacity on P          (the bottleneck)
        push b along P                          (2|P| residual updates)

Why BFS matters. Plain Ford–Fulkerson with arbitrary paths can take
time proportional to the flow value (or never stop, with irrational
capacities). With shortest paths, the BFS distance d_f(s, v) never
decreases for any v, and each augmentation saturates at least one
"critical" edge, which cannot be critical again until d_f(s, u) has
grown by 2. Each of the E edges is therefore critical O(V) times:

    O(V·E) augmentations × O(E) per BFS  =  O(V·E²)

independent of the capacities. The benchmark plots the path length
across augmentations to show the monotonicity directly.

capacity_scaling_max_flow (Ahuja–Orlin) restricts the search to the
Δ-residual graph (arcs with residual ≥ Δ), halving Δ from the largest
power of two ≤ U down to 1. Each phase has O(E) augmentations, giving
O(E² log U): big-capacity paths first, so far fewer augmentations when
capacities are spread over a wide range.

Both run in place on a FlowNetwork (see network.py) and return a
FlowResult with
    iterations                  number of augmentations
    stats["bottlenecks"]        bottleneck of each augmentation
    stats["path_lengths"]       edges on each augmenting path
    stats["residual_updates"]   residual capacities changed (2 per path edge)
    stats["nodes_scanned"]      nodes dequeued across all BFS calls
and, with record=True, a trace of {"path", "bottleneck", "value",
"flows"} per augmentation for animate_flow.
"""
from typing import List, Optional

from src.flow.core.network import FlowNetwork, FlowResult, check_terminals


def _bfs_path(net: FlowNetwork, s: int, t: int, threshold, parent: List[int],
              stamp: List[int], mark: int) -> int:
    """
    BFS from s over arcs with residual capacity >= threshold. Fills
    parent[v] = edge used to reach v; returns the number of nodes
    dequeued, or −(that number) − 1 if t was not reached.
    `stamp` avoids re-clearing parent between searches.
    """
    adj, to, cap = net.adj, net.to, net.cap
    stamp[s] = mark
    queue = [s]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for e in adj[u]:
            if cap[e] >= threshold:
                v = to[e]
                if stamp[v] != mark:
                    stamp[v] = mark
                    parent[v] = e
                    if v == t:
                        return head
                    queue.append(v)
    return -head - 1


def _augment(net: FlowNetwork, s: int, t: int, parent: List[int]):
    """Push the bottleneck along the parent path; return (b, path)."""
    to, cap = net.to, net.cap
    path = []
    v = t
    while v != s:
        e = parent[v]
        path.append(e)
        v = to[e ^ 1]
    b = min(cap[e] for e in path)
    for e in path:
        cap[e] -= b
        cap[e ^ 1] += b
    path.reverse()
    return b, path


def _forward_flows(net: FlowNetwork) -> List:
    base, cap = net.base, net.cap
    return [base[e] - cap[e] for e in range(0, len(base), 2)]


def _augmenting_paths(net: FlowNetwork, s: int, t: int, thresholds,
                      algorithm: str, record: bool,
                      max_iterations: Optional[int]) -> FlowResult:
    check_terminals(net, s, t)
    net.reset()
    n = net.n
    parent = [-1] * n
    stamp = [0] * n
    mark = 0
    value = 0
    bottlenecks, lengths, trace = [], [], []
    scanned = 0
    phases = []
    for threshold in thresholds:
        count = 0
        while max_iterations is None or len(bottlenecks) < max_iterations:
            mark += 1
            visited = _bfs_path(net, s, t, threshold, parent, stamp, mark)
            if visited < 0:
                scanned += -visited - 1
                break
            scanned += visited
            b, path = _augment(net, s, t, parent)
            value += b
            count += 1
            bottlenecks.append(b)
            lengths.append(len(path))
            if record:
                trace.append({"path": path, "bottleneck": b, "value": value,
                              "threshold": threshold,
                              "flows": _forward_flows(net)})
        phases.append((threshold, count))
    stats = {"bottlenecks": bottlenecks, "path_lengths": lengths,
             "residual_updates": 2 * sum(lengths), "nodes_scanned": scanned}
    if algorithm == "capacity_scaling":
        stats["phases"] = phases
    return net.result(s, algorithm, iterations=len(bottlenecks),
                      stats=stats, trace=trace)


def edmonds_karp(net: FlowNetwork, s: int, t: int, record: bool = False,
                 max_iterations: Optional[int] = None) -> FlowResult:
    """
    Maximum s–t flow by shortest augmenting paths. O(V·E²).

    `max_iterations` stops early (the flow is then feasible but maybe
    not maximum) — used to animate the first few steps of big inputs.
    """
    tiny = _smallest_positive(net)
    return _augmenting_paths(net, s, t, [tiny], "edmonds_karp", record,
                             max_iterations)


def capacity_scaling_max_flow(net: FlowNetwork, s: int, t: int,
                              record: bool = False) -> FlowResult:
    """
    Maximum flow by Δ-scaling augmenting paths. O(E² log U).

    Phases use Δ = 2^k, …, 2, 1; a final phase with Δ = the smallest
    residual quantum finishes fractional capacities (it finds nothing
    when every capacity is an integer).
    """
    positive = [c for c in net.base if c > 0]
    if not positive:
        return _augmenting_paths(net, s, t, [], "capacity_scaling", record, None)
    delta = 1
    while delta * 2 <= max(positive):
        delta *= 2
    thresholds = []
    while delta >= 1:
        thresholds.append(delta)
        delta //= 2
    if any(c != int(c) for c in positive):
        thresholds.append(_smallest_positive(net))
    return _augmenting_paths(net, s, t, thresholds, "capacity_scaling",
                             record, None)


def _smallest_positive(net: FlowNetwork):
    """
    Threshold that counts any genuinely positive residual: 1 for integer
    capacities, a small tolerance otherwise (floating-point leftovers
    below it are treated as saturated).
    """
    if all(c == int(c) for c in net.base):
        return 1
    return 1e-9 * max(max(net.base, default=1), 1)
