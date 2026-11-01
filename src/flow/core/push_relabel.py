"""
Push–relabel (Goldberg–Tarjan) maximum flow.

Instead of whole augmenting paths, push–relabel keeps a *preflow*:
nodes may hold more inflow than outflow (an excess), and work is
local to one node at a time.

    height[s] = n, every other height 0; saturate all edges out of s
    while some node u ∉ {s, t} has excess > 0 (u is *active*):
        discharge(u):
            push   along an admissible residual arc u → v
                   (height[u] = height[v] + 1): move min(excess, cap)
            relabel when u has no admissible arc:
                   height[u] = 1 + min height over residual arcs u → v

Heights are a valid labeling (height[u] ≤ height[v] + 1 on every
residual arc), so height[u] < n is a lower bound on u's distance to t
and height[u] − n a lower bound on its distance back to s. Heights
never exceed 2n − 1, giving O(V²) relabels, O(V·E) saturating pushes
and O(V³) non-saturating pushes for the FIFO rule — O(V³) overall,
versus O(V·E²) for Edmonds–Karp. Highest-label selection improves the
non-saturating bound to O(V²·√E).

Variants (``variant=``):
    "fifo"     active nodes processed in queue order
    "highest"  always discharge the active node with the greatest
               height (bucketed by height)

Two standard heuristics turn the theory into a fast code; both can be
switched off to measure their effect:
    global relabeling   every ~n relabels, recompute exact heights by
                        reverse BFS from t (and from s for nodes that
                        can no longer reach t)
    gap heuristic       if no node has height h < n, nobody above h can
                        reach t; lift them all to n + 1 at once

The algorithm continues until every excess has drained to t or back
to s, so the result is a genuine flow (not just a preflow) and the
flow assignment can be validated like any other.

Counters in FlowResult.stats: pushes, saturating_pushes, relabels,
global_relabels, gap_nodes (nodes lifted by gaps), discharges (also
FlowResult.iterations). profile_every=k records (operations, flow at
t, active nodes) every k operations in stats["profile"] — the
convergence curves in the benchmark. record=True stores a snapshot
of heights, excesses and flows after every push/relabel for
animate_flow.
"""
from collections import deque
from typing import List

from src.flow.core.network import FlowNetwork, FlowResult, check_terminals

VARIANTS = ("fifo", "highest")


def push_relabel(net: FlowNetwork, s: int, t: int, variant: str = "fifo",
                 global_relabel: bool = True, gap: bool = True,
                 global_relabel_freq: float = 1.0, record: bool = False,
                 profile_every: int = 0) -> FlowResult:
    """
    Maximum s–t flow by push–relabel.

    Args:
        variant: "fifo" or "highest" (highest-label selection).
        global_relabel / gap: enable the two heuristics.
        global_relabel_freq: run a global relabel after this many × n
            relabels.
        record: keep per-operation snapshots (small networks only).
        profile_every: sample (ops, flow at t, active count) this often.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    check_terminals(net, s, t)
    net.reset()
    n = net.n
    adj, to, cap = net.adj, net.to, net.cap
    height = [0] * n
    excess = [0] * n
    current = [0] * n
    sentinel = 2 * n                     # nodes that reach neither s nor t
    height[s] = n
    # integer capacities are exact; with floats, amounts below eps are
    # rounding dust (treated as zero so the loop cannot chase them)
    eps = 0 if all(c == int(c) for c in net.base) else \
        1e-12 * max(max(net.base, default=1), 1)

    stats = {"pushes": 0, "saturating_pushes": 0, "relabels": 0,
             "global_relabels": 0, "gap_nodes": 0, "discharges": 0}
    profile: List = []
    trace: List = []

    # bucket[h] = set of nodes at height h < n (for the gap heuristic)
    bucket = [set() for _ in range(n)] if gap else None

    fifo = variant == "fifo"
    queue: deque = deque()
    active_at = [[] for _ in range(2 * n + 1)] if not fifo else None
    in_active = [False] * n
    state = {"max_active": -1, "num_active": 0, "ops": 0}

    def activate(v: int) -> None:
        if in_active[v] or v == s or v == t:
            return
        in_active[v] = True
        state["num_active"] += 1
        if fifo:
            queue.append(v)
        else:
            h = height[v]
            active_at[h].append(v)
            if h > state["max_active"]:
                state["max_active"] = h

    def snapshot(op: str, u: int, edge: int = -1, amount=0) -> None:
        trace.append({"op": op, "node": u, "edge": edge, "amount": amount,
                      "heights": list(height), "excess": list(excess),
                      "flows": [net.base[e] - cap[e]
                                for e in range(0, len(cap), 2)],
                      "value": excess[t]})

    def sample() -> None:
        profile.append((state["ops"], excess[t], state["num_active"]))

    def rebuild_heights() -> None:
        """Global relabel: exact residual distances to t (or to s, + n)."""
        stats["global_relabels"] += 1
        for v in range(n):
            height[v] = sentinel
        height[t] = 0
        frontier = [t]
        for root, offset in ((t, 0), (s, n)):
            height[root] = offset
            frontier = [root]
            head = 0
            while head < len(frontier):
                v = frontier[head]
                head += 1
                hv = height[v] + 1
                for e in adj[v]:
                    w = to[e]
                    # w → v is residual iff the twin of (v → w) has capacity
                    if height[w] == sentinel and cap[e ^ 1] > eps and w != s:
                        height[w] = hv
                        frontier.append(w)
        height[s] = n
        for v in range(n):
            current[v] = 0
        if gap:
            for b in bucket:
                b.clear()
            for v in range(n):
                if height[v] < n and v != t:
                    bucket[height[v]].add(v)
        # active structures are keyed by height: rebuild them
        if fifo:
            pass                          # queue order does not depend on height
        else:
            for lst in active_at:
                lst.clear()
            state["max_active"] = -1
            for v in range(n):
                if in_active[v]:
                    h = height[v]
                    active_at[h].append(v)
                    if h > state["max_active"]:
                        state["max_active"] = h

    def relabel(u: int) -> None:
        old = height[u]
        best = sentinel
        for e in adj[u]:
            if cap[e] > eps:
                h = height[to[e]]
                if h < best:
                    best = h
        new = best + 1
        stats["relabels"] += 1
        state["ops"] += 1
        if gap and old < n:
            b = bucket[old]
            b.discard(u)
            if not b and old > 0:         # t always occupies height 0
                # gap at `old`: every node above it (and below n) is cut off
                # from t; u itself is above the gap after its relabel
                lifted = n + 1
                if new < lifted:
                    new = lifted
                for h in range(old + 1, n):
                    hb = bucket[h]
                    if hb:
                        for v in hb:
                            if height[v] < lifted:
                                height[v] = lifted
                                current[v] = 0
                                stats["gap_nodes"] += 1
                                if not fifo and in_active[v]:
                                    _move_active(v, h, lifted)
                        hb.clear()
            elif new < n:
                bucket[new].add(u)
        height[u] = new
        current[u] = 0
        if record:
            snapshot("relabel", u)

    def _move_active(v: int, old_h: int, new_h: int) -> None:
        lst = active_at[old_h]
        lst.remove(v)
        active_at[new_h].append(v)
        if new_h > state["max_active"]:
            state["max_active"] = new_h

    # ---- initialise: saturate every arc out of s --------------------
    for e in adj[s]:
        c = cap[e]
        if c > 0:
            v = to[e]
            cap[e] = 0
            cap[e ^ 1] += c
            excess[v] += c
            excess[s] -= c
    if global_relabel:
        rebuild_heights()
        stats["global_relabels"] = 0      # the initial labeling is free
    elif gap:
        for v in range(n):
            if v != s and v != t:
                bucket[0].add(v)
    for e in adj[s]:
        if excess[to[e]] > eps:
            activate(to[e])
    if record:
        snapshot("init", s)

    relabel_limit = max(1, int(global_relabel_freq * n))
    relabels_since = 0

    while state["num_active"]:
        # ---- select an active node ----------------------------------
        if fifo:
            u = queue.popleft()
        else:
            m = state["max_active"]
            while not active_at[m]:
                m -= 1
            state["max_active"] = m
            u = active_at[m].pop()
        in_active[u] = False
        state["num_active"] -= 1
        stats["discharges"] += 1

        # ---- discharge u ---------------------------------------------
        edges = adj[u]
        deg = len(edges)
        while excess[u] > eps:
            i = current[u]
            if i == deg:
                relabel(u)
                relabels_since += 1
                if height[u] >= sentinel:
                    # no residual arc at all: only possible for float dust
                    stats["stranded"] = stats.get("stranded", 0) + excess[u]
                    break
                if global_relabel and relabels_since >= relabel_limit:
                    relabels_since = 0
                    rebuild_heights()
                continue
            e = edges[i]
            r = cap[e]
            if r > eps:
                v = to[e]
                if height[u] == height[v] + 1:
                    d = excess[u] if excess[u] < r else r
                    cap[e] = r - d
                    cap[e ^ 1] += d
                    excess[u] -= d
                    was_idle = excess[v] <= eps
                    excess[v] += d
                    stats["pushes"] += 1
                    state["ops"] += 1
                    if d == r:
                        stats["saturating_pushes"] += 1
                    if was_idle and excess[v] > eps:
                        activate(v)
                    if record:
                        snapshot("push", u, e, d)
                    if profile_every and state["ops"] % profile_every == 0:
                        sample()
                    if d < r:
                        break             # u drained; keep this arc current
                    current[u] = i + 1
                    continue
            current[u] = i + 1
        if excess[u] > eps and height[u] < sentinel:
            activate(u)                   # left early after a global relabel

    if profile_every:
        sample()
    stats["profile"] = profile
    return net.result(s, f"push_relabel_{variant}",
                      iterations=stats["discharges"], stats=stats,
                      trace=trace)
