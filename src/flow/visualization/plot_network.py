"""
Static drawings of flow networks and application results (matplotlib
+ networkx, headless).

    layered_positions   x = BFS distance from s, so flow reads left to right
    plot_network        nodes, arcs labelled "flow/capacity", arc width
                        by utilisation, min-cut arcs highlighted
    plot_matching       bipartite graph with the matching in bold
    plot_segmentation   image, ground truth and segmentation masks
    plot_schedule       Gantt chart of a ScheduleResult
"""
from collections import deque
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.flow.core.network import FlowNetwork

FLOW_COLOR = "#1f77b4"
IDLE_COLOR = "#b0b0b0"
CUT_COLOR = "#d62728"
PATH_COLOR = "#ff7f0e"


def layered_positions(net: FlowNetwork, s: int,
                      t: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
    """Column = BFS hop distance from s over original arcs; t last."""
    depth = {s: 0}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for e in net.adj[u]:
            if not e & 1 and net.to[e] not in depth:
                depth[net.to[e]] = depth[u] + 1
                queue.append(net.to[e])
    far = max(depth.values(), default=0) + 1
    for v in range(net.n):
        depth.setdefault(v, far)
    if t is not None:
        depth[t] = max(d for v, d in depth.items() if v != t) + 1
    columns: Dict[int, List[int]] = {}
    for v in range(net.n):
        columns.setdefault(depth[v], []).append(v)
    pos = {}
    for d, nodes in columns.items():
        k = len(nodes)
        for i, v in enumerate(nodes):
            pos[v] = (float(d), (k - 1) / 2 - i)
    return pos


def _aggregate(net: FlowNetwork, flows: Optional[Sequence[float]] = None):
    """{(u, v): [flow, capacity, [edge ids]]} over forward edges."""
    agg: Dict[Tuple[int, int], list] = {}
    for k, e in enumerate(range(0, len(net.to), 2)):
        u, v = net.to[e + 1], net.to[e]
        f = flows[k] if flows is not None else net.base[e] - net.cap[e]
        entry = agg.setdefault((u, v), [0, 0, []])
        entry[0] += f
        entry[1] += net.base[e]
        entry[2].append(e)
    return agg


def _num(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def draw_flow(ax, net: FlowNetwork, pos: Dict[int, Tuple[float, float]],
              flows: Optional[Sequence[float]] = None,
              highlight: Iterable[int] = (), cut: Iterable[int] = (),
              node_color=None, node_size: int = 650,
              labels: Optional[Dict[int, str]] = None,
              show_labels: bool = True) -> None:
    """Draw net on ax. highlight/cut are forward edge ids (twins are
    mapped to their forward edge)."""
    highlight = {e & ~1 for e in highlight}
    cut = {e & ~1 for e in cut}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(net.n))
    agg = _aggregate(net, flows)
    graph.add_edges_from(agg)
    for (u, v), (f, c, ids) in agg.items():
        if any(e in highlight for e in ids):
            color, width, style = PATH_COLOR, 3.5, "solid"
        elif any(e in cut for e in ids):
            color, width, style = CUT_COLOR, 2.5, "dashed"
        elif f > 1e-12:
            color, width, style = FLOW_COLOR, 1 + 3 * f / max(c, 1e-12), "solid"
        else:
            color, width, style = IDLE_COLOR, 1.0, "solid"
        rad = 0.18 if (v, u) in agg else 0.0
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], ax=ax,
                               edge_color=color, width=width, style=style,
                               arrowsize=14, node_size=node_size,
                               connectionstyle=f"arc3,rad={rad}")
        if show_labels:
            (x1, y1), (x2, y2) = pos[u], pos[v]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if rad:
                mx += rad * (y2 - y1) * 0.5
                my -= rad * (x2 - x1) * 0.5
            ax.text(mx, my, f"{_num(f)}/{_num(c)}", fontsize=8, ha="center",
                    va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85))
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size,
                           node_color=node_color if node_color is not None
                           else "#e8eef7", edgecolors="#333333")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9,
                            labels=labels or {v: str(v) for v in range(net.n)})
    ax.set_axis_off()


def plot_network(net: FlowNetwork, s: int, t: int,
                 pos: Optional[Dict[int, Tuple[float, float]]] = None,
                 show_cut: bool = True, title: Optional[str] = None,
                 labels: Optional[Dict[int, str]] = None,
                 save_path: Optional[str] = None, ax=None):
    """
    Draw the network's current flow: "f/c" on every arc, width by
    utilisation, and (show_cut) the min-cut arcs dashed red with the
    source side shaded. Returns the figure.
    """
    pos = pos or layered_positions(net, s, t)
    own = ax is None
    if own:
        width = 2.2 * (max(x for x, _ in pos.values()) + 1) + 2
        fig, ax = plt.subplots(figsize=(min(width, 16), 6))
    else:
        fig = ax.figure
    cut_edges, colors = (), None
    side = net.min_cut(s)
    if show_cut and t not in side:
        cut_edges = net.cut_edges(side)
        colors = ["#cfe3c8" if v in side else "#f3d9d9" for v in range(net.n)]
    names = dict(labels or {})
    names.setdefault(s, f"s ({s})" if s != 0 else "s")
    names.setdefault(t, f"t ({t})" if t != net.n - 1 else "t")
    for v in range(net.n):
        names.setdefault(v, str(v))
    draw_flow(ax, net, pos, cut=cut_edges, node_color=colors, labels=names)
    value = net.flow_value(s)
    ax.set_title(title or f"flow value {_num(value)}"
                 + (f" = min-cut capacity {_num(net.cut_capacity(side))}"
                    if cut_edges else ""))
    if own:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
    return fig


def plot_matching(left: Sequence[Hashable], right: Sequence[Hashable],
                  edges: Iterable[Tuple], pairs: Iterable[Tuple],
                  title: Optional[str] = None,
                  save_path: Optional[str] = None, ax=None):
    """Bipartite graph: left column, right column, matched edges bold."""
    left, right = list(left), list(right)
    pairs = {(a, b) for a, b in pairs}
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * max(len(left),
                                                            len(right)))))
    else:
        fig = ax.figure
    ly = {v: -i for i, v in enumerate(left)}
    ry = {v: -i * (len(left) - 1) / max(len(right) - 1, 1)
          for i, v in enumerate(right)}
    for a, b, *_ in edges:
        matched = (a, b) in pairs
        ax.plot([0, 1], [ly[a], ry[b]], color=FLOW_COLOR if matched else
                IDLE_COLOR, lw=2.6 if matched else 0.8,
                zorder=2 if matched else 1)
    matched_l = {a for a, _ in pairs}
    matched_r = {b for _, b in pairs}
    ax.scatter([0] * len(left), [ly[v] for v in left], s=90, zorder=3,
               c=[FLOW_COLOR if v in matched_l else "white" for v in left],
               edgecolors="#333333")
    ax.scatter([1] * len(right), [ry[v] for v in right], s=90, zorder=3,
               c=[FLOW_COLOR if v in matched_r else "white" for v in right],
               edgecolors="#333333")
    if len(left) + len(right) <= 60:
        for v in left:
            ax.text(-0.06, ly[v], str(v), ha="right", va="center", fontsize=8)
        for v in right:
            ax.text(1.06, ry[v], str(v), ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.35, 1.35)
    ax.set_axis_off()
    ax.set_title(title or f"maximum matching: {len(pairs)} pairs")
    if own:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
    return fig


def plot_segmentation(image: np.ndarray, masks: Dict[str, np.ndarray],
                      truth: Optional[np.ndarray] = None,
                      title: Optional[str] = None,
                      save_path: Optional[str] = None):
    """Input image, optional ground truth, then each labelled mask with
    its outline drawn over the image."""
    panels = [("input", image, "gray")]
    if truth is not None:
        panels.append(("ground truth", truth, "gray"))
    fig, axes = plt.subplots(1, len(panels) + len(masks),
                             figsize=(3.4 * (len(panels) + len(masks)), 3.6),
                             squeeze=False)
    axes = axes[0]
    for ax, (label, arr, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(label)
    for ax, (label, mask) in zip(axes[len(panels):], masks.items()):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.45)
        ax.contour(mask.astype(float), levels=[0.5], colors="red",
                   linewidths=1)
        ax.set_title(label)
    for ax in axes:
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_schedule(schedule, title: Optional[str] = None,
                  save_path: Optional[str] = None):
    """Gantt chart: one row per machine, one coloured block per job-slot."""
    machines = sorted(schedule.machines)
    jobs = sorted({job for row in schedule.machines.values()
                   for _, job in row}, key=str)
    cmap = plt.get_cmap("tab20")
    color = {job: cmap(i % 20) for i, job in enumerate(jobs)}
    horizon = len(schedule.slots)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * horizon + 2),
                                    1 + 0.6 * len(machines)))
    for m in machines:
        for slot, job in schedule.machines[m]:
            ax.barh(m, 1, left=slot, color=color[job], edgecolor="white")
            ax.text(slot + 0.5, m, str(job), ha="center", va="center",
                    fontsize=7)
    ax.set_yticks(machines)
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.set_xlim(0, max(horizon, 1))
    ax.set_xlabel("time slot")
    ax.invert_yaxis()
    status = "feasible" if schedule.feasible else \
        f"infeasible ({schedule.scheduled}/{schedule.required} units)"
    ax.set_title(title or f"schedule: {status}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
