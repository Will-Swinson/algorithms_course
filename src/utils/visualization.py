# File: src/utils/visualization.py
"""
Graph and traversal-order visualization built on networkx + matplotlib.

draw_traversal() renders the graph with nodes numbered and colored by
visit order, so BFS's expanding rings and DFS's deep probing are
visible at a glance. Kept separate from the algorithms: they return
plain visit-order lists, and this module only consumes them.

plot_dp_comparison() (Week 5) draws the recursive-vs-DP dashboard:
time, calls, speedup, and peak memory against input size.

plot_metric_panels() and plot_table_heatmap() (Week 6) are the
general-purpose versions: any grid of line charts with optional
dashed reference curves, and a DP table drawn as an annotated heatmap.
"""
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import networkx as nx

from src.graphs.graph import Graph


def _to_networkx(graph: Graph) -> "nx.Graph":
    """Convert our Graph into a networkx graph for layout/drawing."""
    nx_graph = nx.DiGraph() if graph.directed else nx.Graph()
    nx_graph.add_nodes_from(graph.nodes())
    for node in graph.nodes():
        for neighbor, weight in graph.get_neighbors(node):
            nx_graph.add_edge(node, neighbor, weight=weight)
    return nx_graph


def draw_graph(graph: Graph, save_path: str, title: str = "Graph",
               show_weights: bool = False, seed: int = 42) -> None:
    """
    Draw the graph with a spring layout and save it as a PNG.

    Args:
        graph: Graph to draw (best under ~100 nodes for legibility).
        save_path: Output PNG path.
        title: Figure title.
        show_weights: Label edges with their weights.
        seed: Layout seed, so the same graph always draws the same way.
    """
    nx_graph = _to_networkx(graph)
    positions = nx.spring_layout(nx_graph, seed=seed)

    plt.figure(figsize=(9, 7))
    nx.draw_networkx(nx_graph, positions, node_color="lightsteelblue",
                     node_size=550, font_size=9, edge_color="gray",
                     arrows=graph.directed)
    if show_weights:
        labels = nx.get_edge_attributes(nx_graph, "weight")
        nx.draw_networkx_edge_labels(nx_graph, positions,
                                     edge_labels=labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def draw_traversal(graph: Graph, order: List[Any], save_path: str,
                   title: str = "Traversal order",
                   path_edges: Optional[List] = None,
                   seed: int = 42) -> None:
    """
    Draw the graph with traversal order shown two ways: each visited
    node is labeled "label\\n#k" (k = visit position) and shaded from
    light (early) to dark (late). Unvisited nodes stay gray — which
    makes disconnected components obvious.

    Args:
        graph: The traversed graph.
        order: Visit order as returned by bfs()/dfs_iterative()/etc.
        save_path: Output PNG path.
        title: Figure title.
        path_edges: Optional [(u, v), ...] to highlight (e.g. a
            shortest path from reconstruct_path()).
        seed: Layout seed for reproducible drawings.
    """
    nx_graph = _to_networkx(graph)
    positions = nx.spring_layout(nx_graph, seed=seed)
    rank = {node: i for i, node in enumerate(order)}

    colors = []
    labels = {}
    for node in nx_graph.nodes():
        if node in rank:
            # 0.15 (early, light) -> 0.9 (late, dark) on the Blues map
            shade = 0.15 + 0.75 * (rank[node] / max(len(order) - 1, 1))
            colors.append(plt.cm.Blues(shade))
            labels[node] = f"{node}\n#{rank[node] + 1}"
        else:
            colors.append("lightgray")
            labels[node] = f"{node}\n–"

    plt.figure(figsize=(9, 7))
    nx.draw_networkx(nx_graph, positions, node_color=colors, labels=labels,
                     node_size=700, font_size=8, edge_color="gray",
                     arrows=graph.directed)
    if path_edges:
        nx.draw_networkx_edges(nx_graph, positions, edgelist=path_edges,
                               edge_color="crimson", width=2.5,
                               arrows=graph.directed)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Week 5: recursive vs dynamic-programming comparison
# ---------------------------------------------------------------------------

DP_COLORS = {
    "recursive": "tab:red",
    "memoization": "tab:blue",
    "tabulation": "tab:green",
    "bottom_up": "tab:green",
    "brute_force": "tab:red",
    "bitmask": "tab:green",
}


def plot_dp_comparison(series: Dict[str, Dict[str, Sequence[float]]],
                       save_path: str, title: str,
                       x_label: str = "Input size (n)",
                       baseline: str = "recursive",
                       log_x: bool = False) -> None:
    """
    Four-panel comparison of solution methods for one problem:
    time (log y), calls (log y), speedup over `baseline`, and peak
    memory. Methods may cover different sizes — typically the
    exponential baseline stops early while DP keeps going.

    Args:
        series: {method: {"sizes": [...], "time": [...],
                 "calls": [...], "memory": [...]}}. "memory" may be
                 omitted for a method or hold None where not measured.
        save_path: Output PNG path.
        title: Figure title.
        x_label: Label for the input-size axis.
        baseline: Method that speedup is measured against.
        log_x: Log-scale the x axis (for sizes spanning decades).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_time, ax_calls), (ax_speed, ax_mem) = axes

    for method, data in series.items():
        color = DP_COLORS.get(method)
        ax_time.plot(data["sizes"], data["time"], marker="o",
                     color=color, label=method)
        ax_calls.plot(data["sizes"], data["calls"], marker="o",
                      color=color, label=method)
        memory = data.get("memory")
        if memory:
            # 0-byte readings (nothing allocated) can't sit on a log axis
            points = [(s, m) for s, m in zip(data["sizes"], memory)
                      if m]
            if points:
                ax_mem.plot([p[0] for p in points],
                            [p[1] / 1024 for p in points],
                            marker="o", color=color, label=method)

    if baseline in series:
        base = dict(zip(series[baseline]["sizes"],
                        series[baseline]["time"]))
        for method, data in series.items():
            if method == baseline:
                continue
            points = [(s, base[s] / t) for s, t in
                      zip(data["sizes"], data["time"]) if s in base and t]
            if points:
                ax_speed.plot([p[0] for p in points],
                              [p[1] for p in points], marker="o",
                              color=DP_COLORS.get(method),
                              label=f"{method} vs {baseline}")

    panels = [
        (ax_time, "Time (s, log scale)", "Execution time"),
        (ax_calls, "Calls / cells evaluated (log scale)",
         "Subproblem evaluations"),
        (ax_speed, "Speedup (×, log scale)",
         f"Speedup over {baseline}"),
        (ax_mem, "Peak memory (KiB, log scale)",
         "Peak heap memory (tracemalloc)"),
    ]
    for ax, y_label, panel_title in panels:
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(panel_title)
        ax.grid(True, which="both", alpha=0.3)
        if ax.get_lines():
            ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Week 6: general metric grids and DP-table heatmaps
# ---------------------------------------------------------------------------

def plot_metric_panels(panels: List[Dict[str, Any]], save_path: str,
                       title: str, ncols: int = 2) -> None:
    """
    Grid of line charts, one per panel dict:

        {"title": str, "xlabel": str, "ylabel": str,
         "series": {label: (xs, ys)},          # solid lines with markers
         "refs": {label: (xs, ys)},            # optional dashed guides
         "colors": {label: color},             # optional, series + refs
         "logx": bool, "logy": bool}           # optional, default False

    Points whose y is None or non-positive on a log axis are dropped,
    so partially measured series still plot.
    """
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols,
                                                     4.5 * nrows),
                             squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        logy = panel.get("logy", False)
        colors = panel.get("colors", {})

        def clean(xs, ys):
            return [(x, y) for x, y in zip(xs, ys)
                    if y is not None and (y > 0 or not logy)]

        for label, (xs, ys) in panel["series"].items():
            pts = clean(xs, ys)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker="o", label=label, color=colors.get(label))
        for label, (xs, ys) in panel.get("refs", {}).items():
            pts = clean(xs, ys)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        linestyle="--", color=colors.get(label, "gray"),
                        alpha=0.7, label=label)
        if panel.get("logx"):
            ax.set_xscale("log")
            # label the measured sizes rather than sparse powers of 10
            xs_all = sorted({x for xs, _ in panel["series"].values()
                             for x in xs})
            if 0 < len(xs_all) <= 12:
                ax.set_xticks(xs_all)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        if logy:
            ax.set_yscale("log")
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel["ylabel"])
        ax.grid(True, which="both", alpha=0.3)
        if ax.get_lines():
            ax.legend(fontsize=9)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_table_heatmap(table: List[List[Optional[float]]], save_path: str,
                       title: str, row_labels: Sequence[Any],
                       col_labels: Sequence[Any],
                       annotations: Optional[List[List[str]]] = None,
                       cmap: str = "YlOrRd") -> None:
    """
    Draw a DP table as a heatmap. None cells (unused, e.g. the lower
    triangle of an interval-DP table) are left blank; `annotations`
    optionally overrides each cell's printed text.
    """
    import numpy as np

    data = np.array([[np.nan if v is None else v for v in row]
                     for row in table], dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(col_labels),
                                    0.8 + 0.7 * len(row_labels)))
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    finite = data[~np.isnan(data)]
    threshold = finite.max() * 0.6 if finite.size else 0
    for i, row in enumerate(table):
        for j, value in enumerate(row):
            if value is None:
                continue
            text = (annotations[i][j] if annotations
                    else f"{value:,.0f}")
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
