"""
Animations of the max-flow algorithms, built from the per-step traces
the solvers record with record=True.

    animate_edmonds_karp   one frame per augmentation: the shortest
                           augmenting path in orange, its bottleneck,
                           and the flow so far on every arc
    animate_push_relabel   one frame per push / relabel: each node is
                           drawn at its current *height* (y axis), so
                           relabels visibly lift nodes and flow runs
                           downhill; node colour shows excess (orange =
                           active), the pushed arc is highlighted
    storyboard             the same frames as a static grid PNG, for
                           reports where a GIF cannot play

GIFs are written with matplotlib's PillowWriter. Meant for small
networks (≲ 15 nodes); bigger traces are subsampled to max_frames.
"""
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.flow.core import edmonds_karp, push_relabel
from src.flow.core.network import FlowNetwork
from src.flow.visualization.plot_network import draw_flow, layered_positions

ACTIVE_COLOR = "#ffb347"
IDLE_NODE = "#e8eef7"
TERMINAL_COLOR = "#cfe3c8"


def _subsample(frames: List, max_frames: int) -> List:
    if len(frames) <= max_frames:
        return frames
    step = (len(frames) - 1) / (max_frames - 1)
    return [frames[round(i * step)] for i in range(max_frames)]


# ----------------------------------------------------------------------
# Edmonds–Karp
# ----------------------------------------------------------------------
def edmonds_karp_frames(net: FlowNetwork, s: int, t: int) -> List[Dict]:
    """Frame dicts: initial zero flow, then one per augmentation."""
    result = edmonds_karp(net, s, t, record=True)
    zero = [0] * net.num_edges
    frames = [{"flows": zero, "path": [], "title": "start: zero flow"}]
    for k, step in enumerate(result.trace, 1):
        frames.append({"flows": step["flows"], "path": step["path"],
                       "title": f"augmentation {k}: path of "
                                f"{len(step['path'])} arcs, bottleneck "
                                f"{step['bottleneck']:g}, |f| = "
                                f"{step['value']:g}"})
    frames.append({"flows": frames[-1]["flows"], "path": [],
                   "cut": net.cut_edges(result.min_cut),
                   "title": f"done: max flow {result.value:g} = min-cut "
                            f"capacity (cut arcs dashed)"})
    return frames


def _draw_ek(ax, net, pos, frame, s, t) -> None:
    ax.clear()
    colors = [TERMINAL_COLOR if v in (s, t) else IDLE_NODE
              for v in range(net.n)]
    draw_flow(ax, net, pos, flows=frame["flows"], highlight=frame["path"],
              cut=frame.get("cut", ()), node_color=colors)
    ax.set_title(frame["title"], fontsize=10)


# ----------------------------------------------------------------------
# push–relabel
# ----------------------------------------------------------------------
def push_relabel_frames(net: FlowNetwork, s: int, t: int,
                        variant: str = "fifo") -> List[Dict]:
    """
    Frame dicts from a recorded push–relabel run. Global relabeling is
    switched off so every height change comes from a visible relabel
    (the gap heuristic stays on).
    """
    result = push_relabel(net, s, t, variant=variant, global_relabel=False,
                          record=True)
    frames = []
    for step in result.trace:
        op, u = step["op"], step["node"]
        if op == "init":
            title = f"init: h(s) = {net.n}, saturate every arc out of s"
            edge = []
        elif op == "push":
            v = net.to[step["edge"]]
            kind = "cancel" if step["edge"] & 1 else "push"
            title = f"{kind} {step['amount']:g} from {u} to {v}"
            edge = [step["edge"]]
        else:
            title = f"relabel {u}: height -> {step['heights'][u]}"
            edge = []
        frames.append({"flows": step["flows"], "heights": step["heights"],
                       "excess": step["excess"], "edge": edge,
                       "node": u if op == "relabel" else None,
                       "title": title + f"   (flow at t = {step['value']:g})"})
    return frames


def _height_positions(base_pos, heights) -> Dict[int, Tuple[float, float]]:
    """x from the layered layout, y = height; nodes sharing a spot are
    nudged apart."""
    pos, used = {}, {}
    for v, (x, _) in base_pos.items():
        key = (x, heights[v])
        k = used.get(key, 0)
        used[key] = k + 1
        shift = 0.42 * ((k + 1) // 2) * (1 if k % 2 else -1)
        pos[v] = (x + shift, float(heights[v]))
    return pos


def _draw_pr(ax, net, base_pos, frame, s, t, max_height) -> None:
    ax.clear()
    excess = frame["excess"]
    colors = []
    for v in range(net.n):
        if v in (s, t):
            colors.append(TERMINAL_COLOR)
        elif excess[v] > 1e-12:
            colors.append(ACTIVE_COLOR)
        else:
            colors.append(IDLE_NODE)
    if frame["node"] is not None:
        colors[frame["node"]] = "#d62728"
    pos = _height_positions(base_pos, frame["heights"])
    labels = {v: (f"{v}\n+{excess[v]:g}" if v not in (s, t)
                  and excess[v] > 1e-12 else str(v)) for v in range(net.n)}
    draw_flow(ax, net, pos, flows=frame["flows"], highlight=frame["edge"],
              node_color=colors, labels=labels, node_size=560)
    ax.set_axis_on()
    ax.set_ylim(-0.7, max_height + 0.7)
    ax.set_ylabel("height label")
    ax.set_xticks([])
    ax.set_yticks(range(0, max_height + 1))
    ax.tick_params(left=True, labelleft=True)
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_title(frame["title"], fontsize=10)


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def animate_edmonds_karp(net: FlowNetwork, s: int, t: int,
                         save_path: Optional[str] = None,
                         interval: int = 1300, max_frames: int = 60):
    """FuncAnimation of Edmonds–Karp; saved as GIF if save_path given."""
    pos = layered_positions(net, s, t)
    frames = _subsample(edmonds_karp_frames(net, s, t), max_frames)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    anim = FuncAnimation(fig, lambda i: _draw_ek(ax, net, pos, frames[i], s, t),
                         frames=len(frames), interval=interval)
    _finish(anim, fig, save_path, interval)
    return anim


def animate_push_relabel(net: FlowNetwork, s: int, t: int,
                         variant: str = "fifo",
                         save_path: Optional[str] = None,
                         interval: int = 700, max_frames: int = 120):
    """FuncAnimation of push–relabel (heights on the y axis)."""
    base_pos = layered_positions(net, s, t)
    frames = _subsample(push_relabel_frames(net, s, t, variant), max_frames)
    top = max(max(f["heights"]) for f in frames)
    fig, ax = plt.subplots(figsize=(9, 6))
    anim = FuncAnimation(fig, lambda i: _draw_pr(ax, net, base_pos, frames[i],
                                                 s, t, top),
                         frames=len(frames), interval=interval)
    _finish(anim, fig, save_path, interval)
    return anim


def _finish(anim, fig, save_path, interval) -> None:
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=max(1, round(1000 / interval))))
        plt.close(fig)


def storyboard(net: FlowNetwork, s: int, t: int, algorithm: str = "ek",
               panels: int = 6, ncols: int = 3,
               save_path: Optional[str] = None, title: Optional[str] = None):
    """Static grid of evenly spaced frames ("ek" or "pr")."""
    if algorithm not in ("ek", "pr"):
        raise ValueError("algorithm must be 'ek' or 'pr'")
    base_pos = layered_positions(net, s, t)
    frames = (edmonds_karp_frames(net, s, t) if algorithm == "ek"
              else push_relabel_frames(net, s, t))
    frames = _subsample(frames, panels)
    nrows = (len(frames) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows),
                             squeeze=False)
    top = max(max(f["heights"]) for f in frames) if algorithm == "pr" else 0
    for ax, frame in zip(axes.flat, frames):
        if algorithm == "ek":
            _draw_ek(ax, net, base_pos, frame, s, t)
        else:
            _draw_pr(ax, net, base_pos, frame, s, t, top)
        ax.title.set_fontsize(8.5)
    for ax in list(axes.flat)[len(frames):]:
        ax.axis("off")
    fig.suptitle(title or ("Edmonds–Karp augmentations" if algorithm == "ek"
                           else "Push–relabel: heights and excess"),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (4.2 * nrows)))
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    return fig
