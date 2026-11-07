"""
Suffix tree drawings.

    draw_suffix_tree   the compressed tree: leaves (boxes, labelled with
                       their suffix start) evenly spaced in
                       lexicographic order, internal nodes centred over
                       their children, edges labelled with their
                       substrings and suffix links drawn as dashed red
                       arrows
    draw_ukkonen_steps one small panel per Ukkonen phase, with the
                       active point and the number of implicit
                       suffixes after each phase
    write_dot          SuffixTree.to_dot() to a .dot file; render with
                       Graphviz (`dot -Tpng tree.dot -o tree.png`) when
                       it is installed — the matplotlib drawing needs
                       no external tools
"""
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

from src.strings.core.suffix_tree import ROOT, SuffixTree

EDGE = "#4a4a4a"
LINK = "#de2d26"
LEAF = "#fdd0a2"
INTERNAL = "#c6dbef"


def _layout(children: Dict[int, List[int]], depth_of) -> Dict[int, Tuple[float, float]]:
    pos: Dict[int, Tuple[float, float]] = {}
    next_x = [0.0]

    def place(u):
        kids = children.get(u, [])
        if not kids:
            pos[u] = (next_x[0], -depth_of(u))
            next_x[0] += 1
            return
        for v in kids:
            place(v)
        pos[u] = (sum(pos[v][0] for v in kids) / len(kids), -depth_of(u))

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10_000))
    try:
        place(ROOT)
    finally:
        sys.setrecursionlimit(limit)
    return pos


def _tex_safe(s: str) -> str:
    """Escape '$' so matplotlib does not read a pair of them as math."""
    return s.replace("$", r"\$")


def _label(s: str, terminator: str, max_len: int) -> str:
    s = s.replace(terminator, "$")
    s = s if len(s) <= max_len else s[:max_len - 1] + "…"
    return _tex_safe(s)


def _draw(ax, edges: List[Tuple[int, int, str]], leaves: Dict[int, int],
          links: Dict[int, int], node_depth: Dict[int, int], terminator: str,
          max_label: int, highlight: Optional[int] = None) -> None:
    children: Dict[int, List[int]] = {}
    for u, v, label in edges:
        children.setdefault(u, []).append(v)
    order = {v: label for _, v, label in edges}
    for u in children:
        children[u].sort(key=lambda v: order[v].replace(terminator, "￿"))
    pos = _layout(children, lambda u: node_depth.get(u, 0))
    for u, v, label in edges:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        ax.plot([x1, x2], [y1, y2], color=EDGE, lw=1.1, zorder=1)
        # label 60% of the way down: edges fan out, so labels crowd less
        ax.text(x1 + 0.6 * (x2 - x1), y1 + 0.6 * (y2 - y1),
                _label(label, terminator, max_label),
                fontsize=8, family="monospace", ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    for u, target in links.items():
        if u in pos and target in pos and u != target:
            ax.annotate("", xy=pos[target], xytext=pos[u],
                        arrowprops=dict(arrowstyle="->", color=LINK, lw=0.9,
                                        linestyle="--",
                                        connectionstyle="arc3,rad=0.25"), zorder=2)
    for u, (x, y) in pos.items():
        if u in leaves:
            ax.text(x, y - 0.1, str(leaves[u]), ha="center", va="top", fontsize=8,
                    zorder=4, bbox=dict(boxstyle="square,pad=0.2", fc=LEAF, ec=EDGE,
                                        lw=0.6))
        else:
            ax.scatter([x], [y], s=60 if u == ROOT else 36, zorder=4,
                       color="#d62728" if u == highlight else INTERNAL,
                       edgecolors=EDGE, linewidths=0.8)
    ax.axis("off")
    ax.margins(0.08)


def draw_suffix_tree(tree: SuffixTree, title: Optional[str] = None,
                     save_path: Optional[str] = None, show_links: bool = True,
                     max_label: int = 10, ax=None):
    """Draw a (small) suffix tree. Node height = string depth."""
    edges = [(u, v, tree.edge_label(v)) for u, v in tree.edges()]
    leaves = {u: tree.suffix_index[u] for u in range(tree.num_nodes)
              if tree.is_leaf(u)}
    links = {u: tree.link[u] for u in range(1, tree.num_nodes)
             if not tree.is_leaf(u)} if show_links else {}
    depth = dict(enumerate(_node_depths(tree)))
    own = ax is None
    if own:
        width = max(6, 0.7 * len(leaves) + 2)
        fig, ax = plt.subplots(figsize=(min(width, 22), 5.5))
    _draw(ax, edges, leaves, links, depth, tree.terminator, max_label)
    shown = _tex_safe(tree.text.replace("\n", "⏎"))
    ax.set_title(title or f"Suffix tree of '{shown}\\$': {tree.num_nodes} nodes, "
                 f"{len(leaves)} leaves (dashed: suffix links)")
    if own:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
    return ax.figure


def _node_depths(tree: SuffixTree) -> List[int]:
    """Tree depth (edges from the root) of every node."""
    depth = [0] * tree.num_nodes
    for u, v in tree.edges():
        depth[v] = depth[u] + 1
    return depth


def draw_ukkonen_steps(text: str, save_path: Optional[str] = None,
                       ncols: int = 4, max_label: int = 8):
    """One panel per phase of Ukkonen's algorithm on `text` + $."""
    tree = SuffixTree(text, record=True)
    phases = tree.trace
    nrows = (len(phases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows),
                             squeeze=False)
    s = tree.s
    for ax, ph in zip(axes.flat, phases):
        i = ph["phase"]
        edges = [(u, v, s[a:b]) for u, v, a, b in ph["snapshot"]]
        children = {}
        for u, v, _ in edges:
            children.setdefault(u, []).append(v)
        depth = {ROOT: 0}
        stack = [ROOT]
        while stack:
            u = stack.pop()
            for v in children.get(u, []):
                depth[v] = depth[u] + 1
                stack.append(v)
        leaves = {v: tree.suffix_index[v] for _, v, _ in edges
                  if v not in children}
        node, edge, length, remainder = ph["after"]
        _draw(ax, edges, leaves, {}, depth, tree.terminator, max_label,
              highlight=node)
        edge_char = _tex_safe(s[edge].replace(tree.terminator, "$")) if length else "-"
        ax.set_title(f"phase {i}: add '{_tex_safe(s[i].replace(tree.terminator, '$'))}'  "
                     f"[{_tex_safe(s[:i + 1].replace(tree.terminator, '$'))}]\n"
                     f"explicit {ph['explicit']}, implicit {ph['implicit']}; "
                     f"active ({'root' if node == ROOT else node}, "
                     f"'{edge_char}', {length})", fontsize=8.5)
    for ax in list(axes.flat)[len(phases):]:
        ax.axis("off")
    fig.suptitle(f"Ukkonen's algorithm on '{_tex_safe(text)}': red node = active node",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (3.6 * nrows)))
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    return fig


def write_dot(tree: SuffixTree, path: str, **kw) -> str:
    source = tree.to_dot(**kw)
    with open(path, "w") as fh:
        fh.write(source + "\n")
    return path
