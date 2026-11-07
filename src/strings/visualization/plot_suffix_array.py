"""
Suffix array and LCP visualizations (matplotlib, headless).

    plot_suffix_array   the sorted-suffix table of a short text next to
                        its LCP bar chart: shared prefixes of adjacent
                        suffixes are shaded, so repeats are visible
    plot_lcp_profile    LCP along the suffix array of a long text, with
                        the longest repeat marked
    plot_lcp_histogram  LCP distributions of several texts (log scale)
    plot_lcp_heatmap    pairwise LCP between the first suffixes in SA
                        order — the matrix the LCP array summarises
    plot_kmp_trace      KMP scan as a grid: rows = steps, each mismatch
                        → fallback highlighted
"""
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

from src.strings.core.kmp import kmp_trace, prefix_function
from src.strings.core.suffix_array import SuffixArray

SHADE = "#9ecae1"
BAR = "#3182bd"
HIGHLIGHT = "#e6550d"


def _show(text: str) -> str:
    """Printable, and with '$' escaped so matplotlib skips math mode."""
    return text.replace("\n", "⏎").replace("\x00", "$").replace("$", r"\$")


def plot_suffix_array(text: str, title: Optional[str] = None,
                      save_path: Optional[str] = None, max_rows: int = 40):
    """Sorted suffix table (rank, SA, suffix) and LCP bars side by side."""
    sa = SuffixArray(text)
    rows = sa.table(width=len(text))[:max_rows]
    n = len(rows)
    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(11, 0.34 * n + 1.6),
                                     gridspec_kw={"width_ratios": [3, 1]},
                                     sharey=True)
    longest = max((r[2] for r in rows), default=0)
    char_w = 0.62
    for y, (rank, start, lcp, suffix) in enumerate(rows):
        ax_t.text(0, y, f"{rank:>3}", family="monospace", va="center", fontsize=9)
        ax_t.text(1.6, y, f"{start:>3}", family="monospace", va="center",
                  fontsize=9, color="#555555")
        shown = _show(suffix)
        if lcp:
            ax_t.add_patch(plt.Rectangle((3.2 - 0.1, y - 0.42),
                                         lcp * char_w + 0.05, 0.84,
                                         color=HIGHLIGHT if lcp == longest else SHADE,
                                         alpha=0.6, lw=0))
        ax_t.text(3.2, y, shown, family="monospace", va="center", fontsize=9)
    ax_t.text(0, -1, "rank", fontweight="bold", fontsize=9)
    ax_t.text(1.6, -1, "SA", fontweight="bold", fontsize=9)
    ax_t.text(3.2, -1, "suffix (shaded: prefix shared with the row above)",
              fontweight="bold", fontsize=9)
    ax_t.set_xlim(-0.3, 3.6 + char_w * max(len(text) + 2, 34))
    ax_t.set_ylim(n - 0.4, -1.6)
    ax_t.axis("off")
    ax_b.barh(range(n), [r[2] for r in rows],
              color=[HIGHLIGHT if r[2] == longest and longest else BAR for r in rows])
    for y, r in enumerate(rows):
        ax_b.text(r[2] + 0.1, y, str(r[2]), va="center", fontsize=8)
    ax_b.set_xlim(0, max(longest, 1) * 1.25 + 0.5)
    ax_b.set_xlabel("LCP with previous suffix")
    ax_b.set_title("LCP array", fontsize=10)
    ax_b.tick_params(left=False, labelleft=False)
    ax_b.grid(True, axis="x", alpha=0.3)
    fig.suptitle(title or f"Suffix array of '{_show(text)}': longest repeat "
                 f"'{_show(sa.longest_repeated_substring())}'", fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_lcp_profile(lcp: Sequence[int], ax=None, title: str = "LCP profile",
                     color: str = BAR):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 3.5))
    lcp = np.asarray(lcp)
    ax.plot(np.arange(len(lcp)), lcp, lw=0.6, color=color)
    if len(lcp):
        r = int(lcp.argmax())
        ax.scatter([r], [lcp[r]], color=HIGHLIGHT, zorder=3,
                   label=f"longest repeat: {lcp[r]} chars")
        ax.legend(fontsize=8)
    ax.set_xlabel("suffix rank")
    ax.set_ylabel("LCP")
    ax.set_title(title)
    return ax.figure


def plot_lcp_histogram(lcps: Dict[str, Sequence[int]], ax=None,
                       title: str = "LCP distribution", bins: int = 40):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))
    top = max((max(v) for v in lcps.values() if len(v)), default=1)
    edges = np.linspace(0, top + 1, bins + 1)
    for label, values in lcps.items():
        ax.hist(values, bins=edges, histtype="step", lw=1.6,
                label=f"{label} (mean {np.mean(values):.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("LCP of adjacent suffixes")
    ax.set_ylabel("suffixes")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax.figure


def plot_lcp_heatmap(text: str, size: int = 24, ax=None,
                     title: Optional[str] = None):
    """Pairwise LCP of the first `size` suffixes in sorted order. The
    LCP array is its first off-diagonal; any other entry is the minimum
    of the LCP values between the two ranks (the RMQ property)."""
    sa = SuffixArray(text)
    k = min(size, len(text))
    starts = [int(x) for x in sa.sa[:k]]
    mat = np.zeros((k, k), dtype=int)
    for a in range(k):
        for b in range(k):
            i, j, h = starts[a], starts[b], 0
            while i + h < len(text) and j + h < len(text) and text[i + h] == text[j + h]:
                h += 1
            mat[a, b] = h
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    np.fill_diagonal(mat, 0)
    im = ax.imshow(mat, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, label="LCP")
    labels = [_show(text[s:s + 8]) for s in starts]
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=90, fontsize=6, family="monospace")
    ax.set_yticklabels(labels, fontsize=6, family="monospace")
    ax.set_title(title or "Pairwise LCP of sorted suffixes")
    return ax.figure


def plot_kmp_trace(text: str, pattern: str, title: Optional[str] = None,
                   save_path: Optional[str] = None, max_steps: int = 30):
    """Each row is one KMP step: the pattern drawn at its current
    alignment, the compared character outlined (green match, red
    mismatch), and fallbacks annotated with the LPS jump."""
    steps = kmp_trace(text, pattern)[:max_steps]
    lps = prefix_function(pattern)
    rows = len(steps) + 2
    fig, ax = plt.subplots(figsize=(0.32 * (len(text) + 14), 0.3 * rows + 1.2))
    for x, c in enumerate(text):
        ax.text(x, 0, c, ha="center", va="center", family="monospace",
                fontweight="bold")
    for y, step in enumerate(steps, start=1):
        shift = step["i"] - step["j"]
        for k, c in enumerate(pattern):
            x = shift + k
            color = "#333333"
            if k == step["j"]:
                ok = step["action"] in ("match", "found")
                ax.add_patch(plt.Rectangle((x - 0.45, y - 0.42), 0.9, 0.84,
                                           fill=False, lw=1.4,
                                           ec="#31a354" if ok else "#de2d26"))
            elif k > step["j"]:
                color = "#bbbbbb"
            ax.text(x, y, c, ha="center", va="center", family="monospace",
                    color=color, fontsize=9)
        note = {"match": "", "found": f"match at {step.get('start')}",
                "advance": "mismatch at j=0: next i",
                "fallback": f"fallback j={step['j']}→lps[{step['j'] - 1}]={step['next_j']}"}
        ax.text(len(text) + 0.8, y, note[step["action"]], va="center", fontsize=8,
                color="#de2d26" if step["action"] == "fallback" else "#333333")
    ax.set_xlim(-1, len(text) + 13)
    ax.set_ylim(rows - 0.5, -0.8)
    ax.axis("off")
    ax.set_title(title or f"KMP scan of {pattern!r}  (lps = {lps})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
