"""Test suite for suffix array / suffix tree / KMP figures (headless)."""

import os


from src.strings.core.suffix_array import SuffixArray
from src.strings.core.suffix_tree import SuffixTree
from src.strings.visualization import (
    draw_suffix_tree, draw_ukkonen_steps, plot_kmp_trace, plot_lcp_heatmap,
    plot_lcp_histogram, plot_lcp_profile, plot_suffix_array, write_dot,
)


def saved(path):
    return os.path.exists(path) and os.path.getsize(path) > 2_000


def test_suffix_array_plot(tmp_path):
    path = str(tmp_path / "sa.png")
    fig = plot_suffix_array("banana$x$", save_path=path)    # '$' must not break mathtext
    assert saved(path)
    assert "banana" in fig._suptitle.get_text()


def test_lcp_plots(tmp_path):
    sa = SuffixArray("mississippi" * 20)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3)
    plot_lcp_profile(sa.lcp, ax=axes[0])
    plot_lcp_histogram({"a": sa.lcp, "b": [0, 1, 2]}, ax=axes[1])
    plot_lcp_heatmap("mississippi", size=8, ax=axes[2])
    path = str(tmp_path / "lcp.png")
    fig.savefig(path)
    assert saved(path)
    assert plot_lcp_profile([0, 3, 1]).axes


def test_draw_suffix_tree(tmp_path):
    path = str(tmp_path / "st.png")
    draw_suffix_tree(SuffixTree("mississippi"), save_path=path)
    assert saved(path)
    draw_suffix_tree(SuffixTree("a$b"), show_links=False, save_path=path)
    assert saved(path)


def test_ukkonen_steps(tmp_path):
    path = str(tmp_path / "uk.png")
    fig = draw_ukkonen_steps("abcab", save_path=path)
    assert saved(path)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == 6 and "implicit" in titles[3]


def test_kmp_trace_plot(tmp_path):
    path = str(tmp_path / "kmp.png")
    plot_kmp_trace("abababd", "ababd", save_path=path)
    assert saved(path)


def test_write_dot(tmp_path):
    path = write_dot(SuffixTree("banana"), str(tmp_path / "t.dot"))
    content = open(path).read()
    assert content.startswith("digraph") and content.strip().endswith("}")
