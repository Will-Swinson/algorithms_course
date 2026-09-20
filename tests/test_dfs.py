# File: tests/test_dfs.py
"""Test suite for depth-first search (iterative and recursive)."""

import random

import pytest

from src.graphs.graph import Graph, ADJACENCY_LIST, ADJACENCY_MATRIX
from src.graphs.dfs import dfs_iterative, dfs_recursive, dfs_all

REPRESENTATIONS = [ADJACENCY_LIST, ADJACENCY_MATRIX]
DFS_VERSIONS = [dfs_iterative, dfs_recursive]


def _sample_graph(rep):
    """
        a -- b -- d
        |    |
        c    e        f -- g   (separate component)
    """
    g = Graph(directed=False, representation=rep)
    for u, v in [("a", "b"), ("a", "c"), ("b", "d"), ("b", "e"),
                 ("f", "g")]:
        g.add_edge(u, v)
    return g


@pytest.mark.parametrize("dfs", DFS_VERSIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestDFSCore:

    def test_starts_at_start(self, dfs, rep):
        assert dfs(_sample_graph(rep), "a")[0] == "a"

    def test_visits_component_exactly_once(self, dfs, rep):
        order = dfs(_sample_graph(rep), "a")
        assert sorted(order) == ["a", "b", "c", "d", "e"]
        assert len(order) == len(set(order))

    def test_does_not_cross_components(self, dfs, rep):
        order = dfs(_sample_graph(rep), "a")
        assert "f" not in order and "g" not in order

    def test_depth_property_on_path_graph(self, dfs, rep):
        """On a path 0-1-2-3-4, DFS from 0 must walk straight down."""
        g = Graph(representation=rep)
        for i in range(4):
            g.add_edge(i, i + 1)
        assert dfs(g, 0) == [0, 1, 2, 3, 4]

    def test_cycle_terminates(self, dfs, rep):
        g = Graph(representation=rep)
        for u, v in [(0, 1), (1, 2), (2, 0)]:
            g.add_edge(u, v)
        assert sorted(dfs(g, 0)) == [0, 1, 2]

    def test_directed_respects_edge_direction(self, dfs, rep):
        g = Graph(directed=True, representation=rep)
        g.add_edge("a", "b")
        g.add_edge("c", "a")
        assert sorted(dfs(g, "a")) == ["a", "b"]

    def test_single_node(self, dfs, rep):
        g = Graph(representation=rep)
        g.add_node("only")
        assert dfs(g, "only") == ["only"]

    def test_missing_start_raises(self, dfs, rep):
        with pytest.raises(KeyError):
            dfs(_sample_graph(rep), "ghost")


class TestVersionsAgree:

    @pytest.mark.parametrize("rep", REPRESENTATIONS)
    def test_iterative_matches_recursive_exactly(self, rep):
        """The iterative version pushes neighbors reversed precisely so
        both versions produce the SAME order — verify on random graphs."""
        random.seed(42)
        for trial in range(10):
            g = Graph(representation=rep)
            n = 40
            for node in range(n):
                g.add_node(node)
            for _ in range(80):
                u, v = random.randrange(n), random.randrange(n)
                if u != v:
                    g.add_edge(u, v)
            assert dfs_iterative(g, 0) == dfs_recursive(g, 0)

    def test_iterative_survives_deep_path_beyond_recursion_limit(self):
        """A 5000-node path overflows Python's call stack recursively;
        the iterative version must handle it."""
        g = Graph()
        n = 5000
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        order = dfs_iterative(g, 0)
        assert len(order) == n
        assert order == list(range(n))


class TestDFSAll:

    @pytest.mark.parametrize("rep", REPRESENTATIONS)
    def test_covers_disconnected_graph(self, rep):
        order = dfs_all(_sample_graph(rep))
        assert sorted(order) == ["a", "b", "c", "d", "e", "f", "g"]
        assert len(order) == len(set(order))

    def test_empty_graph(self):
        assert dfs_all(Graph()) == []
