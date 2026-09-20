# File: tests/test_bfs.py
"""Test suite for breadth-first search."""

import pytest

from src.graphs.graph import Graph, ADJACENCY_LIST, ADJACENCY_MATRIX
from src.graphs.bfs import bfs, bfs_all, bfs_shortest_paths

REPRESENTATIONS = [ADJACENCY_LIST, ADJACENCY_MATRIX]


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


@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestBFSOrder:

    def test_starts_at_start(self, rep):
        assert bfs(_sample_graph(rep), "a")[0] == "a"

    def test_level_order(self, rep):
        """Every node at distance d must appear before every node at
        distance d+1 — the defining BFS property."""
        order = bfs(_sample_graph(rep), "a")
        depth = {"a": 0, "b": 1, "c": 1, "d": 2, "e": 2}
        depths_in_order = [depth[n] for n in order]
        assert depths_in_order == sorted(depths_in_order)

    def test_visits_component_exactly_once(self, rep):
        order = bfs(_sample_graph(rep), "a")
        assert sorted(order) == ["a", "b", "c", "d", "e"]
        assert len(order) == len(set(order))

    def test_does_not_cross_components(self, rep):
        order = bfs(_sample_graph(rep), "a")
        assert "f" not in order and "g" not in order

    def test_single_node_graph(self, rep):
        g = Graph(representation=rep)
        g.add_node("only")
        assert bfs(g, "only") == ["only"]

    def test_cycle_terminates(self, rep):
        g = Graph(representation=rep)
        for u, v in [(0, 1), (1, 2), (2, 0)]:  # triangle
            g.add_edge(u, v)
        assert sorted(bfs(g, 0)) == [0, 1, 2]

    def test_directed_respects_edge_direction(self, rep):
        g = Graph(directed=True, representation=rep)
        g.add_edge("a", "b")
        g.add_edge("c", "a")  # points INTO a; unreachable from a
        assert sorted(bfs(g, "a")) == ["a", "b"]

    def test_missing_start_raises(self, rep):
        with pytest.raises(KeyError):
            bfs(_sample_graph(rep), "ghost")


@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestBFSAll:

    def test_covers_disconnected_graph(self, rep):
        order = bfs_all(_sample_graph(rep))
        assert sorted(order) == ["a", "b", "c", "d", "e", "f", "g"]
        assert len(order) == len(set(order))

    def test_empty_graph(self, rep):
        assert bfs_all(Graph(representation=rep)) == []


@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestBFSShortestPaths:

    def test_predecessors_give_fewest_edge_paths(self, rep):
        preds = bfs_shortest_paths(_sample_graph(rep), "a")
        assert preds["a"] is None
        assert preds["d"] == "b"       # a-b-d is the only 2-hop route
        assert preds["c"] == "a"

    def test_unreachable_nodes_absent(self, rep):
        preds = bfs_shortest_paths(_sample_graph(rep), "a")
        assert "f" not in preds

    def test_finds_shorter_of_two_routes(self, rep):
        g = Graph(representation=rep)
        # a-b-c-d (3 hops) vs a-d (1 hop)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]:
            g.add_edge(u, v)
        preds = bfs_shortest_paths(g, "a")
        assert preds["d"] == "a"       # direct edge wins
