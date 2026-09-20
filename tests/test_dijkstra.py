# File: tests/test_dijkstra.py
"""Test suite for Dijkstra's algorithm (heap- and list-based)."""

import random

import pytest

from src.graphs.graph import Graph
from src.graphs.dijkstra import (
    dijkstra, dijkstra_list_based, reconstruct_path, INFINITY,
)
from src.utils.graph_generator import generate_graph

VERSIONS = [dijkstra, dijkstra_list_based]


def _weighted_graph():
    """Classic textbook example (directed):

        a --1--> b --2--> c
        a --4--> c        (longer direct route)
        c --1--> d
        e                 (disconnected)
    """
    g = Graph(directed=True)
    g.add_edge("a", "b", 1)
    g.add_edge("b", "c", 2)
    g.add_edge("a", "c", 4)
    g.add_edge("c", "d", 1)
    g.add_node("e")
    return g


@pytest.mark.parametrize("algo", VERSIONS, ids=lambda f: f.__name__)
class TestDistances:

    def test_source_distance_is_zero(self, algo):
        distances, _ = algo(_weighted_graph(), "a")
        assert distances["a"] == 0

    def test_prefers_cheaper_multi_hop_route(self, algo):
        """a->b->c costs 3, beating the direct a->c edge of 4 — the
        case where greedy-by-weight differs from BFS hop counting."""
        distances, preds = algo(_weighted_graph(), "a")
        assert distances["c"] == 3
        assert preds["c"] == "b"

    def test_all_distances(self, algo):
        distances, _ = algo(_weighted_graph(), "a")
        assert distances == {"a": 0, "b": 1, "c": 3, "d": 4,
                             "e": INFINITY}

    def test_disconnected_node_is_infinite(self, algo):
        distances, preds = algo(_weighted_graph(), "a")
        assert distances["e"] == INFINITY
        assert preds["e"] is None

    def test_undirected_graph(self, algo):
        g = Graph(directed=False)
        g.add_edge("x", "y", 5)
        g.add_edge("y", "z", 5)
        distances, _ = algo(g, "z")
        assert distances["x"] == 10  # traverses edges both ways

    def test_missing_source_raises(self, algo):
        with pytest.raises(KeyError):
            algo(_weighted_graph(), "ghost")

    def test_negative_weight_raises(self, algo):
        g = Graph(directed=True)
        g.add_edge("a", "b", -1)
        with pytest.raises(ValueError):
            algo(g, "a")

    def test_zero_weight_edges_allowed(self, algo):
        g = Graph(directed=True)
        g.add_edge("a", "b", 0)
        g.add_edge("b", "c", 2)
        distances, _ = algo(g, "a")
        assert distances == {"a": 0, "b": 0, "c": 2}

    def test_single_node(self, algo):
        g = Graph()
        g.add_node("only")
        distances, preds = algo(g, "only")
        assert distances == {"only": 0}
        assert preds == {"only": None}


class TestVersionsAgree:

    def test_heap_and_list_agree_on_random_graphs(self):
        """Two independent implementations must produce identical
        distances on random weighted graphs — each is the other's
        oracle."""
        for seed in range(5):
            g = generate_graph(60, 240, directed=True, weighted=True,
                               seed=seed)
            heap_dist, _ = dijkstra(g, 0)
            list_dist, _ = dijkstra_list_based(g, 0)
            assert heap_dist == list_dist

    def test_agrees_with_bfs_on_unit_weights(self):
        """With all weights 1, shortest weight == fewest hops, so
        Dijkstra must match BFS layer depths."""
        from src.graphs.bfs import bfs_shortest_paths
        g = generate_graph(50, 150, directed=False, weighted=False,
                           seed=7)
        distances, _ = dijkstra(g, 0)
        preds = bfs_shortest_paths(g, 0)

        def bfs_depth(node):
            depth = 0
            while preds[node] is not None:
                node = preds[node]
                depth += 1
            return depth

        for node in g.nodes():
            assert distances[node] == bfs_depth(node)


class TestPathReconstruction:

    def test_full_path(self):
        distances, preds = dijkstra(_weighted_graph(), "a")
        assert reconstruct_path(preds, "a", "d") == ["a", "b", "c", "d"]

    def test_source_to_itself(self):
        _, preds = dijkstra(_weighted_graph(), "a")
        assert reconstruct_path(preds, "a", "a") == ["a"]

    def test_unreachable_target_returns_none(self):
        _, preds = dijkstra(_weighted_graph(), "a")
        assert reconstruct_path(preds, "a", "e") is None

    def test_path_cost_matches_distance(self):
        g = generate_graph(40, 160, directed=True, weighted=True, seed=3)
        distances, preds = dijkstra(g, 0)
        for target in g.nodes():
            if distances[target] == INFINITY or target == 0:
                continue
            path = reconstruct_path(preds, 0, target)
            cost = sum(g.get_weight(path[i], path[i + 1])
                       for i in range(len(path) - 1))
            assert cost == pytest.approx(distances[target])
