# File: tests/test_graph_representation.py
"""Test suite for the Graph class: both representations must behave
identically through the shared API."""

import pytest

from src.graphs.graph import Graph, ADJACENCY_LIST, ADJACENCY_MATRIX

REPRESENTATIONS = [ADJACENCY_LIST, ADJACENCY_MATRIX]


@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestNodes:

    def test_add_and_query_nodes(self, rep):
        g = Graph(representation=rep)
        for n in ["a", "b", "c"]:
            g.add_node(n)
        assert len(g) == 3
        assert g.has_node("a")
        assert "b" in g
        assert not g.has_node("ghost")
        assert g.nodes() == ["a", "b", "c"]  # insertion order

    def test_add_duplicate_node_is_noop(self, rep):
        g = Graph(representation=rep)
        g.add_node("a")
        g.add_edge("a", "b")
        g.add_node("a")  # must not wipe existing edges
        assert len(g) == 2
        assert g.has_edge("a", "b")

    def test_remove_node_removes_incident_edges(self, rep):
        g = Graph(representation=rep)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.remove_node("b")
        assert not g.has_node("b")
        assert len(g) == 2
        assert g.num_edges() == 0
        assert g.get_neighbors("a") == []
        assert g.get_neighbors("c") == []

    def test_remove_missing_node_raises(self, rep):
        with pytest.raises(KeyError):
            Graph(representation=rep).remove_node("ghost")

    def test_empty_graph(self, rep):
        g = Graph(representation=rep)
        assert len(g) == 0
        assert g.nodes() == []
        assert g.num_edges() == 0


@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestUndirectedEdges:

    def test_edge_is_symmetric(self, rep):
        g = Graph(directed=False, representation=rep)
        g.add_edge("a", "b")
        assert g.has_edge("a", "b")
        assert g.has_edge("b", "a")
        assert g.num_edges() == 1  # counted once

    def test_remove_edge_removes_both_directions(self, rep):
        g = Graph(directed=False, representation=rep)
        g.add_edge("a", "b")
        g.remove_edge("b", "a")
        assert not g.has_edge("a", "b")
        assert not g.has_edge("b", "a")
        assert g.has_node("a") and g.has_node("b")  # nodes survive

    def test_neighbors_both_sides(self, rep):
        g = Graph(directed=False, representation=rep)
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert sorted(n for n, _ in g.get_neighbors("a")) == ["b", "c"]
        assert [n for n, _ in g.get_neighbors("b")] == ["a"]


@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestDirectedEdges:

    def test_edge_is_one_way(self, rep):
        g = Graph(directed=True, representation=rep)
        g.add_edge("a", "b")
        assert g.has_edge("a", "b")
        assert not g.has_edge("b", "a")
        assert g.num_edges() == 1
        assert g.get_neighbors("b") == []

    def test_two_way_needs_two_edges(self, rep):
        g = Graph(directed=True, representation=rep)
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        assert g.num_edges() == 2
        assert g.has_edge("b", "a")


@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestWeights:

    def test_default_weight_is_one(self, rep):
        g = Graph(representation=rep)
        g.add_edge("a", "b")
        assert g.get_weight("a", "b") == 1

    def test_explicit_weight_stored_both_ways(self, rep):
        g = Graph(directed=False, representation=rep)
        g.add_edge("a", "b", weight=2.5)
        assert g.get_weight("a", "b") == 2.5
        assert g.get_weight("b", "a") == 2.5
        assert g.get_neighbors("a") == [("b", 2.5)]

    def test_readding_edge_updates_weight(self, rep):
        g = Graph(representation=rep)
        g.add_edge("a", "b", weight=1)
        g.add_edge("a", "b", weight=9)
        assert g.get_weight("a", "b") == 9
        assert g.num_edges() == 1

    def test_get_weight_missing_edge_raises(self, rep):
        g = Graph(representation=rep)
        g.add_node("a")
        g.add_node("b")
        with pytest.raises(KeyError):
            g.get_weight("a", "b")


@pytest.mark.parametrize("rep", REPRESENTATIONS)
class TestErrorsAndStr:

    def test_get_neighbors_missing_node_raises(self, rep):
        with pytest.raises(KeyError):
            Graph(representation=rep).get_neighbors("ghost")

    def test_remove_missing_edge_raises(self, rep):
        g = Graph(representation=rep)
        g.add_node("a")
        g.add_node("b")
        with pytest.raises(KeyError):
            g.remove_edge("a", "b")

    def test_str_mentions_nodes_and_edges(self, rep):
        g = Graph(representation=rep)
        g.add_edge("a", "b", weight=3)
        text = str(g)
        assert "a" in text and "b" in text
        assert "2 nodes" in text and "1 edges" in text

    def test_add_edge_autocreates_nodes(self, rep):
        g = Graph(representation=rep)
        g.add_edge("x", "y")
        assert g.has_node("x") and g.has_node("y")


def test_invalid_representation_raises():
    with pytest.raises(ValueError):
        Graph(representation="hologram")


def test_representations_agree_on_same_edges():
    """Build the same random graph both ways; every query must agree."""
    import random
    random.seed(42)
    edges = [(random.randrange(30), random.randrange(30),
              random.randint(1, 9)) for _ in range(120)]

    g_list = Graph(directed=True, representation=ADJACENCY_LIST)
    g_matrix = Graph(directed=True, representation=ADJACENCY_MATRIX)
    for graph in (g_list, g_matrix):
        for n in range(30):
            graph.add_node(n)
        for u, v, w in edges:
            if u != v:
                graph.add_edge(u, v, w)

    assert g_list.nodes() == g_matrix.nodes()
    assert g_list.num_edges() == g_matrix.num_edges()
    for node in g_list.nodes():
        assert sorted(g_list.get_neighbors(node)) == \
            sorted(g_matrix.get_neighbors(node))
