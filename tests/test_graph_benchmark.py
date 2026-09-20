# File: tests/test_graph_benchmark.py
"""Test suite for the graph generator and visualization utilities the
Week 4 benchmark depends on."""

import os

import pytest

from src.graphs.graph import ADJACENCY_MATRIX
from src.graphs.bfs import bfs
from src.utils.graph_generator import (
    generate_graph, generate_sparse_graph, generate_dense_graph,
    generate_random_graph,
)
from src.utils.visualization import draw_graph, draw_traversal


class TestGenerateGraph:

    def test_node_and_edge_counts(self):
        g = generate_graph(50, 100, seed=1)
        assert len(g) == 50
        assert g.num_edges() == 100

    def test_connected_graph_is_fully_reachable(self):
        g = generate_graph(80, 160, connected=True, seed=2)
        assert len(bfs(g, 0)) == 80

    def test_directed_connected_reachable_from_root(self):
        g = generate_graph(50, 150, directed=True, connected=True, seed=3)
        assert len(bfs(g, 0)) == 50

    def test_seed_reproducibility(self):
        g1 = generate_graph(30, 60, weighted=True, seed=42)
        g2 = generate_graph(30, 60, weighted=True, seed=42)
        for node in g1.nodes():
            assert sorted(g1.get_neighbors(node)) == \
                sorted(g2.get_neighbors(node))

    def test_weighted_weights_in_range(self):
        g = generate_graph(20, 40, weighted=True, weight_range=(5, 10),
                           seed=4)
        for node in g.nodes():
            for _, w in g.get_neighbors(node):
                assert 5 <= w <= 10

    def test_unweighted_weights_are_one(self):
        g = generate_graph(20, 40, weighted=False, seed=5)
        for node in g.nodes():
            for _, w in g.get_neighbors(node):
                assert w == 1

    def test_edge_count_clamped_to_maximum(self):
        g = generate_graph(5, 1000, seed=6)  # max undirected = 10
        assert g.num_edges() == 10

    def test_no_self_loops(self):
        g = generate_graph(25, 100, seed=7)
        for node in g.nodes():
            assert not g.has_edge(node, node)

    def test_matrix_representation_supported(self):
        g = generate_graph(20, 40, representation=ADJACENCY_MATRIX, seed=8)
        assert g.representation == ADJACENCY_MATRIX
        assert len(bfs(g, 0)) == 20

    def test_invalid_node_count_raises(self):
        with pytest.raises(ValueError):
            generate_graph(0, 5)


class TestDensityHelpers:

    def test_sparse_has_average_degree_four(self):
        g = generate_sparse_graph(100, seed=9)
        assert g.num_edges() == 200  # E = 2V

    def test_dense_has_half_of_possible_edges(self):
        g = generate_dense_graph(40, seed=10)
        assert g.num_edges() == (40 * 39 // 2) // 2

    def test_random_probability_scales_edges(self):
        g = generate_random_graph(40, edge_probability=0.5, seed=11)
        assert g.num_edges() == round((40 * 39 // 2) * 0.5)

    def test_bad_probability_raises(self):
        with pytest.raises(ValueError):
            generate_random_graph(10, edge_probability=1.5)


class TestVisualization:
    """Smoke tests: the drawing helpers must produce non-empty PNGs."""

    def test_draw_graph_writes_png(self, tmp_path):
        g = generate_sparse_graph(15, seed=12)
        out = str(tmp_path / "graph.png")
        draw_graph(g, out, title="test")
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_draw_traversal_writes_png(self, tmp_path):
        g = generate_sparse_graph(15, seed=13)
        out = str(tmp_path / "traversal.png")
        draw_traversal(g, bfs(g, 0), out, title="bfs")
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_draw_traversal_handles_unvisited_nodes(self, tmp_path):
        from src.graphs.graph import Graph
        g = Graph()
        g.add_edge("a", "b")
        g.add_node("island")  # never visited from "a"
        out = str(tmp_path / "partial.png")
        draw_traversal(g, bfs(g, "a"), out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
