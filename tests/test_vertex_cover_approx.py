"""Test suite for Vertex Cover: the matching 2-approximation, max-degree
greedy, exact branch and bound, and the edge-list graph helpers."""

import itertools
import random

import pytest

from src.complexity.vertex_cover_approx import (
    exact_vertex_cover, greedy_vertex_cover, is_vertex_cover,
    vertex_cover_2approx,
)
from src.graphs.graph import Graph
from src.utils.graph_utils import (
    adjacency_sets, complement_edges, complete_bipartite_edges,
    edges_from_graph, normalize_edges, random_graph_edges, vertices_of,
)
from src.utils.timer import CallStats

ALL = [vertex_cover_2approx, greedy_vertex_cover, exact_vertex_cover]


def min_cover_size(edges):
    verts = sorted(vertices_of(edges))
    for k in range(len(verts) + 1):
        for combo in itertools.combinations(verts, k):
            if is_vertex_cover(edges, combo):
                return k


def random_edges(seed, max_n=9):
    rng = random.Random(seed)
    n = rng.randint(2, max_n)
    return random_graph_edges(n, rng.uniform(1, n - 1), seed=seed)


class TestValidity:

    @pytest.mark.parametrize("solver", ALL)
    def test_empty_and_single_edge(self, solver):
        assert solver([]) == set()
        cover = solver([(1, 2)])
        assert is_vertex_cover([(1, 2)], cover)

    @pytest.mark.parametrize("solver", ALL)
    def test_string_vertices(self, solver):
        edges = [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")]
        assert is_vertex_cover(edges, solver(edges))

    @pytest.mark.parametrize("solver", ALL)
    @pytest.mark.parametrize("seed", range(40))
    def test_always_a_cover(self, solver, seed):
        edges = random_graph_edges(30, 4, seed=seed)
        assert is_vertex_cover(edges, solver(edges))

    def test_is_vertex_cover(self):
        assert is_vertex_cover([(1, 2), (2, 3)], {2})
        assert not is_vertex_cover([(1, 2), (2, 3)], {1})


class TestExact:

    @pytest.mark.parametrize("seed", range(60))
    def test_matches_enumeration(self, seed):
        edges = random_edges(seed)
        assert len(exact_vertex_cover(edges)) == min_cover_size(edges)

    @pytest.mark.parametrize("n,size", [(3, 2), (5, 4), (7, 6)])
    def test_complete_graph(self, n, size):
        edges = list(itertools.combinations(range(n), 2))
        assert len(exact_vertex_cover(edges)) == size

    def test_star_and_path(self):
        assert exact_vertex_cover([(0, i) for i in range(1, 8)]) == {0}
        path = [(i, i + 1) for i in range(9)]       # 10 vertices
        assert len(exact_vertex_cover(path)) == 5

    def test_counts_branch_nodes(self):
        stats = CallStats()
        exact_vertex_cover(random_graph_edges(60, 4, seed=1), stats)
        assert stats.calls >= 1


class TestApproximation:

    @pytest.mark.parametrize("seed", range(60))
    def test_within_factor_two(self, seed):
        edges = random_edges(seed)
        opt = min_cover_size(edges)
        assert len(vertex_cover_2approx(edges)) <= 2 * opt

    def test_cover_is_endpoints_of_a_matching(self):
        edges = random_graph_edges(40, 3, seed=2)
        cover = vertex_cover_2approx(edges)
        assert len(cover) % 2 == 0

    @pytest.mark.parametrize("half", [1, 3, 10])
    def test_tight_on_complete_bipartite(self, half):
        edges = complete_bipartite_edges(half, half)
        assert len(vertex_cover_2approx(edges)) == 2 * half
        assert len(exact_vertex_cover(edges)) == half

    def test_greedy_beats_matching_on_star(self):
        star = [(0, i) for i in range(1, 10)]
        assert greedy_vertex_cover(star) == {0}
        assert len(vertex_cover_2approx(star)) == 2

    def test_linear_time_scale(self):
        edges = random_graph_edges(100_000, 4, seed=3)
        assert is_vertex_cover(edges, vertex_cover_2approx(edges))


class TestGraphUtils:

    def test_normalize(self):
        assert normalize_edges([(1, 2), (2, 1), (3, 3), (2, 3)]) == \
            [(1, 2), (2, 3)]

    def test_adjacency_with_isolated(self):
        adj = adjacency_sets([(1, 2)], vertices=[1, 2, 9])
        assert adj == {1: {2}, 2: {1}, 9: set()}

    def test_complement(self):
        comp = complement_edges([0, 1, 2, 3], [(0, 1), (2, 3)])
        assert sorted(comp) == [(0, 2), (0, 3), (1, 2), (1, 3)]

    def test_random_graph_degree(self):
        edges = random_graph_edges(400, 4, seed=1)
        assert 2 * len(edges) / 400 == pytest.approx(4, rel=0.15)
        big = random_graph_edges(5_000, 4, seed=1)
        assert len(set(big)) == len(big) and all(u < v for u, v in big)
        assert random_graph_edges(1) == []

    def test_from_week4_graph(self):
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert sorted(map(sorted, edges_from_graph(g))) == \
            [["a", "b"], ["b", "c"]]

    def test_bipartite_shape(self):
        edges = complete_bipartite_edges(2, 3)
        assert len(edges) == 6 and vertices_of(edges) == set(range(5))
