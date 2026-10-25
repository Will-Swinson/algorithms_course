"""Test suite for metric TSP: MST doubling, Christofides (optimal and
greedy matching), Euler circuits, shortcutting, and the matchings."""

import math

import pytest

from src.approximation.tsp_metric_approx import (
    christofides_tour, eulerian_circuit, greedy_perfect_matching,
    min_weight_perfect_matching, mst_doubling_tour, mst_edges, mst_weight,
    odd_degree_vertices, optimal_tour, shortcut, tour_cost,
)
from src.utils.matrix_utils import euclidean_distance_matrix, random_points

METHODS = [mst_doubling_tour, christofides_tour,
           lambda d: christofides_tour(d, matching="greedy")]


def cities(n, seed):
    return euclidean_distance_matrix(random_points(n, seed=seed))


def assert_valid(tour, n):
    assert tour[0] == tour[-1] == 0
    assert sorted(tour[:-1]) == list(range(n))


def brute_matching(dist, verts):
    """Minimum perfect matching cost by recursion (small inputs)."""
    if not verts:
        return 0.0
    first, rest = verts[0], verts[1:]
    return min(dist[first][v] + brute_matching(
        dist, [u for u in rest if u != v]) for v in rest)


class TestBuildingBlocks:

    def test_euler_circuit_uses_every_edge_once(self):
        # two triangles sharing vertex 0 (all degrees even)
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
        walk = eulerian_circuit(5, edges)
        assert walk[0] == walk[-1] == 0
        used = sorted(tuple(sorted(p)) for p in zip(walk, walk[1:]))
        assert used == sorted(tuple(sorted(e)) for e in edges)

    def test_euler_on_doubled_tree(self):
        tree = [(0, 1), (1, 2), (1, 3)]
        walk = eulerian_circuit(4, tree + tree)
        assert len(walk) == 2 * len(tree) + 1

    def test_shortcut(self):
        assert shortcut([0, 1, 2, 1, 3, 1, 0]) == [0, 1, 2, 3, 0]

    def test_odd_degree_count_is_even(self):
        for seed in range(10):
            d = cities(15, seed)
            odd = odd_degree_vertices(15, mst_edges(d))
            assert len(odd) % 2 == 0

    @pytest.mark.parametrize("seed", range(10))
    def test_blossom_matches_brute_force(self, seed):
        d = cities(12, seed)
        odd = odd_degree_vertices(12, mst_edges(d))
        m = min_weight_perfect_matching(d, odd)
        assert sorted(v for e in m for v in e) == sorted(odd)
        assert sum(d[u][v] for u, v in m) == pytest.approx(
            brute_matching(d, odd))

    def test_greedy_matching_is_perfect(self):
        d = cities(10, 1)
        m = greedy_perfect_matching(d, list(range(10)))
        assert sorted(v for e in m for v in e) == list(range(10))

    @pytest.mark.parametrize("match", [min_weight_perfect_matching,
                                       greedy_perfect_matching])
    def test_matching_rejects_odd_count(self, match):
        with pytest.raises(ValueError):
            match(cities(3, 0), [0, 1, 2])


class TestTours:

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 25])
    def test_valid_tours(self, method, n):
        d = cities(n, n)
        res = method(d)
        assert_valid(res.tour, n)
        assert res.cost == pytest.approx(tour_cost(d, res.tour))

    @pytest.mark.parametrize("seed", range(20))
    def test_guarantees_against_optimum(self, seed):
        d = cities(9, 100 + seed)
        opt = optimal_tour(d)[0]
        doubling = mst_doubling_tour(d)
        chris = christofides_tour(d)
        assert opt - 1e-9 <= doubling.cost <= 2 * opt + 1e-9
        assert opt - 1e-9 <= chris.cost <= 1.5 * opt + 1e-9
        assert chris.matching_cost <= opt / 2 + 1e-9       # key lemma
        assert doubling.mst_cost <= opt + 1e-9

    def test_certified_ratio_uses_mst(self):
        d = cities(30, 2)
        res = christofides_tour(d)
        assert res.mst_cost == pytest.approx(mst_weight(d))
        assert res.certified_ratio == pytest.approx(res.cost / res.mst_cost)
        assert res.certified_ratio <= 2 + 1e-9

    def test_square(self):
        # MST is a path; its odd ends get matched by the one missing
        # side, so Christofides finds the perimeter. Doubling may cross
        # a diagonal (4 + √2 − 1 ≈ 4.83), still within its bound.
        d = euclidean_distance_matrix([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert christofides_tour(d).cost == pytest.approx(4)
        assert 4 - 1e-9 <= mst_doubling_tour(d).cost <= 2 * 4

    def test_doubling_walk_is_twice_tree(self):
        d = cities(10, 3)
        res = mst_doubling_tour(d)
        walk_length = sum(d[a][b] for a, b in zip(res.euler_walk,
                                                  res.euler_walk[1:]))
        assert walk_length == pytest.approx(2 * res.mst_cost)
        assert res.cost <= walk_length + 1e-9

    def test_bad_matching_option(self):
        with pytest.raises(ValueError):
            christofides_tour(cities(5, 0), matching="random")

    def test_invalid_matrix(self):
        with pytest.raises(ValueError):
            mst_doubling_tour([])
        with pytest.raises(ValueError):
            christofides_tour([[0, 1]])

    def test_collinear_cities(self):
        d = euclidean_distance_matrix([(i, 0) for i in range(6)])
        assert christofides_tour(d).cost == pytest.approx(10)
        assert all(math.isfinite(m(d).cost) for m in METHODS)
