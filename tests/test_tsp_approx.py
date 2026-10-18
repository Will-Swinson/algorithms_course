"""Test suite for the TSP heuristics: nearest neighbor (single and
repeated), MST double-tree 2-approximation, 2-opt, and the MST lower
bound, checked against Week 6's exact Held–Karp."""

import math

import pytest

from src.complexity.traveling_salesman_approx import (
    mst_parents, mst_tour, mst_weight, nearest_neighbor_tour, optimal_tour,
    repeated_nearest_neighbor, two_opt,
)
from src.dp_advanced.bitmask_traveling_salesman import tour_cost
from src.utils.matrix_utils import euclidean_distance_matrix, random_points
from src.utils.timer import CallStats

HEURISTICS = [nearest_neighbor_tour, repeated_nearest_neighbor, mst_tour,
              lambda d: two_opt(d, nearest_neighbor_tour(d)[1])]

SQUARE = euclidean_distance_matrix([(0, 0), (0, 1), (1, 1), (1, 0)])


def cities(n, seed):
    return euclidean_distance_matrix(random_points(n, seed=seed))


def assert_valid(tour, n, start=0):
    assert tour[0] == tour[-1] == start
    assert sorted(tour[:-1]) == list(range(n))


class TestValidity:

    @pytest.mark.parametrize("heuristic", HEURISTICS)
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 9, 30])
    def test_valid_tour_and_cost(self, heuristic, n):
        d = cities(n, n)
        cost, tour = heuristic(d)
        assert_valid(tour, n)
        assert cost == pytest.approx(tour_cost(d, tour))

    def test_nearest_neighbor_other_start(self):
        cost, tour = nearest_neighbor_tour(cities(8, 1), start=5)
        assert_valid(tour, 8, start=5)

    @pytest.mark.parametrize("bad", [[], [[0, 1]]])
    def test_invalid_matrix(self, bad):
        with pytest.raises(ValueError):
            nearest_neighbor_tour(bad)

    def test_invalid_start(self):
        with pytest.raises(ValueError):
            nearest_neighbor_tour(SQUARE, start=4)

    def test_square_is_solved(self):
        for heuristic in HEURISTICS:
            assert heuristic(SQUARE)[0] == pytest.approx(4)


class TestNearestNeighbor:

    def test_greedy_choice(self):
        d = [[0, 1, 9, 2],
             [1, 0, 3, 9],
             [9, 3, 0, 4],
             [2, 9, 4, 0]]
        assert nearest_neighbor_tour(d) == (10, [0, 1, 2, 3, 0])

    def test_repeated_never_worse(self):
        for seed in range(10):
            d = cities(15, seed)
            assert repeated_nearest_neighbor(d)[0] <= \
                nearest_neighbor_tour(d)[0] + 1e-9

    def test_missing_roads_still_visit_all(self):
        inf = float("inf")
        d = [[0, 1, inf], [1, 0, inf], [inf, inf, 0]]
        cost, tour = nearest_neighbor_tour(d)
        assert_valid(tour, 3)
        assert cost == inf


class TestMST:

    def test_parents_form_tree(self):
        parent = mst_parents(cities(20, 3))
        assert parent[0] == -1 and all(0 <= p < 20 for p in parent[1:])

    def test_weight_known(self):
        assert mst_weight(SQUARE) == pytest.approx(3)
        line = euclidean_distance_matrix([(0, 0), (1, 0), (3, 0)])
        assert mst_weight(line) == pytest.approx(3)

    @pytest.mark.parametrize("seed", range(15))
    def test_lower_bound_and_two_approx(self, seed):
        d = cities(9, seed)
        opt = optimal_tour(d)[0]
        lb = mst_weight(d)
        assert lb <= opt + 1e-9
        assert mst_tour(d)[0] <= 2 * opt + 1e-9
        assert mst_tour(d)[0] <= 2 * lb + 1e-9


class TestTwoOpt:

    def test_uncrosses_square(self):
        crossed = [0, 2, 1, 3, 0]                   # diagonals cross
        assert tour_cost(SQUARE, crossed) == pytest.approx(2 + 2 * math.sqrt(2))
        cost, tour = two_opt(SQUARE, crossed)
        assert cost == pytest.approx(4)
        assert_valid(tour, 4)

    @pytest.mark.parametrize("seed", range(10))
    def test_never_worse_and_near_optimal(self, seed):
        d = cities(10, seed)
        start_cost, start = nearest_neighbor_tour(d)
        cost, _ = two_opt(d, start)
        assert cost <= start_cost + 1e-9
        assert cost <= 1.25 * optimal_tour(d)[0]

    def test_two_optimal_result(self):
        d = cities(25, 7)
        cost, tour = two_opt(d, nearest_neighbor_tour(d)[1])
        assert two_opt(d, tour)[0] == pytest.approx(cost)

    def test_pass_cap_and_stats(self):
        d = cities(40, 2)
        start = nearest_neighbor_tour(d)[1]
        one = two_opt(d, start, max_passes=1)[0]
        full = two_opt(d, start)[0]
        assert full <= one + 1e-9
        stats = CallStats()
        two_opt(d, start, max_passes=1, stats=stats)
        assert stats.calls == 40 * 37 // 2          # n(n-3)/2 moves per pass

    def test_small_tours_unchanged(self):
        d = cities(3, 1)
        tour = [0, 2, 1, 0]
        assert two_opt(d, tour)[1] == tour


class TestRatios:

    @pytest.mark.parametrize("seed", range(10))
    def test_heuristics_bounded_by_optimum(self, seed):
        d = cities(10, 100 + seed)
        opt = optimal_tour(d)[0]
        for heuristic in HEURISTICS:
            assert heuristic(d)[0] >= opt - 1e-9
