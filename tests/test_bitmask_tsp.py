"""Test suite for bitmask-DP TSP, the brute-force baseline, and the
bit-manipulation helpers."""

import math
import random

import pytest

from src.dp_advanced.bitmask_traveling_salesman import (
    MAX_BITMASK_CITIES, MAX_BRUTE_FORCE_CITIES, tour_cost, tsp_bitmask,
    tsp_brute_force,
)
from src.utils import bitmask_utils as bits
from src.utils.matrix_utils import (
    INF, euclidean_distance_matrix, random_points,
)
from src.utils.timer import CallStats

SOLVERS = [tsp_bitmask, tsp_brute_force]

FOUR = [[0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]]


def cities(n, seed):
    return euclidean_distance_matrix(random_points(n, seed=seed))


def asymmetric(n, seed):
    rng = random.Random(seed)
    return [[0 if i == j else rng.randint(1, 50) for j in range(n)]
            for i in range(n)]


def assert_valid_tour(tour, n):
    assert tour[0] == tour[-1] == 0
    assert sorted(tour[:-1]) == list(range(n))


@pytest.mark.parametrize("solve", SOLVERS)
class TestCorrectness:

    def test_classic_four_city(self, solve):
        cost, tour = solve(FOUR)
        assert cost == 80
        assert_valid_tour(tour, 4)
        assert tour_cost(FOUR, tour) == 80

    def test_single_city(self, solve):
        assert solve([[0]]) == (0, [0])

    def test_two_cities(self, solve):
        assert solve([[0, 3], [5, 0]]) == (8, [0, 1, 0])

    def test_reported_cost_matches_tour(self, solve):
        dist = cities(7, seed=3)
        cost, tour = solve(dist)
        assert_valid_tour(tour, 7)
        assert math.isclose(tour_cost(dist, tour), cost)

    def test_square_corners(self, solve):
        dist = euclidean_distance_matrix([(0, 0), (0, 1), (1, 1), (1, 0)])
        cost, _ = solve(dist)
        assert math.isclose(cost, 4.0)  # perimeter, not the diagonals

    def test_asymmetric_distances(self, solve):
        dist = [[0, 1, 9], [9, 0, 1], [1, 9, 0]]
        # 0->1->2->0 costs 3; the reverse direction costs 27
        assert solve(dist) == (3, [0, 1, 2, 0])

    def test_no_tour_when_city_unreachable(self, solve):
        dist = [[0, 1, INF], [1, 0, INF], [INF, INF, 0]]
        assert solve(dist) == (INF, None)

    def test_missing_edges_routed_around(self, solve):
        dist = [[0, 1, INF, 1], [1, 0, 1, INF],
                [INF, 1, 0, 1], [1, INF, 1, 0]]
        cost, tour = solve(dist)
        assert cost == 4
        assert_valid_tour(tour, 4)

    def test_empty_raises(self, solve):
        with pytest.raises(ValueError):
            solve([])

    def test_non_square_raises(self, solve):
        with pytest.raises(ValueError):
            solve([[0, 1], [1]])


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("n", [3, 5, 8])
def test_bitmask_matches_brute_force(n, seed):
    for dist in (cities(n, seed), asymmetric(n, seed)):
        dp_cost, dp_tour = tsp_bitmask(dist)
        bf_cost, _ = tsp_brute_force(dist)
        assert math.isclose(dp_cost, bf_cost)
        assert math.isclose(tour_cost(dist, dp_tour), bf_cost)


class TestLimitsAndWork:

    def test_bitmask_handles_sixteen_cities(self):
        cost, tour = tsp_bitmask(cities(16, seed=1))
        assert_valid_tour(tour, 16)
        assert cost > 0

    def test_size_limits(self):
        too_big = [[0] * (MAX_BITMASK_CITIES + 1)
                   for _ in range(MAX_BITMASK_CITIES + 1)]
        with pytest.raises(ValueError):
            tsp_bitmask(too_big)
        n = MAX_BRUTE_FORCE_CITIES + 1
        with pytest.raises(ValueError):
            tsp_brute_force([[0] * n for _ in range(n)])

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_brute_force_tries_n_minus_1_factorial_tours(self, n):
        stats = CallStats()
        tsp_brute_force(cities(n, seed=2), stats=stats)
        assert stats.calls == math.factorial(n - 1)

    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_bitmask_transitions_within_n_squared_2_to_n(self, n):
        stats = CallStats()
        tsp_bitmask(cities(n, seed=2), stats=stats)
        assert 0 < stats.calls <= n * n * 2 ** n

    def test_bitmask_beats_brute_force_work_at_ten(self):
        dp, bf = CallStats(), CallStats()
        dist = cities(10, seed=4)
        tsp_bitmask(dist, stats=dp)
        tsp_brute_force(dist, stats=bf)
        assert dp.calls * 10 < bf.calls


class TestBitmaskUtils:

    def test_bit_and_membership(self):
        mask = bits.bit(0) | bits.bit(2) | bits.bit(3)
        assert mask == 0b1101
        assert bits.has_bit(mask, 2) and not bits.has_bit(mask, 1)

    def test_set_and_clear(self):
        assert bits.set_bit(0b0001, 3) == 0b1001
        assert bits.clear_bit(0b1001, 0) == 0b1000
        assert bits.clear_bit(0b1000, 1) == 0b1000  # already clear

    def test_full_mask(self):
        assert bits.full_mask(0) == 0
        assert bits.full_mask(4) == 0b1111
        with pytest.raises(ValueError):
            bits.full_mask(-1)

    def test_popcount(self):
        assert bits.popcount(0) == 0
        assert bits.popcount(0b101101) == 4

    def test_iter_bits_in_order(self):
        assert list(bits.iter_bits(0b101001)) == [0, 3, 5]
        assert list(bits.iter_bits(0)) == []

    def test_round_trip(self):
        elements = [1, 4, 7, 19]
        assert bits.to_list(bits.mask_from(elements)) == elements

    def test_mask_from_rejects_negative(self):
        with pytest.raises(ValueError):
            bits.mask_from([-1])

    def test_format_mask(self):
        assert bits.format_mask(0b1101, 5) == "01101"

    def test_every_subset_enumerated_once(self):
        n = 4
        subsets = {frozenset(bits.iter_bits(m))
                   for m in range(bits.full_mask(n) + 1)}
        assert len(subsets) == 2 ** n
