"""Test suite for Uncapacitated Facility Location: star greedy, local
search, exhaustive search, and the metric instance generator."""

import itertools
import math

import pytest

from src.approximation.facility_location_greedy import (
    MAX_EXACT_FACILITIES, FacilityInstance, evaluate,
    exact_facility_location, greedy_facility_location,
    local_search_facility_location, random_metric_instance,
)

# two clusters far apart: open one facility in each
TWO_CLUSTERS = FacilityInstance(
    [10, 10, 10],
    [[1, 1, 100, 100],
     [100, 100, 1, 1],
     [50, 50, 50, 50]])

SOLVERS = [greedy_facility_location, local_search_facility_location,
           exact_facility_location]


def brute_force(inst):
    m = inst.num_facilities
    return min(evaluate(inst, combo).cost for r in range(1, m + 1)
               for combo in itertools.combinations(range(m), r))


class TestInstance:

    @pytest.mark.parametrize("kwargs", [
        dict(opening_costs=[], connection=[]),
        dict(opening_costs=[1], connection=[[1], [2]]),
        dict(opening_costs=[1, 1], connection=[[1, 2], [1]]),
        dict(opening_costs=[-1], connection=[[1]]),
    ])
    def test_validation(self, kwargs):
        with pytest.raises(ValueError):
            FacilityInstance(**kwargs)

    def test_evaluate(self):
        sol = evaluate(TWO_CLUSTERS, {0, 1})
        assert sol.cost == 24 and sol.assignment == [0, 0, 1, 1]
        assert sol.opening_cost == 20 and sol.connection_cost == 4
        with pytest.raises(ValueError):
            evaluate(TWO_CLUSTERS, set())

    def test_metric_generator(self):
        inst = random_metric_instance(4, 6, seed=1)
        assert inst.num_facilities == 4 and inst.num_clients == 6
        f, c = inst.facility_points[2], inst.client_points[3]
        assert inst.connection[2][3] == pytest.approx(math.dist(f, c))
        assert random_metric_instance(4, 6, seed=1).opening_costs == \
            inst.opening_costs


class TestSolvers:

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_two_clusters(self, solver):
        sol = solver(TWO_CLUSTERS)
        assert sol.open == frozenset({0, 1}) and sol.cost == 24

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_single_facility(self, solver):
        inst = FacilityInstance([5], [[1, 2, 3]])
        assert solver(inst).cost == 11

    def test_free_facilities_all_open(self):
        inst = FacilityInstance([0, 0], [[0, 5], [5, 0]])
        assert exact_facility_location(inst).cost == 0
        assert greedy_facility_location(inst).cost == 0

    @pytest.mark.parametrize("seed", range(25))
    def test_exact_matches_brute_force(self, seed):
        inst = random_metric_instance(6, 12, seed=seed)
        assert exact_facility_location(inst).cost == \
            pytest.approx(brute_force(inst))

    @pytest.mark.parametrize("seed", range(25))
    def test_greedy_within_metric_bound(self, seed):
        inst = random_metric_instance(7, 20, seed=seed)
        opt = exact_facility_location(inst).cost
        sol = greedy_facility_location(inst)
        assert sol.cost == pytest.approx(evaluate(inst, sol.open).cost)
        assert opt - 1e-9 <= sol.cost <= 1.861 * opt
        assert sol.steps >= len(sol.open)

    @pytest.mark.parametrize("seed", range(25))
    def test_local_search_within_three(self, seed):
        inst = random_metric_instance(7, 20, seed=seed)
        opt = exact_facility_location(inst).cost
        assert local_search_facility_location(inst).cost <= 3 * opt + 1e-9

    def test_local_search_never_worse_than_start(self):
        for seed in range(10):
            inst = random_metric_instance(10, 30, seed=seed)
            start = greedy_facility_location(inst)
            improved = local_search_facility_location(inst, start=start.open)
            assert improved.cost <= start.cost + 1e-9

    def test_local_search_is_local_optimum(self):
        inst = random_metric_instance(8, 25, seed=3)
        sol = local_search_facility_location(inst)
        again = local_search_facility_location(inst, start=sol.open)
        assert again.steps == 0 and again.cost == pytest.approx(sol.cost)

    def test_greedy_reconnects_to_nearest(self):
        inst = random_metric_instance(8, 25, seed=4)
        sol = greedy_facility_location(inst)
        for j, i in enumerate(sol.assignment):
            assert inst.connection[i][j] == min(
                inst.connection[k][j] for k in sol.open)

    def test_exact_limit(self):
        inst = FacilityInstance([1] * (MAX_EXACT_FACILITIES + 1),
                                [[1]] * (MAX_EXACT_FACILITIES + 1))
        with pytest.raises(ValueError):
            exact_facility_location(inst)
