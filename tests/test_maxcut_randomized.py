"""Test suite for Max-Cut: random cut (expectation and variance),
derandomization by conditional expectations, local search, and
exhaustive search."""

import itertools
import random
import statistics

import pytest

from src.approximation.maxcut_randomized import (
    MAX_EXACT_VERTICES, as_weighted, cut_value, derandomized_cut,
    exact_maxcut, expected_random_cut, local_search_cut, random_cut,
    total_weight,
)
from src.utils.graph_utils import random_graph_edges, random_weighted_edges

TRIANGLE = [(0, 1), (1, 2), (0, 2)]
SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]       # bipartite: OPT = 4


def brute_force(edges, n):
    best = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        best = max(best, cut_value(edges, dict(enumerate(bits))))
    return best


class TestBasics:

    def test_weights(self):
        assert as_weighted([(0, 1), (1, 2, 5), (3, 3)]) == \
            [(0, 1, 1.0), (1, 2, 5.0)]
        assert total_weight([(0, 1), (1, 2, 5)]) == 6

    def test_cut_value(self):
        assert cut_value(SQUARE, {0: 0, 1: 1, 2: 0, 3: 1}) == 4
        assert cut_value(SQUARE, {0: 0, 1: 0, 2: 0, 3: 0}) == 0

    def test_expected_formulas(self):
        mean, var = expected_random_cut([(0, 1, 2), (1, 2, 4)])
        assert mean == 3 and var == (4 + 16) / 4


class TestExact:

    def test_known(self):
        assert exact_maxcut(TRIANGLE).value == 2
        assert exact_maxcut(SQUARE).value == 4

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_brute_force(self, seed):
        edges = random_weighted_edges(9, 3, seed=seed)
        res = exact_maxcut(edges, range(9))
        assert res.value == pytest.approx(brute_force(edges, 9))
        assert cut_value(edges, res.side) == pytest.approx(res.value)

    def test_trivial_and_limit(self):
        assert exact_maxcut([], [0]).value == 0
        big = [(i, i + 1) for i in range(MAX_EXACT_VERTICES)]
        with pytest.raises(ValueError):
            exact_maxcut(big)


class TestRandomCut:

    def test_isolated_vertices_get_sides(self):
        res = random_cut([(0, 1)], vertices=range(4), rng=random.Random(1))
        assert set(res.side) == {0, 1, 2, 3}

    def test_mean_and_variance_match_theory(self):
        edges = random_weighted_edges(25, 4, seed=3)
        mean, var = expected_random_cut(edges)
        values = [random_cut(edges, rng=random.Random(s)).value
                  for s in range(6_000)]
        assert statistics.mean(values) == pytest.approx(mean, rel=0.02)
        assert statistics.pvariance(values) == pytest.approx(var, rel=0.1)

    def test_expectation_at_least_half_opt(self):
        edges = random_graph_edges(14, 4, seed=5)
        opt = exact_maxcut(edges).value
        mean, _ = expected_random_cut(edges)
        assert mean >= opt / 2


class TestDeterministicGuarantees:

    @pytest.mark.parametrize("seed", range(30))
    def test_derandomized_at_least_half_w(self, seed):
        edges = random_weighted_edges(12, 4, seed=seed)
        res = derandomized_cut(edges)
        assert res.value >= total_weight(edges) / 2 - 1e-9
        assert res.value == pytest.approx(cut_value(edges, res.side))

    @pytest.mark.parametrize("seed", range(30))
    def test_local_search_at_least_half_w(self, seed):
        edges = random_weighted_edges(12, 4, seed=seed)
        res = local_search_cut(edges, rng=random.Random(seed))
        assert res.value >= total_weight(edges) / 2 - 1e-9
        assert res.value <= exact_maxcut(edges).value + 1e-9

    def test_local_optimum_no_improving_flip(self):
        edges = random_weighted_edges(30, 4, seed=7)
        res = local_search_cut(edges, rng=random.Random(2))
        for v in res.side:
            flipped = dict(res.side)
            flipped[v] ^= 1
            assert cut_value(edges, flipped) <= res.value + 1e-9

    def test_local_search_improves_bad_start(self):
        start = {v: 0 for v in range(4)}
        res = local_search_cut(SQUARE, start=start)
        assert res.value == 4 and res.flips >= 1

    def test_bipartite_found_exactly(self):
        assert derandomized_cut(SQUARE).value == 4
