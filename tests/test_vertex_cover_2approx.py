"""Test suite for the Week 9 Vertex Cover 2-approximations: greedy edge
picking with its matching certificate, and LP relaxation + rounding.

(Named test_vertex_cover_2approx.py because Week 8 already has
tests/test_vertex_cover_approx.py.)"""

import itertools
import random

import pytest

from src.approximation.vertex_cover_2approx import (
    cover_weight, exact_vertex_cover, is_vertex_cover,
    lp_rounding_vertex_cover, lp_vertex_cover, matching_vertex_cover,
)
from src.utils.graph_utils import complete_bipartite_edges, random_graph_edges


def min_weight_cover(edges, weights):
    verts = sorted({v for e in edges for v in e})
    best = None
    for r in range(len(verts) + 1):
        for combo in itertools.combinations(verts, r):
            if is_vertex_cover(edges, combo):
                w = sum(weights[v] for v in combo)
                best = w if best is None else min(best, w)
    return best


class TestMatchingCover:

    def test_empty(self):
        res = matching_vertex_cover([])
        assert res.cover == set() and res.cost == 0
        assert res.certified_ratio == 1.0

    def test_matching_is_disjoint_and_explains_cover(self):
        edges = random_graph_edges(50, 4, seed=1)
        res = matching_vertex_cover(edges)
        ends = [v for e in res.matching for v in e]
        assert len(ends) == len(set(ends))          # disjoint edges
        assert set(ends) == res.cover               # C = endpoints of M
        assert res.cost == 2 * res.lower_bound

    @pytest.mark.parametrize("seed", range(50))
    def test_cover_and_two_approx(self, seed):
        rng = random.Random(seed)
        edges = random_graph_edges(rng.randint(2, 14), rng.uniform(1, 5),
                                   seed=seed)
        res = matching_vertex_cover(edges)
        opt = len(exact_vertex_cover(edges))
        assert is_vertex_cover(edges, res.cover)
        assert res.lower_bound <= opt               # |M| ≤ OPT
        assert res.cost <= 2 * opt
        assert res.certified_ratio == pytest.approx(2.0) or not edges

    @pytest.mark.parametrize("half", [1, 4, 12])
    def test_tight_example(self, half):
        edges = complete_bipartite_edges(half, half)
        assert matching_vertex_cover(edges).cost == 2 * half
        assert len(exact_vertex_cover(edges)) == half

    def test_self_loops_ignored(self):
        assert matching_vertex_cover([(1, 1), (1, 2)]).cover == {1, 2}


class TestLPRounding:

    def test_lp_is_half_integral(self):
        x, _ = lp_vertex_cover(random_graph_edges(30, 4, seed=2))
        assert set(round(v, 6) for v in x.values()) <= {0.0, 0.5, 1.0}

    def test_triangle(self):
        x, value = lp_vertex_cover([(0, 1), (1, 2), (0, 2)])
        assert value == pytest.approx(1.5)
        assert lp_rounding_vertex_cover([(0, 1), (1, 2), (0, 2)]).cost == 3

    def test_star_lp_is_integral(self):
        star = [(0, i) for i in range(1, 6)]
        res = lp_rounding_vertex_cover(star)
        assert res.cover == {0} and res.lower_bound == pytest.approx(1)

    @pytest.mark.parametrize("seed", range(30))
    def test_unweighted_two_approx(self, seed):
        edges = random_graph_edges(12, 3, seed=seed)
        res = lp_rounding_vertex_cover(edges)
        opt = len(exact_vertex_cover(edges))
        assert is_vertex_cover(edges, res.cover)
        assert res.lower_bound <= opt + 1e-9
        assert res.cost <= 2 * res.lower_bound + 1e-9

    @pytest.mark.parametrize("seed", range(30))
    def test_weighted_two_approx(self, seed):
        rng = random.Random(seed)
        edges = random_graph_edges(9, 3, seed=seed)
        weights = {v: rng.randint(1, 20) for v in range(9)}
        res = lp_rounding_vertex_cover(edges, weights)
        opt = min_weight_cover(edges, weights)
        assert is_vertex_cover(edges, res.cover)
        assert res.cost == cover_weight(res.cover, weights)
        assert res.lower_bound <= opt + 1e-9
        assert res.cost <= 2 * opt + 1e-9

    def test_weights_change_the_answer(self):
        # heavy center: LP prefers the five leaves
        star = [(0, i) for i in range(1, 6)]
        weights = {0: 100, **{i: 1 for i in range(1, 6)}}
        assert lp_rounding_vertex_cover(star, weights).cover == \
            {1, 2, 3, 4, 5}

    def test_empty_graph(self):
        assert lp_rounding_vertex_cover([]).cover == set()
