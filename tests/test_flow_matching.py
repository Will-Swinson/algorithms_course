"""Test suite for bipartite matching and min-cost assignment via flow."""

import itertools

import networkx as nx
import pytest

from src.flow.applications import (
    assignment_from_matrix, bipartite_matching, is_matching,
    min_cost_assignment,
)
from src.flow.utils.generators import bipartite_graph

SOLVERS = ["ek", "scaling", "pr", "pr-highest"]


def hopcroft_karp_size(n_left, n_right, edges):
    graph = nx.Graph()
    left = [("l", a) for a in range(n_left)]
    graph.add_nodes_from(left)
    graph.add_nodes_from(("r", b) for b in range(n_right))
    graph.add_edges_from((("l", a), ("r", b)) for a, b in edges)
    return len(nx.bipartite.hopcroft_karp_matching(graph, left)) // 2


@pytest.mark.parametrize("solver", SOLVERS)
class TestMatching:

    def test_perfect(self, solver):
        r = bipartite_matching("abc", "xyz", [("a", "x"), ("a", "y"),
                                              ("b", "y"), ("c", "z")], solver)
        assert r.size == 3 and r.as_dict() == {"a": "x", "b": "y", "c": "z"}

    def test_needs_augmenting_path(self, solver):
        # greedy a-x would block b; the max matching re-routes a to y
        r = bipartite_matching("ab", "xy", [("a", "x"), ("a", "y"),
                                            ("b", "x")], solver)
        assert r.size == 2

    def test_star_and_empty(self, solver):
        assert bipartite_matching("abcd", "x", [(c, "x") for c in "abcd"],
                                  solver).size == 1
        assert bipartite_matching("ab", "xy", [], solver).size == 0

    def test_duplicates_ignored(self, solver):
        r = bipartite_matching("a", "x", [("a", "x"), ("a", "x")], solver)
        assert r.size == 1

    @pytest.mark.parametrize("seed", range(12))
    def test_matches_hopcroft_karp(self, solver, seed):
        n_left, n_right = 20 + seed, 15 + 2 * seed
        edges = bipartite_graph(n_left, n_right, 1.5 + seed % 3, seed=seed)
        r = bipartite_matching(range(n_left), range(n_right), edges, solver)
        assert r.size == hopcroft_karp_size(n_left, n_right, edges)
        assert is_matching(r.pairs, edges)


class TestMatchingErrors:

    def test_unknown_vertex(self):
        with pytest.raises(ValueError):
            bipartite_matching("a", "x", [("a", "q")])

    def test_unknown_solver(self):
        with pytest.raises(ValueError):
            bipartite_matching("a", "x", [("a", "x")], "simplex")

    def test_large(self):
        edges = bipartite_graph(3_000, 3_000, 3.0, seed=1)
        r = bipartite_matching(range(3_000), range(3_000), edges)
        assert r.size == hopcroft_karp_size(3_000, 3_000, edges)


class TestAssignment:

    @pytest.mark.parametrize("seed", range(10))
    def test_matrix_brute_force(self, seed):
        import random
        rng = random.Random(seed)
        n = rng.randint(1, 6)
        costs = [[rng.randint(0, 30) for _ in range(n)] for _ in range(n)]
        r = assignment_from_matrix(costs)
        best = min(sum(costs[i][p[i]] for i in range(n))
                   for p in itertools.permutations(range(n)))
        assert r.size == n and r.cost == best
        assert r.cost == sum(costs[a][b] for a, b in r.pairs)

    def test_rectangular(self):
        r = assignment_from_matrix([[4, 1, 3], [2, 0, 5]])
        assert r.size == 2 and r.cost == 1 + 2

    def test_prefers_maximum_matching_over_cheapness(self):
        # matching both is mandatory even though a-y is expensive
        r = min_cost_assignment("ab", "xy", [("a", "x", 1), ("a", "y", 50),
                                             ("b", "x", 2)])
        assert r.size == 2 and r.cost == 52

    def test_maximize_weight_with_negative_costs(self):
        weights = {("a", "x"): 3, ("a", "y"): 5, ("b", "x"): 4, ("b", "y"): 1}
        r = min_cost_assignment("ab", "xy", [(a, b, -w) for (a, b), w in
                                             weights.items()])
        assert -r.cost == 9

    def test_empty_matrix(self):
        assert assignment_from_matrix([]).size == 0


def test_is_matching():
    assert is_matching([("a", "x"), ("b", "y")])
    assert not is_matching([("a", "x"), ("a", "y")])
    assert not is_matching([("a", "x"), ("b", "x")])
    assert not is_matching([("a", "z")], edges=[("a", "x")])
