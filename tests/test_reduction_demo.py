"""Test suite for the reduction framework: 3-SAT → Vertex Cover,
Vertex Cover → Independent Set → Clique, 3-SAT → Subset Sum, chains,
verifiers, and the complexity-class catalog.

The core check for every reduction: on random formulas, the source and
target instances have the same yes/no answer (solved exactly), a
target solution maps back to a valid source solution, and a source
solution maps forward to a valid target solution."""

import logging
import random

import pytest

from src.complexity.reduction_demo import (
    CATALOG, Clique, IndependentSet, IndependentSetToClique, ReductionChain,
    SubsetSum, ThreeSAT, ThreeSatToSubsetSum, ThreeSatToVertexCover,
    VertexCover, VertexCoverToIndependentSet, check_reduction, classify,
    format_catalog, solve_clique, solve_independent_set, solve_vertex_cover,
    verify_3sat, verify_clique, verify_independent_set, verify_subset_sum,
    verify_vertex_cover,
)

SAT = ThreeSAT([(1, -2, 3), (-1, 2, 3), (-1, -2, -3)], 3)
# x1 and ¬x1 and (x2 ∨ x3): unsatisfiable
UNSAT = ThreeSAT([(1,), (-1,), (2, 3)], 3)
# all 8 sign patterns over x1, x2, x3: unsatisfiable, every clause size 3
ALL_EIGHT = ThreeSAT([(a * 1, b * 2, c * 3) for a in (1, -1)
                      for b in (1, -1) for c in (1, -1)], 3)


def full_chain():
    return ReductionChain([ThreeSatToVertexCover(),
                           VertexCoverToIndependentSet(),
                           IndependentSetToClique()])


REDUCTIONS = [ThreeSatToVertexCover, ThreeSatToSubsetSum, full_chain]


def random_formula(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    clauses = [tuple(v if rng.random() < 0.5 else -v
                     for v in rng.sample(range(1, n + 1),
                                         rng.randint(1, min(3, n))))
               for _ in range(rng.randint(1, 12))]
    return ThreeSAT(clauses, n)


class TestInstances:

    def test_threesat_validation(self):
        with pytest.raises(ValueError):
            ThreeSAT([(1, 2, 3, 4)], 4)
        with pytest.raises(ValueError):
            ThreeSAT([()], 1)
        with pytest.raises(ValueError):
            ThreeSAT([(5,)], 2)

    def test_verifiers(self):
        g = [(1, 2), (2, 3), (1, 3), (3, 4)]
        assert verify_vertex_cover(VertexCover([1, 2, 3, 4], g, 2), {1, 3})
        assert not verify_vertex_cover(VertexCover([1, 2, 3, 4], g, 1), {3})
        assert verify_independent_set(IndependentSet([1, 2, 3, 4], g, 2),
                                      {1, 4})
        assert not verify_independent_set(IndependentSet([1, 2, 3, 4], g, 2),
                                          {1, 2})
        assert verify_clique(Clique([1, 2, 3, 4], g, 3), {1, 2, 3})
        assert not verify_clique(Clique([1, 2, 3, 4], g, 3), {2, 3, 4})
        assert verify_subset_sum(SubsetSum([3, 5, 7], 12), [1, 2])
        assert not verify_subset_sum(SubsetSum([3, 5, 7], 12), [0, 0])
        assert verify_3sat(SAT, {1: True, 2: True, 3: False})
        assert not verify_3sat(SAT, {1: True, 2: True})   # incomplete

    def test_problem_solvers(self):
        tri = [(1, 2), (2, 3), (1, 3)]
        assert solve_vertex_cover(VertexCover([1, 2, 3], tri, 1)) is None
        assert len(solve_vertex_cover(VertexCover([1, 2, 3], tri, 2))) == 2
        assert len(solve_independent_set(IndependentSet([1, 2, 3, 4], tri,
                                                        2))) == 2
        assert solve_clique(Clique([1, 2, 3, 4], tri, 3)) == {1, 2, 3}
        assert solve_clique(Clique([1, 2, 3, 4], tri, 4)) is None


class TestThreeSatToVertexCover:

    def test_sizes(self):
        vc = ThreeSatToVertexCover().reduce(SAT)
        n, m = 3, 3
        assert len(vc.vertices) == 2 * n + 3 * m
        assert len(vc.edges) == n + 3 * m + 3 * m
        assert vc.k == n + 2 * m

    def test_clause_sizes_one_and_two(self):
        phi = ThreeSAT([(1,), (1, -2)], 2)
        vc = ThreeSatToVertexCover().reduce(phi)
        assert vc.k == 2 + 0 + 1
        assert len(vc.vertices) == 4 + 1 + 2

    def test_forward_cover_has_size_k(self):
        r = ThreeSatToVertexCover()
        vc = r.reduce(SAT)
        cover = r.forward({1: True, 2: True, 3: False})
        assert len(cover) == vc.k and verify_vertex_cover(vc, cover)

    def test_recover_reads_variable_gadgets(self):
        r = ThreeSatToVertexCover()
        r.reduce(SAT)
        cover = r.forward({1: False, 2: False, 3: True})
        assert r.recover(cover) == {1: False, 2: False, 3: True}

    def test_unsat_has_no_small_cover(self):
        for phi in (UNSAT, ALL_EIGHT):
            vc = ThreeSatToVertexCover().reduce(phi)
            assert solve_vertex_cover(vc) is None

    def test_must_reduce_first(self):
        with pytest.raises(RuntimeError):
            ThreeSatToVertexCover().recover(set())


class TestOtherReductions:

    def test_vc_to_is_complements(self):
        r = VertexCoverToIndependentSet()
        ind = r.reduce(VertexCover([1, 2, 3], [(1, 2), (2, 3)], 1))
        assert ind.k == 2 and ind.edges == [(1, 2), (2, 3)]
        assert r.recover({1, 3}) == {2}
        assert r.forward({2}) == {1, 3}

    def test_is_to_clique_complements_graph(self):
        r = IndependentSetToClique()
        cl = r.reduce(IndependentSet([1, 2, 3], [(1, 2)], 2))
        assert sorted(cl.edges) == [(1, 3), (2, 3)] and cl.k == 2

    def test_subset_sum_numbers(self):
        r = ThreeSatToSubsetSum()
        ss = r.reduce(ThreeSAT([(1, -2, 3), (-1, 2, -3)], 3))
        # columns x1 x2 x3 C1 C2
        assert r.numbers[r.labels.index("y1")] == 10010
        assert r.numbers[r.labels.index("z1")] == 10001
        assert ss.target == 11133
        assert len(ss.numbers) == 2 * 3 + 2 * 2

    def test_subset_sum_duplicate_literal(self):
        phi = ThreeSAT([(1, 1, 2)], 2)
        r = ThreeSatToSubsetSum()
        ss = r.reduce(phi)
        assert r.numbers[r.labels.index("y1")] == 102     # two hits in C1
        chosen = r.forward({1: True, 2: False})
        assert verify_subset_sum(ss, chosen)

    def test_chain_name_and_round_trip(self):
        chain = full_chain()
        clique = chain.reduce(SAT)
        assert chain.name == ("3-SAT → Vertex Cover → Independent Set → "
                              "Clique")
        assert isinstance(clique, Clique)
        fwd = chain.forward({1: True, 2: True, 3: False})
        assert verify_clique(clique, fwd)
        assert verify_3sat(SAT, chain.recover(fwd))
        assert any(s.title.startswith("[3-SAT → Vertex Cover]")
                   for s in chain.steps)

    def test_empty_chain(self):
        with pytest.raises(ValueError):
            ReductionChain([])


class TestValidation:

    @pytest.mark.parametrize("make", REDUCTIONS)
    @pytest.mark.parametrize("seed", range(60))
    def test_random_formulas(self, make, seed):
        check = check_reduction(make(), random_formula(seed))
        assert check.ok, check.notes

    @pytest.mark.parametrize("make", REDUCTIONS)
    @pytest.mark.parametrize("phi", [SAT, UNSAT, ALL_EIGHT])
    def test_fixed_formulas(self, make, phi):
        check = check_reduction(make(), phi)
        assert check.ok
        assert check.source_yes == (phi is SAT)

    def test_detects_broken_reduction(self):
        class Broken(ThreeSatToVertexCover):
            def _reduce(self, phi):
                vc = super()._reduce(phi)
                return VertexCover(vc.vertices, vc.edges, vc.k - 1)
        assert not check_reduction(Broken(), SAT).ok


class TestExplanations:

    def test_steps_recorded(self):
        r = ThreeSatToVertexCover()
        r.reduce(SAT)
        titles = [s.title for s in r.steps]
        assert titles[0] == "Input formula"
        assert any("Variable gadgets" in t for t in titles)
        assert "k = n + Σ(|C_j| − 1) = 9" in r.explain()

    def test_verbose_prints(self, capsys):
        ThreeSatToSubsetSum(verbose=True).reduce(SAT)
        out = capsys.readouterr().out
        assert "Step 1. Input formula" in out and "Target" in out

    def test_steps_logged(self, caplog):
        with caplog.at_level(logging.INFO,
                             logger="src.complexity.reduction_demo"):
            ThreeSatToVertexCover().reduce(SAT)
        assert any("Variable gadgets" in r.message for r in caplog.records)

    def test_steps_reset_between_reductions(self):
        r = ThreeSatToVertexCover()
        r.reduce(SAT)
        first = len(r.steps)
        r.reduce(UNSAT)
        assert len(r.steps) == first


class TestCatalog:

    def test_classify(self):
        assert classify("2-sat").complexity_class == "P"
        assert classify("3-SAT").complexity_class == "NP-complete"
        assert classify("TSP (find").complexity_class == "NP-hard"
        with pytest.raises(KeyError):
            classify("chess")

    def test_format_lists_every_problem(self):
        text = format_catalog()
        assert all(p.name in text for p in CATALOG)
