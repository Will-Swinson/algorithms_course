"""Test suite for the SAT solvers: brute force, backtracking/DPLL,
linear-time 2-SAT, WalkSAT, and the random-formula generators."""

import random

import pytest

from src.complexity.satisfiability_solver import (
    PHASE_TRANSITION_RATIO, backtracking_sat, brute_force_sat,
    count_satisfied, format_cnf, is_satisfied, planted_ksat,
    random_assignment, random_ksat, two_sat, validate_cnf, walksat,
)
from src.utils.timer import CallStats

# x1 ∨ x2, ¬x1 ∨ x2, x1 ∨ ¬x2, ¬x1 ∨ ¬x2: every assignment fails one
UNSAT_2 = [(1, 2), (-1, 2), (1, -2), (-1, -2)]
SAT_3 = [(1, -2, 3), (-1, 2, -3), (2, 3, -1)]

EXACT = [
    brute_force_sat,
    backtracking_sat,
    lambda cnf, n: backtracking_sat(cnf, n, unit_propagation=False),
]


class TestBasics:

    def test_is_satisfied(self):
        assert is_satisfied([(1, -2)], {1: True, 2: True})
        assert not is_satisfied([(1, -2)], {1: False, 2: True})
        assert is_satisfied([], {})
        assert not is_satisfied([()], {})          # empty clause

    def test_count_satisfied(self):
        assert count_satisfied(UNSAT_2, {1: True, 2: True}) == 3

    def test_format(self):
        assert format_cnf([(1, -2), (3,)]) == "(x1 ∨ ¬x2) ∧ (x3)"
        assert format_cnf([]) == "⊤"

    @pytest.mark.parametrize("cnf,n", [([(0, 1)], 2), ([(3,)], 2),
                                       ([(1,)], -1)])
    def test_validate_rejects(self, cnf, n):
        with pytest.raises(ValueError):
            validate_cnf(cnf, n)


class TestExactSolvers:

    @pytest.mark.parametrize("solver", EXACT)
    def test_satisfiable(self, solver):
        result = solver(SAT_3, 3)
        assert result is not None and is_satisfied(SAT_3, result)
        assert set(result) == {1, 2, 3}

    @pytest.mark.parametrize("solver", EXACT)
    def test_unsatisfiable(self, solver):
        assert solver(UNSAT_2, 2) is None

    @pytest.mark.parametrize("solver", EXACT)
    def test_empty_formula_and_clause(self, solver):
        assert solver([], 2) == {1: False, 2: False}
        assert solver([()], 1) is None

    @pytest.mark.parametrize("seed", range(60))
    def test_solvers_agree_on_random_formulas(self, seed):
        rng = random.Random(seed)
        n = rng.randint(1, 8)
        cnf = random_ksat(n, rng.randint(1, 45), min(3, n), seed=seed)
        answers = [solver(cnf, n) for solver in EXACT]
        assert len({a is None for a in answers}) == 1
        for a in answers:
            assert a is None or is_satisfied(cnf, a)

    def test_unit_propagation_prunes(self):
        cnf = random_ksat(18, round(PHASE_TRANSITION_RATIO * 18), seed=3)
        with_up, without = CallStats(), CallStats()
        backtracking_sat(cnf, 18, stats=with_up)
        backtracking_sat(cnf, 18, unit_propagation=False, stats=without)
        assert with_up.calls < without.calls

    def test_brute_force_counts_all_on_unsat(self):
        stats = CallStats()
        brute_force_sat(UNSAT_2, 2, stats)
        assert stats.calls == 4

    def test_backtracking_depth_bounded_by_n(self):
        stats = CallStats()
        backtracking_sat(random_ksat(12, 50, seed=1), 12, stats=stats)
        assert 1 <= stats.max_depth <= 13


class TestTwoSat:

    def test_known(self):
        assert two_sat(UNSAT_2, 2) is None
        cnf = [(1, 2), (-1, 3), (-2, -3)]
        assert is_satisfied(cnf, two_sat(cnf, 3))

    def test_unit_clauses(self):
        assert two_sat([(1,), (-1,)], 1) is None
        assert two_sat([(1,), (-1, 2)], 2) == {1: True, 2: True}

    def test_rejects_three_literals(self):
        with pytest.raises(ValueError):
            two_sat([(1, 2, 3)], 3)

    @pytest.mark.parametrize("seed", range(80))
    def test_matches_brute_force(self, seed):
        rng = random.Random(seed)
        n = rng.randint(2, 9)
        cnf = random_ksat(n, rng.randint(1, 3 * n), 2, seed=seed)
        fast, slow = two_sat(cnf, n), brute_force_sat(cnf, n)
        assert (fast is None) == (slow is None)
        assert fast is None or is_satisfied(cnf, fast)

    def test_large_formula_no_recursion_limit(self):
        n = 50_000
        chain = [(-i, i + 1) for i in range(1, n)] + [(1,)]
        result = two_sat(chain, n)
        assert result is not None and all(result.values())


class TestHeuristics:

    def test_walksat_solves_planted(self):
        cnf, _ = planted_ksat(100, 300, seed=4)
        result = walksat(cnf, 100, rng=random.Random(1))
        assert result.solved and is_satisfied(cnf, result.assignment)
        assert result.fraction == 1.0

    def test_walksat_never_solves_unsat(self):
        result = walksat(UNSAT_2, 2, max_flips=500, rng=random.Random(0))
        assert not result.solved
        assert result.satisfied == 3 and result.flips == 500

    def test_walksat_reports_best_assignment(self):
        cnf = random_ksat(30, 200, seed=9)          # ratio 6.7: UNSAT
        result = walksat(cnf, 30, max_flips=2_000, rng=random.Random(2))
        assert count_satisfied(cnf, result.assignment) == result.satisfied

    def test_walksat_is_seeded(self):
        cnf, _ = planted_ksat(60, 250, seed=5)
        a = walksat(cnf, 60, rng=random.Random(7))
        b = walksat(cnf, 60, rng=random.Random(7))
        assert a.assignment == b.assignment and a.flips == b.flips

    def test_walksat_bad_noise(self):
        with pytest.raises(ValueError):
            walksat(SAT_3, 3, noise=1.5)

    def test_random_assignment_near_seven_eighths(self):
        cnf = random_ksat(200, 4_000, seed=6)
        fracs = [random_assignment(cnf, 200, random.Random(s)).fraction
                 for s in range(20)]
        assert sum(fracs) / 20 == pytest.approx(7 / 8, abs=0.01)


class TestGenerators:

    def test_random_ksat_shape(self):
        cnf = random_ksat(10, 40, 3, seed=1)
        assert len(cnf) == 40
        assert all(len({abs(l) for l in c}) == 3 for c in cnf)
        assert random_ksat(10, 40, 3, seed=1) == cnf

    def test_planted_is_satisfied_by_hidden(self):
        cnf, hidden = planted_ksat(25, 120, seed=2)
        assert len(cnf) == 120 and is_satisfied(cnf, hidden)

    @pytest.mark.parametrize("gen", [random_ksat, planted_ksat])
    def test_bad_k(self, gen):
        with pytest.raises(ValueError):
            gen(2, 5, k=3)
