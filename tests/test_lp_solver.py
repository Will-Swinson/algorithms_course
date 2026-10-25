"""Test suite for the two-phase simplex LP solver."""

import itertools
import random

import numpy as np
import pytest

from src.utils.lp_solver import (
    INFEASIBLE, OPTIMAL, UNBOUNDED, solve_lp,
)


def brute_force_min(c, A, b):
    """min c·x s.t. A x ≥ b, x ≥ 0 by enumerating basic solutions."""
    n = len(c)
    rows = list(zip(A, b)) + [(np.eye(n)[i], 0.0) for i in range(n)]
    best = None
    for combo in itertools.combinations(range(len(rows)), n):
        M = np.array([rows[i][0] for i in combo], dtype=float)
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        x = np.linalg.solve(M, np.array([rows[i][1] for i in combo]))
        if (x >= -1e-9).all() and all(np.dot(a, x) >= bb - 1e-9
                                      for a, bb in zip(A, b)):
            value = float(np.dot(c, x))
            best = value if best is None else min(best, value)
    return best


class TestKnownLPs:

    def test_textbook_maximization(self):
        # max 3x + 5y s.t. x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18 → 36 at (2, 6)
        r = solve_lp([-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]],
                     b_ub=[4, 12, 18])
        assert r.status == OPTIMAL and r.success
        assert r.objective == pytest.approx(-36)
        assert r.x == pytest.approx([2, 6])

    def test_equality(self):
        r = solve_lp([1, 2], A_eq=[[1, 1]], b_eq=[3])
        assert r.objective == pytest.approx(3) and r.x == pytest.approx([3, 0])

    def test_covering_lp_is_fractional(self):
        # vertex cover LP of a triangle: all 1/2, value 3/2
        a = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        r = solve_lp([1, 1, 1], A_ge=a, b_ge=[1, 1, 1])
        assert r.objective == pytest.approx(1.5)
        assert r.x == pytest.approx([0.5, 0.5, 0.5])

    def test_negative_rhs_is_normalized(self):
        # -x ≤ -2  ⇔  x ≥ 2
        r = solve_lp([1], A_ub=[[-1]], b_ub=[-2])
        assert r.objective == pytest.approx(2)

    def test_infeasible(self):
        r = solve_lp([1, 1], A_ge=[[1, 1]], b_ge=[2], A_ub=[[1, 1]],
                     b_ub=[1])
        assert r.status == INFEASIBLE and r.x is None

    def test_unbounded(self):
        assert solve_lp([-1, 0], A_ub=[[0, 1]], b_ub=[1]).status == UNBOUNDED

    def test_no_constraints(self):
        assert solve_lp([1, 2]).objective == 0
        assert solve_lp([-1]).status == UNBOUNDED

    def test_redundant_equalities(self):
        r = solve_lp([1, 1], A_eq=[[1, 1], [2, 2]], b_eq=[2, 4])
        assert r.success and r.objective == pytest.approx(2)

    def test_shape_errors(self):
        with pytest.raises(ValueError):
            solve_lp([1, 1], A_ub=[[1, 1, 1]], b_ub=[1])
        with pytest.raises(ValueError):
            solve_lp([1, 1], A_ub=[[1, 1]], b_ub=[1, 2])


class TestAgainstBruteForce:

    @pytest.mark.parametrize("seed", range(120))
    def test_random_covering_lps(self, seed):
        rng = random.Random(seed)
        n, m = rng.randint(1, 4), rng.randint(1, 5)
        A = [[rng.randint(0, 3) for _ in range(n)] for _ in range(m)]
        b = [rng.randint(1, 5) for _ in range(m)]
        c = [rng.randint(1, 6) for _ in range(n)]
        r = solve_lp(c, A_ge=A, b_ge=b)
        expected = brute_force_min(c, A, b)
        if expected is None:
            assert r.status == INFEASIBLE
        else:
            assert r.success and r.objective == pytest.approx(expected)
            assert (np.asarray(A) @ r.x >= np.asarray(b) - 1e-7).all()
            assert (r.x >= 0).all()

    def test_degenerate_set_cover_lp_terminates(self):
        rng = np.random.default_rng(0)
        a = (rng.random((120, 60)) < 0.05).astype(float)
        a[a.sum(axis=1) == 0, 0] = 1
        r = solve_lp(rng.uniform(1, 10, 60), A_ge=a, b_ge=np.ones(120))
        assert r.success
        assert (a @ r.x >= 1 - 1e-7).all()
