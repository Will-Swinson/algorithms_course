"""Test suite for numerical stability utilities: errors, norms,
condition numbers, test matrices, summation, and cancellation."""

import importlib
import math

import numpy as np
import pytest

N = importlib.import_module("src.fft_analyzer.core.numerical")
RNG = np.random.default_rng(4)


class TestErrors:

    def test_machine_epsilon(self):
        assert N.machine_epsilon() == np.finfo(float).eps == N.EPS
        assert N.machine_epsilon(np.float32) == np.finfo(np.float32).eps

    def test_absolute_relative(self):
        assert N.absolute_error([1.0, 2.0], [1.0, 2.5]) == 0.5
        assert N.relative_error([1.1], [1.0]) == pytest.approx(0.1)
        assert N.relative_error([0.0], [0.0]) == 0.0
        assert N.relative_error([1.0], [0.0]) == math.inf

    def test_ulp_distance(self):
        assert N.ulp_distance(1.0, 1.0) == 0
        assert N.ulp_distance(1.0, 1.0 + N.EPS) == 1
        assert N.ulp_distance(-0.0, 0.0) == 0
        assert N.ulp_distance(-5e-324, 5e-324) == 2
        with pytest.raises(ValueError):
            N.ulp_distance(math.nan, 1.0)

    def test_digits_correct(self):
        assert N.digits_correct(1.0001, 1.0) == pytest.approx(4, abs=0.01)
        assert N.digits_correct(1.0, 1.0) == 16.0


class TestNorms:

    @pytest.mark.parametrize("p,ref", [(1, 1), (2, 2), (np.inf, np.inf)])
    def test_vector(self, p, ref):
        x = RNG.normal(size=20)
        assert N.vector_norm(x, p) == pytest.approx(np.linalg.norm(x, ref))

    def test_vector_no_overflow(self):
        assert N.vector_norm([1e300, 1e300]) == pytest.approx(math.sqrt(2) * 1e300)
        assert N.vector_norm([]) == 0.0
        with pytest.raises(ValueError):
            N.vector_norm([1], 3)

    @pytest.mark.parametrize("p", [1, 2, np.inf, "fro"])
    def test_matrix(self, p):
        A = RNG.normal(size=(6, 4))
        assert N.matrix_norm(A, p) == pytest.approx(np.linalg.norm(A, p), rel=1e-6)
        with pytest.raises(ValueError):
            N.matrix_norm(A, 3)


class TestConditioning:

    @pytest.mark.parametrize("n", [2, 6, 10, 20])
    def test_random_matches_numpy(self, n):
        A = RNG.normal(size=(n, n))
        for p in (1, 2, np.inf):
            assert N.condition_number(A, p) == pytest.approx(np.linalg.cond(A, p), rel=1e-4)

    @pytest.mark.parametrize("n", [4, 8, 10, 12])
    def test_hilbert_matches_numpy(self, n):
        assert math.log10(N.condition_number(N.hilbert(n))) == \
            pytest.approx(math.log10(np.linalg.cond(N.hilbert(n))), abs=0.1)

    def test_singular_is_infinite(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert N.condition_number(A, 1) == math.inf
        assert N.condition_number(A, 2) > 1e15
        assert N.estimate_condition_1(A) == math.inf

    @pytest.mark.parametrize("seed", range(10))
    def test_hager_estimate(self, seed):
        A = np.random.default_rng(seed).normal(size=(15, 15))
        est, exact = N.estimate_condition_1(A), np.linalg.cond(A, 1)
        assert 0.3 * exact <= est <= exact * (1 + 1e-9)

    @pytest.mark.parametrize("kappa", [10, 1e6, 1e12])
    def test_matrix_with_condition(self, kappa):
        A = N.matrix_with_condition(25, kappa, seed=1)
        assert np.linalg.cond(A) == pytest.approx(kappa, rel=1e-3)

    def test_check_conditioning_verdicts(self):
        assert N.check_conditioning(np.eye(4)).verdict == "well-conditioned"
        assert N.check_conditioning(N.hilbert(4)).verdict == "moderately ill-conditioned"
        r = N.check_conditioning(N.hilbert(9))
        assert r.verdict == "severely ill-conditioned" and r.ill_conditioned
        assert 10 < r.digits_lost < 13 and "κ" in str(r)
        assert N.check_conditioning(N.hilbert(13)).verdict == "numerically singular"

    def test_error_bounded_by_kappa(self):
        for n in (4, 6, 8, 10):
            H = N.hilbert(n)
            x = np.ones(n)
            from src.fft_analyzer.core.matrix import solve
            err = N.relative_error(solve(H, H @ x), x)
            assert err < 10 * N.condition_number(H) * N.EPS


class TestMatrices:

    def test_hilbert_exact_inverse(self):
        for n in (2, 4, 6):
            inv = np.array(N.hilbert_exact_inverse(n), dtype=float)
            assert np.allclose(inv @ N.hilbert(n), np.eye(n), atol=1e-6)
        assert N.hilbert_exact_inverse(2)[0][0] == 4

    def test_vandermonde(self):
        V = N.vandermonde([1, 2, 3])
        assert V.tolist() == [[1, 1, 1], [1, 2, 4], [1, 3, 9]]
        assert N.vandermonde([1, 2], increasing=False).tolist() == [[1, 1], [2, 1]]


class TestSummationAndCancellation:

    def test_sums(self):
        vals = [0.1] * 100_000
        exact = 10_000.0
        naive = abs(N.naive_sum(vals) - exact)
        assert abs(N.kahan_sum(vals) - exact) < naive / 100
        assert abs(N.pairwise_sum(vals) - exact) < naive
        assert N.pairwise_sum([1, 2, 3]) == 6

    def test_variance(self):
        data = [1e9 + v for v in (4.0, 7.0, 13.0, 16.0)]
        assert N.welford_variance(data) == pytest.approx(30.0)
        assert abs(N.naive_variance(data) - 30.0) > 1.0

    def test_quadratic(self):
        small_naive = N.quadratic_roots_naive(1, 1e8, 1)[0]
        big, small = N.quadratic_roots_stable(1, 1e8, 1)
        assert small == pytest.approx(-1e-8, rel=1e-12)
        assert abs(small_naive + 1e-8) / 1e-8 > 0.1
        assert big == pytest.approx(-1e8)
        assert sorted(N.quadratic_roots_stable(1, -3, 2)) == [1.0, 2.0]

    def test_cancellation_examples(self):
        cases = N.cancellation_examples()
        assert len(cases) == 6
        for c in cases:
            assert c.stable_error <= 1e-10
            assert c.naive_error > 100 * max(c.stable_error, 1e-16)
        assert N.stability_ratio(1e-16, 1e-6) == pytest.approx(1e10)
        assert N.stability_ratio(0.0, 1.0) == math.inf
