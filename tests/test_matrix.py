"""Test suite for matrix multiplication (classical, Strassen), LU with
partial pivoting, solves, determinants, QR, and eigenvalue methods."""

import importlib

import numpy as np
import pytest

M = importlib.import_module("src.fft_analyzer.core.matrix")
RNG = np.random.default_rng(1)


class TestMultiplication:

    @pytest.mark.parametrize("shape", [(1, 1, 1), (3, 4, 5), (16, 16, 16), (7, 2, 9)])
    def test_classical(self, shape):
        n, m, p = shape
        A, B = RNG.normal(size=(n, m)), RNG.normal(size=(m, p))
        assert np.allclose(M.matmul_classical(A, B), A @ B)

    @pytest.mark.parametrize("n", [1, 2, 5, 16, 33, 64])
    @pytest.mark.parametrize("cutoff", [1, 4, 16])
    def test_strassen_square(self, n, cutoff):
        A, B = RNG.normal(size=(n, n)), RNG.normal(size=(n, n))
        assert np.allclose(M.strassen(A, B, cutoff), A @ B)

    def test_strassen_rectangular_and_base(self):
        A, B = RNG.normal(size=(10, 3)), RNG.normal(size=(3, 7))
        assert np.allclose(M.strassen(A, B, 2), A @ B)
        assert np.allclose(M.strassen(A, B, 2, base=np.matmul), A @ B)

    def test_shape_errors(self):
        with pytest.raises(ValueError):
            M.matmul_classical(np.ones((2, 3)), np.ones((2, 3)))
        with pytest.raises(ValueError):
            M.strassen(np.ones((2, 3)), np.ones((2, 3)))

    def test_multiplication_count(self):
        assert M.strassen_multiplications(8, 8) == 512
        assert M.strassen_multiplications(16, 8) == 7 * 512
        assert M.strassen_multiplications(256, 32) < 256 ** 3


class TestLU:

    @pytest.mark.parametrize("n", [1, 2, 5, 20, 50])
    def test_factorization(self, n):
        A = RNG.normal(size=(n, n))
        r = M.lu_decompose(A)
        assert np.allclose(r.P @ A, r.L @ r.U)
        assert np.allclose(np.tril(r.L), r.L) and np.allclose(np.diag(r.L), 1)
        assert np.allclose(np.triu(r.U), r.U)
        assert np.abs(r.L).max() <= 1 + 1e-12          # partial pivoting bound

    def test_pivoting_needed(self):
        A = [[0.0, 1.0], [1.0, 1.0]]
        r = M.lu_decompose(A)
        assert r.swaps == 1 and np.allclose(r.P @ np.array(A), r.L @ r.U)
        with pytest.raises(ValueError, match="zero pivot"):
            M.lu_no_pivot(A)

    def test_tiny_pivot_instability(self):
        A = [[1e-20, 1.0], [1.0, 1.0]]
        assert np.allclose(M.solve(A, [1.0, 2.0]), [1.0, 1.0])
        bad = M.solve(A, [1.0, 2.0], pivot=False)
        assert abs(bad[0] - 1.0) > 0.5                  # catastrophically wrong
        assert M.lu_no_pivot(A).growth > 1e19

    @pytest.mark.parametrize("n", [3, 10, 40])
    def test_solve_det_inverse(self, n):
        A = RNG.normal(size=(n, n))
        b = RNG.normal(size=n)
        assert np.allclose(A @ M.solve(A, b), b)
        assert np.isclose(M.determinant(A), np.linalg.det(A))
        assert np.allclose(M.inverse(A) @ A, np.eye(n))
        B = RNG.normal(size=(n, 3))
        assert np.allclose(A @ M.lu_solve(M.lu_decompose(A), B), B)

    def test_singular(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert M.determinant(A) == 0.0
        with pytest.raises(ValueError):
            M.inverse(A)
        with pytest.raises(ValueError):
            M.lu_decompose(np.ones((2, 3)))

    def test_triangular_solves(self):
        L = np.tril(RNG.normal(size=(5, 5))) + 5 * np.eye(5)
        U = L.T.copy()
        b = RNG.normal(size=5)
        assert np.allclose(L @ M.forward_substitution(L, b), b)
        assert np.allclose(U @ M.back_substitution(U, b), b)


class TestQR:

    @pytest.mark.parametrize("method", list(M.QR_METHODS))
    @pytest.mark.parametrize("shape", [(5, 5), (12, 4), (30, 30)])
    def test_reduced_qr(self, method, shape):
        A = RNG.normal(size=shape)
        Q, R = M.qr(A, method)
        assert Q.shape == shape and R.shape == (shape[1], shape[1])
        assert np.allclose(Q @ R, A)
        assert np.allclose(Q.T @ Q, np.eye(shape[1]), atol=1e-10)
        assert np.allclose(np.triu(R), R)

    def test_householder_full_and_wide(self):
        A = RNG.normal(size=(6, 3))
        Q, R = M.qr_householder(A, full=True)
        assert Q.shape == (6, 6) and np.allclose(Q @ R, A) and np.allclose(Q.T @ Q, np.eye(6))
        W = RNG.normal(size=(3, 5))
        Q, R = M.qr_householder(W)
        assert np.allclose(Q @ R, W)

    def test_orthogonality_ranking_on_ill_conditioned(self):
        from src.fft_analyzer.core.numerical import matrix_with_condition
        A = matrix_with_condition(30, 1e10, seed=2)
        loss = {m: np.linalg.norm(f(A)[0].T @ f(A)[0] - np.eye(30))
                for m, f in M.QR_METHODS.items()}
        assert loss["householder"] < 1e-13
        assert loss["householder"] < loss["modified_gram_schmidt"] < loss["gram_schmidt"]

    def test_dependent_columns_and_bad_method(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(ValueError):
            M.qr_gram_schmidt(A)
        with pytest.raises(ValueError):
            M.qr(A, "givens")

    def test_least_squares(self):
        A = RNG.normal(size=(200, 5))
        b = RNG.normal(size=200)
        assert np.allclose(M.least_squares(A, b), np.linalg.lstsq(A, b, rcond=None)[0])

    def test_least_squares_tall_without_forming_q(self):
        A = RNG.normal(size=(50_000, 6))
        b = A @ np.arange(6.0)
        assert np.allclose(M.least_squares(A, b), np.arange(6.0))

    def test_reflector_helpers(self):
        A = RNG.normal(size=(7, 4))
        vs, R = M.householder_reflectors(A)
        b = RNG.normal(size=7)
        assert np.allclose(M.apply_q(vs, M.apply_qt(vs, b)), b)


class TestEigen:

    def _sym(self, n, seed=3):
        B = np.random.default_rng(seed).normal(size=(n, n))
        return B @ B.T

    def test_power_method(self):
        S = self._sym(10)
        r = M.power_method(S)
        assert r.converged and np.isclose(r.value, np.linalg.eigvalsh(S)[-1], rtol=1e-8)
        assert np.allclose(np.abs(r.vector), np.abs(np.linalg.eigh(S)[1][:, -1]), atol=1e-4)
        assert len(r.history) == r.iterations + 1

    def test_power_method_rate(self):
        Q, _ = np.linalg.qr(RNG.normal(size=(20, 20)))
        fast = (Q * np.r_[1.0, 0.3, np.full(18, 0.1)]) @ Q.T
        slow = (Q * np.r_[1.0, 0.95, np.full(18, 0.1)]) @ Q.T
        assert M.power_method(fast).iterations < M.power_method(slow).iterations / 5

    def test_power_method_limits(self):
        r = M.power_method(np.diag([1.0, 0.999999]), max_iter=5)
        assert not r.converged and r.iterations == 5
        z = M.power_method(np.zeros((3, 3)))
        assert z.value == 0.0

    def test_inverse_iteration_and_qr_algorithm(self):
        S = self._sym(8)
        ev = np.linalg.eigvalsh(S)
        assert np.isclose(M.inverse_power_method(S, 0.0).value, ev[0], rtol=1e-8)
        assert np.isclose(M.inverse_power_method(S, ev[3] + 1e-3).value, ev[3], rtol=1e-8)
        assert np.allclose(M.qr_eigenvalues(S), ev[::-1], rtol=1e-6)
        with pytest.raises(ValueError):
            M.qr_eigenvalues(RNG.normal(size=(3, 3)))
