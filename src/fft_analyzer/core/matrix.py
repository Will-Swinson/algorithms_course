"""
Dense matrix algorithms: multiplication, LU, QR, determinants, linear
solves, and power-method eigenvalues.

Multiplication
    matmul_classical  the triple loop on Python lists, i-k-j order (the
                      inner loop walks a row of B and a row of C, which
                      is what a cache — and Python's list indexing —
                      prefers). n³ multiplications.
    strassen          7 recursive products of half-size blocks instead
                      of 8: O(n^log₂7) ≈ O(n^2.807). Odd sizes are padded.
                      Below `cutoff` it hands off to `base` (the classical
                      loop by default). Fewer multiplications, but more
                      additions and a weaker error bound
                      (‖ΔC‖ ≲ c·n^{log₂12}·ε·‖A‖‖B‖ instead of n·ε·|A||B|
                      elementwise) — the stability-vs-performance trade.

LU with partial pivoting: PA = LU. At each column the row with the
largest |pivot| is swapped up, so every multiplier |l_ij| ≤ 1 and the
growth factor stays small in practice. Without pivoting a tiny pivot
(e.g. [[1e-20, 1], [1, 1]]) produces multipliers of 10²⁰ and destroys
the answer — lu_no_pivot exists to show it. Solving Ax = b is then two
triangular solves (O(n²) each), and det(A) = sign(P)·Π u_ii.

QR (A = QR, Q orthonormal columns, R upper triangular):
    classical Gram–Schmidt   orthogonality loss grows like κ(A)²·ε
    modified Gram–Schmidt    subtracts projections one at a time: κ(A)·ε
    Householder reflections  backward stable: ‖QᵀQ − I‖ ≈ ε regardless
                             of κ(A)
(the benchmark plots exactly these three slopes).

Eigenvalues: the power method multiplies by A and normalizes; the
component along the dominant eigenvector grows by |λ₁/λ₂| each step,
so convergence is linear with ratio |λ₂/λ₁|. Inverse iteration with a
shift σ finds the eigenvalue nearest σ; the unshifted QR algorithm
(A_{k+1} = R_k Q_k) gives all eigenvalues of a symmetric matrix.
"""
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# multiplication
# ----------------------------------------------------------------------
def _as_lists(A) -> List[List[float]]:
    return [list(map(float, row)) for row in np.asarray(A, dtype=float)]


def matmul_classical(A, B) -> np.ndarray:
    """Pure-Python triple loop (i-k-j order). O(n·m·p)."""
    a, b = _as_lists(A), _as_lists(B)
    n, m = len(a), len(a[0]) if a else 0
    if m != len(b):
        raise ValueError(f"shapes {np.shape(A)} and {np.shape(B)} do not align")
    p = len(b[0]) if b else 0
    c = [[0.0] * p for _ in range(n)]
    for i in range(n):
        ci, ai = c[i], a[i]
        for k in range(m):
            aik = ai[k]
            if aik:
                bk = b[k]
                for j in range(p):
                    ci[j] += aik * bk[j]
    return np.array(c).reshape(n, p)


def strassen(A, B, cutoff: int = 64,
             base: Optional[Callable] = None) -> np.ndarray:
    """
    Strassen multiplication of square or rectangular matrices (padded
    to a common power-of-two size). `base` multiplies blocks at or
    below `cutoff` (default: matmul_classical).
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shapes {A.shape} and {B.shape} do not align")
    base = base or matmul_classical
    n = max(A.shape + B.shape)
    size = 1
    while size < n:
        size *= 2
    Ap = np.zeros((size, size))
    Bp = np.zeros((size, size))
    Ap[:A.shape[0], :A.shape[1]] = A
    Bp[:B.shape[0], :B.shape[1]] = B
    C = _strassen(Ap, Bp, max(1, cutoff), base)
    return C[:A.shape[0], :B.shape[1]]


def _strassen(A, B, cutoff, base):
    n = A.shape[0]
    if n <= cutoff:
        return np.asarray(base(A, B))
    h = n // 2
    a11, a12, a21, a22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    b11, b12, b21, b22 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]
    m1 = _strassen(a11 + a22, b11 + b22, cutoff, base)
    m2 = _strassen(a21 + a22, b11, cutoff, base)
    m3 = _strassen(a11, b12 - b22, cutoff, base)
    m4 = _strassen(a22, b21 - b11, cutoff, base)
    m5 = _strassen(a11 + a12, b22, cutoff, base)
    m6 = _strassen(a21 - a11, b11 + b12, cutoff, base)
    m7 = _strassen(a12 - a22, b21 + b22, cutoff, base)
    C = np.empty((n, n))
    C[:h, :h] = m1 + m4 - m5 + m7
    C[:h, h:] = m3 + m5
    C[h:, :h] = m2 + m4
    C[h:, h:] = m1 - m2 + m3 + m6
    return C


def strassen_multiplications(n: int, cutoff: int) -> int:
    """Scalar multiplications Strassen performs on n × n (n a power of
    two), versus n³ for the classical algorithm."""
    if n <= cutoff:
        return n ** 3
    return 7 * strassen_multiplications(n // 2, cutoff)


# ----------------------------------------------------------------------
# LU
# ----------------------------------------------------------------------
@dataclass
class LUResult:
    L: np.ndarray
    U: np.ndarray
    perm: np.ndarray            # PA = LU with P = I[perm]
    swaps: int
    growth: float               # max|U| / max|A| (element growth factor)

    @property
    def P(self) -> np.ndarray:
        return np.eye(len(self.perm))[self.perm]


def _square(A) -> np.ndarray:
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {A.shape}")
    return A


def lu_decompose(A, pivot: bool = True, tol: float = 0.0) -> LUResult:
    """
    Doolittle LU (unit lower-triangular L) with partial pivoting.
    Raises ValueError for a (numerically) singular matrix.
    """
    U = _square(A)
    n = len(U)
    L = np.eye(n)
    perm = np.arange(n)
    swaps = 0
    scale = np.abs(U).max() if n else 0.0
    for k in range(n - 1):
        if pivot:
            p = k + int(np.argmax(np.abs(U[k:, k])))
            if p != k:
                U[[k, p], k:] = U[[p, k], k:]
                L[[k, p], :k] = L[[p, k], :k]
                perm[[k, p]] = perm[[p, k]]
                swaps += 1
        if abs(U[k, k]) <= tol:
            if pivot:
                raise ValueError("matrix is singular to working precision")
            raise ValueError(f"zero pivot at step {k}: LU without pivoting fails")
        factors = U[k + 1:, k] / U[k, k]
        L[k + 1:, k] = factors
        U[k + 1:, k:] -= np.outer(factors, U[k, k:])
        U[k + 1:, k] = 0.0
    if n and abs(U[-1, -1]) <= tol and pivot:
        raise ValueError("matrix is singular to working precision")
    growth = float(np.abs(U).max() / scale) if scale else 1.0
    return LUResult(L, U, perm, swaps, growth)


def lu_no_pivot(A) -> LUResult:
    return lu_decompose(A, pivot=False)


def forward_substitution(L, b, unit: bool = False) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    b = np.array(b, dtype=float)
    n = len(L)
    x = np.zeros_like(b)
    for i in range(n):
        s = b[i] - L[i, :i] @ x[:i]
        x[i] = s if unit else s / L[i, i]
    return x


def back_substitution(U, b) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    b = np.array(b, dtype=float)
    n = len(U)
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def lu_solve(lu: LUResult, b) -> np.ndarray:
    """Solve Ax = b given PA = LU (b may have several columns)."""
    b = np.asarray(b, dtype=float)
    y = forward_substitution(lu.L, b[lu.perm], unit=True)
    return back_substitution(lu.U, y)


def solve(A, b, pivot: bool = True) -> np.ndarray:
    return lu_solve(lu_decompose(A, pivot), b)


def determinant(A) -> float:
    """det(A) = (−1)^swaps · Π u_ii (0 for singular matrices; ±inf when
    it overflows a double — use log_determinant for big matrices)."""
    sign, logdet = log_determinant(A)
    if sign == 0:
        return 0.0
    return sign * math.exp(logdet) if logdet < 709.0 else sign * math.inf


def log_determinant(A) -> Tuple[float, float]:
    """(sign, log|det A|) from the LU diagonal: the determinant of even a
    modest matrix (a 400×400 Gaussian one is ~10^400) overflows a double,
    its logarithm never does."""
    try:
        lu = lu_decompose(A)
    except ValueError:
        return 0.0, -np.inf
    d = np.diag(lu.U)
    sign = (-1) ** lu.swaps * float(np.prod(np.sign(d)))
    return sign, float(np.sum(np.log(np.abs(d))))


def inverse(A) -> np.ndarray:
    lu = lu_decompose(A)
    return lu_solve(lu, np.eye(len(lu.U)))


# ----------------------------------------------------------------------
# QR
# ----------------------------------------------------------------------
def _matrix(A) -> np.ndarray:
    A = np.array(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    return A


def qr_gram_schmidt(A) -> Tuple[np.ndarray, np.ndarray]:
    """Classical Gram–Schmidt: project each column on all previous q's
    computed from the *original* column. Reduced QR (m ≥ n)."""
    A = _matrix(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        R[:j, j] = Q[:, :j].T @ A[:, j]
        v -= Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            raise ValueError("columns are linearly dependent")
        Q[:, j] = v / R[j, j]
    return Q, R


def qr_modified_gram_schmidt(A) -> Tuple[np.ndarray, np.ndarray]:
    """Modified Gram–Schmidt: remove each q's component from all
    remaining columns as soon as q is known."""
    V = _matrix(A)
    m, n = V.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        if R[j, j] == 0:
            raise ValueError("columns are linearly dependent")
        Q[:, j] = V[:, j] / R[j, j]
        R[j, j + 1:] = Q[:, j] @ V[:, j + 1:]
        V[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
    return Q, R


def householder_reflectors(A) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Triangularize A with Householder reflections H_k = I − 2v_kv_kᵀ
    (‖v_k‖ = 1, acting on rows k…m−1). v = x + sign(x₁)‖x‖e₁ — the sign
    avoids cancellation. Returns (reflectors, R as m×n). Q is never
    formed: O(mn²) work and O(mn) memory even for very tall A.
    """
    R = _matrix(A)
    m, n = R.shape
    vs: List[np.ndarray] = []
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        norm = np.linalg.norm(x)
        if norm == 0:
            vs.append(np.zeros(m - k))
            continue
        v = x.copy()
        v[0] += np.copysign(norm, x[0] if x[0] != 0 else 1.0)
        v /= np.linalg.norm(v)
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        R[k + 1:, k] = 0.0
        vs.append(v)
    return vs, R


def apply_qt(vs: List[np.ndarray], b) -> np.ndarray:
    """Qᵀb = H_k ⋯ H_1 b, applying the stored reflectors."""
    y = np.array(b, dtype=float)
    for k, v in enumerate(vs):
        y[k:] -= 2.0 * np.outer(v, v @ y[k:]).reshape(y[k:].shape)
    return y


def apply_q(vs: List[np.ndarray], y) -> np.ndarray:
    """Qy = H_1 ⋯ H_k y."""
    out = np.array(y, dtype=float)
    for k in range(len(vs) - 1, -1, -1):
        v = vs[k]
        out[k:] -= 2.0 * np.outer(v, v @ out[k:]).reshape(out[k:].shape)
    return out


def qr_householder(A, full: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Householder QR. Reduced factors (Q m×n, R n×n) by default — Q is
    built by applying the reflectors to the first n columns of I —
    or full factors (Q m×m, R m×n).
    """
    vs, R = householder_reflectors(A)
    m, n = R.shape
    if full:
        return apply_q(vs, np.eye(m)), R
    k = min(m, n)
    return apply_q(vs, np.eye(m, k)), R[:k, :]


QR_METHODS = {"gram_schmidt": qr_gram_schmidt,
              "modified_gram_schmidt": qr_modified_gram_schmidt,
              "householder": qr_householder}


def qr(A, method: str = "householder") -> Tuple[np.ndarray, np.ndarray]:
    try:
        return QR_METHODS[method](A)
    except KeyError:
        raise ValueError(f"unknown QR method {method!r}; choose from "
                         f"{list(QR_METHODS)}") from None


def least_squares(A, b) -> np.ndarray:
    """min ‖Ax − b‖₂ via Householder QR: Rx = (Qᵀb)[:n], without
    forming Q (works for very tall A, e.g. fitting a model to a whole
    audio signal)."""
    vs, R = householder_reflectors(A)
    n = R.shape[1]
    return back_substitution(R[:n, :n], apply_qt(vs, b)[:n])


# ----------------------------------------------------------------------
# eigenvalues
# ----------------------------------------------------------------------
@dataclass
class EigenResult:
    value: float
    vector: np.ndarray
    iterations: int
    converged: bool
    history: List[float] = field(default_factory=list)


def power_method(A, x0: Optional[Sequence[float]] = None, tol: float = 1e-10,
                 max_iter: int = 10_000, seed: int = 0) -> EigenResult:
    """
    Dominant eigenpair by repeated multiplication. The eigenvalue
    estimate is the Rayleigh quotient xᵀAx (x normalized). Stops when
    the residual ‖Ax − λx‖ ≤ tol·|λ|. (Testing only the change in λ is
    not enough: when |λ₂| ≈ |λ₁| the Rayleigh quotient barely moves while
    x is still far from the eigenvector.) The residual shrinks like
    |λ₂/λ₁|^k, so expect about log(tol)/log|λ₂/λ₁| iterations.
    """
    A = _square(A)
    n = len(A)
    x = np.random.default_rng(seed).normal(size=n) if x0 is None \
        else np.array(x0, dtype=float)
    x /= np.linalg.norm(x)
    lam = float(x @ A @ x)
    history = [lam]
    for it in range(1, max_iter + 1):
        y = A @ x
        norm = np.linalg.norm(y)
        if norm == 0:
            return EigenResult(0.0, x, it, True, history)
        x = y / norm
        Ax = A @ x
        new = float(x @ Ax)
        history.append(new)
        if np.linalg.norm(Ax - new * x) <= tol * max(abs(new), 1e-300):
            return EigenResult(new, x, it, True, history)
        lam = new
    return EigenResult(lam, x, max_iter, False, history)


def inverse_power_method(A, shift: float = 0.0, tol: float = 1e-12,
                         max_iter: int = 1_000, seed: int = 0) -> EigenResult:
    """Eigenpair closest to `shift`: power method on (A − σI)⁻¹,
    factorized once with LU."""
    A = _square(A)
    n = len(A)
    lu = lu_decompose(A - shift * np.eye(n))
    x = np.random.default_rng(seed).normal(size=n)
    x /= np.linalg.norm(x)
    lam = float(x @ A @ x)
    history = [lam]
    for it in range(1, max_iter + 1):
        y = lu_solve(lu, x)
        x = y / np.linalg.norm(y)
        new = float(x @ A @ x)
        history.append(new)
        if abs(new - lam) <= tol * max(abs(new), 1.0):
            return EigenResult(new, x, it, True, history)
        lam = new
    return EigenResult(lam, x, max_iter, False, history)


def qr_eigenvalues(A, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """
    All eigenvalues of a symmetric matrix by the unshifted QR algorithm
    (A_{k+1} = R_k Q_k is similar to A_k and converges to diagonal
    form). Sorted descending.
    """
    Ak = _square(A)
    if not np.allclose(Ak, Ak.T):
        raise ValueError("qr_eigenvalues expects a symmetric matrix")
    for _ in range(max_iter):
        Q, R = qr_householder(Ak)
        Ak = R @ Q
        off = np.linalg.norm(Ak - np.diag(np.diag(Ak)))
        if off <= tol * max(np.linalg.norm(Ak), 1e-300):
            break
    return np.sort(np.diag(Ak))[::-1]
