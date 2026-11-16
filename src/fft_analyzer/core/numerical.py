"""
Numerical stability utilities: errors, norms, conditioning, and the
classic ways floating-point arithmetic loses accuracy.

Errors and norms
    absolute_error, relative_error, ulp_distance (how many representable
    doubles apart two numbers are), digits_correct (−log₁₀ rel. error)
    vector_norm (p = 1, 2, ∞), matrix_norm (1, ∞, Frobenius, 2)

Machine precision: machine_epsilon() finds the smallest ε with
1 + ε ≠ 1 by halving (2⁻⁵² for IEEE double); every elementary
operation is exact up to a relative error of u = ε/2.

Conditioning. κ(A) = ‖A‖·‖A⁻¹‖ bounds how much a relative change in
the data can change the solution of Ax = b:

    ‖δx‖/‖x‖  ≲  κ(A) · ‖δb‖/‖b‖,

so a stable solver still loses about log₁₀ κ(A) of its ~16 digits.
    condition_number       exact κ_p from the inverse (our LU) or, for
                           p = 2, from singular values via the power
                           method on AᵀA and inverse iteration
    estimate_condition_1   Hager–Higham estimator: ‖A⁻¹‖₁ from a few
                           LU solves instead of forming A⁻¹ (O(n²) once
                           A is factored)
    check_conditioning     "is this matrix ill-conditioned?" — κ, digits
                           lost, and a verdict (κ > 1/√ε ≈ 6.7e7 means
                           half the digits are gone; κ > 1/ε means none
                           are left)

Ill-conditioned test matrices: hilbert(n) (κ grows like e^{3.5n}),
vandermonde, and matrix_with_condition(n, κ) built from U·diag(σ)·Vᵀ.

Catastrophic cancellation: subtracting nearly equal numbers keeps the
absolute error but wipes out the leading digits, so the *relative*
error explodes. cancellation_examples() runs six textbook cases, each
with a naive formula, an algebraically equal stable one, and an exact
reference (fractions or extended precision).
"""
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence

import numpy as np

from src.fft_analyzer.core.matrix import inverse, lu_decompose, lu_solve, power_method

EPS = float(np.finfo(float).eps)


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------
def machine_epsilon(dtype=float) -> float:
    """Smallest power of two ε with 1 + ε > 1 in the given type."""
    one = np.array(1, dtype=dtype)
    eps = np.array(1, dtype=dtype)
    while one + eps / np.array(2, dtype=dtype) > one:
        eps = eps / np.array(2, dtype=dtype)
    return float(eps)


def absolute_error(approx, exact) -> float:
    return float(np.max(np.abs(np.asarray(approx) - np.asarray(exact))))


def relative_error(approx, exact, ord=None) -> float:
    """‖approx − exact‖ / ‖exact‖ (componentwise |·| for scalars)."""
    approx, exact = np.asarray(approx, dtype=complex), np.asarray(exact, dtype=complex)
    denom = np.linalg.norm(exact.ravel(), ord)
    diff = np.linalg.norm((approx - exact).ravel(), ord)
    if denom == 0:
        return 0.0 if diff == 0 else math.inf
    return float(diff / denom)


def ulp_distance(a: float, b: float) -> int:
    """Number of representable doubles between a and b."""
    if math.isnan(a) or math.isnan(b):
        raise ValueError("NaN has no ULP distance")
    ia = np.array(a, dtype=np.float64).view(np.int64).item()
    ib = np.array(b, dtype=np.float64).view(np.int64).item()
    # map the sign-magnitude bit patterns onto a monotone integer line
    ia = ia if ia >= 0 else -(ia & 0x7FFFFFFFFFFFFFFF)
    ib = ib if ib >= 0 else -(ib & 0x7FFFFFFFFFFFFFFF)
    return abs(ia - ib)


def digits_correct(approx, exact) -> float:
    """−log₁₀(relative error), capped at ~16 for exact answers."""
    err = relative_error(approx, exact)
    return 16.0 if err == 0 else float(min(16.0, max(0.0, -math.log10(err))))


# ----------------------------------------------------------------------
# norms
# ----------------------------------------------------------------------
def vector_norm(x, p=2) -> float:
    x = np.abs(np.asarray(x, dtype=complex).ravel())
    if p == 1:
        return float(x.sum())
    if p == 2:
        scale = x.max() if x.size else 0.0      # scaled to avoid overflow
        return float(scale * np.sqrt(np.sum((x / scale) ** 2))) if scale else 0.0
    if p in (np.inf, "inf"):
        return float(x.max()) if x.size else 0.0
    raise ValueError("p must be 1, 2 or inf")


def matrix_norm(A, p=2) -> float:
    """Induced 1-norm (max column sum), ∞-norm (max row sum), 'fro',
    or 2-norm (largest singular value, by the power method on AᵀA)."""
    A = np.asarray(A, dtype=float)
    if p == 1:
        return float(np.abs(A).sum(axis=0).max())
    if p in (np.inf, "inf"):
        return float(np.abs(A).sum(axis=1).max())
    if p == "fro":
        return vector_norm(A, 2)
    if p == 2:
        return math.sqrt(max(power_method(A.T @ A, tol=1e-12).value, 0.0))
    raise ValueError("p must be 1, 2, inf or 'fro'")


# ----------------------------------------------------------------------
# conditioning
# ----------------------------------------------------------------------
def singular_value_extremes(A, iterations: int = 300) -> tuple:
    """
    (σ_max, σ_min) without forming AᵀA (which would square κ and lose
    every digit for κ ≳ 10⁸):
      σ_max² — power iteration with x ← Aᵀ(Ax)
      σ_min² — power iteration on (AᵀA)⁻¹ = A⁻¹A⁻ᵀ, applied as two
               triangular-solve pairs with the LU factors of A and Aᵀ
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    rng = np.random.default_rng(1)

    def dominant(apply) -> float:
        x = rng.normal(size=n)
        x /= np.linalg.norm(x)
        lam = 0.0
        for _ in range(iterations):
            y = apply(x)
            norm = np.linalg.norm(y)
            if not np.isfinite(norm) or norm == 0:
                return norm
            new = norm                       # ‖Mx‖ → λ_max for unit x
            x = y / norm
            if abs(new - lam) <= 1e-12 * new:
                return new
            lam = new
        return lam

    s_max = math.sqrt(dominant(lambda x: A.T @ (A @ x)))
    try:
        lu, lut = lu_decompose(A), lu_decompose(A.T)
    except ValueError:
        return s_max, 0.0
    inv_lam = dominant(lambda x: lu_solve(lu, lu_solve(lut, x)))
    if not np.isfinite(inv_lam) or inv_lam == 0:
        return s_max, 0.0
    return s_max, 1.0 / math.sqrt(inv_lam)


def condition_number(A, p=2) -> float:
    """κ_p(A) = ‖A‖_p ‖A⁻¹‖_p (∞ for a singular matrix)."""
    A = np.asarray(A, dtype=float)
    if p == 2:
        s_max, s_min = singular_value_extremes(A)
        return math.inf if s_min == 0 else s_max / s_min
    try:
        Ainv = inverse(A)
    except ValueError:
        return math.inf
    return matrix_norm(A, p) * matrix_norm(Ainv, p)


def estimate_condition_1(A, max_iter: int = 5) -> float:
    """
    Hager's estimator (Higham's variant) for κ₁(A): maximize ‖A⁻¹x‖₁
    over the unit 1-ball by a few steps of gradient ascent, each needing
    one solve with A and one with Aᵀ. Exact or a slight underestimate.
    """
    A = np.asarray(A, dtype=float)
    n = len(A)
    try:
        lu = lu_decompose(A)
        lut = lu_decompose(A.T)
    except ValueError:
        return math.inf
    x = np.full(n, 1.0 / n)
    est = 0.0
    for _ in range(max_iter):
        y = lu_solve(lu, x)
        new = float(np.abs(y).sum())
        xi = np.where(y >= 0, 1.0, -1.0)
        z = lu_solve(lut, xi)
        j = int(np.argmax(np.abs(z)))
        if new <= est or np.abs(z[j]) <= z @ x:
            est = max(est, new)
            break
        est = new
        x = np.zeros(n)
        x[j] = 1.0
    return matrix_norm(A, 1) * est


@dataclass
class ConditionReport:
    kappa: float
    digits_lost: float
    digits_left: float
    verdict: str

    @property
    def ill_conditioned(self) -> bool:
        return self.verdict != "well-conditioned"

    def __str__(self) -> str:
        return (f"κ ≈ {self.kappa:.3g}: {self.verdict} (expect to lose "
                f"~{self.digits_lost:.1f} of 16 digits)")


def check_conditioning(A, p=2) -> ConditionReport:
    """Is this matrix ill-conditioned? κ ≤ 10⁴: well-conditioned;
    up to 1/√ε: moderately ill-conditioned; up to 1/ε: severely;
    beyond: numerically singular."""
    kappa = condition_number(A, p)
    lost = math.log10(kappa) if math.isfinite(kappa) and kappa > 0 else math.inf
    if kappa <= 1e4:
        verdict = "well-conditioned"
    elif kappa <= 1 / math.sqrt(EPS):
        verdict = "moderately ill-conditioned"
    elif kappa < 1 / EPS:
        verdict = "severely ill-conditioned"
    else:
        verdict = "numerically singular"
    return ConditionReport(kappa, lost, max(0.0, -math.log10(EPS) - lost), verdict)


# ----------------------------------------------------------------------
# test matrices
# ----------------------------------------------------------------------
def hilbert(n: int) -> np.ndarray:
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1)


def hilbert_exact_inverse(n: int) -> List[List[Fraction]]:
    """Exact inverse of the n×n Hilbert matrix (integer entries)."""
    out = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = Fraction(
                (-1) ** (i + j) * (i + j + 1) * math.comb(n + i, n - j - 1)
                * math.comb(n + j, n - i - 1) * math.comb(i + j, i) ** 2)
    return out


def vandermonde(x: Sequence[float], increasing: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    powers = np.arange(len(x))
    V = x[:, None] ** powers[None, :]
    return V if increasing else V[:, ::-1]


def matrix_with_condition(n: int, kappa: float, seed: int = 0) -> np.ndarray:
    """Random n×n matrix with 2-norm condition number exactly kappa
    (singular values log-spaced from 1 down to 1/kappa)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    sigma = np.logspace(0, -math.log10(kappa), n)
    return (U * sigma) @ V.T


# ----------------------------------------------------------------------
# summation and cancellation
# ----------------------------------------------------------------------
def naive_sum(values: Sequence[float]) -> float:
    total = 0.0
    for v in values:
        total += v
    return total


def kahan_sum(values: Sequence[float]) -> float:
    """Compensated summation: carries the low-order bits each addition
    loses, so the error is O(ε) instead of O(nε)."""
    total, comp = 0.0, 0.0
    for v in values:
        y = v - comp
        t = total + y
        comp = (t - total) - y
        total = t
    return total


def pairwise_sum(values: Sequence[float]) -> float:
    """Recursive halving (what numpy.sum does): error O(ε log n)."""
    values = list(values)
    if len(values) <= 8:
        return naive_sum(values)
    mid = len(values) // 2
    return pairwise_sum(values[:mid]) + pairwise_sum(values[mid:])


def naive_variance(x: Sequence[float]) -> float:
    """E[x²] − E[x]²: cancels catastrophically when the mean is large."""
    n = len(x)
    s, s2 = 0.0, 0.0
    for v in x:
        s += v
        s2 += v * v
    return (s2 - s * s / n) / (n - 1)


def welford_variance(x: Sequence[float]) -> float:
    """One-pass, numerically stable (Welford 1962)."""
    mean, m2 = 0.0, 0.0
    for k, v in enumerate(x, 1):
        delta = v - mean
        mean += delta / k
        m2 += delta * (v - mean)
    return m2 / (len(x) - 1)


def quadratic_roots_naive(a: float, b: float, c: float):
    d = math.sqrt(b * b - 4 * a * c)
    return (-b + d) / (2 * a), (-b - d) / (2 * a)


def quadratic_roots_stable(a: float, b: float, c: float):
    """q = −(b + sign(b)√D)/2: no subtraction of nearly equal numbers;
    the second root comes from Vieta (x₁x₂ = c/a)."""
    d = math.sqrt(b * b - 4 * a * c)
    q = -0.5 * (b + math.copysign(d, b))
    return q / a, c / q


@dataclass
class CancellationCase:
    name: str
    naive: float
    stable: float
    exact: float

    @property
    def naive_error(self) -> float:
        return abs(self.naive - self.exact) / abs(self.exact)

    @property
    def stable_error(self) -> float:
        return abs(self.stable - self.exact) / abs(self.exact)


def cancellation_examples() -> List[CancellationCase]:
    """Six classic losses of significance, with naive, stable and
    exact (high-precision) results."""
    import decimal
    D = decimal.Decimal
    ctx = decimal.Context(prec=60)
    cases = []

    x = 1e-8                                       # (1 − cos x)/x²
    exact = 0.5 - x * x / 24
    cases.append(CancellationCase("(1 - cos x)/x^2, x = 1e-8",
                                  (1 - math.cos(x)) / x ** 2,
                                  2 * math.sin(x / 2) ** 2 / x ** 2, exact))

    x = 1e12                                       # √(x+1) − √x
    exact_d = ctx.sqrt(D(x) + 1) - ctx.sqrt(D(x))
    cases.append(CancellationCase("sqrt(x+1) - sqrt(x), x = 1e12",
                                  math.sqrt(x + 1) - math.sqrt(x),
                                  1 / (math.sqrt(x + 1) + math.sqrt(x)),
                                  float(exact_d)))

    a, b, c = 1.0, 1e8, 1.0                        # small root of x² + 1e8x + 1
    exact_small = float((-D(b) + ctx.sqrt(D(b) ** 2 - 4 * D(a) * D(c))) / (2 * D(a)))
    cases.append(CancellationCase("small root of x^2 + 1e8 x + 1",
                                  quadratic_roots_naive(a, b, c)[0],
                                  quadratic_roots_stable(a, b, c)[1], exact_small))

    data = [1e9 + v for v in (4.0, 7.0, 13.0, 16.0)]   # variance, large mean
    exact_var = float(np.var([4.0, 7.0, 13.0, 16.0], ddof=1))
    cases.append(CancellationCase("variance of 1e9 + {4, 7, 13, 16}",
                                  naive_variance(data), welford_variance(data),
                                  exact_var))

    vals = [0.1] * 1_000_000                       # sum of 0.1, a million times
    exact_sum = float(sum(Fraction(0.1) for _ in range(10)) * 100_000)
    cases.append(CancellationCase("sum of 0.1 x 10^6", naive_sum(vals),
                                  kahan_sum(vals), exact_sum))

    x = -20.0                                      # e^x by Taylor series
    terms = [x ** k / math.factorial(k) for k in range(120)]
    taylor = sum(terms)
    via_reciprocal = 1 / sum((-x) ** k / math.factorial(k) for k in range(120))
    cases.append(CancellationCase("exp(-20) by Taylor series", taylor,
                                  via_reciprocal, math.exp(-20)))
    return cases


def stability_ratio(stable_error: float, naive_error: float) -> float:
    return naive_error / stable_error if stable_error else math.inf
