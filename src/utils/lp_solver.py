"""
A small linear-programming solver: the two-phase simplex method.

    minimize    c · x
    subject to  A_ub x ≤ b_ub
                A_ge x ≥ b_ge
                A_eq x = b_eq
                x ≥ 0

Week 9 uses it to solve LP relaxations — the integer program with
"x ∈ {0, 1}" loosened to "0 ≤ x ≤ 1" — whose optimum is a lower bound
on the integer optimum and whose fractional values guide rounding.

How it works (tableau form, numpy arrays):

1. Standard form. Every row gets b ≥ 0 (multiply by −1 and flip the
   inequality if needed). A ≤ row gains a slack variable (+s), a ≥ row
   a surplus variable (−s); both turn the row into an equation.
2. Phase 1. Rows without a ready-made basic variable (≥ and = rows)
   get an artificial variable. Minimizing the sum of artificials finds
   a basic feasible solution; if that minimum is above zero, the LP
   is infeasible.
3. Phase 2. Drop the artificials and minimize the real objective from
   that starting vertex.

Each pivot moves to an adjacent vertex of the feasible polytope with
an objective at least as good. The entering column is the one with the
most negative reduced cost (Dantzig's rule), switching to Bland's
smallest-index rule after a run of degenerate pivots — covering LPs
are highly degenerate, and Bland's rule cannot cycle.

Worst case the simplex method visits exponentially many vertices, but
on the covering LPs here it takes a few hundred pivots. Dense numpy
tableaus keep each pivot O(rows × columns).
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"
ITERATION_LIMIT = "iteration_limit"


@dataclass
class LPResult:
    status: str
    x: Optional[np.ndarray]          # values of the original variables
    objective: Optional[float]
    iterations: int                  # pivots across both phases

    @property
    def success(self) -> bool:
        return self.status == OPTIMAL


def _as_matrix(a: Optional[Sequence], n: int) -> np.ndarray:
    if a is None:
        return np.zeros((0, n))
    m = np.asarray(a, dtype=float)
    if m.ndim != 2 or m.shape[1] != n:
        raise ValueError(f"constraint matrix must have {n} columns")
    return m


def _as_vector(b: Optional[Sequence], rows: int) -> np.ndarray:
    if b is None:
        return np.zeros(0)
    v = np.asarray(b, dtype=float).ravel()
    if v.shape[0] != rows:
        raise ValueError(f"right-hand side must have {rows} entries")
    return v


def solve_lp(c: Sequence[float],
             A_ub: Optional[Sequence] = None, b_ub: Optional[Sequence] = None,
             A_ge: Optional[Sequence] = None, b_ge: Optional[Sequence] = None,
             A_eq: Optional[Sequence] = None, b_eq: Optional[Sequence] = None,
             max_iter: int = 100_000, tol: float = 1e-9) -> LPResult:
    """
    Minimize c·x subject to the given constraints and x ≥ 0.

    Returns:
        LPResult with status "optimal", "infeasible", "unbounded", or
        "iteration_limit"; x and objective are set only when optimal.
    """
    c = np.asarray(c, dtype=float).ravel()
    n = c.shape[0]
    blocks = []                       # (A, b, sense) with sense in ≤ ≥ =
    for A, b, sense in ((A_ub, b_ub, "<="), (A_ge, b_ge, ">="),
                        (A_eq, b_eq, "=")):
        A = _as_matrix(A, n)
        blocks.append((A, _as_vector(b, A.shape[0]), sense))

    rows_A, rows_b, senses = [], [], []
    for A, b, sense in blocks:
        for i in range(A.shape[0]):
            a, rhs, s = A[i], b[i], sense
            if rhs < 0:                          # keep b ≥ 0
                a, rhs = -a, -rhs
                s = {"<=": ">=", ">=": "<=", "=": "="}[s]
            rows_A.append(a)
            rows_b.append(rhs)
            senses.append(s)
    m = len(rows_A)
    if m == 0:
        if np.any(c < -tol):
            return LPResult(UNBOUNDED, None, None, 0)
        return LPResult(OPTIMAL, np.zeros(n), 0.0, 0)

    n_slack = sum(s != "=" for s in senses)
    n_art = sum(s != "<=" for s in senses)
    width = n + n_slack + n_art
    T = np.zeros((m + 1, width + 1))
    basis = np.empty(m, dtype=int)
    slack_col, art_col = n, n + n_slack
    art_cols = []
    for i, (a, rhs, s) in enumerate(zip(rows_A, rows_b, senses)):
        T[i, :n] = a
        T[i, -1] = rhs
        if s == "<=":
            T[i, slack_col] = 1.0
            basis[i] = slack_col
            slack_col += 1
        else:
            if s == ">=":
                T[i, slack_col] = -1.0
                slack_col += 1
            T[i, art_col] = 1.0
            basis[i] = art_col
            art_cols.append(art_col)
            art_col += 1

    iterations = 0

    # ---- Phase 1: minimize the sum of artificial variables -------------
    if art_cols:
        cost1 = np.zeros(width)
        cost1[art_cols] = 1.0
        _set_objective(T, basis, cost1)
        status, it = _simplex(T, basis, width, max_iter, tol)
        iterations += it
        if status != OPTIMAL:
            return LPResult(status, None, None, iterations)
        if -T[-1, -1] > 1e-7 * max(1.0, float(np.abs(rows_b).max())):
            return LPResult(INFEASIBLE, None, None, iterations)
        # Drive any artificial still basic (at value 0) out of the basis
        art_set = set(art_cols)
        keep = np.ones(m, dtype=bool)
        for i in range(m):
            if basis[i] in art_set:
                row = T[i, :n + n_slack]
                nz = np.where(np.abs(row) > tol)[0]
                if nz.size:
                    _pivot(T, basis, i, int(nz[0]))
                else:
                    keep[i] = False              # redundant constraint
        if not keep.all():
            T = np.vstack([T[:-1][keep], T[-1:]])
            basis = basis[keep]
        T = np.delete(T, art_cols, axis=1)
        width -= len(art_cols)

    # ---- Phase 2: minimize the real objective ---------------------------
    cost2 = np.zeros(width)
    cost2[:n] = c
    _set_objective(T, basis, cost2)
    status, it = _simplex(T, basis, width, max_iter - iterations, tol)
    iterations += it
    if status != OPTIMAL:
        return LPResult(status, None, None, iterations)

    x = np.zeros(width)
    x[basis] = T[:-1, -1]
    x = np.where(np.abs(x) < tol, 0.0, x)[:n]
    return LPResult(OPTIMAL, x, float(c @ x), iterations)


def _set_objective(T: np.ndarray, basis: np.ndarray,
                   cost: np.ndarray) -> None:
    """Bottom row = reduced costs c_j − c_B·B⁻¹A_j; corner = −c_B·B⁻¹b."""
    cb = cost[basis]
    T[-1, :-1] = cost - cb @ T[:-1, :-1]
    T[-1, -1] = -cb @ T[:-1, -1]


def _pivot(T: np.ndarray, basis: np.ndarray, row: int, col: int) -> None:
    T[row] /= T[row, col]
    factors = T[:, col].copy()
    factors[row] = 0.0
    T -= np.outer(factors, T[row])
    basis[row] = col


def _simplex(T: np.ndarray, basis: np.ndarray, width: int, max_iter: int,
             tol: float) -> tuple:
    """Pivot until no reduced cost is negative. Returns (status, pivots)."""
    degenerate_run = 0
    for it in range(max_iter):
        reduced = T[-1, :width]
        negative = np.where(reduced < -tol)[0]
        if negative.size == 0:
            return OPTIMAL, it
        bland = degenerate_run > 50
        col = int(negative[0]) if bland else int(np.argmin(reduced))
        column = T[:-1, col]
        positive = column > tol
        if not positive.any():
            return UNBOUNDED, it
        ratios = np.full(column.shape, np.inf)
        ratios[positive] = T[:-1, -1][positive] / column[positive]
        best = ratios.min()
        ties = np.where(ratios <= best + tol)[0]
        # Bland: among tied rows, leave by the smallest basic index
        row = int(ties[np.argmin(basis[ties])])
        degenerate_run = degenerate_run + 1 if best <= tol else 0
        _pivot(T, basis, row, col)
    return ITERATION_LIMIT, max_iter
