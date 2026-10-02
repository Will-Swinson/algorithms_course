"""
Matrix Chain Multiplication — interval DP.

Matrix multiplication is associative, so A1·A2·A3 can be computed as
(A1·A2)·A3 or A1·(A2·A3) with the same result — but not the same cost.
Multiplying p×q by q×r takes p·q·r scalar multiplications, and for
dimensions 10×100, 100×5, 5×50 the two orders cost 7,500 vs 75,000.
Given dimensions p[0..n] (matrix A_i is p[i-1] × p[i]), find the
cheapest parenthesization.

INTERVAL DP: the subproblems are contiguous sub-chains A_i..A_j. Any
parenthesization of A_i..A_j has a LAST multiplication, splitting it
at some k into (A_i..A_k)(A_k+1..A_j):

    m[i][i] = 0
    m[i][j] = min over i <= k < j of
              m[i][k] + m[k+1][j] + p[i-1]·p[k]·p[j]

Optimal substructure: in an optimal order, both halves must
themselves be ordered optimally, or swapping in a cheaper half would
lower the total. Overlap: m[2][4] is needed by m[1][4], m[2][5],
m[2][6], ... The table is filled by INCREASING CHAIN LENGTH, since
every interval depends only on strictly shorter ones.

1. mcm_recursive: tries every split recursively. Θ(3ⁿ) calls —
   exponential (the number of orders is the Catalan number C(n-1)).
2. mcm_memoized: same recursion, each (i, j) cached. O(n³) time.
3. mcm_bottom_up: fills the table by chain length. O(n³) time,
   O(n²) space, no recursion.

split[i][j] records the best k, and parenthesize() turns it into the
order itself, e.g. "((A1(A2A3))((A4A5)A6))".
"""
from typing import List, Optional, Sequence, Tuple

from src.dp._recursion import recursion_depth
from src.utils.timer import CallStats

INF = float("inf")
Table = List[List[int]]


def _validate(p: Sequence[int]) -> None:
    if len(p) < 2:
        raise ValueError("need at least 2 dimensions (one matrix), "
                         f"got {len(p)}")
    for d in p:
        if isinstance(d, bool) or not isinstance(d, int):
            raise TypeError(f"dimensions must be ints, got {d!r}")
        if d <= 0:
            raise ValueError(f"dimensions must be positive, got {d}")


def mcm_recursive(p: Sequence[int],
                  stats: Optional[CallStats] = None) -> int:
    """
    Minimum multiplication cost by trying every split, no caching.

    Args:
        p: Dimensions; matrix A_i is p[i-1] × p[i] (len(p) = n + 1).
        stats: Optional counter for calls and recursion depth.

    Returns:
        The minimum number of scalar multiplications.

    Raises:
        TypeError / ValueError: On invalid dimensions.

    Complexity: Θ(3ⁿ) calls (exactly 3^(n-1)), O(n) stack depth.
    """
    _validate(p)

    def solve(i: int, j: int) -> int:
        if stats is not None:
            stats.enter()
        best = 0
        if i < j:
            best = INF
            for k in range(i, j):
                cost = (solve(i, k) + solve(k + 1, j)
                        + p[i - 1] * p[k] * p[j])
                if cost < best:
                    best = cost
        if stats is not None:
            stats.exit()
        return best

    return solve(1, len(p) - 1)


def _memoized_tables(p: Sequence[int], stats: Optional[CallStats]
                     ) -> Tuple[Table, Table]:
    n = len(p) - 1
    cost: List[List[Optional[int]]] = [[None] * (n + 1)
                                       for _ in range(n + 1)]
    split = [[0] * (n + 1) for _ in range(n + 1)]

    def solve(i: int, j: int) -> int:
        if stats is not None:
            stats.enter()
        if cost[i][j] is None:
            if i == j:
                cost[i][j] = 0
            else:
                best, best_k = INF, i
                for k in range(i, j):
                    c = solve(i, k) + solve(k + 1, j) + p[i - 1] * p[k] * p[j]
                    if c < best:
                        best, best_k = c, k
                cost[i][j] = best
                split[i][j] = best_k
        if stats is not None:
            stats.exit()
        return cost[i][j]

    with recursion_depth(n + 1):
        solve(1, n)
    return cost, split


def mcm_memoized(p: Sequence[int],
                 stats: Optional[CallStats] = None) -> int:
    """
    Minimum cost top-down: the recursion with each interval (i, j)
    solved once and cached in an (n+1) × (n+1) grid.

    Args / Returns / Raises: as mcm_recursive.

    Complexity: O(n³) time, O(n²) space, O(n) recursion depth.
    """
    _validate(p)
    cost, _ = _memoized_tables(p, stats)
    return cost[1][len(p) - 1]


def build_mcm_tables(p: Sequence[int],
                     stats: Optional[CallStats] = None
                     ) -> Tuple[Table, Table]:
    """
    Fill the bottom-up cost and split tables (1-indexed, row/column 0
    unused).

    cost[i][j]  = minimum scalar multiplications for A_i..A_j
    split[i][j] = the k achieving it (last multiplication is
                  (A_i..A_k)(A_k+1..A_j))

    Args:
        p: Dimensions.
        stats: Optional counter; receives one count per split point
            evaluated (the inner loop), n(n²-1)/6 in total.

    Complexity: O(n³) time, O(n²) space.
    """
    _validate(p)
    n = len(p) - 1
    cost = [[0] * (n + 1) for _ in range(n + 1)]
    split = [[0] * (n + 1) for _ in range(n + 1)]
    evaluated = 0
    for length in range(2, n + 1):              # shorter chains first
        for i in range(1, n - length + 2):
            j = i + length - 1
            best, best_k = INF, i
            p_i, p_j = p[i - 1], p[j]
            cost_i = cost[i]
            for k in range(i, j):
                c = cost_i[k] + cost[k + 1][j] + p_i * p[k] * p_j
                if c < best:
                    best, best_k = c, k
            cost_i[j] = best
            split[i][j] = best_k
            evaluated += length - 1
    if stats is not None:
        stats.count(evaluated)
    return cost, split


def mcm_bottom_up(p: Sequence[int],
                  stats: Optional[CallStats] = None) -> int:
    """
    Minimum cost bottom-up: the top-right cell of the cost table.

    Args / Returns / Raises: as mcm_recursive.

    Complexity: O(n³) time, O(n²) space, no recursion.
    """
    cost, _ = build_mcm_tables(p, stats)
    return cost[1][len(p) - 1]


def parenthesize(split: Table, i: int, j: int) -> str:
    """
    The optimal order for A_i..A_j as a string, e.g. "((A1A2)A3)".

    Args:
        split: Split table from build_mcm_tables() / matrix_chain_order().
        i, j: 1-indexed chain bounds.
    """
    if i == j:
        return f"A{i}"
    k = split[i][j]
    return f"({parenthesize(split, i, k)}{parenthesize(split, k + 1, j)})"


def _tables(p: Sequence[int], method: str) -> Tuple[Table, Table]:
    _validate(p)
    if method == "bottom_up":
        return build_mcm_tables(p)
    if method == "memoized":
        return _memoized_tables(p, None)
    raise ValueError(f"method must be 'bottom_up' or 'memoized', "
                     f"got {method!r}")


def matrix_chain_order(p: Sequence[int], method: str = "bottom_up"
                       ) -> Tuple[int, str]:
    """
    Minimum cost AND the optimal parenthesization.

    Args:
        p: Dimensions.
        method: "bottom_up" (default) or "memoized" — both fill the
            same kind of split table.

    Returns:
        (minimum scalar multiplications, parenthesization string).

    Raises:
        ValueError: On an unknown method or invalid dimensions.
    """
    cost, split = _tables(p, method)
    n = len(p) - 1
    return cost[1][n], parenthesize(split, 1, n)


def split_table(p: Sequence[int], method: str = "bottom_up") -> Table:
    """The split table alone (e.g. for matrix_utils.multiply_chain())."""
    return _tables(p, method)[1]
