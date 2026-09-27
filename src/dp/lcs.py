"""
Longest Common Subsequence — the longest sequence of characters that
appears, in order but not necessarily contiguously, in both strings.
"ABCBDAB" and "BDCABA" share "BCBA" (length 4).

Comparing the prefix lengths i of X and j of Y:

    L(i, j) = 0                                 if i == 0 or j == 0
            = L(i-1, j-1) + 1                   if X[i-1] == Y[j-1]
            = max(L(i-1, j), L(i, j-1))         otherwise

OPTIMAL SUBSTRUCTURE: if the last characters match, some LCS ends with
that character, and the rest of it is an LCS of the shorter prefixes.
If they don't, an LCS drops the last character of X or of Y.
OVERLAPPING SUBPROBLEMS: L(i-1, j) and L(i, j-1) both need
L(i-1, j-1), and the sharing compounds at every level.

1. Naive recursion: O(2^(m+n)) in the worst case (no matches, so
   every call branches twice).
2. Top-down (memoization): each of the (m+1)(n+1) prefix pairs solved
   once. O(m × n) time and space; recursion up to m + n deep.
3. Bottom-up (tabulation): fill the (m+1) × (n+1) table row by row.
   O(m × n) time and space; reconstruct_lcs() walks it backward to
   recover an actual subsequence, not just its length.
"""
from typing import List, Optional, Tuple

from src.dp._recursion import recursion_depth
from src.utils.timer import CallStats


def _validate(x: str, y: str) -> None:
    for name, s in (("x", x), ("y", y)):
        if not isinstance(s, str):
            raise TypeError(f"{name} must be a str, got {type(s).__name__}")


def lcs_recursive(x: str, y: str,
                  stats: Optional[CallStats] = None) -> int:
    """
    LCS length by direct recursion on prefix lengths.

    Args:
        x, y: The two strings.
        stats: Optional counter for calls and recursion depth.

    Returns:
        Length of the longest common subsequence.

    Raises:
        TypeError: If either input is not a str.

    Complexity: O(2^(m+n)) time worst case, O(m + n) stack space.
    """
    _validate(x, y)

    def solve(i: int, j: int) -> int:
        if stats is not None:
            stats.enter()
        if i == 0 or j == 0:
            result = 0
        elif x[i - 1] == y[j - 1]:
            result = solve(i - 1, j - 1) + 1
        else:
            result = max(solve(i - 1, j), solve(i, j - 1))
        if stats is not None:
            stats.exit()
        return result

    with recursion_depth(len(x) + len(y) + 1):
        return solve(len(x), len(y))


def lcs_memo(x: str, y: str, stats: Optional[CallStats] = None) -> int:
    """
    LCS length top-down: the naive recursion with each L(i, j) cached.

    The cache is a preallocated (m+1) × (n+1) grid (-1 = unsolved)
    rather than a dict: nearly every cell gets visited for LCS, and a
    list of ints is several times smaller than a dict of tuple keys.

    Args / Returns / Raises: as lcs_recursive.

    Complexity: O(m × n) time and space, O(m + n) stack depth.
    """
    _validate(x, y)
    memo = [[-1] * (len(y) + 1) for _ in range(len(x) + 1)]

    def solve(i: int, j: int) -> int:
        if stats is not None:
            stats.enter()
        if memo[i][j] < 0:
            if i == 0 or j == 0:
                memo[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                memo[i][j] = solve(i - 1, j - 1) + 1
            else:
                memo[i][j] = max(solve(i - 1, j), solve(i, j - 1))
        if stats is not None:
            stats.exit()
        return memo[i][j]

    with recursion_depth(len(x) + len(y) + 1):
        return solve(len(x), len(y))


def build_lcs_table(x: str, y: str,
                    stats: Optional[CallStats] = None) -> List[List[int]]:
    """
    Fill the bottom-up table: table[i][j] = LCS length of x[:i], y[:j].

    Args:
        x, y: The two strings.
        stats: Optional counter; receives one count per cell filled.

    Returns:
        The (m+1) × (n+1) table.

    Raises:
        TypeError: If either input is not a str.

    Complexity: O(m × n) time and space.
    """
    _validate(x, y)
    m, n = len(x), len(y)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        xi = x[i - 1]
        above = table[i - 1]
        row = table[i]
        for j in range(1, n + 1):
            if xi == y[j - 1]:
                row[j] = above[j - 1] + 1
            else:
                row[j] = max(above[j], row[j - 1])
    if stats is not None:
        stats.count(m * n)
    return table


def lcs_tabulation(x: str, y: str,
                   stats: Optional[CallStats] = None) -> int:
    """
    LCS length bottom-up: the bottom-right cell of the table.

    Args / Returns / Raises: as lcs_recursive.

    Complexity: O(m × n) time and space, no recursion.
    """
    return build_lcs_table(x, y, stats)[-1][-1]


def reconstruct_lcs(table: List[List[int]], x: str, y: str) -> str:
    """
    Recover one longest common subsequence from a filled table.

    Walk from table[m][n] toward table[0][0]: on a character match,
    that character is in the LCS (step diagonally); otherwise step
    toward whichever neighbor holds the larger value — the direction
    the optimum came from. Characters are collected backward, then
    reversed.

    Args:
        table: Output of build_lcs_table(x, y).
        x, y: The same strings the table was built from.

    Returns:
        An LCS of x and y (ties between equal-length answers are broken
        by preferring to drop a character of x).

    Complexity: O(m + n).
    """
    chars: List[str] = []
    i, j = len(x), len(y)
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            chars.append(x[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    chars.reverse()
    return "".join(chars)


def lcs(x: str, y: str) -> Tuple[int, str]:
    """Convenience wrapper: (LCS length, one LCS string)."""
    table = build_lcs_table(x, y)
    return table[-1][-1], reconstruct_lcs(table, x, y)
