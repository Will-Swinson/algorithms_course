"""
Knuth–Morris–Pratt: deterministic O(n + m) pattern matching.

The prefix function (LPS, "longest proper prefix that is also a
suffix") of the pattern P:

    lps[i] = length of the longest proper prefix of P[0..i]
             that is also a suffix of P[0..i]

e.g. P = "ababaca" → lps = [0, 0, 1, 2, 3, 0, 1]. Built in O(m) by
the same fallback idea the search uses.

Search. Keep j = number of pattern characters currently matched.

    match    T[i] == P[j]  →  i += 1, j += 1   (report a hit when j = m,
                                               then j = lps[m − 1])
    mismatch j > 0         →  j = lps[j − 1]   (fallback: reuse the
                                               longest border; i stays)
    mismatch j = 0         →  i += 1

The text index i never moves backwards. Each comparison either
advances i (at most n times) or shrinks j by a fallback, and j can
only shrink as much as it has grown, so there are at most 2n
comparisons — O(n + m) total, no matter how repetitive the input.
Brute force, by contrast, restarts one position later after every
mismatch: O(nm) on inputs like T = a^n, P = a^(m−1)b.

kmp_trace() records every comparison and fallback so the behaviour can
be printed or plotted (the demo shows it).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class MatchResult:
    """positions: 0-based starts of every (possibly overlapping)
    occurrence; comparisons: character comparisons performed."""
    positions: List[int]
    comparisons: int = 0
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.positions)


def prefix_function(pattern: Sequence) -> List[int]:
    """LPS array of `pattern` in O(m)."""
    m = len(pattern)
    lps = [0] * m
    k = 0
    for i in range(1, m):
        c = pattern[i]
        while k and pattern[k] != c:
            k = lps[k - 1]
        if pattern[k] == c:
            k += 1
        lps[i] = k
    return lps


def kmp_search(text: Sequence, pattern: Sequence,
               lps: List[int] = None) -> MatchResult:
    """
    All occurrences of `pattern` in `text` (overlapping allowed).
    Pass a precomputed `lps` to skip preprocessing on repeated queries.
    An empty pattern matches at every position 0..n.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return MatchResult(list(range(n + 1)), 0)
    if m > n:
        return MatchResult([], 0)
    lps = prefix_function(pattern) if lps is None else lps
    positions = []
    comparisons = 0
    fallbacks = 0
    j = 0
    last = m - 1
    for i in range(n):
        c = text[i]
        comparisons += 1
        while j and pattern[j] != c:
            j = lps[j - 1]
            fallbacks += 1
            comparisons += 1
        if pattern[j] == c:
            if j == last:
                positions.append(i - last)
                j = lps[last]
            else:
                j += 1
    return MatchResult(positions, comparisons, {"fallbacks": fallbacks})


def brute_force_search(text: Sequence, pattern: Sequence) -> MatchResult:
    """Baseline: try every alignment, compare left to right. O(nm)."""
    n, m = len(text), len(pattern)
    if m == 0:
        return MatchResult(list(range(n + 1)), 0)
    positions = []
    comparisons = 0
    for s in range(n - m + 1):
        j = 0
        while j < m:
            comparisons += 1
            if text[s + j] != pattern[j]:
                break
            j += 1
        if j == m:
            positions.append(s)
    return MatchResult(positions, comparisons)


def kmp_trace(text: Sequence, pattern: Sequence) -> List[Dict]:
    """
    Every step of the KMP scan: {"i", "j", "char", "action", "next_j"}
    where action is "match", "found", "fallback" (j ← lps[j−1]) or
    "advance" (mismatch with j = 0). Meant for short inputs.
    """
    m = len(pattern)
    if m == 0:
        return []
    lps = prefix_function(pattern)
    steps = []
    j = 0
    for i, c in enumerate(text):
        while j and pattern[j] != c:
            steps.append({"i": i, "j": j, "char": c, "action": "fallback",
                          "next_j": lps[j - 1]})
            j = lps[j - 1]
        if pattern[j] == c:
            if j == m - 1:
                steps.append({"i": i, "j": j, "char": c, "action": "found",
                              "next_j": lps[m - 1], "start": i - m + 1})
                j = lps[m - 1]
            else:
                steps.append({"i": i, "j": j, "char": c, "action": "match",
                              "next_j": j + 1})
                j += 1
        else:
            steps.append({"i": i, "j": j, "char": c, "action": "advance",
                          "next_j": 0})
    return steps


def format_trace(text: str, pattern: str, limit: int = 40) -> str:
    """Human-readable alignment view of the first `limit` trace steps."""
    lines = []
    for step in kmp_trace(text, pattern)[:limit]:
        i, j = step["i"], step["j"]
        align = " " * (i - j) + pattern
        if step["action"] == "fallback":
            note = f"mismatch '{step['char']}' vs '{pattern[j]}': j {j} -> lps[{j - 1}] = {step['next_j']}"
        elif step["action"] == "advance":
            note = f"mismatch '{step['char']}' at j = 0: move to i = {i + 1}"
        elif step["action"] == "found":
            note = f"MATCH at {step['start']}; continue with j = {step['next_j']}"
        else:
            note = "match"
        lines.append(f"i={i:>3} j={j:>2}  {text}\n{'':12}{align}   {note}")
    return "\n".join(lines)
