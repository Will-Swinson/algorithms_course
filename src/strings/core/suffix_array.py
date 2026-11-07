"""
Suffix array + LCP array: a sorted index of every suffix.

SA[r] is the start of the r-th smallest suffix of T; LCP[r] is the
length of the longest common prefix of suffixes SA[r − 1] and SA[r]
(LCP[0] = 0). For T = "banana":

    r  SA  LCP  suffix
    0   5   0   a
    1   3   1   ana
    2   1   3   anana
    3   0   0   banana
    4   4   0   na
    5   2   2   nana

Construction (build_suffix_array, method=...):
    "doubling"  prefix doubling, numpy-vectorized. Rank suffixes by
                their first k characters, then by their first 2k using
                the pair (rank[i], rank[i + k]), packed into one int64
                key and sorted with np.argsort. log₂(longest repeat)
                rounds of O(n log n) — fast because each round runs in C.
    "python"    the same doubling with Python's sort on tuple keys:
                O(n log² n), shown for comparison.
    "naive"     sort the suffixes themselves: O(n² log n) worst case
                and O(n²) memory traffic; small inputs only.
(Kärkkäinen–Sanders DC3 is O(n) but its constant factors in Python
lose to the numpy doubling at every size measured, so it is omitted.)

LCP: Kasai et al. — walk suffixes in text order; the LCP drops by at
most 1 from one text position to the next, so the total work is O(n).

Search: all suffixes beginning with P form one contiguous block of SA.
Two binary searches find its ends, each comparing P with a suffix
prefix in O(m): O(m log n) per query, and counting needs no scan of the
occurrences at all.

Derived queries: longest repeated substring = max LCP; number of
distinct substrings = n(n+1)/2 − ΣLCP; longest common substring of
two strings = max LCP between adjacent suffixes from different
strings in the suffix array of A + sep + B.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

METHODS = ("doubling", "python", "naive")


def _codes(text) -> np.ndarray:
    if isinstance(text, str):
        if text.isascii():
            return np.frombuffer(text.encode("ascii"), dtype=np.uint8).astype(np.int64)
        return np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    return np.asarray(text, dtype=np.int64)


def suffix_array_doubling(text) -> Tuple[np.ndarray, int]:
    """(SA as int64 array, number of doubling rounds)."""
    s = _codes(text)
    n = len(s)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    _, rank = np.unique(s, return_inverse=True)
    rank = rank.astype(np.int64)
    k, rounds = 1, 0
    order = np.argsort(rank, kind="stable")
    if n == 1:
        return order, 0
    while True:
        rounds += 1
        # pack (rank[i], rank[i + k] + 1) into one int64 key: one sort
        # of a single key is ~2.5x faster than np.lexsort on two
        second = np.zeros(n, dtype=np.int64)
        if k < n:
            second[:n - k] = rank[k:] + 1
        key = (rank << 32) | second
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[order] = np.concatenate(
            ([0], np.cumsum(sorted_key[1:] != sorted_key[:-1])))
        rank = new_rank
        if rank[order[-1]] == n - 1 or k >= n:
            return order, rounds
        k *= 2


def suffix_array_python(text) -> List[int]:
    """Prefix doubling with Python sorts (O(n log² n))."""
    n = len(text)
    rank = [ord(c) for c in text] if isinstance(text, str) else list(text)
    sa = list(range(n))
    k = 1
    while n > 1:
        key = lambda i: (rank[i], rank[i + k] if i + k < n else -1)  # noqa: E731
        sa.sort(key=key)
        new = [0] * n
        for r in range(1, n):
            new[sa[r]] = new[sa[r - 1]] + (key(sa[r]) != key(sa[r - 1]))
        rank = new
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


def suffix_array_naive(text) -> List[int]:
    """Sort the suffixes directly (small inputs)."""
    return sorted(range(len(text)), key=lambda i: text[i:])


def kasai_lcp(text, sa: Sequence[int]) -> List[int]:
    """LCP[r] = lcp(suffix SA[r−1], suffix SA[r]); LCP[0] = 0. O(n)."""
    n = len(text)
    sa = [int(x) for x in sa]
    rank = [0] * n
    for r, i in enumerate(sa):
        rank[i] = r
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            h = 0
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and text[i + h] == text[j + h]:
            h += 1
        lcp[r] = h
        if h:
            h -= 1
    return lcp


def build_suffix_array(text, method: str = "doubling") -> np.ndarray:
    if method == "doubling":
        return suffix_array_doubling(text)[0]
    if method == "python":
        return np.asarray(suffix_array_python(text), dtype=np.int64)
    if method == "naive":
        return np.asarray(suffix_array_naive(text), dtype=np.int64)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


@dataclass
class SuffixArrayStats:
    n: int
    rounds: int
    mean_lcp: float
    max_lcp: int
    distinct_substrings: int


class SuffixArray:
    """
    Suffix array index over `text` with an optional LCP array.

        sa = SuffixArray("banana")
        sa.find("ana")      -> [1, 3]
        sa.count("an")      -> 2      (O(m log n), no scan)
    """

    def __init__(self, text: str, method: str = "doubling", lcp: bool = True):
        self.text = text
        self.method = method
        if method == "doubling":
            self.sa, self.rounds = suffix_array_doubling(text)
        else:
            self.sa, self.rounds = build_suffix_array(text, method), 0
        self.lcp: Optional[List[int]] = kasai_lcp(text, self.sa) if lcp else None

    def __len__(self) -> int:
        return len(self.text)

    def ensure_lcp(self) -> List[int]:
        if self.lcp is None:
            self.lcp = kasai_lcp(self.text, self.sa)
        return self.lcp

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def range(self, pattern: str) -> Tuple[int, int]:
        """[lo, hi) block of SA whose suffixes start with `pattern`."""
        text, sa, m = self.text, self.sa, len(pattern)
        n = len(sa)
        lo, hi = 0, n
        while lo < hi:                      # first suffix >= pattern
            mid = (lo + hi) // 2
            start = sa[mid]
            if text[start:start + m] < pattern:
                lo = mid + 1
            else:
                hi = mid
        first = lo
        hi = n
        while lo < hi:                      # first suffix whose prefix > pattern
            mid = (lo + hi) // 2
            start = sa[mid]
            if text[start:start + m] <= pattern:
                lo = mid + 1
            else:
                hi = mid
        return first, lo

    def count(self, pattern: str) -> int:
        if not pattern:
            return len(self.text) + 1
        lo, hi = self.range(pattern)
        return hi - lo

    def find(self, pattern: str) -> List[int]:
        """Sorted start positions of every occurrence."""
        if not pattern:
            return list(range(len(self.text) + 1))
        lo, hi = self.range(pattern)
        return np.sort(self.sa[lo:hi]).tolist()

    def contains(self, pattern: str) -> bool:
        return self.count(pattern) > 0

    # ------------------------------------------------------------------
    # LCP-based queries
    # ------------------------------------------------------------------
    def longest_repeated_substring(self) -> str:
        lcp = self.ensure_lcp()
        if not lcp:
            return ""
        r = max(range(len(lcp)), key=lcp.__getitem__)
        start = int(self.sa[r])
        return self.text[start:start + lcp[r]]

    def distinct_substrings(self) -> int:
        n = len(self.text)
        return n * (n + 1) // 2 - sum(self.ensure_lcp())

    def stats(self) -> SuffixArrayStats:
        lcp = self.ensure_lcp()
        n = len(self.text)
        return SuffixArrayStats(n, self.rounds,
                                sum(lcp) / n if n else 0.0,
                                max(lcp, default=0),
                                self.distinct_substrings())

    def suffix(self, rank: int, width: Optional[int] = None) -> str:
        start = int(self.sa[rank])
        end = len(self.text) if width is None else start + width
        return self.text[start:end]

    def table(self, width: int = 20) -> List[Tuple[int, int, int, str]]:
        """(rank, SA, LCP, suffix prefix) rows, for display."""
        lcp = self.ensure_lcp()
        return [(r, int(self.sa[r]), lcp[r], self.suffix(r, width))
                for r in range(len(self.sa))]


def longest_common_substring(a: str, b: str, separator: str = "\x01") -> Tuple[str, int, int]:
    """
    (substring, start in a, start in b) of a longest common substring,
    via the suffix array of a + separator + b: the answer is the
    largest LCP between neighbouring suffixes that start in different
    strings. The separator must occur in neither string.
    """
    if separator in a or separator in b:
        raise ValueError("separator occurs in the input")
    joined = a + separator + b
    sa = SuffixArray(joined)
    lcp = sa.lcp
    cut = len(a)
    best, where = 0, (0, 0)
    for r in range(1, len(joined)):
        i, j = int(sa.sa[r - 1]), int(sa.sa[r])
        if (i < cut) != (j < cut) and lcp[r] > best:
            best = lcp[r]
            where = (i, j) if i < cut else (j, i)
    if best == 0:
        return "", -1, -1
    ia, jb = where
    return a[ia:ia + best], ia, jb - cut - 1
