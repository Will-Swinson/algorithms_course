"""
Rabin–Karp: pattern matching by rolling fingerprints.

Hash the pattern once, then slide a window over the text, updating the
window's hash in O(1) per step (utils/rolling_hash.py). Only windows
whose hash equals the pattern's are compared character by character.

    expected time  O(n + m + (occ + spurious)·m)
    worst case     O(nm)   — every window a hash hit, e.g. T = a^n,
                             P = a^m (every window is a real match) or
                             a deliberately bad modulus/base

Correctness modes:
    verify=True   (Las Vegas)  hash hits are checked, so the answer is
                  always exact; collisions cost time only. Every hit
                  that fails the check is counted as `spurious`.
    verify=False  (Monte Carlo) hash hits are reported as matches; a
                  collision becomes a false positive. With a random
                  base and prime q, each window is a false positive
                  with probability ≤ (m − 1)/q, so the expected count
                  over a scan is ≤ n(m − 1)/q — about 10⁻¹² per
                  megabyte for q = 2^61 − 1.

Multiple patterns. Hash every pattern; patterns of the same length
share one rolling pass and a dict lookup per window. P patterns of k
distinct lengths cost O(k·n + Σ|P|) expected, instead of P full scans
— the batch case where Rabin–Karp beats KMP (one automaton per
pattern) by roughly P/k.
"""
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from src.strings.core.kmp import MatchResult
from src.strings.utils.rolling_hash import DEFAULT_MODULUS, RollingHash, as_codes


def _hasher(base: Optional[int], modulus: int, seed: Optional[int]) -> RollingHash:
    return RollingHash(base=base, modulus=modulus,
                       rng=random.Random(seed) if seed is not None else None)


def rabin_karp(text, pattern, base: Optional[int] = None,
               modulus: int = DEFAULT_MODULUS, verify: bool = True,
               seed: Optional[int] = None,
               hasher: Optional[RollingHash] = None) -> MatchResult:
    """
    All occurrences of `pattern` in `text`.

    stats: hash_hits (windows whose hash matched), spurious (hits that
    were not real matches — collisions), false_positives (verify=False:
    reported positions that are not real matches; computed afterwards
    for measurement only), windows (windows hashed).
    """
    hasher = hasher or _hasher(base, modulus, seed)
    t, p = as_codes(text), as_codes(pattern)
    n, m = len(t), len(p)
    if m == 0:
        return MatchResult(list(range(n + 1)), 0, {"hash_hits": 0, "spurious": 0,
                                                   "windows": 0})
    if m > n:
        return MatchResult([], 0, {"hash_hits": 0, "spurious": 0, "windows": 0})
    b, q = hasher.base, hasher.modulus
    target = hasher.hash(p)
    high = hasher.power(m)
    p_bytes = bytes(p) if isinstance(t, bytes) and isinstance(p, bytes) else None
    h = 0
    for x in t[:m]:
        h = (h * b + x) % q
    positions = []
    hits = spurious = comparisons = 0
    last = n - m
    i = 0
    while True:
        if h == target:
            hits += 1
            if verify:
                comparisons += m
                window = t[i:i + m]
                if (window == p_bytes) if p_bytes is not None else \
                        list(window) == list(p):
                    positions.append(i)
                else:
                    spurious += 1
            else:
                positions.append(i)
        if i == last:
            break
        h = ((h - t[i] * high) * b + t[i + m]) % q
        i += 1
    stats = {"hash_hits": hits, "spurious": spurious, "windows": n - m + 1}
    if not verify:
        stats["false_positives"] = sum(1 for s in positions
                                       if list(t[s:s + m]) != list(p))
    return MatchResult(positions, comparisons, stats)


def rabin_karp_multi(text, patterns: Iterable, base: Optional[int] = None,
                     modulus: int = DEFAULT_MODULUS, verify: bool = True,
                     seed: Optional[int] = None,
                     hasher: Optional[RollingHash] = None) -> Dict[str, MatchResult]:
    """
    Search many patterns in one rolling pass per distinct length.
    Returns {pattern: MatchResult}; duplicate patterns share a result.
    """
    hasher = hasher or _hasher(base, modulus, seed)
    t = as_codes(text)
    n = len(t)
    b, q = hasher.base, hasher.modulus
    patterns = list(dict.fromkeys(patterns))
    results = {p: MatchResult([], 0, {"hash_hits": 0, "spurious": 0})
               for p in patterns}
    by_length: Dict[int, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
    for p in patterns:
        if len(p) == 0:
            results[p] = MatchResult(list(range(n + 1)), 0, {"hash_hits": 0,
                                                             "spurious": 0})
        elif len(p) <= n:
            by_length[len(p)][hasher.hash(p)].append(p)
    for m, table in by_length.items():
        coded = {pat: as_codes(pat) for pats in table.values() for pat in pats}
        high = hasher.power(m)
        h = 0
        for x in t[:m]:
            h = (h * b + x) % q
        last = n - m
        i = 0
        get = table.get
        while True:
            bucket = get(h)
            if bucket is not None:
                window = list(t[i:i + m]) if verify else None
                for pat in bucket:
                    r = results[pat]
                    r.stats["hash_hits"] += 1
                    if not verify:
                        r.positions.append(i)
                        continue
                    r.comparisons += m
                    if window == list(coded[pat]):
                        r.positions.append(i)
                    else:
                        r.stats["spurious"] += 1
            if i == last:
                break
            h = ((h - t[i] * high) * b + t[i + m]) % q
            i += 1
    return results


def expected_spurious_hits(n: int, m: int, modulus: int) -> float:
    """Expected collisions in a scan if window hashes behave like
    uniform values mod q: (n − m + 1)/q (≤ n(m−1)/q is the worst-case
    bound for a random base)."""
    return max(n - m + 1, 0) / modulus
