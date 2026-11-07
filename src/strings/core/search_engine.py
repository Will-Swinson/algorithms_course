"""
StringSearchEngine: one interface over four matchers.

    engine = StringSearchEngine(text)
    engine.find("pattern", method="kmp")    # scan, O(n + m)
    engine.find("pattern", method="rk")     # scan, rolling hash
    engine.find("pattern", method="sa")     # index, O(m log n + occ)
    engine.find("pattern", method="tree")   # index, O(m + occ)

Scanning methods pay per query and nothing up front; indexing methods
pay once (build) and then answer each query in time that hardly
depends on n. The engine builds an index lazily on its first use,
times the build ("preprocessing"), and keeps per-method statistics of
every query, so it can answer the practical question: for the number
of queries you expect, which method is cheapest overall?

    total(method, q) = preprocessing(method) + q · mean_search(method)

recommend(q) compares those totals (measuring unmeasured methods on
a few probe patterns first) and returns the cheapest method with the
break-even query counts behind the choice — e.g. "SA is fastest for
repeated queries: its 0.6 s build pays for itself after 15 queries".

The suffix tree is capped at tree_max_chars (default 2 M characters)
because a Python suffix tree needs ~270 bytes per character.
"""
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from src.strings.core.kmp import kmp_search, prefix_function
from src.strings.core.rabin_karp import rabin_karp, rabin_karp_multi
from src.strings.core.suffix_array import SuffixArray
from src.strings.core.suffix_tree import SuffixTree
from src.strings.utils.rolling_hash import RollingHash

METHODS = ("kmp", "rk", "sa", "tree")
INDEXED = ("sa", "tree")
NAMES = {"kmp": "KMP", "rk": "Rabin-Karp", "sa": "suffix array",
         "tree": "suffix tree"}


@dataclass
class SearchResult:
    pattern: str
    method: str
    positions: List[int]
    search_time: float               # seconds for this query
    preprocess_time: float           # one-off build time of the method's index
                                     # (pattern preprocessing for KMP/RK)

    @property
    def count(self) -> int:
        return len(self.positions)


@dataclass
class MethodStats:
    method: str
    queries: int = 0
    total_search_time: float = 0.0
    preprocess_time: float = 0.0     # index build (0 for scans)
    pattern_preprocess_time: float = 0.0
    index_memory: Optional[int] = None   # bytes, if measured

    @property
    def mean_search_time(self) -> float:
        return self.total_search_time / self.queries if self.queries else float("nan")


@dataclass
class Recommendation:
    method: str
    expected_queries: int
    reason: str
    totals: Dict[str, float] = field(default_factory=dict)
    break_even: Dict[str, Optional[float]] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.reason


class StringSearchEngine:

    def __init__(self, text: str, tree_max_chars: int = 2_000_000,
                 seed: int = 0, measure_memory: bool = False):
        self.text = text
        self.tree_max_chars = tree_max_chars
        self.measure_memory = measure_memory
        self.hasher = RollingHash(rng=random.Random(seed))
        self._sa: Optional[SuffixArray] = None
        self._tree: Optional[SuffixTree] = None
        self.stats: Dict[str, MethodStats] = {m: MethodStats(m) for m in METHODS}

    # ------------------------------------------------------------------
    # indexes
    # ------------------------------------------------------------------
    def _build(self, method: str):
        if method == "tree" and len(self.text) > self.tree_max_chars:
            raise ValueError(f"text has {len(self.text):,} characters; the suffix "
                             f"tree is limited to {self.tree_max_chars:,} "
                             "(raise tree_max_chars to override)")
        build = (lambda: SuffixArray(self.text, lcp=False)) if method == "sa" \
            else (lambda: SuffixTree(self.text))
        if self.measure_memory:
            tracemalloc.start()
        start = time.perf_counter()
        index = build()
        elapsed = time.perf_counter() - start
        if self.measure_memory:
            self.stats[method].index_memory = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        self.stats[method].preprocess_time = elapsed
        return index

    def preprocess(self, method: str) -> float:
        """Build the index for `method` now (no-op for scans); returns
        its build time."""
        self._check(method)
        if method == "sa" and self._sa is None:
            self._sa = self._build("sa")
        elif method == "tree" and self._tree is None:
            self._tree = self._build("tree")
        return self.stats[method].preprocess_time

    @property
    def suffix_array(self) -> SuffixArray:
        self.preprocess("sa")
        return self._sa

    @property
    def suffix_tree(self) -> SuffixTree:
        self.preprocess("tree")
        return self._tree

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    @staticmethod
    def _check(method: str) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    def find(self, pattern: str, method: str = "kmp") -> SearchResult:
        """All occurrence positions of `pattern` with the given method."""
        self._check(method)
        self.preprocess(method)
        st = self.stats[method]
        start = time.perf_counter()
        pattern_prep = 0.0
        if method == "kmp":
            lps = prefix_function(pattern)
            pattern_prep = time.perf_counter() - start
            positions = kmp_search(self.text, pattern, lps).positions
        elif method == "rk":
            positions = rabin_karp(self.text, pattern, hasher=self.hasher).positions
        elif method == "sa":
            positions = self._sa.find(pattern)
        else:
            positions = self._tree.find(pattern)
        elapsed = time.perf_counter() - start
        st.queries += 1
        st.total_search_time += elapsed
        st.pattern_preprocess_time += pattern_prep
        return SearchResult(pattern, method, positions, elapsed,
                            st.preprocess_time if method in INDEXED else pattern_prep)

    def count(self, pattern: str, method: str = "sa") -> int:
        """Number of occurrences; the indexes count without listing."""
        self._check(method)
        if method == "sa":
            return self.suffix_array.count(pattern)
        if method == "tree":
            return self.suffix_tree.count(pattern)
        return self.find(pattern, method).count

    def find_many(self, patterns: Iterable[str], method: str = "rk") -> Dict[str, List[int]]:
        """Batch search; Rabin–Karp shares one pass per pattern length."""
        patterns = list(dict.fromkeys(patterns))
        if method == "rk":
            start = time.perf_counter()
            results = rabin_karp_multi(self.text, patterns, hasher=self.hasher)
            st = self.stats["rk"]
            st.queries += len(patterns)
            st.total_search_time += time.perf_counter() - start
            return {p: r.positions for p, r in results.items()}
        return {p: self.find(p, method).positions for p in patterns}

    def compare(self, pattern: str, methods: Sequence[str] = METHODS) -> Dict[str, SearchResult]:
        """Run one query with every method and check they agree."""
        results = {m: self.find(pattern, m) for m in methods
                   if m != "tree" or len(self.text) <= self.tree_max_chars}
        answers = {tuple(r.positions) for r in results.values()}
        if len(answers) > 1:
            raise AssertionError(f"methods disagree on {pattern!r}")
        return results

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def summary(self) -> List[Dict]:
        rows = []
        for m in METHODS:
            st = self.stats[m]
            rows.append({"method": m, "queries": st.queries,
                         "preprocess_s": st.preprocess_time,
                         "mean_search_s": st.mean_search_time,
                         "index_bytes": st.index_memory})
        return rows

    def _probe(self, methods: Sequence[str], probes: int) -> None:
        n = len(self.text)
        width = max(1, min(12, n))
        step = max(1, (n - width) // max(probes, 1))
        samples = [self.text[i:i + width] for i in range(0, max(n - width, 0) + 1, step)][:probes]
        for m in methods:
            for p in samples or [self.text]:
                self.find(p, m)

    def recommend(self, expected_queries: int = 1, probes: int = 3,
                  methods: Optional[Sequence[str]] = None) -> Recommendation:
        """
        Cheapest method for `expected_queries` queries, from measured
        build and query times (methods without measurements are probed
        with a few patterns cut from the text).
        """
        if expected_queries < 1:
            raise ValueError("expected_queries must be >= 1")
        methods = list(methods or METHODS)
        if "tree" in methods and len(self.text) > self.tree_max_chars:
            methods.remove("tree")
        self._probe([m for m in methods if self.stats[m].queries == 0], probes)
        totals = {m: self.stats[m].preprocess_time
                  + expected_queries * self.stats[m].mean_search_time
                  for m in methods}
        best = min(totals, key=totals.get)
        scans = [m for m in methods if m not in INDEXED]
        fastest_scan = min(scans, key=lambda m: self.stats[m].mean_search_time) \
            if scans else None
        break_even: Dict[str, Optional[float]] = {}
        for m in methods:
            if m in INDEXED and fastest_scan:
                gain = (self.stats[fastest_scan].mean_search_time
                        - self.stats[m].mean_search_time)
                break_even[m] = self.stats[m].preprocess_time / gain if gain > 0 else None
        best_st = self.stats[best]
        if best in INDEXED:
            q_star = break_even.get(best)
            reason = (f"{NAMES[best]} is fastest for repeated queries: its "
                      f"{best_st.preprocess_time:.3g} s build pays for itself after "
                      f"~{q_star:.0f} queries ({best_st.mean_search_time * 1e3:.3g} ms "
                      f"per query vs {self.stats[fastest_scan].mean_search_time * 1e3:.3g}"
                      f" ms for {NAMES[fastest_scan]})")
        else:
            others = [m for m in methods if m in INDEXED and break_even.get(m)]
            if others:
                q_star = min(break_even[m] for m in others)
                reason = (f"{NAMES[best]} is cheapest for {expected_queries} "
                          f"quer{'y' if expected_queries == 1 else 'ies'}: no index "
                          f"to build; an index would pay off only after "
                          f"~{q_star:.0f} queries")
            else:
                reason = (f"{NAMES[best]} is cheapest: it needs no preprocessing "
                          "and the indexes do not answer faster on this text")
        return Recommendation(best, expected_queries, reason, totals, break_even)
