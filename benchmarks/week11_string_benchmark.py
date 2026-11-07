"""
Week 11 benchmark: string search with KMP, Rabin–Karp, suffix arrays
and suffix trees.

Datasets are generated (seeded): English-like text (Zipf words), DNA
(with repeats) and, for LCP statistics, server logs and uniform random
text. Sizes: small (10 KB), 100 KB, medium (1 MB and 5 MB). The
optional 100 MB tier is left out: pure-Python scans of 100 MB take
minutes per query and the suffix tree would need ~27 GB.

Every query's answer is cross-checked across all methods (and
Python's str.find, the C reference); the `valid` column records it.

1. Search latency and preprocessing per method and dataset size; the
   cumulative cost of q queries (when does an index pay off?);
   latency vs pattern length; the worst case a^n vs a^(m−1)b.
2. Memory: tracemalloc peak while building each index.
3. Rabin–Karp collisions: spurious hits vs modulus against the
   (n − m + 1)/q prediction, Monte Carlo false positives, bad
   base/modulus choices, and batch (multi-pattern) matching.
4. Suffix array + LCP: construction time (numpy doubling, Python
   doubling, naive sort, Kasai LCP, Ukkonen tree), doubling rounds,
   LCP distributions per text type.
5. Applications: plagiarism score vs copied fraction, DNA exact and
   approximate search, autocomplete top-k latency.

Outputs (benchmarks/results/):
    runtime_comparison.png      latency, preprocessing, crossover,
                                pattern length, worst case
    collision_analysis.png      RK collisions and batch matching
    sa_lcp_stats.png            SA/LCP construction and statistics
    heatmap_dataset_sizes.png   method × dataset-size heatmaps
    string_applications.png     plagiarism, DNA, autocomplete
    suffix_array_example.png    SA + LCP of "mississippi"
    suffix_tree_example.png     suffix tree of "mississippi" (+ .dot)
    ukkonen_steps.png           Ukkonen phases on "abcabxabcd"
    kmp_trace.png               KMP fallbacks on a short example
    string_comparison_table.csv one row per configuration

Usage:
    python benchmarks/week11_string_benchmark.py          # ~4 min
    python benchmarks/week11_string_benchmark.py --quick  # ~40 s
"""
import argparse
import csv
import gc
import os
import random
import statistics
import sys
import time
import tracemalloc
from typing import Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.strings.applications import (
    Autocomplete, GenomeIndex, SortedPrefixIndex, compare_documents,
)
from src.strings.applications.dna_search import brute_force_mutations
from src.strings.core import (
    SuffixArray, SuffixTree, brute_force_search, kasai_lcp, kmp_search,
    prefix_function, rabin_karp, rabin_karp_multi,
)
from src.strings.core.suffix_array import (
    suffix_array_doubling, suffix_array_naive, suffix_array_python,
)
from src.strings.utils.generators import (
    dna_sequence, english_like_text, log_lines, mutate, random_text,
    sample_patterns,
)
from src.strings.utils.rolling_hash import MERSENNE_61, RollingHash
from src.strings.visualization import (
    draw_suffix_tree, draw_ukkonen_steps, plot_kmp_trace, plot_suffix_array,
    write_dot,
)
from src.utils.visualization import plot_mixed_panels

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_NAME = "string_comparison_table.csv"
SEED = 11

FIELDS = ["experiment", "dataset", "method", "n", "pattern_length",
          "parameter", "queries", "preprocess_s", "mean_search_s",
          "std_search_s", "memory_mb", "value", "value2", "value3", "valid",
          "note"]

METHODS = ["naive", "kmp", "rk", "sa", "tree", "builtin"]
LABELS = {"naive": "brute force", "kmp": "KMP", "rk": "Rabin–Karp",
          "sa": "suffix array", "tree": "suffix tree",
          "builtin": "str.find (C reference)"}
COLORS = {"naive": "#7f7f7f", "kmp": "#1f77b4", "rk": "#ff7f0e",
          "sa": "#2ca02c", "tree": "#9467bd", "builtin": "#8c564b"}

SIZES = [10_000, 100_000, 1_000_000, 5_000_000]
SIZE_NAMES = {10_000: "10 KB", 100_000: "100 KB", 1_000_000: "1 MB",
              5_000_000: "5 MB"}
DATASETS = {"english": english_like_text, "dna": dna_sequence}
LIMITS = {"naive": 1_000_000, "tree": 1_000_000}
QUERIES = 20
PATTERN_LENGTH = 12
MODULI = [7, 31, 101, 1_009, 10_007, 100_003, 1_000_003, 1_000_000_007, MERSENNE_61]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def make_row(**kw) -> Dict:
    row = {f: "" for f in FIELDS}
    row.update(kw)
    return row


def dataset(kind: str, n: int, seed: int = SEED) -> str:
    if kind == "dna":
        return dna_sequence(n, repeat_fraction=0.1, seed=seed)
    if kind == "logs":
        return log_lines(n, seed=seed)
    if kind == "random":
        return random_text(n, seed=seed)
    return english_like_text(n, seed=seed)


def builtin_find(text: str, pattern: str) -> List[int]:
    out, i = [], text.find(pattern)
    while i >= 0:
        out.append(i)
        i = text.find(pattern, i + 1)
    return out


def peak(func: Callable[[], object]):
    """(seconds, peak traced bytes, result) of one call under tracemalloc
    (the time is inflated by tracing; use it for memory only)."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = func()
        elapsed = time.perf_counter() - start
        _, top = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return elapsed, top, result


def mean_std(xs: Sequence[float]):
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


class Searchers:
    """The six methods over one text, indexes built on demand (timed)."""

    def __init__(self, text: str, seed: int = SEED):
        self.text = text
        self.hasher = RollingHash(rng=random.Random(seed))
        self.sa: Optional[SuffixArray] = None
        self.tree: Optional[SuffixTree] = None
        self.prep: Dict[str, float] = {}
        self.memory: Dict[str, float] = {}

    def build(self, method: str, measure_memory: bool = False) -> None:
        if method == "sa" and self.sa is None:
            make = lambda: SuffixArray(self.text, lcp=False)  # noqa: E731
        elif method == "tree" and self.tree is None:
            make = lambda: SuffixTree(self.text)  # noqa: E731
        else:
            return
        gc.collect()
        start = time.perf_counter()
        index = make()
        elapsed = time.perf_counter() - start
        if measure_memory:
            # a second build under tracemalloc: tracing slows Python-heavy
            # builds (the suffix tree) ~6x, so it must not be the timed one
            _, top, _ = peak(make)
            self.memory[method] = top / 2 ** 20
        setattr(self, method, index)
        self.prep[method] = elapsed

    def search(self, method: str, pattern: str) -> List[int]:
        if method == "naive":
            return brute_force_search(self.text, pattern).positions
        if method == "kmp":
            return kmp_search(self.text, pattern).positions
        if method == "rk":
            return rabin_karp(self.text, pattern, hasher=self.hasher).positions
        if method == "sa":
            return self.sa.find(pattern)
        if method == "tree":
            return self.tree.find(pattern)
        return builtin_find(self.text, pattern)


def time_queries(searchers: Searchers, method: str, patterns: Sequence[str],
                 reference: Dict[str, List[int]]):
    """Per-query times (fast index queries are repeated and averaged).
    The cyclic garbage collector is paused while timing, as timeit does:
    otherwise a collection that walks the suffix tree's ~10^6 node
    dicts can land inside a microsecond query."""
    times, ok = [], True
    gc.collect()
    gc.disable()
    try:
        return _time_queries(searchers, method, patterns, reference)
    finally:
        gc.enable()


def _time_queries(searchers, method, patterns, reference):
    times, ok = [], True
    for p in patterns:
        start = time.perf_counter()
        result = searchers.search(method, p)
        elapsed = time.perf_counter() - start
        if elapsed < 1e-3:                          # average tiny timings
            reps = max(3, int(2e-3 / max(elapsed, 1e-7)))
            start = time.perf_counter()
            for _ in range(min(reps, 200)):
                searchers.search(method, p)
            elapsed = (time.perf_counter() - start) / min(reps, 200)
        times.append(elapsed)
        ok &= result == reference[p]
    return times, ok


# ----------------------------------------------------------------------
# 1 + 2. latency, preprocessing, memory
# ----------------------------------------------------------------------
def bench_search(sizes: Sequence[int] = SIZES, kinds: Sequence[str] = ("english", "dna"),
                 queries: int = QUERIES, limits: Optional[Dict[str, int]] = None,
                 samples: Optional[Dict] = None) -> List[Dict]:
    limits = LIMITS if limits is None else limits
    rows = []
    for kind in kinds:
        for n in sizes:
            text = dataset(kind, n)
            patterns = sample_patterns(text, queries, PATTERN_LENGTH, seed=SEED + n)
            reference = {p: builtin_find(text, p) for p in patterns}
            s = Searchers(text)
            for method in METHODS:
                if n > limits.get(method, float("inf")):
                    continue
                if method in ("sa", "tree"):
                    s.build(method, measure_memory=n <= 1_000_000)
                times, ok = time_queries(s, method, patterns, reference)
                mean, std = mean_std(times)
                if method == "kmp":
                    prep = statistics.mean(
                        _timeit(lambda p=p: prefix_function(p)) for p in patterns)
                elif method == "rk":
                    prep = statistics.mean(
                        _timeit(lambda p=p: s.hasher.hash(p)) for p in patterns)
                else:
                    prep = s.prep.get(method, 0.0)
                row = make_row(experiment="search", dataset=kind, method=method,
                               n=n, pattern_length=PATTERN_LENGTH,
                               queries=len(patterns), preprocess_s=prep,
                               mean_search_s=mean, std_search_s=std,
                               memory_mb=s.memory.get(method, ""),
                               value=statistics.mean(len(reference[p]) for p in patterns),
                               valid=ok, note="value = mean matches per query")
                rows.append(row)
                print(f"  {kind:7s} {SIZE_NAMES.get(n, n):>6} {method:8s} "
                      f"search {mean * 1e3:9.3f} ms  prep {prep:8.4f} s  "
                      f"mem {row['memory_mb'] and format(row['memory_mb'], '.1f')} "
                      f"valid={ok}")
            if samples is not None and n == 1_000_000 and kind == "english":
                samples["crossover_text"] = (kind, n)
    return rows


def _timeit(func) -> float:
    start = time.perf_counter()
    for _ in range(50):
        func()
    return (time.perf_counter() - start) / 50


def bench_pattern_length(n: int = 1_000_000, lengths=(4, 8, 16, 32, 64, 128),
                         queries: int = 10) -> List[Dict]:
    text = dataset("english", n)
    s = Searchers(text)
    s.build("sa")
    if n <= LIMITS["tree"]:
        s.build("tree")
    rows = []
    for m in lengths:
        patterns = sample_patterns(text, queries, m, seed=SEED + m)
        reference = {p: builtin_find(text, p) for p in patterns}
        for method in ("kmp", "rk", "sa", "tree", "builtin"):
            if method == "tree" and s.tree is None:
                continue
            times, ok = time_queries(s, method, patterns, reference)
            mean, std = mean_std(times)
            rows.append(make_row(experiment="pattern_length", dataset="english",
                                 method=method, n=n, pattern_length=m,
                                 queries=queries, mean_search_s=mean,
                                 std_search_s=std, valid=ok))
        print(f"  length {m:4d}: " + ", ".join(
            f"{r['method']} {r['mean_search_s'] * 1e3:.3f} ms" for r in rows[-5:]))
    return rows


def bench_worst_case(n: int = 100_000, lengths=(2, 4, 8, 16, 32, 64)) -> List[Dict]:
    """T = a^n, P = a^(m−1)b: no match, but brute force compares ~m
    characters at every alignment."""
    text = "a" * n
    rows = []
    for m in lengths:
        pattern = "a" * (m - 1) + "b"
        for method, run in (("naive", lambda: brute_force_search(text, pattern)),
                            ("kmp", lambda: kmp_search(text, pattern)),
                            ("rk", lambda: rabin_karp(text, pattern, seed=1))):
            start = time.perf_counter()
            r = run()
            elapsed = time.perf_counter() - start
            rows.append(make_row(experiment="worst_case", dataset="a^n",
                                 method=method, n=n, pattern_length=m,
                                 queries=1, mean_search_s=elapsed,
                                 value=r.comparisons, valid=r.positions == [],
                                 note="value = character comparisons"))
        print(f"  worst case m={m:3d}: " + ", ".join(
            f"{r['method']} {r['value']:,} cmp" for r in rows[-3:]))
    # RK's own worst case: every window is a hash hit (P = a^m, all real matches)
    for m in lengths:
        r = rabin_karp(text, "a" * m, seed=1)
        rows.append(make_row(experiment="worst_case", dataset="a^n", method="rk",
                             n=n, pattern_length=m, parameter="all windows match",
                             queries=1, value=r.comparisons,
                             valid=len(r.positions) == n - m + 1,
                             note="value = character comparisons"))
    return rows


# ----------------------------------------------------------------------
# 3. Rabin–Karp collisions and batch matching
# ----------------------------------------------------------------------
def bench_collisions(n: int = 1_000_000, moduli: Sequence[int] = MODULI,
                     patterns: int = 5, length: int = PATTERN_LENGTH) -> List[Dict]:
    text = dataset("english", n)
    absent = [p for p in sample_patterns(text, 4 * patterns, length,
                                         present_fraction=0.0, seed=SEED)
              if p not in text][:patterns]
    rows = []
    windows = n - length + 1
    for q in moduli:
        spur, fp, times = [], [], []
        for k, p in enumerate(absent):
            hasher = RollingHash(modulus=q, rng=random.Random(SEED + k))
            start = time.perf_counter()
            r = rabin_karp(text, p, hasher=hasher)
            times.append(time.perf_counter() - start)
            spur.append(r.stats["spurious"])
            fp.append(rabin_karp(text, p, hasher=hasher, verify=False)
                      .stats["false_positives"])
        rows.append(make_row(experiment="collision_modulus", dataset="english",
                             method="rk", n=n, pattern_length=length,
                             parameter=f"q={q}", queries=len(absent),
                             mean_search_s=statistics.mean(times),
                             value=statistics.mean(spur),
                             value2=statistics.mean(fp), value3=windows / q,
                             valid=True,
                             note="value = spurious hits per verified scan; "
                                  "value2 = Monte Carlo false positives; "
                                  "value3 = expected (n - m + 1)/q"))
        print(f"  q={q:<20d} spurious/scan {statistics.mean(spur):12.2f} "
              f"(expected {windows / q:12.2f}) time {statistics.mean(times):.3f} s")
    # bad parameter choices against good ones of the same size
    present = sample_patterns(text, patterns, length, present_fraction=1.0, seed=SEED + 1)
    configs = [
        ("prime q=65,521, random base", RollingHash(modulus=65_521, rng=random.Random(1))),
        ("q=2^16, base 256", RollingHash(base=256, modulus=1 << 16)),
        ("q=2^61-1, random base", RollingHash(modulus=MERSENNE_61, rng=random.Random(1))),
        ("base 1 (character sum)", RollingHash(base=1, modulus=MERSENNE_61)),
        ("base = q (only last char)", RollingHash(base=1_000_003, modulus=1_000_003)),
    ]
    for label, hasher in configs:
        spur = []
        for p in absent + present:
            spur.append(rabin_karp(text, p, hasher=hasher).stats["spurious"])
        rows.append(make_row(experiment="collision_params", dataset="english",
                             method="rk", n=n, pattern_length=length,
                             parameter=label, queries=len(spur),
                             value=statistics.mean(spur), valid=True,
                             note="value = spurious hash hits per scan"))
        print(f"  {label:30s} spurious/scan {statistics.mean(spur):12.1f}")
    return rows


def bench_birthday(moduli=(101, 1_009, 10_007, 100_003, 1_000_003), trials: int = 30) -> List[Dict]:
    """Random strings hashed until two collide: ~√(πq/2) tries."""
    rows = []
    for q in moduli:
        tries = []
        for t in range(trials):
            rng = random.Random(SEED * 1000 + t)
            hasher = RollingHash(modulus=q, rng=rng)
            seen = {}
            k = 0
            while True:
                k += 1
                s = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=12))
                h = hasher.hash(s)
                if h in seen and seen[h] != s:
                    break
                seen[h] = s
            tries.append(k)
        rows.append(make_row(experiment="birthday", dataset="random", method="rk",
                             parameter=f"q={q}", queries=trials,
                             value=statistics.mean(tries), valid=True,
                             note=f"theory sqrt(pi q / 2) = {np.sqrt(np.pi * q / 2):.1f}"))
    return rows


def bench_batch(n: int = 1_000_000, counts=(1, 10, 50, 100, 500),
                length: int = PATTERN_LENGTH) -> List[Dict]:
    text = dataset("english", n)
    rows = []
    for count in counts:
        patterns = sample_patterns(text, count, length, seed=SEED + count)
        start = time.perf_counter()
        multi = rabin_karp_multi(text, patterns, seed=1)
        t_multi = time.perf_counter() - start
        timings = {"rk-multi": t_multi}
        for method, run in (("kmp", lambda p: kmp_search(text, p)),
                            ("rk", lambda p: rabin_karp(text, p, seed=1))):
            if count > 100:
                sub = patterns[:20]          # extrapolate per-pattern cost
                start = time.perf_counter()
                for p in sub:
                    run(p)
                timings[method] = (time.perf_counter() - start) / len(sub) * count
            else:
                start = time.perf_counter()
                for p in patterns:
                    run(p)
                timings[method] = time.perf_counter() - start
        ok = all(multi[p].positions == builtin_find(text, p) for p in patterns[:50])
        for method, t in timings.items():
            rows.append(make_row(experiment="batch", dataset="english", method=method,
                                 n=n, pattern_length=length, parameter=f"P={count}",
                                 queries=count, mean_search_s=t, valid=ok,
                                 note="total time for all patterns"
                                      + (" (extrapolated from 20)" if count > 100
                                         and method != "rk-multi" else "")))
        print(f"  batch P={count:4d}: RK multi {t_multi:.2f} s, KMP {timings['kmp']:.2f} s, "
              f"RK one-by-one {timings['rk']:.2f} s")
    return rows


# ----------------------------------------------------------------------
# 4. suffix array / LCP
# ----------------------------------------------------------------------
def bench_construction(sizes=(1_000, 10_000, 100_000, 1_000_000, 5_000_000),
                       limits=None) -> List[Dict]:
    limits = limits or {"naive": 10_000, "python": 100_000, "tree": 1_000_000}
    rows = []
    for n in sizes:
        text = dataset("english", n)
        builds = {"doubling": lambda: suffix_array_doubling(text),
                  "python": lambda: suffix_array_python(text),
                  "naive": lambda: suffix_array_naive(text),
                  "tree": lambda: SuffixTree(text)}
        sa = None
        for method, build in builds.items():
            if n > limits.get(method, float("inf")):
                continue
            start = time.perf_counter()
            result = build()
            elapsed = time.perf_counter() - start
            rounds = result[1] if method == "doubling" else ""
            if method == "doubling":
                sa = result[0]
            rows.append(make_row(experiment="construction", dataset="english",
                                 method=method, n=n, preprocess_s=elapsed,
                                 value=rounds, valid=True,
                                 note="value = doubling rounds" if rounds != "" else ""))
            del result
        start = time.perf_counter()
        kasai_lcp(text, sa)
        rows.append(make_row(experiment="construction", dataset="english",
                             method="kasai", n=n,
                             preprocess_s=time.perf_counter() - start, valid=True))
        print(f"  build n={n:>9,}: " + ", ".join(
            f"{r['method']} {r['preprocess_s']:.3f}s" for r in rows
            if r["n"] == n and r["experiment"] == "construction"))
    return rows


def bench_lcp(n: int = 1_000_000, kinds=("english", "dna", "logs", "random"),
              samples: Optional[Dict] = None) -> List[Dict]:
    rows = []
    for kind in kinds:
        text = dataset(kind, n)
        sa = SuffixArray(text)
        st = sa.stats()
        lcp = np.asarray(sa.lcp)
        if samples is not None:
            samples[("lcp", kind)] = lcp
        rows.append(make_row(experiment="lcp", dataset=kind, method="sa", n=n,
                             parameter=f"rounds={sa.rounds}", value=st.mean_lcp,
                             value2=st.max_lcp, value3=float(np.percentile(lcp, 99)),
                             valid=True,
                             note=f"value = mean LCP; value2 = max LCP; value3 = "
                                  f"99th percentile; distinct substrings "
                                  f"{st.distinct_substrings:.4g}"))
        print(f"  lcp {kind:8s} mean {st.mean_lcp:6.2f} max {st.max_lcp:6d} "
              f"rounds {sa.rounds}")
    return rows


# ----------------------------------------------------------------------
# 5. applications
# ----------------------------------------------------------------------
def bench_plagiarism(fractions=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8), length: int = 20_000,
                     k: int = 5) -> List[Dict]:
    rows = []
    source = english_like_text(4 * length, seed=SEED)
    for f in fractions:
        rng = random.Random(SEED + int(f * 100))
        own = english_like_text(length, seed=SEED + 500)
        copied = int(f * length)
        pieces, pos = [], 0
        while pos < copied:                  # copy in chunks of ~1,000 chars
            size = min(1000, copied - pos)
            start = rng.randrange(0, len(source) - size)
            pieces.append(source[start:start + size])
            pos += size
        rest = own[:length - copied]
        cut = len(rest) // (len(pieces) + 1) if pieces else len(rest)
        doc = []
        for i, piece in enumerate(pieces):
            doc.append(rest[i * cut:(i + 1) * cut])
            doc.append(piece)
        doc.append(rest[len(pieces) * cut:])
        doc = "".join(doc)
        start = time.perf_counter()
        report = compare_documents(doc, source, k=k)
        elapsed = time.perf_counter() - start
        rows.append(make_row(experiment="plagiarism", dataset="english",
                             method="rk+sa", n=len(doc), parameter=f"copied={f}",
                             mean_search_s=elapsed, value=report.containment_a,
                             value2=report.coverage_a, value3=report.jaccard,
                             valid=True,
                             note=f"value = containment; value2 = character "
                                  f"coverage; value3 = Jaccard; spurious hash "
                                  f"hits {report.spurious_hash_hits}"))
        print(f"  plagiarism copied {f:.1f}: containment {report.containment_a:.3f} "
              f"coverage {report.coverage_a:.3f} jaccard {report.jaccard:.3f} "
              f"({elapsed:.2f} s)")
    return rows


def bench_dna(sizes=(100_000, 1_000_000), genes: int = 20, gene_length: int = 30,
              max_mismatches=(0, 1, 2, 3)) -> List[Dict]:
    rows = []
    for n in sizes:
        genome = dataset("dna", n)
        rng = random.Random(SEED + n)
        starts = [rng.randrange(0, n - gene_length) for _ in range(genes)]
        queries = [mutate(genome[s:s + gene_length], 2, seed=SEED + i)[0]
                   for i, s in enumerate(starts)]
        start = time.perf_counter()
        index = GenomeIndex(genome, method="sa")
        build = time.perf_counter() - start
        for k in max_mismatches:
            found, times = 0, []
            for q, s in zip(queries, starts):
                t0 = time.perf_counter()
                hits = index.find_mutations(q, k, both_strands=False)
                times.append(time.perf_counter() - t0)
                found += any(h.position == s for h in hits)
            rows.append(make_row(experiment="dna", dataset="dna", method="sa",
                                 n=n, pattern_length=gene_length,
                                 parameter=f"k={k}", queries=genes,
                                 preprocess_s=build,
                                 mean_search_s=statistics.mean(times),
                                 value=found / genes, valid=True,
                                 note="value = planted 2-mutation genes recovered"))
        brute = []
        for q in queries[:3]:
            t0 = time.perf_counter()
            brute_force_mutations(genome, q, 2)
            brute.append(time.perf_counter() - t0)
        rows.append(make_row(experiment="dna", dataset="dna", method="brute force",
                             n=n, pattern_length=gene_length, parameter="k=2",
                             queries=3, mean_search_s=statistics.mean(brute),
                             valid=True, note="Hamming scan of every window"))
        print(f"  dna n={n:>9,}: build {build:.2f} s, " + ", ".join(
            f"k={r['parameter'][2:]} {r['mean_search_s'] * 1e3:.2f} ms "
            f"recall {r['value']:.2f}" for r in rows[-5:-1])
            + f", brute k=2 {statistics.mean(brute) * 1e3:.0f} ms")
    return rows


def zipf_vocabulary(size: int, seed: int = SEED) -> Dict[str, float]:
    """`size` distinct pseudo-words (3–10 letters) with Zipf weights."""
    rng = random.Random(seed)
    letters = "etaoinshrdlcumwfgypbvkjxqz"
    weights = [1 / (i + 1) for i in range(len(letters))]   # skewed letters
    vocab: Dict[str, float] = {}
    while len(vocab) < size:
        word = "".join(rng.choices(letters, weights=weights, k=rng.randint(3, 10)))
        if word not in vocab:
            vocab[word] = 1_000_000 / (len(vocab) + 1)       # rank-based Zipf
    return vocab


def bench_autocomplete(sizes=(10_000, 100_000, 500_000), queries: int = 300,
                       k: int = 10) -> List[Dict]:
    rows = []
    for size in sizes:
        vocab = zipf_vocabulary(size)
        start = time.perf_counter()
        ac = Autocomplete(vocab)
        build = time.perf_counter() - start
        base = SortedPrefixIndex(ac.frequency)
        rng = random.Random(SEED)
        terms = list(vocab)
        # users type popular words: sample by frequency, prefixes of 1–3 chars
        prefixes = [t[:rng.randint(1, 3)] for t in
                    rng.choices(terms, weights=list(vocab.values()), k=queries)]
        for method, run in (("trie", lambda p: ac.complete(p, k)),
                            ("sorted+scan", lambda p: base.complete(p, k))):
            start = time.perf_counter()
            for p in prefixes:
                run(p)
            elapsed = (time.perf_counter() - start) / queries
            rows.append(make_row(experiment="autocomplete", dataset="zipf words",
                                 method=method, n=len(ac), parameter=f"k={k}",
                                 queries=queries,
                                 preprocess_s=build if method == "trie" else "",
                                 mean_search_s=elapsed, valid=True,
                                 value=statistics.mean(ac.count_prefix(p)
                                                       for p in prefixes),
                                 note="n = vocabulary size; value = mean "
                                      "completions per prefix"))
        same = all([c.term for c in ac.complete(p, k)] ==
                   [t for t, _ in base.complete(p, k)] for p in prefixes[:50])
        rows[-1]["valid"] = rows[-2]["valid"] = same
        print(f"  autocomplete vocab {len(ac):>7,}: trie "
              f"{rows[-2]['mean_search_s'] * 1e3:.3f} ms, scan "
              f"{rows[-1]['mean_search_s'] * 1e3:.3f} ms, "
              f"{rows[-1]['value']:,.0f} completions/prefix, agree={same}")
    return rows


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
def _series(rows, experiment, key="n", value="mean_search_s", methods=METHODS,
            scale=1.0, **match):
    out = {}
    for m in methods:
        pts = sorted((r[key], r[value] * scale) for r in rows
                     if r["experiment"] == experiment and r["method"] == m
                     and r[value] != "" and all(r[k] == v for k, v in match.items()))
        if pts:
            out[LABELS.get(m, m)] = ([p[0] for p in pts], [p[1] for p in pts])
    return out


def _colors():
    return {LABELS[m]: c for m, c in COLORS.items()}


def plot_runtime(rows: List[Dict], results_dir: str) -> None:
    panels = []
    for kind in ("english", "dna"):
        series = _series(rows, "search", scale=1e3, dataset=kind)
        if series:
            panels.append({"title": f"Search latency per query ({kind})",
                           "xlabel": "text size (characters)", "ylabel": "ms",
                           "series": series, "colors": _colors(),
                           "logx": True, "logy": True})
    prep = _series(rows, "search", value="preprocess_s",
                   methods=["kmp", "rk", "sa", "tree"], dataset="english")
    if prep:
        panels.append({"title": "Preprocessing (english): pattern vs index",
                       "xlabel": "text size (characters)", "ylabel": "seconds",
                       "series": prep, "colors": _colors(), "logx": True,
                       "logy": True})
    cross = [r for r in rows if r["experiment"] == "search"
             and r["dataset"] == "english" and r["n"] == 1_000_000]
    if cross:
        qs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        series = {}
        for r in cross:
            if r["method"] == "builtin":
                continue
            p = r["preprocess_s"] if r["method"] in ("sa", "tree") else 0.0
            series[LABELS[r["method"]]] = (qs, [p + q * r["mean_search_s"] for q in qs])
        panels.append({"title": "Total cost of q queries (english, 1 MB)",
                       "xlabel": "queries", "ylabel": "seconds (build + queries)",
                       "series": series, "colors": _colors(), "logx": True,
                       "logy": True})
    length = _series(rows, "pattern_length", key="pattern_length", scale=1e3,
                     methods=["kmp", "rk", "sa", "tree", "builtin"])
    if length:
        panels.append({"title": "Latency vs pattern length (english, 1 MB)",
                       "xlabel": "pattern length", "ylabel": "ms",
                       "series": length, "colors": _colors(), "logx": True,
                       "logy": True})
    worst = _series(rows, "worst_case", key="pattern_length", value="value",
                    methods=["naive", "kmp", "rk"], dataset="a^n", parameter="")
    if worst:
        n = next(r["n"] for r in rows if r["experiment"] == "worst_case")
        # RK with P = a^(m-1)b gets no hash hits (0 comparisons, off the log
        # axis); with P = a^m every window is a real hit and it performs
        # exactly brute force's m(n - m + 1) comparisons
        worst.pop("Rabin–Karp", None)
        worst["brute force = Rabin–Karp with P = a^m"] = worst.pop("brute force")
        panels.append({"title": f"Worst case T = a^n (n = {n:,}), P = a^(m−1)b"
                                f"\n(Rabin–Karp: no hash hits, 0 comparisons)",
                       "xlabel": "pattern length m",
                       "ylabel": "character comparisons", "series": worst,
                       "colors": dict(_colors(), **{
                           "brute force = Rabin–Karp with P = a^m": "#d62728"}),
                       "logx": True, "logy": True})
    plot_mixed_panels(panels, os.path.join(results_dir, "runtime_comparison.png"),
                      "String search: latency, preprocessing and crossover", ncols=2)


def plot_collisions(rows: List[Dict], results_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    mod = sorted((int(r["parameter"][2:]), r["value"], r["value2"],
                  r["mean_search_s"], r["n"], r["pattern_length"])
                 for r in rows if r["experiment"] == "collision_modulus")
    ax = axes[0, 0]
    if mod:
        qs = np.array([m[0] for m in mod], dtype=float)
        windows = mod[0][4] - mod[0][5] + 1
        ax.plot(qs, [max(m[1], 1e-3) for m in mod], "o-", color=COLORS["rk"],
                label="measured spurious hits (verified)")
        ax.plot(qs, [max(m[2], 1e-3) for m in mod], "s--", color="#d62728",
                label="Monte Carlo false positives (no verify)")
        ax.plot(qs, windows / qs, "k:", label="theory (n − m + 1)/q")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, None)
        ax.text(0.98, 0.02, "points at 10⁻³ = zero observed", transform=ax.transAxes,
                ha="right", fontsize=8, color="#555555")
    ax.set_title("Spurious hash hits per scan vs modulus q (1 MB)")
    ax.set_xlabel("modulus q")
    ax.set_ylabel("hits per scan")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if mod:
        ax.plot([m[0] for m in mod], [m[3] for m in mod], "o-", color=COLORS["rk"])
        ax.set_xscale("log")
    ax.set_title("Rabin–Karp scan time vs modulus (verification cost)")
    ax.set_xlabel("modulus q")
    ax.set_ylabel("seconds per scan")

    ax = axes[1, 0]
    params = [r for r in rows if r["experiment"] == "collision_params"]
    if params:
        vals = [max(r["value"], 0.01) for r in params]
        bars = ax.barh(range(len(params)), vals,
                       color=["#2ca02c" if "random" in r["parameter"] else "#d62728"
                              for r in params])
        for b, r in zip(bars, params):
            ax.text(b.get_width() * 1.1, b.get_y() + b.get_height() / 2,
                    f"{r['value']:,.1f}", va="center", fontsize=8)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([r["parameter"] for r in params], fontsize=8)
        ax.set_xscale("log")
        ax.invert_yaxis()
    ax.set_title("Base/modulus choices (green: good, red: bad)")
    ax.set_xlabel("spurious hits per scan (log; 0 shown at 0.01)")

    ax = axes[1, 1]
    for method, label in (("rk-multi", "Rabin–Karp, one pass for all patterns"),
                          ("kmp", "KMP, one scan per pattern"),
                          ("rk", "Rabin–Karp, one scan per pattern")):
        pts = sorted((int(r["parameter"][2:]), r["mean_search_s"]) for r in rows
                     if r["experiment"] == "batch" and r["method"] == method)
        if pts:
            ax.plot(*zip(*pts), "o-", label=label,
                    color={"rk-multi": "#d62728"}.get(method, COLORS.get(method)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Batch matching: P patterns of one length (1 MB)")
    ax.set_xlabel("number of patterns P")
    ax.set_ylabel("total seconds")
    ax.legend(fontsize=8)
    for a in axes.flat:
        a.grid(True, which="both", alpha=0.3)
    fig.suptitle("Rabin–Karp: collisions, parameters and batch matching",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(results_dir, "collision_analysis.png"), dpi=150)
    plt.close(fig)


def plot_sa_lcp(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axes[0, 0]
    labels = {"doubling": "SA prefix doubling (numpy)",
              "python": "SA prefix doubling (Python sort)",
              "naive": "SA naive suffix sort", "kasai": "LCP (Kasai)",
              "tree": "suffix tree (Ukkonen)"}
    colors = {"doubling": "#2ca02c", "python": "#98df8a", "naive": "#7f7f7f",
              "kasai": "#17becf", "tree": "#9467bd"}
    for method, label in labels.items():
        pts = sorted((r["n"], r["preprocess_s"]) for r in rows
                     if r["experiment"] == "construction" and r["method"] == method)
        if pts:
            ax.plot(*zip(*pts), "o-", label=label, color=colors[method])
    ns = np.array([1e3, 5e6])
    ref = [r for r in rows if r["experiment"] == "construction"
           and r["method"] == "doubling" and r["n"] == 1_000_000]
    if ref:
        c = ref[0]["preprocess_s"] / (1e6 * np.log2(1e6))
        ax.plot(ns, c * ns * np.log2(ns), "k:", lw=1, label="∝ n log n")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Index construction time (english)")
    ax.set_xlabel("text size (characters)")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    lcps = {k[1]: v for k, v in samples.items() if k[0] == "lcp"}
    if lcps:
        top = max(int(np.percentile(v, 99.9)) for v in lcps.values()) + 2
        edges = np.arange(0, top + 1)
        for kind, values in lcps.items():
            ax.hist(np.minimum(values, top), bins=edges, histtype="step", lw=1.5,
                    label=f"{kind} (mean {values.mean():.1f}, max {values.max()})")
        ax.set_yscale("log")
    ax.set_title("LCP distribution of adjacent suffixes (1 MB)")
    ax.set_xlabel("LCP (clipped at the 99.9th percentile)")
    ax.set_ylabel("suffixes")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    lcp_rows = [r for r in rows if r["experiment"] == "lcp"]
    if lcp_rows:
        kinds = [r["dataset"] for r in lcp_rows]
        rounds = [int(r["parameter"].split("=")[1]) for r in lcp_rows]
        maxes = [int(r["value2"]) for r in lcp_rows]
        x = np.arange(len(kinds))
        ax.bar(x - 0.2, rounds, 0.4, label="doubling rounds", color="#2ca02c")
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, maxes, 0.4, label="max LCP (longest repeat)",
                color="#9467bd")
        ax2.set_yscale("log")
        ax2.set_ylabel("max LCP (log)")
        ax.plot(x - 0.2, np.ceil(np.log2(np.maximum(maxes, 1) + 1)), "k_",
                markersize=18, mew=2, label="⌈log₂(max LCP + 1)⌉")
        ax.set_xticks(x)
        ax.set_xticklabels(kinds)
        ax.set_ylabel("rounds")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("Doubling rounds track the longest repeat")

    ax = axes[1, 1]
    mem = sorted((r["n"], r["method"], r["memory_mb"]) for r in rows
                 if r["experiment"] == "search" and r["dataset"] == "english"
                 and r["memory_mb"] != "")
    for method in ("sa", "tree"):
        pts = [(n, m) for n, meth, m in mem if meth == method]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] * 2 ** 20 / p[0] for p in pts],
                    "o-", color=COLORS[method], label=LABELS[method])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Index memory per text character (tracemalloc peak)")
    ax.set_xlabel("text size (characters)")
    ax.set_ylabel("bytes per character")
    ax.legend(fontsize=8)
    for a in axes.flat:
        a.grid(True, which="both", alpha=0.3)
    fig.suptitle("Suffix arrays and LCP: construction, structure, memory",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(results_dir, "sa_lcp_stats.png"), dpi=150)
    plt.close(fig)


def plot_heatmaps(rows: List[Dict], results_dir: str, queries_total: int = 100) -> None:
    search = [r for r in rows if r["experiment"] == "search"]
    kinds = list(dict.fromkeys(r["dataset"] for r in search))
    columns = [(k, n) for k in kinds
               for n in sorted({r["n"] for r in search if r["dataset"] == k})]
    methods = [m for m in METHODS if any(r["method"] == m for r in search)]
    if not columns:
        return

    def grid(value):
        mat = np.full((len(methods), len(columns)), np.nan)
        for r in search:
            if (r["dataset"], r["n"]) in columns:
                v = value(r)
                if v is not None:
                    mat[methods.index(r["method"]), columns.index((r["dataset"], r["n"]))] = v
        return mat

    panels = [
        ("Search latency per query (ms)", grid(lambda r: r["mean_search_s"] * 1e3)),
        (f"Total for {queries_total} queries incl. index build (s)",
         grid(lambda r: (r["preprocess_s"] if r["method"] in ("sa", "tree") else 0)
              + queries_total * r["mean_search_s"])),
        ("Index memory (MB, tracemalloc peak)",
         grid(lambda r: r["memory_mb"] if r["memory_mb"] != "" else None)),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.2 * len(panels) + 0.6))
    for ax, (title, mat) in zip(axes, panels):
        shown = np.where(np.isnan(mat), np.nan, np.log10(np.maximum(mat, 1e-6)))
        im = ax.imshow(shown, cmap="viridis_r", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", color="#999999")
                    continue
                txt = f"{v:.3g}" if v >= 0.01 else f"{v:.2g}"
                dark = shown[i, j] > np.nanmean(shown)    # viridis_r: high = dark
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if dark else "black")
        # mark the best method per column
        for j in range(mat.shape[1]):
            col = mat[:, j]
            if not np.all(np.isnan(col)):
                i = int(np.nanargmin(col))
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec="#d62728", lw=2))
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([f"{k}\n{SIZE_NAMES.get(n, f'{n:,}')}" for k, n in columns])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([LABELS[m] for m in methods])
        ax.set_title(title + "  (red box: lowest in column; colour: log scale)")
        fig.colorbar(im, ax=ax, fraction=0.02, label="log10")
    fig.suptitle("Methods × dataset sizes", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(results_dir, "heatmap_dataset_sizes.png"), dpi=150)
    plt.close(fig)


def plot_applications(rows: List[Dict], results_dir: str) -> None:
    panels = []
    plag = sorted((float(r["parameter"].split("=")[1]), r["value"], r["value2"],
                   r["value3"]) for r in rows if r["experiment"] == "plagiarism")
    if plag:
        xs = [p[0] for p in plag]
        panels.append({"title": "Plagiarism: detected vs actual copied share",
                       "xlabel": "fraction of the document copied",
                       "ylabel": "score",
                       "series": {"containment (5-word phrases)": (xs, [p[1] for p in plag]),
                                  "character coverage": (xs, [p[2] for p in plag]),
                                  "Jaccard": (xs, [p[3] for p in plag])},
                       "refs": {"ideal": ([0, max(xs)], [0, max(xs)])}})
    dna = {}
    for n in sorted({r["n"] for r in rows if r["experiment"] == "dna"}):
        pts = sorted((int(r["parameter"][2:]), r["mean_search_s"] * 1e3) for r in rows
                     if r["experiment"] == "dna" and r["n"] == n and r["method"] == "sa")
        if pts:
            dna[f"SA, genome {n:,}"] = ([p[0] for p in pts], [p[1] for p in pts])
        brute = [r for r in rows if r["experiment"] == "dna" and r["n"] == n
                 and r["method"] == "brute force"]
        if brute:
            dna[f"brute force k=2, {n:,}"] = ([2], [brute[0]["mean_search_s"] * 1e3])
    if dna:
        panels.append({"title": "DNA: approximate gene search (30 bp)",
                       "xlabel": "allowed mismatches k", "ylabel": "ms per gene",
                       "series": dna, "logy": True})
    auto = _series(rows, "autocomplete", methods=["trie", "sorted+scan"], scale=1e3)
    if auto:
        panels.append({"title": "Autocomplete top-10 latency",
                       "xlabel": "vocabulary size", "ylabel": "ms per query",
                       "series": auto, "logx": True, "logy": True})
    recall = {}
    for n in sorted({r["n"] for r in rows if r["experiment"] == "dna"}):
        pts = sorted((int(r["parameter"][2:]), r["value"]) for r in rows
                     if r["experiment"] == "dna" and r["n"] == n and r["method"] == "sa")
        if pts:
            recall[f"genome {n:,}"] = ([p[0] for p in pts], [p[1] for p in pts])
    if recall:
        panels.append({"title": "DNA: planted genes with 2 mutations recovered",
                       "xlabel": "allowed mismatches k", "ylabel": "recall",
                       "series": recall})
    plot_mixed_panels(panels, os.path.join(results_dir, "string_applications.png"),
                      "String applications", ncols=2)


def plot_examples(results_dir: str) -> None:
    plot_suffix_array("mississippi", save_path=os.path.join(
        results_dir, "suffix_array_example.png"))
    tree = SuffixTree("mississippi")
    draw_suffix_tree(tree, save_path=os.path.join(results_dir,
                                                  "suffix_tree_example.png"))
    write_dot(tree, os.path.join(results_dir, "suffix_tree_example.dot"))
    draw_ukkonen_steps("abcabxabcd", save_path=os.path.join(results_dir,
                                                            "ukkonen_steps.png"))
    plot_kmp_trace("abababcabababd", "ababd", save_path=os.path.join(
        results_dir, "kmp_trace.png"))


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def print_summary(rows: List[Dict]) -> None:
    print("\nFastest per-query method and break-even vs KMP (english):")
    for n in SIZES:
        own = {r["method"]: r for r in rows if r["experiment"] == "search"
               and r["dataset"] == "english" and r["n"] == n}
        if "kmp" not in own:
            continue
        kmp = own["kmp"]["mean_search_s"]
        parts = []
        for m in ("sa", "tree"):
            if m in own and kmp > own[m]["mean_search_s"]:
                q = own[m]["preprocess_s"] / (kmp - own[m]["mean_search_s"])
                parts.append(f"{LABELS[m]} pays off after {q:.0f} queries")
        print(f"  {SIZE_NAMES[n]:>6}: KMP {kmp * 1e3:.2f} ms/query; " + "; ".join(parts))
    invalid = [r for r in rows if r["valid"] is False]
    print(f"\nValidation: {len(invalid)} configuration(s) disagreed with str.find"
          if invalid else "\nValidation: every method agreed with str.find on every query")


def run(results_dir: str = RESULTS_DIR, quick: bool = False) -> List[Dict]:
    os.makedirs(results_dir, exist_ok=True)
    samples: Dict = {}
    rows: List[Dict] = []
    print("1. Search latency, preprocessing, memory")
    rows += bench_search([n for n in SIZES if not quick or n <= 1_000_000],
                         queries=8 if quick else QUERIES, samples=samples)
    rows += bench_pattern_length(n=100_000 if quick else 1_000_000,
                                 queries=5 if quick else 10)
    rows += bench_worst_case(n=20_000 if quick else 100_000)
    print("2. Rabin-Karp collisions and batch matching")
    rows += bench_collisions(n=200_000 if quick else 1_000_000,
                             patterns=3 if quick else 5)
    rows += bench_birthday(trials=10 if quick else 30)
    rows += bench_batch(n=200_000 if quick else 1_000_000,
                        counts=(1, 10, 50) if quick else (1, 10, 50, 100, 500))
    print("3. Suffix array / LCP")
    rows += bench_construction(sizes=(1_000, 10_000, 100_000) if quick else
                               (1_000, 10_000, 100_000, 1_000_000, 5_000_000))
    rows += bench_lcp(n=100_000 if quick else 1_000_000, samples=samples)
    print("4. Applications")
    rows += bench_plagiarism(length=5_000 if quick else 20_000)
    rows += bench_dna(sizes=(100_000,) if quick else (100_000, 1_000_000))
    rows += bench_autocomplete(sizes=(10_000, 50_000) if quick else
                               (10_000, 100_000, 500_000))

    csv_path = os.path.join(results_dir, CSV_NAME)
    write_csv(rows, csv_path)
    print(f"\nSaved {csv_path} ({len(rows)} rows)")
    print_summary(rows)
    plot_runtime(rows, results_dir)
    plot_collisions(rows, results_dir)
    plot_sa_lcp(rows, samples, results_dir)
    plot_heatmaps(rows, results_dir)
    plot_applications(rows, results_dir)
    plot_examples(results_dir)
    print(f"Saved plots to {results_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true",
                        help="smaller texts and fewer queries")
    args = parser.parse_args()
    start = time.perf_counter()
    run(quick=args.quick)
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
