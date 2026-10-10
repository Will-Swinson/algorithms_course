"""
Week 7 benchmark: randomized algorithms vs deterministic counterparts.

Two kinds of randomness are kept apart (see src/utils/random_utils.py):
inputs are generated from fixed seeds, and each trial gives the
algorithm its own seeded random.Random. Holding the input fixed while
varying only the algorithm's seed isolates the variance that the
algorithm's coin flips cause.

Experiments (results land in benchmarks/results/):

1. QuickSort: randomized pivot vs first-element pivot vs Week 2's
   quick_sort vs Python's sorted(), n = 100 .. 100,000 on random input,
   and on sorted input (the deterministic worst case). Comparisons vs
   the exact expectation 2(n+1)H_n − 4n, runtime mean/variance, and the
   comparison-count distribution.              -> quicksort_comparison.png
2. Selection: randomized QuickSelect vs median-of-medians vs sort, for
   k = min, median, max. Comparisons vs Knuth's exact expectation,
   recursion depth, and the probability of finishing within c·n
   comparisons.                                -> selection_runtime.png
3. Primality: Fermat vs Miller–Rabin error rates on Carmichael numbers
   for k = 1 .. 20 rounds, exact per-round liar fractions for every
   odd composite up to 3,000, and speed vs trial division from 10^6
   to 10^300.                                  -> primality_accuracy.png
4. Bloom filter: false-positive rate vs load factor, bits per item,
   and number of hash functions, against (1 − e^(−kn/m))^k; memory vs
   a Python set.                               -> bloom_filter_fp_rate.png
5. MinHash: estimation error vs signature size k = 16 .. 512 against
   sqrt(J(1−J)/k), on simulated token sets and near-duplicate text.
                                               -> minhash_similarity_plot.png

One row per configuration: summary_table.csv

Usage:
    python benchmarks/week7_randomized_benchmark.py          # ~1 min
    python benchmarks/week7_randomized_benchmark.py --quick  # ~20 s
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.randomized import (
    BloomFilter, MinHash, PartitionStats, deterministic_quicksort,
    fermat_test, is_prime_trial_division, jaccard_similarity, liar_fraction,
    median_of_medians_select, miller_rabin, randomized_quicksort,
    randomized_select, shingles, sort_select,
)
from src.sorting import quick_sort
from src.utils.probability_analysis import (
    bloom_false_positive_rate, expected_quickselect_comparisons,
    expected_quicksort_comparisons, harmonic, miller_rabin_error_bound,
    minhash_standard_error, summarize,
    wilson_interval,
)
from src.utils.random_utils import random_array, random_set_pair
from src.utils.timer import peak_memory, time_call
from src.utils.visualization import plot_mixed_panels

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_NAME = "summary_table.csv"
SEED = 42

FIELDS = ["experiment", "method", "paradigm", "input_size", "parameter",
          "trials", "mean_time_s", "std_time_s", "var_time_s2",
          "mean_work", "std_work", "expected_work", "work_unit",
          "mean_depth", "success_rate", "error_metric", "error",
          "error_theory", "error_ci_high"]

LAS_VEGAS, MONTE_CARLO, DETERMINISTIC = "las_vegas", "monte_carlo", \
    "deterministic"

QS_SIZES = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
QS_TRIALS = 20
QS_WORST_SIZES = [100, 300, 1_000, 2_000, 3_000]   # first-pivot on sorted
QS_DIST_N, QS_DIST_TRIALS = 10_000, 300

SEL_SIZES = QS_SIZES
SEL_TRIALS = 20
SEL_DIST_N, SEL_DIST_TRIALS = 10_000, 500
SEL_BUDGETS = [round(2 + 0.25 * i, 2) for i in range(17)]   # c = 2 .. 6

PRIME_DIGITS_TRIAL = [6, 8, 10, 12, 14]
PRIME_DIGITS_MC = PRIME_DIGITS_TRIAL + [20, 50, 100, 200, 300]
ROUNDS = [1, 2, 3, 5, 10, 20]
CARMICHAEL = [561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841,
              29341, 41041, 46657, 52633, 62745, 63973, 75361]
CARMICHAEL_TRIALS = 2_000
LIAR_LIMIT = 3_000

BLOOM_BITS = 1 << 16
BLOOM_LOADS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
BLOOM_LOAD_KS = [3, 7]
BLOOM_N = 10_000
BLOOM_BITS_PER_ITEM = [2, 4, 6, 8, 10, 12, 16, 20]
BLOOM_K_SWEEP = list(range(1, 13))
BLOOM_K_SWEEP_BPI = 8
BLOOM_PROBES = 50_000

MINHASH_KS = [16, 32, 64, 128, 256, 512]
MINHASH_JS = [0.1, 0.3, 0.5, 0.7, 0.9]
MINHASH_SET_SIZE = 500
MINHASH_TRIALS = 50

# Base paragraphs for the near-duplicate text experiment
BASE_TEXTS = [
    "Randomized algorithms make random choices during execution to gain "
    "simplicity and speed. A Las Vegas algorithm always returns the right "
    "answer but its running time is a random variable, while a Monte Carlo "
    "algorithm runs in fixed time and may return a wrong answer with small "
    "probability that we can drive down by repeating it several times.",
    "Search engines detect near duplicate web pages by breaking each page "
    "into overlapping shingles of words and comparing the resulting sets. "
    "Comparing every pair exactly is far too slow at web scale, so each "
    "page is reduced to a short MinHash signature whose agreement rate "
    "estimates the Jaccard similarity of the underlying shingle sets.",
    "A Bloom filter answers set membership questions using a bit array and "
    "several hash functions. It never misses an element that was inserted "
    "but occasionally reports an element that was never added, trading a "
    "tunable false positive rate for memory savings that make it popular in "
    "databases, caches, network routers, and distributed storage systems.",
]
TEXT_MUTATION_RATES = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]
TEXT_SHINGLE = 2
TEXT_KS = [16, 128, 512]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def make_row(experiment: str, method: str, paradigm: str, size: int,
             parameter: str = "", **fields) -> Dict:
    row = {f: None for f in FIELDS}
    row.update(experiment=experiment, method=method, paradigm=paradigm,
               input_size=size, parameter=parameter, **fields)
    _print_row(row)
    return row


def _print_row(row: Dict) -> None:
    parts = [f"  {row['experiment']:16s} {row['method']:24s} "
             f"n={row['input_size']:<7} {row['parameter']:14s}"]
    if row["mean_time_s"] is not None:
        parts.append(f"{row['mean_time_s'] * 1e3:10.3f} ms "
                     f"± {(row['std_time_s'] or 0) * 1e3:7.3f}")
    if row["mean_work"] is not None:
        parts.append(f"{row['mean_work']:>14,.0f} {row['work_unit'] or ''}")
    if row["error"] is not None:
        parts.append(f"{row['error_metric']}={row['error']:.3g}")
    print(" ".join(parts), flush=True)


def timed(func: Callable[[], object]) -> Tuple[float, object]:
    """(seconds, result) of one call."""
    start = time.perf_counter()
    result = func()
    return time.perf_counter() - start, result


def trial_fields(times: Sequence[float],
                 work: Optional[Sequence[float]] = None,
                 depths: Optional[Sequence[float]] = None) -> Dict:
    """Mean/variance columns for repeated trials."""
    t = summarize(times)
    fields = {"trials": len(times), "mean_time_s": t.mean,
              "std_time_s": t.std, "var_time_s2": t.variance}
    if work:
        w = summarize(work)
        fields.update(mean_work=w.mean, std_work=w.std)
    if depths:
        fields["mean_depth"] = summarize(depths).mean
    return fields


# ---------------------------------------------------------------------------
# 1. QuickSort
# ---------------------------------------------------------------------------

SortFunc = Callable[..., List]


def run_sort_trials(func: SortFunc, arr: List, trials: int,
                    seed: int, counted: bool = True
                    ) -> Tuple[List[float], List[int], List[int]]:
    """Run func on the same input `trials` times, each with its own
    seeded rng. Returns (times, comparisons, depths)."""
    times, comps, depths = [], [], []
    expected = sorted(arr)
    func(arr)                     # warm-up: caches, allocator
    for t in range(trials):
        rng = random.Random(seed * 1_000 + t)
        if counted:
            stats = PartitionStats()
            seconds, out = timed(lambda: func(arr, stats=stats, rng=rng))
            comps.append(stats.comparisons)
            depths.append(stats.max_depth)
        else:
            seconds, out = timed(lambda: func(arr))
        assert out == expected, f"{func.__name__} returned an unsorted list"
        times.append(seconds)
    return times, comps, depths


def bench_quicksort(sizes: Sequence[int] = QS_SIZES,
                    trials: int = QS_TRIALS,
                    worst_sizes: Sequence[int] = QS_WORST_SIZES,
                    dist_n: int = QS_DIST_N,
                    dist_trials: int = QS_DIST_TRIALS,
                    samples: Optional[Dict] = None) -> List[Dict]:
    print("\n=== QuickSort: random pivot vs first-element pivot ===")
    rows = []
    methods = [
        ("randomized_quicksort", randomized_quicksort, LAS_VEGAS, True),
        ("deterministic_quicksort", deterministic_quicksort, DETERMINISTIC,
         True),
        ("week2_quick_sort", quick_sort, LAS_VEGAS, False),
        ("python_sorted", sorted, DETERMINISTIC, False),
    ]
    for n in sizes:
        arr = random_array(n, "random", seed=SEED + n)
        expected = expected_quicksort_comparisons(n)
        for name, func, paradigm, counted in methods:
            times, comps, depths = run_sort_trials(func, arr, trials, n,
                                                   counted)
            rows.append(make_row(
                "quicksort_random", name, paradigm, n,
                **trial_fields(times, comps, depths),
                expected_work=expected if counted else None,
                work_unit="comparisons" if counted else None))

    for n in sizes:
        arr = random_array(n, "sorted")
        cases = [("randomized_quicksort", randomized_quicksort, LAS_VEGAS,
                  trials)]
        if n in worst_sizes:
            # deterministic: the counts never vary, 3 runs time it
            cases.append(("deterministic_quicksort", deterministic_quicksort,
                          DETERMINISTIC, min(trials, 3)))
        for name, func, paradigm, runs in cases:
            times, comps, depths = run_sort_trials(func, arr, runs, n)
            rows.append(make_row(
                "quicksort_sorted", name, paradigm, n,
                **trial_fields(times, comps, depths),
                expected_work=(expected_quicksort_comparisons(n)
                               if paradigm == LAS_VEGAS
                               else n * (n - 1) / 2),
                work_unit="comparisons"))

    if samples is not None and dist_trials:
        arr = random_array(dist_n, "random", seed=SEED + dist_n)
        times, comps, depths = run_sort_trials(randomized_quicksort, arr,
                                               dist_trials, 7)
        samples["qs_dist"] = {"n": dist_n, "times": times,
                              "comparisons": comps, "depths": depths}
        rows.append(make_row(
            "quicksort_dist", "randomized_quicksort", LAS_VEGAS, dist_n,
            **trial_fields(times, comps, depths),
            expected_work=expected_quicksort_comparisons(dist_n),
            work_unit="comparisons"))
    return rows


# ---------------------------------------------------------------------------
# 2. Selection
# ---------------------------------------------------------------------------

def k_values(n: int) -> Dict[str, int]:
    return {"min": 1, "median": (n + 1) // 2, "max": n}


def bench_selection(sizes: Sequence[int] = SEL_SIZES,
                    trials: int = SEL_TRIALS,
                    dist_n: int = SEL_DIST_N,
                    dist_trials: int = SEL_DIST_TRIALS,
                    samples: Optional[Dict] = None) -> List[Dict]:
    print("\n=== Selection: QuickSelect vs median-of-medians vs sort ===")
    rows = []
    for n in sizes:
        arr = random_array(n, "random", seed=SEED + n)
        for label, k in k_values(n).items():
            param = f"k={label}"
            # randomized: vary the algorithm's seed
            times, comps, depths = [], [], []
            for t in range(trials):
                rng = random.Random(n * 1_000 + t)
                stats = PartitionStats()
                seconds, value = timed(
                    lambda: randomized_select(arr, k, stats, rng))
                assert value == k - 1
                times.append(seconds)
                comps.append(stats.comparisons)
                depths.append(stats.max_depth)
            within = sum(c <= 4 * n for c in comps) / trials
            rows.append(make_row(
                "selection", "randomized_select", LAS_VEGAS, n, param,
                **trial_fields(times, comps, depths),
                expected_work=expected_quickselect_comparisons(n, k),
                work_unit="comparisons", success_rate=within))

            # deterministic: counts are fixed, a few runs time it
            stats = PartitionStats()
            assert median_of_medians_select(arr, k, stats) == k - 1
            mom_times = [timed(lambda: median_of_medians_select(arr, k))[0]
                         for _ in range(3)]
            rows.append(make_row(
                "selection", "median_of_medians", DETERMINISTIC, n, param,
                **trial_fields(mom_times),
                mean_work=stats.comparisons, std_work=0.0,
                mean_depth=stats.max_depth, work_unit="comparisons"))

            sort_times = [timed(lambda: sort_select(arr, k))[0]
                          for _ in range(3)]
            rows.append(make_row("selection", "sort_select", DETERMINISTIC,
                                 n, param, **trial_fields(sort_times)))

    if samples is not None and dist_trials:
        arr = random_array(dist_n, "random", seed=SEED + dist_n)
        k = (dist_n + 1) // 2
        comps, depths = [], []
        for t in range(dist_trials):
            stats = PartitionStats()
            randomized_select(arr, k, stats, random.Random(9_000 + t))
            comps.append(stats.comparisons)
            depths.append(stats.max_depth)
        samples["sel_dist"] = {"n": dist_n, "comparisons": comps,
                               "depths": depths}
        for c in SEL_BUDGETS:
            rows.append(make_row(
                "selection_budget", "randomized_select", LAS_VEGAS, dist_n,
                f"c={c}", trials=dist_trials,
                expected_work=expected_quickselect_comparisons(dist_n, k),
                work_unit="comparisons",
                success_rate=sum(x <= c * dist_n for x in comps)
                / dist_trials))
    return rows


# ---------------------------------------------------------------------------
# 3. Primality
# ---------------------------------------------------------------------------

def next_prime(start: int) -> int:
    """Smallest prime >= start (Miller–Rabin with 40 seeded rounds;
    error < 1e-24)."""
    n = start | 1
    rng = random.Random(SEED)
    while not miller_rabin(n, 40, rng):
        n += 2
    return n


def odd_composites(limit: int) -> List[int]:
    return [n for n in range(9, limit + 1, 2)
            if not is_prime_trial_division(n)]


def bench_primality(digits_trial: Sequence[int] = PRIME_DIGITS_TRIAL,
                    digits_mc: Sequence[int] = PRIME_DIGITS_MC,
                    rounds: Sequence[int] = ROUNDS,
                    carmichael: Sequence[int] = CARMICHAEL,
                    carmichael_trials: int = CARMICHAEL_TRIALS,
                    liar_limit: int = LIAR_LIMIT,
                    samples: Optional[Dict] = None) -> List[Dict]:
    print("\n=== Primality: Fermat & Miller–Rabin vs trial division ===")
    rows = []

    # Speed on primes (the slow case: every round runs)
    for d in sorted(set(digits_trial) | set(digits_mc)):
        p = next_prime(10 ** (d - 1) + 1)
        param = f"{d} digits"
        if d in digits_trial:
            assert is_prime_trial_division(p)
            seconds = time_call(lambda: is_prime_trial_division(p),
                                min_time=0.2)
            rows.append(make_row("primality_speed", "trial_division",
                                 DETERMINISTIC, d, param,
                                 mean_time_s=seconds,
                                 mean_work=math.isqrt(p) / 3,
                                 work_unit="divisions"))
        if d in digits_mc:
            for k in [5, 10, 20]:
                for name, test in [("miller_rabin", miller_rabin),
                                   ("fermat", fermat_test)]:
                    if name == "fermat" and k != 20:
                        continue
                    rng = random.Random(SEED)
                    seconds = time_call(lambda: test(p, k, rng),
                                        min_time=0.2)
                    rows.append(make_row("primality_speed", name,
                                         MONTE_CARLO, d, f"k={k}",
                                         mean_time_s=seconds, mean_work=k,
                                         work_unit="rounds"))

    # Error rates on Carmichael numbers (the hardest composites for Fermat)
    fermat_liars = {n: liar_fraction(n, "fermat") for n in carmichael}
    mr_liars = {n: liar_fraction(n, "miller_rabin") for n in carmichael}
    for k in rounds:
        for name, test, liars in [("fermat", fermat_test, fermat_liars),
                                  ("miller_rabin", miller_rabin, mr_liars)]:
            rng = random.Random(SEED + k)
            errors = sum(test(n, k, rng) for n in carmichael
                         for _ in range(carmichael_trials))
            total = len(carmichael) * carmichael_trials
            predicted = sum(f ** k for f in liars.values()) / len(liars)
            rows.append(make_row(
                "primality_error", name, MONTE_CARLO, len(carmichael),
                f"k={k}", trials=total, error_metric="false_prime_rate",
                error=errors / total, error_theory=predicted,
                error_ci_high=wilson_interval(errors, total)[1],
                expected_work=(miller_rabin_error_bound(k)
                               if name == "miller_rabin" else None),
                work_unit="4^-k bound" if name == "miller_rabin" else None))

    # Exact per-round liar fraction for every small odd composite
    composites = odd_composites(liar_limit)
    fermat_frac = [liar_fraction(n, "fermat") for n in composites]
    mr_frac = [liar_fraction(n, "miller_rabin") for n in composites]
    for name, fracs in [("fermat", fermat_frac), ("miller_rabin", mr_frac)]:
        worst = max(range(len(fracs)), key=fracs.__getitem__)
        rows.append(make_row(
            "primality_liars", name, MONTE_CARLO, liar_limit,
            f"worst n={composites[worst]}", trials=len(composites),
            error_metric="max_liar_fraction", error=fracs[worst],
            error_theory=0.25 if name == "miller_rabin" else None))
    if samples is not None:
        samples["liars"] = {"n": composites, "fermat": fermat_frac,
                            "miller_rabin": mr_frac,
                            "carmichael_fermat": fermat_liars}
    return rows


# ---------------------------------------------------------------------------
# 4. Bloom filter
# ---------------------------------------------------------------------------

def bloom_trial(num_bits: int, num_hashes: int, num_items: int,
                probes: int, seed: int) -> Tuple[float, BloomFilter]:
    """Insert num_items members, probe `probes` non-members; returns
    (measured false-positive rate, filter)."""
    bf = BloomFilter(num_bits, num_hashes, seed=seed)
    bf.update(f"member-{i}" for i in range(num_items))
    false_pos = sum(bf.check(f"probe-{i}") for i in range(probes))
    return false_pos / probes, bf


def set_memory(num_items: int) -> int:
    """Peak bytes to build a Python set of the same member strings."""
    return peak_memory(
        lambda: {f"member-{i}" for i in range(num_items)})[0]


def bench_bloom(loads: Sequence[float] = BLOOM_LOADS,
                load_ks: Sequence[int] = BLOOM_LOAD_KS,
                num_bits: int = BLOOM_BITS,
                num_items: int = BLOOM_N,
                bits_per_item: Sequence[int] = BLOOM_BITS_PER_ITEM,
                k_sweep: Sequence[int] = BLOOM_K_SWEEP,
                probes: int = BLOOM_PROBES) -> List[Dict]:
    print("\n=== Bloom filter: false positives vs load, memory, k ===")
    rows = []

    def add(experiment, parameter, m, k, n):
        start = time.perf_counter()
        fp, bf = bloom_trial(m, k, n, probes, seed=SEED)
        elapsed = time.perf_counter() - start
        lo, hi = wilson_interval(round(fp * probes), probes)
        rows.append(make_row(
            experiment, "bloom_filter", MONTE_CARLO, n, parameter,
            trials=probes, mean_time_s=elapsed / (n + probes),
            mean_work=bf.memory_bytes(), work_unit="bytes",
            error_metric="false_positive_rate", error=fp,
            error_theory=bloom_false_positive_rate(m, n, k),
            error_ci_high=hi))

    for k in load_ks:
        for load in loads:
            n = round(load * num_bits)
            add("bloom_load", f"k={k} load={load}", num_bits, k, n)
    for b in bits_per_item:
        m = b * num_items
        k = max(1, round(b * math.log(2)))
        add("bloom_memory", f"{b} bits/item k={k}", m, k, num_items)
    m = BLOOM_K_SWEEP_BPI * num_items
    for k in k_sweep:
        add("bloom_k", f"k={k}", m, k, num_items)

    rows.append(make_row("bloom_memory", "python_set", DETERMINISTIC,
                         num_items, "exact", mean_work=set_memory(num_items),
                         work_unit="bytes", error_metric="false_positive_rate",
                         error=0.0, error_theory=0.0))
    return rows


# ---------------------------------------------------------------------------
# 5. MinHash
# ---------------------------------------------------------------------------

def mutate_text(text: str, rate: float, seed: int) -> str:
    """Replace each word with probability `rate` by a random filler word."""
    rng = random.Random(seed)
    filler = ["data", "set", "hash", "value", "model", "system", "random",
              "fast", "large", "small", "query", "table", "graph", "node"]
    return " ".join(rng.choice(filler) if rng.random() < rate else w
                    for w in text.split())


def text_corpus() -> List[Tuple[str, str]]:
    """(name, text) for every base paragraph and its mutated copies."""
    docs = []
    for b, base in enumerate(BASE_TEXTS):
        for r, rate in enumerate(TEXT_MUTATION_RATES):
            docs.append((f"doc{b}_{int(rate * 100)}",
                         mutate_text(base, rate, seed=SEED + 10 * b + r)))
    return docs


def bench_minhash(ks: Sequence[int] = MINHASH_KS,
                  jaccards: Sequence[float] = MINHASH_JS,
                  set_size: int = MINHASH_SET_SIZE,
                  trials: int = MINHASH_TRIALS,
                  text_ks: Sequence[int] = TEXT_KS,
                  samples: Optional[Dict] = None) -> List[Dict]:
    print("\n=== MinHash: estimation error vs signature size ===")
    rows = []
    k_max = max(ks)
    estimates: Dict[Tuple[float, int], List[float]] = {}
    for target in jaccards:
        a, b = random_set_pair(set_size, target, seed=SEED)
        true_j = jaccard_similarity(a, b)
        for t in range(trials):
            mh = MinHash(k_max, seed=1_000 * t + int(target * 100))
            sig_a, sig_b = mh.signature(a), mh.signature(b)
            # the first k hashes of a k_max family are a valid k-family
            for k in ks:
                estimates.setdefault((true_j, k), []).append(
                    MinHash.similarity(sig_a[:k], sig_b[:k]))
        for k in ks:
            errs = [e - true_j for e in estimates[(true_j, k)]]
            rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
            rows.append(make_row(
                "minhash_synthetic", "minhash", MONTE_CARLO, set_size,
                f"J={true_j:.3f} k={k}", trials=trials,
                mean_work=sum(estimates[(true_j, k)]) / trials,
                expected_work=true_j, work_unit="estimated J",
                error_metric="rmse", error=rmse,
                error_theory=minhash_standard_error(true_j, k)))

    # Timing: signature cost grows with k·|S|; comparison only with k
    a, b = random_set_pair(set_size, 0.5, seed=SEED)
    exact_time = time_call(lambda: jaccard_similarity(a, b), min_time=0.1)
    rows.append(make_row("minhash_time", "exact_jaccard", DETERMINISTIC,
                         set_size, "", mean_time_s=exact_time))
    for k in ks:
        mh = MinHash(k, seed=SEED)
        sig_a, sig_b = mh.signature(a), mh.signature(b)
        rows.append(make_row(
            "minhash_time", "minhash_signature", MONTE_CARLO, set_size,
            f"k={k}", mean_time_s=time_call(lambda: mh.signature(a),
                                            min_time=0.1)))
        rows.append(make_row(
            "minhash_time", "minhash_compare", MONTE_CARLO, set_size,
            f"k={k}", mean_time_s=time_call(
                lambda: MinHash.similarity(sig_a, sig_b), min_time=0.1)))

    # Near-duplicate text: every pair of documents
    docs = text_corpus()
    sets = {name: shingles(text, TEXT_SHINGLE) for name, text in docs}
    names = [name for name, _ in docs]
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
    truth = [jaccard_similarity(sets[x], sets[y]) for x, y in pairs]
    text_points = {}
    for k in text_ks:
        mh = MinHash(k, seed=SEED)
        sigs = {name: mh.signature(s) for name, s in sets.items()}
        est = [MinHash.similarity(sigs[x], sigs[y]) for x, y in pairs]
        text_points[k] = est
        errs = [e - j for e, j in zip(est, truth)]
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        theory = math.sqrt(sum(j * (1 - j) for j in truth)
                           / len(truth) / k)
        rows.append(make_row(
            "minhash_text", "minhash", MONTE_CARLO, len(pairs),
            f"k={k}", trials=len(pairs), error_metric="rmse", error=rmse,
            error_theory=theory))

    if samples is not None:
        samples["minhash"] = {"estimates": estimates, "truth": truth,
                              "text": text_points}
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _select(rows, experiment, method, field="mean_time_s",
            parameter: Optional[str] = None):
    pts = sorted((r["input_size"], r[field]) for r in rows
                 if r["experiment"] == experiment and r["method"] == method
                 and (parameter is None or r["parameter"] == parameter))
    return [p[0] for p in pts], [p[1] for p in pts]


def _band(rows, experiment, method, mean_field, std_field):
    pts = sorted((r["input_size"], r[mean_field], r[std_field])
                 for r in rows if r["experiment"] == experiment
                 and r["method"] == method and r[std_field] is not None)
    return ([p[0] for p in pts], [p[1] - p[2] for p in pts],
            [p[1] + p[2] for p in pts])


def _ms(series):
    xs, ys = series
    return xs, [y * 1e3 if y is not None else None for y in ys]


def _ms_band(band):
    xs, lows, highs = band
    return xs, [v * 1e3 for v in lows], [v * 1e3 for v in highs]


def _param_value(parameter: str, key: str) -> float:
    """'k=7 load=0.1' -> value of key."""
    for part in parameter.split():
        if part.startswith(key + "="):
            return float(part.split("=", 1)[1])
    raise KeyError(key)


def plot_quicksort(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    exp = "quicksort_random"
    names = {"randomized_quicksort": "randomized pivot",
             "deterministic_quicksort": "first-element pivot",
             "week2_quick_sort": "Week 2 quick_sort",
             "python_sorted": "sorted() (Timsort, C)"}
    colors = {"randomized pivot": "tab:green",
              "first-element pivot": "tab:red",
              "Week 2 quick_sort": "tab:blue",
              "sorted() (Timsort, C)": "tab:gray"}
    sizes = _select(rows, exp, "randomized_quicksort")[0]
    comps = _select(rows, exp, "randomized_quicksort", "mean_work")
    worst = _select(rows, "quicksort_sorted", "deterministic_quicksort")
    rand_sorted = _select(rows, "quicksort_sorted", "randomized_quicksort")
    ratio = [(n, c / expected_quicksort_comparisons(n))
             for n, c in zip(*comps)]
    cv = [(r["input_size"], r["std_work"] / r["mean_work"],
           r["std_time_s"] / r["mean_time_s"]) for r in rows
          if r["experiment"] == exp
          and r["method"] == "randomized_quicksort"]
    cv.sort()
    panels = [
        {"title": "Runtime on random input (mean ± std band)",
         "xlabel": "n", "ylabel": "Time (ms)", "colors": colors,
         "series": {names[m]: _ms(_select(rows, exp, m)) for m in names},
         "bands": {"randomized pivot": _ms_band(
             _band(rows, exp, "randomized_quicksort", "mean_time_s",
                   "std_time_s"))},
         "logx": True, "logy": True},
        {"title": "Sorted input: the first-pivot worst case",
         "xlabel": "n", "ylabel": "Time (ms)", "colors": colors,
         "series": {"first-element pivot": _ms(worst),
                    "randomized pivot": _ms(rand_sorted)},
         "refs": ({"c·n²": (worst[0], [worst[1][-1] * 1e3 * (n / worst[0][-1])
                                       ** 2 for n in worst[0]])}
                  if worst[0] else {}),
         "logx": True, "logy": True},
        {"title": "Comparisons: measured ÷ 2(n+1)Hₙ − 4n",
         "xlabel": "n", "ylabel": "Measured / expected",
         "series": {"randomized pivot (mean of trials)":
                    ([p[0] for p in ratio], [p[1] for p in ratio])},
         "refs": {"expected = 1": (sizes, [1.0] * len(sizes))},
         "colors": {"randomized pivot (mean of trials)": "tab:green"},
         "logx": True},
        {"title": "Run-to-run spread (same input, new seeds)",
         "xlabel": "n", "ylabel": "Coefficient of variation (std / mean)",
         "series": {"comparisons": ([c[0] for c in cv], [c[1] for c in cv]),
                    "runtime": ([c[0] for c in cv], [c[2] for c in cv])},
         "refs": {"0.648·n / E[C] (theory, comparisons)": (
             [c[0] for c in cv],
             [0.6482 * c[0] / expected_quicksort_comparisons(c[0])
              for c in cv])},
         "logx": True, "logy": True},
    ]
    dist = samples.get("qs_dist")
    if dist:
        n = dist["n"]
        panels.append(
            {"kind": "hist",
             "title": f"Comparison counts, n = {n:,}, "
                      f"{len(dist['comparisons'])} seeds",
             "xlabel": "Comparisons", "ylabel": "Runs",
             "samples": {"randomized pivot": dist["comparisons"]},
             "vlines": {"expected 2(n+1)Hₙ − 4n":
                        expected_quicksort_comparisons(n)},
             "colors": {"randomized pivot": "tab:green",
                        "expected 2(n+1)Hₙ − 4n": "black"}})
        depths = dist["depths"]
        panels.append(
            {"kind": "hist",
             "title": f"Recursion depth, n = {n:,} (first pivot on "
                      f"sorted input: {n:,})",
             "xlabel": "Max recursion depth", "ylabel": "Runs",
             "samples": {"randomized pivot": depths},
             "vlines": {"log₂ n (perfect splits)": math.log2(n)},
             "bins": max(5, max(depths) - min(depths) + 1),
             "colors": {"randomized pivot": "tab:green",
                        "log₂ n (perfect splits)": "black"}})
    plot_mixed_panels(panels,
                      os.path.join(results_dir, "quicksort_comparison.png"),
                      "Randomized vs deterministic QuickSort")


def plot_selection(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    def sel(method, label, field="mean_time_s"):
        return _select(rows, "selection", method, field, f"k={label}")

    colors = {"QuickSelect min": "tab:cyan", "QuickSelect median": "tab:green",
              "QuickSelect max": "tab:olive",
              "median-of-medians (median)": "tab:red",
              "sort + index (median)": "tab:gray"}
    runtime = {f"QuickSelect {label}": _ms(sel("randomized_select", label))
               for label in ["min", "median", "max"]}
    runtime["median-of-medians (median)"] = _ms(sel("median_of_medians",
                                                    "median"))
    runtime["sort + index (median)"] = _ms(sel("sort_select", "median"))

    per_n = {}
    refs = {}
    for label in ["min", "median"]:
        xs, ys = sel("randomized_select", label, "mean_work")
        per_n[f"QuickSelect {label}"] = (xs, [y / x for x, y in zip(xs, ys)])
        refs[f"expected ({label})"] = (xs, [
            expected_quickselect_comparisons(x, k_values(x)[label]) / x
            for x in xs])
    xs, ys = sel("median_of_medians", "median", "mean_work")
    per_n["median-of-medians (median)"] = (xs, [y / x for x, y in
                                                zip(xs, ys)])
    colors.update({"expected (min)": "tab:cyan",
                   "expected (median)": "tab:green"})

    depth = {f"QuickSelect {label}": sel("randomized_select", label,
                                         "mean_depth")
             for label in ["min", "median", "max"]}
    depth["median-of-medians (median)"] = sel("median_of_medians", "median",
                                              "mean_depth")
    xs = depth["QuickSelect min"][0]
    panels = [
        {"title": "Runtime (mean of trials)", "xlabel": "n",
         "ylabel": "Time (ms)", "series": runtime, "colors": colors,
         "logx": True, "logy": True},
        {"title": "Comparisons per element: linear for every k",
         "xlabel": "n", "ylabel": "Comparisons / n", "series": per_n,
         "refs": refs, "colors": colors, "logx": True},
        {"title": "Average recursion depth", "xlabel": "n",
         "ylabel": "Depth (partition rounds)", "series": depth,
         "refs": {"Hₙ ≈ ln n (expected, min)": (xs, [harmonic(x)
                                                     for x in xs])},
         "colors": colors, "logx": True},
    ]
    budget = [(float(r["parameter"][2:]), r["success_rate"]) for r in rows
              if r["experiment"] == "selection_budget"]
    dist = samples.get("sel_dist")
    if budget and dist:
        n = dist["n"]
        mean_c = sum(dist["comparisons"]) / len(dist["comparisons"]) / n
        cs = [b[0] for b in budget]
        panels.append(
            {"title": f"Success probability: P(comparisons ≤ c·n), "
                      f"median, n = {n:,}",
             "xlabel": "Budget c (comparisons / n)",
             "ylabel": "Fraction of runs within budget",
             "series": {"measured": (cs, [b[1] for b in budget])},
             "refs": {"Markov bound 1 − E/(c·n)": (
                 cs, [max(0.0, 1 - mean_c / c) for c in cs])},
             "colors": {"measured": "tab:green"}})
    plot_mixed_panels(panels,
                      os.path.join(results_dir, "selection_runtime.png"),
                      "Randomized QuickSelect vs deterministic selection")


def plot_primality(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    def err(method, field):
        pts = sorted((int(r["parameter"][2:]), r[field]) for r in rows
                     if r["experiment"] == "primality_error"
                     and r["method"] == method)
        return [p[0] for p in pts], [p[1] for p in pts]

    ks = err("fermat", "error")[0]
    colors = {"Fermat measured": "tab:red", "Fermat predicted": "tab:red",
              "Miller–Rabin measured": "tab:green",
              "Miller–Rabin 95% upper bound": "tab:green",
              "4⁻ᵏ bound": "black",
              "trial division": "tab:gray", "Fermat k=20": "tab:red",
              "Miller–Rabin k=5": "tab:olive",
              "Miller–Rabin k=10": "tab:cyan",
              "Miller–Rabin k=20": "tab:green"}
    panels = [
        {"title": "False 'prime' rate on 16 Carmichael numbers",
         "xlabel": "Rounds k", "ylabel": "Error rate",
         "series": {"Fermat measured": err("fermat", "error"),
                    "Miller–Rabin measured": err("miller_rabin", "error")},
         "refs": {"Fermat predicted": err("fermat", "error_theory"),
                  "Miller–Rabin 95% upper bound":
                      err("miller_rabin", "error_ci_high"),
                  "4⁻ᵏ bound": (ks, [miller_rabin_error_bound(k)
                                     for k in ks])},
         "colors": colors, "logy": True},
    ]
    liars = samples.get("liars")
    if liars:
        panels.append(
            {"kind": "scatter",
             "title": f"Exact one-round liar fraction, odd composites "
                      f"≤ {max(liars['n']):,}",
             "xlabel": "n", "ylabel": "Fraction of bases that lie",
             "points": {"Fermat": (liars["n"], liars["fermat"]),
                        "Miller–Rabin": (liars["n"],
                                         liars["miller_rabin"])},
             "refs": {"1/4 (Rabin's bound)": ([9, max(liars["n"])],
                                              [0.25, 0.25])},
             "colors": {"Fermat": "tab:red", "Miller–Rabin": "tab:green",
                        "1/4 (Rabin's bound)": "black"}})

    def speed(method, parameter=None):
        pts = sorted((r["input_size"], r["mean_time_s"] * 1e6) for r in rows
                     if r["experiment"] == "primality_speed"
                     and r["method"] == method
                     and (parameter is None or r["parameter"] == parameter))
        return [p[0] for p in pts], [p[1] for p in pts]

    speed_series = {"trial division": speed("trial_division"),
                    "Fermat k=20": speed("fermat", "k=20")}
    for k in [5, 10, 20]:
        speed_series[f"Miller–Rabin k={k}"] = speed("miller_rabin", f"k={k}")
    panels.append(
        {"title": "Time to certify a prime", "xlabel": "Digits in n",
         "ylabel": "Time (µs)", "series": speed_series, "colors": colors,
         "logy": True})
    td = dict(zip(*speed("trial_division")))
    mr = dict(zip(*speed("miller_rabin", "k=20")))
    common = sorted(set(td) & set(mr))
    panels.append(
        {"title": "Speedup of Miller–Rabin (k=20) over trial division",
         "xlabel": "Digits in n", "ylabel": "Trial division ÷ Miller–Rabin",
         "series": {"speedup": (common, [td[d] / mr[d] for d in common])},
         "refs": {"break-even": (common, [1.0] * len(common))},
         "colors": {"speedup": "tab:green"}, "logy": True})
    plot_mixed_panels(panels,
                      os.path.join(results_dir, "primality_accuracy.png"),
                      "Monte Carlo primality: Fermat vs Miller–Rabin")


def plot_bloom(rows: List[Dict], results_dir: str) -> None:
    def bloom(experiment, key, field, k=None):
        pts = sorted((_param_value(r["parameter"], key), r[field])
                     for r in rows if r["experiment"] == experiment
                     and r["method"] == "bloom_filter"
                     and (k is None or _param_value(r["parameter"], "k") == k))
        return [p[0] for p in pts], [p[1] for p in pts]

    load_series, load_refs, colors = {}, {}, {}
    for k, color in zip(BLOOM_LOAD_KS, ["tab:blue", "tab:purple"]):
        load_series[f"k={k} measured"] = bloom("bloom_load", "load",
                                               "error", k)
        load_refs[f"k={k} theory"] = bloom("bloom_load", "load",
                                           "error_theory", k)
        colors[f"k={k} measured"] = colors[f"k={k} theory"] = color

    mem_rows = sorted((r for r in rows if r["experiment"] == "bloom_memory"
                       and r["method"] == "bloom_filter"),
                      key=lambda r: r["mean_work"])
    bpi = [float(r["parameter"].split()[0]) for r in mem_rows]
    set_row = next((r for r in rows if r["method"] == "python_set"), None)
    n_items = mem_rows[0]["input_size"] if mem_rows else BLOOM_N

    k_x, k_measured = bloom("bloom_k", "k", "error")
    panels = [
        {"title": "False positives vs load factor (m = 65,536 bits)",
         "xlabel": "Load factor n / m (items per bit)",
         "ylabel": "False-positive rate", "series": load_series,
         "refs": load_refs, "colors": colors, "logy": True},
        {"title": f"Memory vs accuracy (n = {n_items:,}, optimal k)",
         "xlabel": "Bits per item (m / n)",
         "ylabel": "False-positive rate",
         "series": {"measured": (bpi, [r["error"] for r in mem_rows])},
         "refs": {"(1 − e^(−kn/m))^k": (bpi, [r["error_theory"]
                                              for r in mem_rows])},
         "colors": {"measured": "tab:blue"}, "logy": True},
        {"title": f"Number of hash functions ({BLOOM_K_SWEEP_BPI} bits/item)",
         "xlabel": "k", "ylabel": "False-positive rate",
         "series": {"measured": (k_x, k_measured)},
         "refs": {"theory": bloom("bloom_k", "k", "error_theory"),
                  f"optimal k = {BLOOM_K_SWEEP_BPI} ln 2 ≈ "
                  f"{BLOOM_K_SWEEP_BPI * math.log(2):.1f}": (
                      [BLOOM_K_SWEEP_BPI * math.log(2)] * 2,
                      [min(k_measured or [1e-3]) / 2,
                       max(k_measured or [1]) * 2])},
         "colors": {"measured": "tab:blue"}, "logy": True},
    ]
    if set_row and mem_rows:
        set_kib = set_row["mean_work"] / 1024
        panels.append(
            {"title": f"Memory for {n_items:,} items: Bloom filter vs set",
             "xlabel": "Bits per item (m / n)", "ylabel": "Memory (KiB)",
             "series": {"Bloom filter": (bpi, [r["mean_work"] / 1024
                                               for r in mem_rows])},
             "refs": {f"Python set of the strings ({set_kib:,.0f} KiB, "
                      "exact)": (bpi, [set_kib] * len(bpi))},
             "colors": {"Bloom filter": "tab:blue"}, "logy": True})
    plot_mixed_panels(panels,
                      os.path.join(results_dir, "bloom_filter_fp_rate.png"),
                      "Bloom filter: false positives vs memory")


def plot_minhash(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    syn = [r for r in rows if r["experiment"] == "minhash_synthetic"]
    js = sorted({_param_value(r["parameter"], "J") for r in syn})
    series, refs, colors = {}, {}, {}
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red",
               "tab:purple"]
    for j, color in zip(js, palette):
        pts = sorted((_param_value(r["parameter"], "k"), r["error"],
                      r["error_theory"]) for r in syn
                     if _param_value(r["parameter"], "J") == j)
        label = f"J = {j:.2f}"
        series[label] = ([p[0] for p in pts], [p[1] for p in pts])
        refs[f"√(J(1−J)/k), J = {j:.2f}"] = ([p[0] for p in pts],
                                             [p[2] for p in pts])
        colors[label] = colors[f"√(J(1−J)/k), J = {j:.2f}"] = color
    panels = [
        {"title": f"RMSE vs signature size ({MINHASH_SET_SIZE}-element "
                  "sets)",
         "xlabel": "Hash functions k", "ylabel": "RMSE of estimate",
         "series": series, "refs": refs, "colors": colors,
         "logx": True, "logy": True},
    ]
    mh = samples.get("minhash")
    if mh:
        truth = mh["truth"]
        text_colors = {16: "tab:red", 128: "tab:orange", 512: "tab:green"}
        panels.append(
            {"kind": "scatter",
             "title": f"Near-duplicate text: {len(truth)} document pairs",
             "xlabel": "True Jaccard (word 2-shingles)",
             "ylabel": "MinHash estimate",
             "points": {f"k = {k}": (truth, est)
                        for k, est in mh["text"].items()},
             "refs": {"y = x": ([0, 1], [0, 1])},
             "colors": {f"k = {k}": text_colors.get(k)
                        for k in mh["text"]}})
        est = mh["estimates"]
        mid = min({j for j, _ in est}, key=lambda j: abs(j - 0.5))
        lo_k, hi_k = min(MINHASH_KS), max(MINHASH_KS)
        if (mid, lo_k) in est and (mid, hi_k) in est:
            panels.append(
                {"kind": "hist",
                 "title": f"Estimates of J = {mid:.3f} across "
                          f"{len(est[(mid, lo_k)])} hash families",
                 "xlabel": "Estimated J", "ylabel": "Trials",
                 "samples": {f"k = {lo_k}": est[(mid, lo_k)],
                             f"k = {hi_k}": est[(mid, hi_k)]},
                 "vlines": {"true J": mid}, "bins": 15,
                 "colors": {f"k = {lo_k}": "tab:red",
                            f"k = {hi_k}": "tab:green", "true J": "black"}})
    timing = [r for r in rows if r["experiment"] == "minhash_time"]
    exact = next((r["mean_time_s"] for r in timing
                  if r["method"] == "exact_jaccard"), None)

    def t(method):
        pts = sorted((_param_value(r["parameter"], "k"),
                      r["mean_time_s"] * 1e3) for r in timing
                     if r["method"] == method)
        return [p[0] for p in pts], [p[1] for p in pts]

    sig = t("minhash_signature")
    panels.append(
        {"title": "Cost: signature O(k·|S|) once, compare O(k)",
         "xlabel": "Hash functions k", "ylabel": "Time (ms)",
         "series": {"build one signature": sig,
                    "compare two signatures": t("minhash_compare")},
         "refs": ({"exact Jaccard of two sets": (sig[0], [exact * 1e3]
                                                 * len(sig[0]))}
                  if exact and sig[0] else {}),
         "logx": True, "logy": True})
    plot_mixed_panels(panels,
                      os.path.join(results_dir, "minhash_similarity_plot.png"),
                      "MinHash: Jaccard estimation error vs signature size")


def make_plots(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    plot_quicksort(rows, samples, results_dir)
    plot_selection(rows, samples, results_dir)
    plot_primality(rows, samples, results_dir)
    plot_bloom(rows, results_dir)
    plot_minhash(rows, samples, results_dir)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r[k]) for k in FIELDS})


def print_summary(rows: List[Dict]) -> None:
    """Condensed console tables for the headline comparisons."""
    print("\n--- QuickSort on random input (mean ms ± std | comparisons / "
          "expected) ---")
    for r in rows:
        if r["experiment"] in ("quicksort_random", "quicksort_sorted"):
            ratio = (f"{r['mean_work'] / r['expected_work']:.3f}"
                     if r["mean_work"] and r["expected_work"] else "—")
            print(f"  {r['experiment']:17s} {r['method']:24s} "
                  f"n={r['input_size']:<7} {r['mean_time_s'] * 1e3:10.3f} ± "
                  f"{r['std_time_s'] * 1e3:7.3f} ms   {ratio}")
    print("\n--- Selection (mean ms | comparisons / n | depth) ---")
    for r in rows:
        if r["experiment"] == "selection":
            per_n = (f"{r['mean_work'] / r['input_size']:.2f}"
                     if r["mean_work"] else "—")
            depth = f"{r['mean_depth']:.1f}" if r["mean_depth"] else "—"
            print(f"  {r['method']:20s} {r['parameter']:9s} "
                  f"n={r['input_size']:<7} {r['mean_time_s'] * 1e3:10.3f} ms "
                  f"{per_n:>6} {depth:>6}")
    print("\n--- Error rates (measured vs theory) ---")
    for r in rows:
        if r["error"] is not None:
            theory = (f"{r['error_theory']:.3g}"
                      if r["error_theory"] is not None else "—")
            print(f"  {r['experiment']:18s} {r['method']:14s} "
                  f"{r['parameter']:22s} {r['error_metric']:20s} "
                  f"{r['error']:.4g} (theory {theory})")


def run(results_dir: str = RESULTS_DIR, quick: bool = False) -> List[Dict]:
    os.makedirs(results_dir, exist_ok=True)
    samples: Dict = {}
    rows: List[Dict] = []
    sizes = [n for n in QS_SIZES if not quick or n <= 30_000]
    rows += bench_quicksort(
        sizes, trials=10 if quick else QS_TRIALS,
        worst_sizes=[n for n in QS_WORST_SIZES if not quick or n <= 2_000],
        dist_trials=100 if quick else QS_DIST_TRIALS, samples=samples)
    rows += bench_selection(sizes, trials=10 if quick else SEL_TRIALS,
                            dist_trials=200 if quick else SEL_DIST_TRIALS,
                            samples=samples)
    rows += bench_primality(
        digits_trial=[d for d in PRIME_DIGITS_TRIAL if not quick or d <= 12],
        carmichael_trials=500 if quick else CARMICHAEL_TRIALS,
        liar_limit=1_500 if quick else LIAR_LIMIT, samples=samples)
    rows += bench_bloom(probes=20_000 if quick else BLOOM_PROBES)
    rows += bench_minhash(trials=8 if quick else MINHASH_TRIALS,
                          samples=samples)

    csv_path = os.path.join(results_dir, CSV_NAME)
    write_csv(rows, csv_path)
    print(f"\nSaved {csv_path} ({len(rows)} rows)")
    print_summary(rows)
    make_plots(rows, samples, results_dir)
    print(f"\nSaved plots to {results_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true",
                        help="smaller sizes and fewer trials")
    args = parser.parse_args()
    start = time.perf_counter()
    run(quick=args.quick)
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
