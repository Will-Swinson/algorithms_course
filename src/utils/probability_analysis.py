"""
Theoretical predictions and summary statistics for Week 7.

Every randomized algorithm this week comes with a formula for how it
should behave on average: expected comparisons for QuickSort and
QuickSelect, an error bound for Miller–Rabin, a false-positive rate for
a Bloom filter, a standard error for MinHash. The benchmark measures
the empirical version of each and plots it against the formula here.

Also: summarize() for mean/variance/confidence intervals of repeated
trials, and wilson_interval() for confidence bounds on an observed
error rate — including the case where zero errors were seen, which is
what Miller–Rabin produces in practice.
"""
import math
from dataclasses import dataclass
from typing import Sequence, Tuple


def harmonic(n: int) -> float:
    """H_n = 1 + 1/2 + ... + 1/n  (H_0 = 0)."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n < 1000:
        return sum(1.0 / i for i in range(1, n + 1))
    # Asymptotic expansion; error < 1e-15 for n >= 1000
    return (math.log(n) + 0.5772156649015329 + 1 / (2 * n)
            - 1 / (12 * n ** 2) + 1 / (120 * n ** 4))


# ---------------------------------------------------------------------------
# Las Vegas algorithms: expected comparisons
# ---------------------------------------------------------------------------

def expected_quicksort_comparisons(n: int) -> float:
    """
    Exact expected comparisons of randomized QuickSort on n distinct
    keys, when partitioning m elements costs m − 1 comparisons:

        E[C_n] = 2(n + 1)·H_n − 4n  ≈  1.386·n·log₂n

    Derivation: elements i < j (in sorted order) are compared exactly
    when the first pivot chosen from {i, ..., j} is i or j, which
    happens with probability 2 / (j − i + 1). Summing over all pairs
    gives the formula.
    """
    if n <= 1:
        return 0.0
    return 2 * (n + 1) * harmonic(n) - 4 * n


def expected_quickselect_comparisons(n: int, k: int) -> float:
    """
    Exact expected comparisons of randomized QuickSelect finding the
    k-th smallest (1-indexed) of n distinct keys (Knuth, TAOCP 5.2.2):

        E[C_{n,k}] = 2·[(n+1)H_n − (n+3−k)H_{n+1−k} − (k+2)H_k + n + 3]

    Special cases: ≈ 2n for k = 1 or k = n, and ≈ 2(1 + ln 2)·n
    ≈ 3.386n for the median — linear in every case.
    """
    if not 1 <= k <= max(n, 1):
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if n <= 1:
        return 0.0
    return 2 * ((n + 1) * harmonic(n) - (n + 3 - k) * harmonic(n + 1 - k)
                - (k + 2) * harmonic(k) + n + 3)


# ---------------------------------------------------------------------------
# Monte Carlo algorithms: error bounds
# ---------------------------------------------------------------------------

def miller_rabin_error_bound(rounds: int) -> float:
    """
    Worst-case probability that Miller–Rabin calls a composite prime:
    at most 1/4 of bases are strong liars, so ≤ 4^(−rounds).
    """
    return 0.25 ** rounds


def bloom_false_positive_rate(num_bits: int, num_items: int,
                              num_hashes: int) -> float:
    """
    Standard approximation of a Bloom filter's false-positive rate:

        p ≈ (1 − e^(−k·n/m))^k

    e^(−kn/m) is the chance one particular bit is still 0 after n
    insertions of k hashes each; a false positive needs all k probed
    bits to be 1.
    """
    if num_bits <= 0 or num_hashes <= 0:
        raise ValueError("num_bits and num_hashes must be positive")
    return (1 - math.exp(-num_hashes * num_items / num_bits)) ** num_hashes


def optimal_bloom_parameters(num_items: int,
                             fp_rate: float) -> Tuple[int, int]:
    """
    (num_bits, num_hashes) minimizing memory for a target FP rate:

        m = −n·ln p / (ln 2)²       k = (m / n)·ln 2

    About 9.6 bits per item buys a 1% false-positive rate, independent
    of how large the items themselves are.
    """
    if num_items <= 0:
        raise ValueError(f"num_items must be positive, got {num_items}")
    if not 0 < fp_rate < 1:
        raise ValueError(f"fp_rate must be in (0, 1), got {fp_rate}")
    m = math.ceil(-num_items * math.log(fp_rate) / math.log(2) ** 2)
    k = max(1, round(m / num_items * math.log(2)))
    return m, k


def minhash_standard_error(jaccard: float, num_hashes: int) -> float:
    """
    Standard deviation of the MinHash estimate.

    Each of the k signature positions agrees with probability J, so the
    estimate is Binomial(k, J) / k with std sqrt(J(1 − J) / k). Halving
    the error costs 4× the hash functions.
    """
    return math.sqrt(jaccard * (1 - jaccard) / num_hashes)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

@dataclass
class Summary:
    """Mean, sample variance, and a normal-approximation 95% CI."""
    n: int
    mean: float
    variance: float
    std: float
    minimum: float
    maximum: float
    ci95_low: float
    ci95_high: float

    @property
    def cv(self) -> float:
        """Coefficient of variation: std / mean (0 if mean is 0)."""
        return self.std / self.mean if self.mean else 0.0


def summarize(samples: Sequence[float]) -> Summary:
    """Summary statistics of repeated trials (sample variance, n − 1)."""
    n = len(samples)
    if n == 0:
        raise ValueError("cannot summarize an empty sample")
    mean = sum(samples) / n
    variance = (sum((x - mean) ** 2 for x in samples) / (n - 1)
                if n > 1 else 0.0)
    std = math.sqrt(variance)
    half = 1.96 * std / math.sqrt(n)
    return Summary(n, mean, variance, std, min(samples), max(samples),
                   mean - half, mean + half)


def wilson_interval(errors: int, trials: int,
                    z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for an error rate.

    Unlike the textbook p ± z·sqrt(p(1−p)/n), this stays meaningful
    when zero errors are observed: 0 errors in 100,000 trials gives an
    upper bound of about 3.8e−5, not a useless "0 ± 0".
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= errors <= trials:
        raise ValueError(f"errors must be in [0, {trials}], got {errors}")
    p = errors / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials
                                   + z * z / (4 * trials * trials))
    return max(0.0, centre - half), min(1.0, centre + half)
