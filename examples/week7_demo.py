"""
Week 7 demo: randomness as a tool, on small readable inputs.

Run from the project root:

    python examples/week7_demo.py

Shows (1) QuickSort on already-sorted input with a fixed vs a random
pivot, (2) QuickSelect finding a median with the comparison count
varying run to run but the answer never changing (Las Vegas),
(3) Fermat being fooled by a Carmichael number where Miller–Rabin is
not (Monte Carlo), (4) a Bloom filter screening usernames in a few
KiB, and (5) MinHash spotting a near-duplicate document.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.randomized import (
    BloomFilter, MinHash, PartitionStats, deterministic_quicksort,
    fermat_test, jaccard_similarity, liar_fraction, median_of_medians_select,
    miller_rabin, randomized_quicksort, randomized_select, shingles,
)
from src.utils.probability_analysis import (
    expected_quicksort_comparisons, expected_quickselect_comparisons,
    miller_rabin_error_bound,
)
from src.utils.random_utils import random_array, random_strings


def header(title):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def demo_quicksort():
    header("1. QuickSort on input that is already sorted (n = 2,000)")
    arr = list(range(2_000))
    fixed = PartitionStats()
    deterministic_quicksort(arr, fixed)
    print(f"  first-element pivot   {fixed.comparisons:>9,} comparisons, "
          f"depth {fixed.max_depth:,}")
    for seed in range(3):
        s = PartitionStats()
        randomized_quicksort(arr, s, random.Random(seed))
        print(f"  random pivot, seed {seed}  {s.comparisons:>9,} comparisons, "
              f"depth {s.max_depth}")
    print(f"  expected for random pivot: 2(n+1)H_n − 4n = "
          f"{expected_quicksort_comparisons(2_000):,.0f}")
    print("  The fixed rule always splits off one element; the random rule "
          "can't be steered.")


def demo_selection():
    header("2. QuickSelect: median of 10,001 numbers (Las Vegas)")
    arr = random_array(10_001, seed=1)
    k = 5_001
    for seed in range(4):
        s = PartitionStats()
        value = randomized_select(arr, k, s, random.Random(seed))
        print(f"  seed {seed}: answer {value}  after {s.comparisons:>6,} "
              f"comparisons, {s.max_depth} rounds")
    mom = PartitionStats()
    median_of_medians_select(arr, k, mom)
    print(f"  median-of-medians: answer {median_of_medians_select(arr, k)} "
          f"after {mom.comparisons:,} comparisons (every time)")
    print(f"  expected QuickSelect cost ≈ "
          f"{expected_quickselect_comparisons(len(arr), k):,.0f} — "
          "random time, never a wrong answer.")


def demo_primality():
    header("3. Primality: the Carmichael number 561 = 3 × 11 × 17 "
           "(Monte Carlo)")
    n, trials = 561, 10_000
    rng = random.Random(2)
    print(f"  {'rounds k':>8}  {'Fermat says prime':>18}  "
          f"{'Miller–Rabin says prime':>24}  {'MR bound 4^-k':>13}")
    for k in [1, 3, 5, 10]:
        f = sum(fermat_test(n, k, rng) for _ in range(trials))
        m = sum(miller_rabin(n, k, rng) for _ in range(trials))
        print(f"  {k:>8}  {f / trials:>17.2%}  {m / trials:>23.2%}  "
              f"{miller_rabin_error_bound(k):>13.1e}")
    print(f"  Per round, {liar_fraction(n, 'fermat'):.0%} of bases fool "
          f"Fermat but only {liar_fraction(n, 'miller_rabin'):.1%} fool "
          "Miller–Rabin.")
    p = 2 ** 127 - 1
    print(f"  2^127 − 1 ({len(str(p))} digits): Miller–Rabin(k=20) says "
          f"{'prime' if miller_rabin(p, 20) else 'composite'} in 20 modular "
          "exponentiations;")
    print("  trial division would need ~4 × 10^18 divisions.")


def demo_bloom():
    header("4. Bloom filter: 'is this username taken?' for 50,000 users")
    taken = random_strings(50_000, length=8, prefix="user_", seed=3)
    bf = BloomFilter.for_capacity(len(taken), fp_rate=0.01, seed=4)
    bf.update(taken)
    probes = random_strings(20_000, length=8, prefix="new_", seed=5)
    false_pos = sum(name in bf for name in probes)
    print(f"  {bf.num_bits:,} bits, k = {bf.num_hashes} hash functions, "
          f"{bf.memory_bytes() / 1024:.0f} KiB "
          f"({bf.num_bits / len(taken):.1f} bits per name)")
    print(f"  every taken name found: {all(n in bf for n in taken)}  "
          "(no false negatives, ever)")
    print(f"  free names wrongly reported taken: {false_pos} of "
          f"{len(probes):,} = {false_pos / len(probes):.2%} "
          f"(designed for 1%, theory {bf.expected_fp_rate():.2%})")


def demo_minhash():
    header("5. MinHash: which document is a near-copy of the original?")
    original = ("The quick brown fox jumps over the lazy dog while the "
                "farmer watches from the porch and the sun sets slowly "
                "behind the hills at the end of a long summer day")
    docs = {
        "light edit": original.replace("quick", "fast")
                              .replace("slowly", "gently"),
        "reordered": ("While the farmer watches from the porch the quick "
                      "brown fox jumps over the lazy dog and the sun sets "
                      "behind the hills slowly at the end of a long day"),
        "unrelated": ("Bloom filters and MinHash sketches let search "
                      "engines and databases compare huge collections "
                      "using only a few bytes per item"),
    }
    mh = MinHash(128, seed=6)
    base = shingles(original, 3)
    base_sig = mh.signature(base)
    print(f"  {'document':12s} {'true Jaccard':>13} {'MinHash (k=128)':>16}")
    for name, text in docs.items():
        s = shingles(text, 3)
        print(f"  {name:12s} {jaccard_similarity(base, s):>13.3f} "
              f"{mh.similarity(base_sig, mh.signature(s)):>16.3f}")
    print("  Each document is stored as 128 numbers, however long it is.")


if __name__ == "__main__":
    demo_quicksort()
    demo_selection()
    demo_primality()
    demo_bloom()
    demo_minhash()
