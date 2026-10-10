# Week 7 Technical Report: Randomized Algorithms

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** Randomized QuickSort, QuickSelect, Primality Testing, Bloom Filters, and MinHash

## Executive Summary

Every earlier week made all of its choices deterministically. This
week each algorithm flips coins on purpose, and every measured
behavior matched its probabilistic prediction.

- **Robustness:** on sorted input at n = 3,000, first-element
  QuickSort made 4.5 million comparisons (120 ms); the random pivot
  made 40,145 (2.6 ms), **46× faster**, and no input order can force
  its worst case.
- **Accuracy of the theory:** mean comparison counts stayed within
  ±2.5% of the exact expectation 2(n+1)Hₙ − 4n at every size.
  QuickSelect stayed linear: 1.8–2.1n comparisons for the min/max and
  2.9–3.6n for the median.
- **Speed with bounded error:** Miller–Rabin with k = 20 certified a
  14-digit prime **462× faster** than trial division, and it handled
  300-digit numbers in 73 ms. It never erred on Carmichael numbers
  from k = 5 up, while Fermat still erred 1.6% of the time at k = 20.
- **Space:** a Bloom filter stored 10,000 keys in 12.2 KiB at 0.85%
  false positives. A Python set of the same strings used 1,019 KiB
  (83× more). MinHash error fell as 1/√k, as predicted.

## Methodology

**Environment:** Apple M3 Pro, macOS 15.7.3, CPython 3.13.5,
single-threaded. pytest ran the 1,568-test suite (439 tests new this
week).

**Two kinds of seeding.** Inputs come from fixed seeds
(`src/utils/random_utils.py`), so every run sees the same arrays and
sets. Each trial then gives the algorithm its own `random.Random(seed)`.
Holding the input fixed and varying only that seed isolates the
variance caused by the algorithm's own coin flips. Every number in
this report is reproducible.

**Data:**
- *Sorting/selection:* distinct keys (a shuffled 0..n−1), so the
  expected-value formulas hold exactly, plus already-sorted arrays.
  Sizes n = 100–100,000, with 20 trials per configuration and
  300–500 trials for the distribution plots.
- *Primality:* the first prime after 10^(d−1) for d = 6–300 digits,
  the 16 Carmichael numbers below 80,000 (2,000 trials each per k),
  and every odd composite up to 3,000 for exact liar counts.
- *Bloom filter:* member strings and disjoint probe strings, with
  50,000 probes per configuration.
- *MinHash:* pairs of 500-element sets with Jaccard similarity set to
  0.1–0.9 (50 hash families each), plus 18 near-duplicate paragraphs
  made by randomly replacing words (153 pairs).

**Metrics:** mean, variance, and coefficient of variation of runtime
and comparisons; exact expectations from
`src/utils/probability_analysis.py`; error rates with Wilson 95%
intervals, which give an upper bound even when zero errors are seen.
Raw data: `benchmarks/results/summary_table.csv` (264 rows, ~1 min run).

**Two measurement traps.** Week 2's Dutch-national-flag partition
swaps elements in from the right, which scrambles sorted input, so
first-element pivoting made only 98k comparisons on sorted n = 2,000
instead of 2.0 million. A Lomuto-style three-way partition keeps order
and reproduces exactly n(n−1)/2. Separately, a small-prime
trial-division shortcut caught every Carmichael number before Fermat
ran, making Fermat look perfect; I removed it.

## Results

### Randomized vs deterministic QuickSort (`quicksort_comparison.png`)

| Input | n | First-element pivot | Random pivot | Measured ÷ expected |
|---|---:|---:|---:|---:|
| random | 10,000 | 9.9 ms, 154,333 cmp | 11.0 ± 1.1 ms, 159,085 cmp | 1.021 |
| random | 100,000 | 125.6 ms | 145.9 ± 30.3 ms | 1.004 |
| **sorted** | 1,000 | 14.2 ms, 499,500 cmp | 0.72 ms | 0.977 |
| **sorted** | 3,000 | **120 ms, 4,498,500 cmp** | **2.6 ms**, 40,145 cmp | 1.016 |
| sorted | 100,000 | — (≈5·10⁹ cmp) | 111.7 ms | 0.990 |

On random input the two pivot rules were within noise of each other
(a few slow trials pushed the random pivot's mean up at n = 100,000;
note its ±30 ms spread); Week 2's quick_sort took 132 ms and
C-implemented `sorted()` was 13–16× faster than all three. On sorted
input the first-element pivot grew as n² (8.4× from n = 1,000 to
3,000, against 9× predicted) and recursed n levels deep; the random
pivot was unaffected.

Over 300 seeds at n = 10,000, comparisons averaged 156,160 against
155,772 expected (+0.25%), with standard deviation 6,962 (theory:
0.648n = 6,482). The distribution is right-skewed; the worst run was
19% above the mean. The coefficient of variation fell from 9.2% at
n = 100 to 3.8% at 100,000, so cost concentrates as n grows. Depth
ranged 26–36, against 10,000 for the first pivot on sorted input.

### Randomized selection (`selection_runtime.png`)

| n = 100,000 | QuickSelect | Comparisons/n | Depth | Median-of-medians | Sort + index |
|---|---:|---:|---:|---:|---:|
| min | 11.2 ms | 2.14 (exp. 2.00) | 12.1 | 62.9 ms, 8.11n | 9.6 ms |
| median | 17.9 ms | 3.58 (exp. 3.39) | 23.6 | 69.0 ms, 8.45n | 9.3 ms |
| max | 9.3 ms | 1.80 (exp. 2.00) | 11.0 | 62.7 ms, 8.31n | 9.2 ms |

Comparisons per element stayed flat from n = 100 to 100,000 for
every k: linear time. Depth for the minimum tracked Hₙ ≈ ln n (12.1
vs 12.1). Median-of-medians, linear in the worst case, made 2.4–4.6×
more comparisons and ran 3.9–6.7× slower. C-implemented sorting beat
both Python routines, a reminder that constant factors decide
practical speed.

**Success probability.** Every Las Vegas run returned the correct
answer; only cost varies. For the median at n = 10,000, the chance of
finishing within c·n comparisons was 58% (c = 3.5), 75% (c = 4), 92%
(c = 5) and 99% (c = 6). Markov's inequality guarantees only about 3%,
15%, 32% and 44%, so the real tail is far thinner.

### Monte Carlo primality (`primality_accuracy.png`)

| Rounds k | Fermat error (Carmichael) | Predicted | Miller–Rabin error | MR 95% upper | 4⁻ᵏ bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 74.8% | 75.1% | 5.36% | 5.6% | 25% |
| 3 | 43.9% | 44.3% | 0.059% | 0.093% | 1.6% |
| 5 | 27.0% | 27.4% | **0** / 32,000 | 0.012% | 0.098% |
| 10 | 9.57% | 9.49% | **0** / 32,000 | 0.012% | 9.5·10⁻⁷ |
| 20 | **1.63%** | 1.69% | **0** / 32,000 | 0.012% | 9·10⁻¹³ |

"Predicted" averages (liar fraction)^k over the 16 numbers, with
liar fractions counted exactly over every base; Fermat stayed within
0.5 points of it. Across all 1,070 odd composites up to 3,000, the
worst Miller–Rabin liar fraction was 23.7% (n = 1891), under Rabin's
1/4 bound; Fermat's worst was 76.6% (n = 2821). At k ≥ 5, 32,000
trials cannot resolve rates below ~10⁻⁴, so the flat green line marks
the experiment's limit, not the algorithm's error.

Trial division's √n/3 divisions grew 10× per two digits (3.8 ms at
12 digits, 37 ms at 14). Miller–Rabin (k = 20) took 62 and 79 µs, with
the crossover near 7 digits, and 73 ms at 300 digits, where trial
division would need ~10¹⁴⁹ divisions.

### Bloom filter (`bloom_filter_fp_rate.png`)

All 38 configurations matched (1 − e^(−kn/m))^k within sampling
error. Raising load from 0.05 to 0.2 multiplied false positives 32×
for k = 3 but 784× for k = 7: more hashes help at light load and hurt
at heavy load. At 8 bits per item the minimum was k = 6 (2.06%), beside
the optimum 8 ln 2 ≈ 5.5. No run ever produced a false negative.

| Bits/item | Memory (10k keys) | FP measured | FP theory |
|---:|---:|---:|---:|
| 4 | 4.9 KiB | 14.5% | 14.7% |
| 10 | 12.2 KiB | 0.85% | 0.82% |
| 16 | 19.5 KiB | 0.034% | 0.046% |
| Python set | 1,019 KiB | 0 | 0 |

### MinHash (`minhash_similarity_plot.png`)

RMSE followed sqrt(J(1−J)/k): at J = 0.5 it fell from 0.147 (k = 16)
to 0.024 (k = 512), against theory's 0.125 and 0.022 — 32× the hashes
for 6× less error. On the 153 text pairs, k = 512 reached RMSE 0.009.
A signature costs O(k·|S|) (1.2–32 ms); comparing two takes 0.7–15 µs
regardless of set size.

## Discussion

**Las Vegas vs Monte Carlo.** QuickSort and QuickSelect are always
correct; randomness affects only their running time. Primality tests,
Bloom filters and MinHash run in fixed time; randomness affects the
answer. The primality tests and Bloom filter err one way only —
"composite" and "absent" are certain — so independent repetition
shrinks error geometrically. MinHash errs both ways and improves only
as 1/√k. Cutting a Las Vegas algorithm off at a budget (say 5n
comparisons) makes it Monte Carlo: here it would answer 92% of the
time.

**Defeating adversarial inputs.** First-element pivoting has a fixed
worst case anyone can construct, and sorted data is common. A random
pivot turns a bad input into a bad sequence of coin flips, which no
adversary controls. Salting the Bloom filter's hash and randomizing
MinHash's coefficients do the same: colliding keys cannot be crafted
without the seed.

**Guarantees vs speed.** Certainty cost 4–7× for median-of-medians
and 462× for trial division at 14 digits, which cannot finish at all
at 300. Meanwhile a 4⁻²⁰ error rate is below the chance of a hardware
fault during the computation.

**Real-world relevance.** *Cryptography:* RSA key generation uses
Miller–Rabin to find primes with hundreds of digits. *Databases and
storage:* Cassandra, Bigtable and LSM-tree engines check a Bloom
filter before reading from disk. *Machine learning and search:*
MinHash with locality-sensitive hashing removes near-duplicate
documents from web crawls and training data. *Streaming analytics:*
Bloom filters, Count-Min sketches and HyperLogLog all trade a bounded
error for memory that does not grow with the size of the stream.

## Conclusion

Weeks 1–6 asked what an algorithm costs in the worst case. Randomized
algorithms ask what it costs in expectation and how likely it is to do
worse. This week the expectations held to within a few percent, the
tails were thin, and the error bounds held. In exchange the algorithms
became simpler, faster, and harder to attack, and in some cases
(300-digit primes, membership tests at 1% of a set's memory) they
became practical where deterministic methods are not. Requiring
expected efficiency instead of exactness is what moves algorithm
design from the Core Techniques of this course into the probabilistic
methods that modern systems rely on.
