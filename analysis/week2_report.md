# Week 2 Performance Analysis Report

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** Divide and Conquer — Merge Sort & QuickSort

## Executive Summary

At n = 50,000 on random data, the divide-and-conquer algorithms beat the
Week 1 quadratic sorts by up to **991×** (quicksort: 55 ms; bubble sort:
54.6 s), and the gap widens with n exactly as O(n²) vs O(n log n) theory
predicts. Doubling-ratio analysis confirmed both complexity classes
empirically, and the optimizations mattered measurably: three-way
partitioning made quicksort **11×** faster than merge sort on
low-diversity data, while randomized pivots kept quicksort's worst case
from ever appearing.

## Methodology

**Hardware:** Apple Silicon Mac (arm64), macOS 15.7.3, CPython 3.13.5
in the project virtual environment, single-threaded.

**Test configuration:** 5 algorithms × 6 data types × 6 sizes
(100, 500, 1,000, 5,000, 10,000, 50,000) = 180 measurements
(`benchmarks/week2_performance.py`, ~30 min wall time). Sizes below
10,000 used 5 timed runs after 2 warmups; sizes ≥ 10,000 used 3 timed
runs after 1 warmup to keep the quadratic algorithms' wall time
manageable — a disclosed trade-off; standard deviations stayed small
(typically < 4% of the mean).

**Data generation:** seeded RNG (seed 42); every algorithm at a given
(type, size) receives an identical fresh copy. Types: random integers,
already sorted, reverse sorted, nearly sorted (95% in place), many
duplicates (10 unique values), few unique (3 values).

**Measurement:** `time.perf_counter()` via the Week 1
`AlgorithmBenchmark` framework; every run's output is verified to be a
sorted permutation of the input before its timing is accepted. Raw
data: `benchmarks/results/comparison_table.csv`.

## Results

### Overall Performance Comparison (n = 50,000, seconds)

| Algorithm | random | sorted | reverse | nearly sorted | many dup. | few unique |
|---|---:|---:|---:|---:|---:|---:|
| bubble_sort | 54.55 | **0.0012** | 64.15 | 30.45 | 47.59 | 40.62 |
| selection_sort | 20.47 | 19.63 | 22.09 | 19.66 | 18.96 | 18.76 |
| insertion_sort | 28.72 | 0.0028 | 53.70 | 1.94 | 23.13 | 17.22 |
| merge_sort | 0.0657 | 0.0411 | 0.0439 | 0.0878 | 0.0593 | 0.0573 |
| quick_sort | 0.0550 | 0.0564 | 0.0570 | 0.0609 | **0.0089** | **0.0050** |

Per-data-type plots: `benchmarks/results/<data_type>.png`.

### O(n²) vs O(n log n) Speedup Analysis

Quicksort's advantage over bubble sort on random data grows steadily
with n — the ratio n²/(n log n) = n/log n in action:

| n | 100 | 500 | 1,000 | 5,000 | 10,000 | 50,000 |
|---|---:|---:|---:|---:|---:|---:|
| speedup | 4.2× | 15.6× | 32.7× | 118× | 233× | **991×** |

Even against the *best* quadratic sort (selection, 20.5 s), quicksort
is 372× faster at n = 50,000.

### Merge Sort Performance

Merge sort is the consistency champion: 41–88 ms across all six data
types at n = 50,000, by far the flattest profile of any algorithm —
matching theory, since it performs the same Θ(n log n) split-and-merge
work regardless of input order. Its cost is space: it is the only
algorithm here allocating new lists at every merge (O(n) auxiliary
memory), which shows up as a modest constant-factor deficit against
quicksort on random data (65.7 ms vs 55.0 ms).

### QuickSort Performance

**Best vs worst case:** the theoretical O(n²) worst case never
appeared. Sorted (56.4 ms) and reverse-sorted (57.0 ms) inputs — the
classic killers of fixed-pivot quicksort — ran at the same speed as
random data (55.0 ms). **Impact of randomization:** with a fixed
first-element pivot, sorted input would have degraded to roughly
selection-sort territory (~20 s, extrapolating the quadratic group);
random pivots make every input ordering equivalent in expectation.
**Effect of three-way partitioning:** dramatic. On few-unique data
(3 values), quicksort finished 50,000 elements in **5.0 ms** — 11×
faster than its own random-data time and 11.5× faster than merge sort —
because the equal-to-pivot block is never recursed into. Scaling from
n = 100 to 50,000 (500×), its few-unique time grew only 416×: effectively
*linear*, the O(n) behavior three-way partitioning predicts.

### Surprising Findings

1. **Bubble sort wins on sorted data.** At 1.2 ms for n = 50,000, its
   early-exit pass beat merge sort (41 ms) and quicksort (56 ms). The
   advantage is fragile — displace 5% of elements and it collapses to
   30.5 s, a 25,000× penalty.
2. **R² alone cannot identify complexity.** The framework's best-fit
   routine labeled *all five* algorithms O(n²) with R² = 1.000 — a
   degree-2 polynomial has enough freedom to fit a smooth n log n curve
   almost perfectly. The doubling ratios (below) separate the classes
   unambiguously; goodness-of-fit does not.
3. **Insertion sort's nearly-sorted advantage fades at scale:** 1.94 s
   at n = 50,000 vs quicksort's 61 ms. Only 5% of elements are out of
   place, but each is displaced an average of n/3 positions, and
   insertion sort pays one shift per position.

## Complexity Validation

The framework reports the average of (time ratio ÷ size ratio) between
consecutive sizes: for O(n²) this equals the size ratio itself (≈ 3.8
for our 5×/2× step pattern); for O(n log n) it is just the slowly
growing log factor (≈ 1.1–1.3).

**Merge sort** — theoretical Θ(n log n) in all cases. Empirical:
doubling ratio **1.149**; from n = 10,000 → 50,000 (5×), time grew
5.68× — the n log n prediction is 5.85×, while quadratic growth would
give 25×. Clear confirmation.

**QuickSort** — theoretical O(n log n) expected, O(n²) worst.
Empirical: doubling ratio **1.238**; the same 5× size step grew time
5.98×. The worst case was not observed — and should not be: with
randomized pivots no fixed input ordering can trigger it, and the
probability of drawing bad pivots at every level simultaneously is
vanishingly small.

**Quadratic control group** — bubble 3.99, selection 3.67, insertion
3.90 doubling ratios, and bubble's 10,000 → 50,000 step grew 25.5× ≈ 5².
Both complexity classes behave exactly as theory predicts.

## Optimization Impact

- **Randomized pivot:** converts quicksort's two historical worst
  cases (sorted, reverse) into ordinary inputs — measured at parity
  with random data instead of an extrapolated ~20+ s.
- **Three-way partitioning:** the difference between quicksort
  *degrading* on duplicates and *exploiting* them: 8.9 ms (many
  duplicates) and 5.0 ms (few unique) vs 55 ms on all-distinct data. A
  two-way partition with random pivots would have gone quadratic here.
- **Insertion-sort cutoff (threshold 10):** a constant-factor win
  visible at the small end — quicksort was the fastest algorithm even
  at n = 100 (40 µs), where recursion overhead would otherwise let the
  simple sorts compete.

## Practical Recommendations

- **Default choice: quicksort** — fastest on random, duplicate-heavy,
  and low-diversity data; ties merge sort elsewhere. Requires the
  randomization + three-way safeguards implemented here.
- **Choose merge sort** when stability matters (equal keys must keep
  their order) or a *guaranteed* O(n log n) worst case is required,
  and O(n) extra memory is acceptable.
- **Insertion sort** stays useful below ~50 elements and as the
  finishing pass inside other algorithms — exactly how quicksort uses
  it here.
- **Bubble and selection sort:** educational baselines only.
- In real Python code, use the built-in `sorted()` (Timsort) — a
  hybrid of merge-sort and insertion-sort ideas validated by these
  same trade-offs.

## Conclusion

Every theoretical claim survived contact with measurement: the two
complexity classes separated by three orders of magnitude at
n = 50,000, doubling ratios matched their predicted signatures, and
each optimization produced a visible, attributable effect. The larger
lesson is that asymptotic class dominates constant factors at scale —
but *within* a class, input distribution and optimizations decide the
winner, and naive curve-fitting (R² = 1.000 for the wrong model) is no
substitute for growth-rate analysis.

## References

1. Amakobe, M. *Advanced Algorithms: A Journey Through Computational
   Problem Solving* — Ch. 2 (Divide and Conquer), §4.4–4.5
   (Recursion-Tree Method, Master Theorem), §1.6 (Algorithm Laboratory).
2. Python Software Foundation. `time.perf_counter()` documentation,
   Python 3.13.
3. Project Week 1 report: `docs/performance_analysis.md`.
