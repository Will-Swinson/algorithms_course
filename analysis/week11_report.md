# Week 11 Technical Report: String Search Algorithms

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** A StringSearchEngine with KMP, Rabin–Karp, suffix arrays and suffix trees

## Executive Summary

I implemented four string matchers in `src/strings/` and put them
behind one `StringSearchEngine` API:
- KMP
- Rabin–Karp
- a suffix array with LCP
- Ukkonen's suffix tree

I then built plagiarism detection, DNA search and autocomplete on top
of them. The benchmark covers English-like text and DNA from 10 KB to
5 MB. It checked every answer against Python's `str.find`, and every
method agreed on every query.

- **Indexes win once queries repeat.** At 1 MB:
  - KMP takes 36 ms per query and the suffix array 0.006 ms.
  - The suffix array's 0.59 s build pays for itself after 16 queries
    (32 at 5 MB).
  - The suffix tree is faster still per query (0.0017 ms), but it
    needs 30 queries to break even and four times the memory.
- **Rabin–Karp's theory held exactly.** With modulus q = 101 a 1 MB
  scan produced 9,860 spurious hash hits; the prediction is 9,901.
  With q = 2⁶¹ − 1 there were none.
- **Batches are Rabin–Karp's strength.** It matched 500 patterns in one
  0.22 s pass; 500 separate KMP scans took 19 s.
- **The applications worked:**
  - Plagiarism containment tracked the copied share within 0.016.
  - Mismatch-tolerant DNA search was ~40,000× faster than a scan.
  - Trie autocomplete was 78× faster than a sorted scan on a 500k-word
    vocabulary.

## Overview of String Matching Complexity

Searching a text of length n for a pattern of length m costs:

| Method | Preprocessing | Query | Space |
|---|---|---|---|
| Brute force | none | O(nm) worst | O(1) |
| KMP | O(m) | O(n + m) | O(m) |
| Rabin–Karp | O(m) | O(n + m) expected, O(nm) worst | O(1) |
| Suffix array + LCP | O(n log n) | O(m log n + occ) | O(n) |
| Suffix tree (Ukkonen) | O(n) | O(m + occ) | O(n), large constant |

Scans pay O(n) for every query. Indexes pay once for preprocessing and
then answer in time that barely depends on n. Every trade-off below
follows from that difference.

## Implementation Details

**KMP** (`core/kmp.py`) precomputes the failure function: for each
prefix of the pattern, the length of its longest proper prefix that is
also a suffix. On a mismatch at pattern index j, the scan falls back
to that length (`lps[j−1]`) and never moves backwards in the text, so
it makes at most 2n comparisons (`kmp_trace.png` shows the fallbacks).

**Rabin–Karp** (`core/rabin_karp.py`, `utils/rolling_hash.py`) slides
a polynomial hash over the text in O(1) per step.
- **Defaults:** the Mersenne prime q = 2⁶¹ − 1 and a random base. For
  two fixed strings the collision probability is then at most
  (m − 1)/q.
- **Two modes:**
  - Verified (Las Vegas): every hash hit is checked, so answers are
    always exact.
  - Unverified (Monte Carlo): hits are reported as matches, so a
    collision becomes a false positive.
- **Multi-pattern search:** patterns of the same length share one pass
  and a dict lookup per window.

**Suffix array** (`core/suffix_array.py`):
- **Construction:** prefix doubling. Each round sorts suffixes by
  (rank[i], rank[i + k]), packed into one int64 key so numpy can sort
  it in C.
  Packing was 2.5× faster than `np.lexsort` on two keys.
- **LCP:** Kasai's algorithm in O(n).
- **Search:** two binary searches find the block of suffixes that start
  with the pattern, so counting needs no scan.

**Suffix tree** (`core/suffix_tree.py`) uses Ukkonen's online
construction:
- The active point and the remainder count track the suffixes that
  are still implicit.
- Suffix links make each extension amortized O(1).
- Leaves share one global end, so extending them all costs nothing.

With `record=True` it logs each phase's explicit extensions, implicit
suffixes and splits (`ukkonen_steps.png`). `to_dot()` exports Graphviz
DOT source; Graphviz is not installed here, so matplotlib draws the
tree.

**The engine** (`core/search_engine.py`):
- **API:** `find(pattern, method="kmp" | "rk" | "sa" | "tree")`, which
  returns positions and counts.
- **Lazy indexes:** each index is built on first use, and the build is
  timed.
- **Statistics:** each method keeps query counts, mean search time and
  preprocessing time.
- **`recommend(q)`:** chooses the method with the lowest
  preprocessing + q × mean query time, and explains why ("suffix
  array is fastest for repeated queries…").

## Benchmark Analysis

**Latency** (`runtime_comparison.png`, `heatmap_dataset_sizes.png`).
Scan times grew linearly with text size. At 1 MB of English:

| Method | Per query |
|---|---|
| Brute force | 60.3 ms |
| KMP | 36.0 ms |
| Rabin–Karp | 217 ms |
| `str.find` (C) | 0.39 ms |
| Suffix array | 0.006 ms |
| Suffix tree | 0.0017 ms |

- **Index queries barely grew with n.** The suffix array took
  0.0041 ms at 10 KB and 0.0065 ms at 5 MB.
- **Pattern length (4–128 characters) hardly affected any method.**
  The exception was very short patterns with many occurrences: there
  the cost of listing every match dominated, and the suffix tree's
  traversal of a large subtree took 1.9 ms.
- **Rabin–Karp was the slowest scan in Python.** It was 6× slower than
  KMP: modular arithmetic on large integers costs more per character
  than KMP's comparison, and q = 2⁶¹ − 1 made the scan 1.6× slower
  than q ≈ 10⁹.

**Worst case.** On T = a¹⁰⁰⁰⁰⁰ with P = a⁶³b:
- Brute force made 6.4 M comparisons; KMP made 200k (32× fewer).
- Rabin–Karp made none, because no window hashed like the pattern.
- With P = aᵐ, every window is a real match, so Rabin–Karp verifies all
  of them and performs exactly brute force's m(n − m + 1) comparisons.
  That is its O(nm) worst case.

**Collisions** (`collision_analysis.png`). Spurious hits followed
(n − m + 1)/q across nine moduli from 7 to 2⁶¹ − 1 (9,860 vs 9,901 at
q = 101; 1,032 vs 991 at q = 1,009). Unverified mode turned exactly
those collisions into false positives. Bad parameters were costly:

| Parameters | Spurious hits per 1 MB scan |
|---|---|
| Base 1 (hash = character sum, so anagrams collide) | 5,020 |
| q = 2¹⁶ with base 256 (only the last two characters count) | 8,860 |
| Base ≡ 0 mod q (only the last character counts) | 50,860 |
| Prime q = 65,521, random base | 13.2 |

The random-base prime of similar size had 13.2 hits, close to its
expected n/q ≈ 15.

**Suffix arrays** (`sa_lcp_stats.png`):
- **Construction:** numpy doubling built 1 MB in 0.60 s and 5 MB in
  6.7 s. The Python-sort version was 4.2× slower at 100 KB. Kasai took
  0.55 s at 1 MB.
- **Rounds:** the number of doubling rounds equalled
  ⌈log₂(longest repeat + 1)⌉ on every text. That was 4 for random
  text, 5 for English (longest repeat 31), 6 for logs (58) and 9 for
  DNA, where planted duplications created a 499-character repeat.

**Memory.** Measured as the tracemalloc peak at 1 MB:
- The suffix array took 70 MB (73 bytes per character).
- The Python suffix tree took 277–316 MB (about 290–330 bytes per
  character).

I measured memory with a separate build because tracing slowed the
tree's construction about 6×. My first run timed builds under
tracemalloc, and that inflated the tree's break-even point from 30 to
209 queries.

## Trade-offs

**Preprocessing vs. runtime.** An index makes sense when the number of
queries exceeds its build time divided by the per-query saving.
- Against KMP, that break-even is 8–32 queries for the suffix array
  and 17–30 for the suffix tree.
- Against C-speed `str.find`, the suffix array needs about 1,550
  queries at 1 MB. Interpreter overhead shifts the crossover, but the
  index still wins eventually.

**Deterministic vs. randomized.** KMP's O(n + m) bound holds for every
input. Rabin–Karp's bound is in expectation over its random base.
- Verified, it is always correct; collisions only cost time.
- Unverified, it is faster but can be wrong. Its error rate (~10⁻¹²
  per megabyte at q = 2⁶¹ − 1) is acceptable for fingerprinting, not
  for exact search.
- In return it batches: 87× faster than KMP for 500 patterns.

**Memory vs. speed.**
- The suffix tree answers queries 3.5× faster than the suffix array
  but needs four times the memory. At 5 MB the tree would need about
  1.5 GB, so the benchmark built only the array there.
- The suffix array also gives LCP-based results cheaply: longest
  repeats, distinct-substring counts and longest common substrings.
- For most uses, the suffix array is the better balance.

## Applications in Search Engines, NLP, Compilers and Bioinformatics

- **Search and plagiarism.** Verified Rabin–Karp fingerprints of
  5-word sequences find shared phrases. Containment matched the copied
  share (0.396 at 40%, 0.784 at 80%); Jaccard stayed low (0.16) because
  the source was four times longer. A suffix array over both documents
  finds the longest verbatim overlaps, which are highlighted.
- **Bioinformatics.**
  - **Input:** FASTA records are indexed once, with both strands
    searched.
  - **Mutations:** search with up to k substitutions splits the gene
    into k + 1 pieces. At least one piece must match exactly, so the
    index finds candidates and each is checked for mismatches.
  - **Speed:** on a 1 Mb genome a 30 bp gene with up to 2 mismatches
    took 0.03 ms, against 1.3 s for a scan.
  - **Recall:** every planted gene carrying 2 mutations was recovered
    at k ≥ 2.
- **NLP and editors (autocomplete).** A compressed prefix trie stores
  each subtree's best score, so a best-first search returns the top-k
  completions without listing them all. Query-frequency weighting
  promotes terms users pick; in the demo, 30 selections moved "watch"
  from outside the top four to second.
- **Compilers.** Keyword scanning is multi-pattern matching (Rabin–Karp
  batches); suffix structures detect duplicated code.

## Conclusion

Algorithmic class mattered most: scans grew with the text, while
index queries stayed in microseconds from 10 KB to 5 MB.

The most useful decision was to measure the break-even point rather
than assume it: the engine's recommendations come from those
measurements, and even a C-implemented scan loses to an index once the
same text is queried often enough.

Validation caught real issues along the way:
- A suffix-tree query containing the terminator character matched the
  end-of-text marker.
- Timing under tracemalloc made the suffix tree look six times more
  expensive to build than it is.
