# Week 5 Technical Report: Dynamic Programming

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** Dynamic Programming — Fibonacci, 0/1 Knapsack, and LCS

## Executive Summary

Dynamic programming turns exponential recursion into polynomial work
by solving each distinct subproblem once and reusing the answer. The
measurements show how large that gap gets. Naive recursive `fib(45)`
made **3.67 billion calls and took 82.4 seconds**. Memoization made 89
calls, tabulation filled 46 cells, and both returned in microseconds,
a **67-million-fold speedup**. The same pattern held for problems with
real decisions: at 22 items the recursive knapsack was 1,558× slower
than tabulation, and recursive LCS on two 14-character strings was
4,631× slower. Past those sizes the recursive versions stop being
practical, while tabulation handled 500-item knapsacks and
1,000-character LCS in about 0.1 s and 0.05 s. Between the two DP
styles, tabulation was 3–8× faster in every large-input test, and for
knapsack it also used 4× less memory.

## Methodology

**Hardware & environment:** Apple M3 Pro (12 cores, 18 GB), macOS
15.7.3, CPython 3.13.5 in the project venv, single-threaded.
matplotlib 3.11 for plots, pytest 9.1 for the test suite (706 tests,
288 of them new this week).

**Implementations** (`src/dp/`): each problem has naive recursive,
top-down memoized, and bottom-up tabulated versions. Knapsack adds
`trace_solution()` to recover the chosen items, and LCS adds
`reconstruct_lcs()` to recover the subsequence itself. Memoized
versions raise Python's recursion limit only for the duration of the
call, since memoized LCS on 1,000-character strings recursed 1,696
frames deep, past CPython's default limit of 1,000.

**Inputs** (all seeded, seed 42):
- *Fibonacci:* n = 10, 20, 30, 35, 40, 45.
- *Knapsack:* weights 1–30, values 1–100, capacity W set to half the
  total weight so the take/skip tree branches as much as possible.
  Recursive runs covered n ≤ 22 and DP ran up to n = 500. A separate
  capacity sweep held n = 100 fixed and varied W from 100 to 5,000.
- *LCS:* pairs of random DNA strings (alphabet ACGT). Recursive runs
  covered lengths 4–14 and DP covered lengths 4–1,000.

**Measurement** (`src/utils/timer.py`): every method is measured in
three separate runs so that one kind of instrumentation cannot distort
another.
1. **Time:** `time.perf_counter()`. Calls under 0.5 s are repeated at
   least 3 times and averaged. In an earlier run a single 0.05 s
   measurement came out 2.7× too high, which is why the floor is 0.5 s.
2. **Calls and recursion depth:** a `CallStats` counter that
   recursive code enters and exits. For tabulation it counts table
   cells filled and reports a depth of 0.
3. **Peak memory:** `tracemalloc`.

Counting naive `fib(40)` and `fib(45)` call by call would mean billions
of instrumented calls, so those two counts come from the exact closed
form 2·F(n+1) − 1. The test suite checks that formula against real
counts. Raw data: `benchmarks/results/dp_vs_recursive_table.csv` (83
rows).

## Results

### Fibonacci (`fibonacci_comparison.png`)

| n | Naive time | Naive calls | Memo time | Memo calls | Tab time | Tab cells | Speedup (tab) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6.5 µs | 177 | 3.0 µs | 19 | 0.4 µs | 11 | 17× |
| 20 | 0.50 ms | 21,891 | 2.7 µs | 39 | 0.6 µs | 21 | 768× |
| 30 | 57.8 ms | 2,692,537 | 4.3 µs | 59 | 0.8 µs | 31 | 69,055× |
| 35 | 649 ms | 29,860,703 | 5.5 µs | 69 | 1.0 µs | 36 | 643,784× |
| 40 | 7.25 s | 331,160,281 | 5.2 µs | 79 | 1.1 µs | 41 | 6.6 M× |
| 45 | **82.4 s** | 3,672,623,805 | 7.3 µs | 89 | 1.2 µs | 46 | **66.7 M×** |

Theory predicts that each +5 step in n multiplies naive work by
φ⁵ ≈ 11.09. The measured time ratios were **11.17×** (35 → 40) and
**11.37×** (40 → 45), so the observed growth is Θ(φⁿ) with a constant
of about 22 ns per call. Memoized calls came to exactly 2n − 1 and
tabulated cells to exactly n + 1, both linear as expected. The
memory panel is less straightforward. `tracemalloc` shows naive
recursion peaking at 320 bytes, because it tracks heap objects and
not interpreter stack frames. The real cost of recursion appears in
the depth column instead, which equals n for both recursive versions.

### 0/1 Knapsack (`knapsack_performance.png`)

| n | W | Recursive | Calls | Memo | Memo calls | Tabulation | Cells | Speedup (tab) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 23 | 1.8 µs | 16 | 4.1 µs | 16 | 3.7 µs | 96 | 0.5× |
| 12 | 93 | 0.30 ms | 4,783 | 0.15 ms | 917 | 47 µs | 1,128 | 6.3× |
| 20 | 141 | 75.8 ms | 1,166,840 | 0.50 ms | 3,471 | 0.12 ms | 2,840 | 625× |
| 22 | 207 | 300 ms | 4,940,293 | 0.96 ms | 5,259 | 0.19 ms | 4,576 | **1,558×** |
| 200 | 1,512 | — | — | 122 ms (32.7 MiB) | 444,093 | 15.7 ms (7.9 MiB) | 302,600 | — |
| 500 | 3,626 | — | — | not run | — | 100 ms (48.2 MiB) | 1,813,500 | — |

Going from 20 to 22 items multiplied recursive time by 3.96×, close to
the 2² = 4 that O(2ⁿ) predicts. Tabulation cost a steady 52–55 ns per
cell: from n = 200 to 500 the cell count grew 6.0× and the time 6.4×.
At n = 4 tabulation was slower than recursion, since building a
96-cell table costs more than 16 calls. The crossover sits around
n = 8.

**Capacity sweep (n = 100):**

| W | Memo calls | Memo time | Tab cells | Tab time |
|---:|---:|---:|---:|---:|
| 100 | 17,217 | 3.2 ms | 10,100 | 0.44 ms |
| 1,000 | 133,387 | 28.2 ms | 100,100 | 5.9 ms |
| 2,500 | 150,243 | 36.2 ms | 250,100 | 14.8 ms |
| 5,000 | 150,243 | 33.6 ms | 500,100 | 27.6 ms |

Tabulation grows linearly in W, which makes the pseudo-polynomial
O(n × W) bound concrete: doubling the numeric capacity doubles the
work without adding a single item. Memoization levels off once W
passes the total item weight (about 1,505), because beyond that point
no new (item, capacity) states can be reached. Top-down DP only solves
the subproblems it actually needs.

### LCS (`lcs_performance.png`)

| Length | Recursive | Calls | Memo | Memo depth | Tabulation | Speedup (tab) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.13 ms | 1,941 | 14 µs | 18 | 6 µs | 23× |
| 12 | 6.17 ms | 95,414 | 23 µs | 22 | 8 µs | 788× |
| 14 | 46.0 ms | 700,405 | 28 µs | 25 | 10 µs | **4,631×** |
| 100 | — | — | 1.52 ms | 179 | 0.42 ms | — |
| 500 | — | — | 42.9 ms | 849 | 12.1 ms | — |
| 1,000 | — | — | 172 ms | 1,696 | 51.8 ms | — |

From length 8 to 14, recursive LCS calls grew 488×, an average of
about 8× for each +2 characters. That is well under the 16× that the
2^(m+n) worst case allows. Each character match takes a single
diagonal branch instead of two, and random DNA matches about a quarter
of the time. Because of this, the recursive cost depends on the input
itself: individual steps ranged from 1.4× (length 8 → 10) to 49×
(10 → 12). Both DP versions
scaled as O(m × n): from length 500 to 1,000 the cells grew 4× and
tabulation time grew 4.3×.

## Discussion

**When and why DP wins.** DP needs two properties. The first is
*overlapping subproblems*: naive `fib(45)` evaluates F(1) more than a
billion times, and there are only 46 distinct subproblems. The second
is *optimal substructure*: the best answer must be built from best
answers to smaller subproblems. For knapsack, once item i has been
taken or skipped, the best use of the remaining capacity is itself a
knapsack problem, so storing K(i, w) cannot lose the optimum. Where
subproblems do not overlap, as in merge sort's disjoint halves from
Week 2, a cache has nothing to reuse and only adds overhead. The
small-input rows show that overhead: at n = 4 both DP versions lost
to plain recursion.

**Memoization vs tabulation.** On large inputs tabulation was 3.3× faster
for LCS, 7.7× for knapsack, and 5–6× for Fibonacci. The work is the
same, but tabulation runs it as a tight loop over list rows, while
memoization pays for a Python function call, a cache lookup, and
bookkeeping at every step (about 275 ns per memoized knapsack call
against 55 ns per table cell). Tabulation also needs no recursion,
whereas memoized LCS at length 1,000 needed a 1,696-frame stack and
would crash under default Python settings. Memoization's advantage is
laziness: it computes only the states that can be reached, as the
capacity sweep showed. It is also easier to write, since it is the
recursive definition with a cache attached.

**Memory trade-offs.** DP trades memory for time, and the
data structure chosen for the cache matters. The knapsack memo uses a
dict keyed by (i, w) tuples and peaked at 32.7 MiB for n = 200, 4.2×
the tabulation table's 7.9 MiB. The LCS memo uses a preallocated grid,
so its memory matched tabulation (10.4 vs 11.4 MiB at length 1,000).
Tabulation also allows further savings. Because each knapsack or LCS
row depends only on the row above it, two rows are enough when only
the value is needed, which cuts space from O(n × W) to O(W). The
catch is that `trace_solution()` and `reconstruct_lcs()` need the full
table to walk backward, so recovering the actual answer costs the
extra memory.

## Case Study: Genome Alignment

LCS is the core of sequence alignment. Needleman–Wunsch global
alignment uses the same 2D table, with scores for matches,
mismatches, and gaps replacing LCS's +1 for a match and 0 otherwise.
`examples/week5_demo.py` compares two 21–22-base DNA fragments and
finds an 18-base conserved subsequence (82% similarity). The recursive
version could not handle real sequences: its 14-character run already
took 46 ms, and the cost multiplies with every added base. Even the
O(m × n) table becomes a problem at genome scale. Two 10⁵-base
sequences need 10¹⁰ cells, which is why production aligners combine
the DP recurrence with the two-row space trick (Hirschberg's
algorithm) and heuristics such as BLAST's seed-and-extend.

## Conclusion

In theory, DP replaces a recurrence's exponential call tree with a
table the size of its distinct subproblem space. In these benchmarks
that meant 82 seconds versus 1 microsecond, and inputs that were out
of reach became routine. Theory also predicted the details: φ⁵ growth
per step for naive Fibonacci, 4× per two knapsack items, and linear
scaling in n × W. Implementation decided the constants. Tabulation's
loops beat memoization's calls, a list grid beat a dict, and Python's
recursion limit determined whether the top-down version worked at all.
DP is only worth using where subproblems overlap, but where they do,
it changes what is computable in practice.
