# Week 3 Performance Analysis Report: Core Data Structures

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** Data Structures — Heaps, AVL Trees, and Hash Tables

## 1. Executive Summary

Structure choice changed search cost by four orders of magnitude on
identical data: at n = 1,000,000 a linear list scan averaged 2,396 µs
per lookup, the AVL tree 1.39 µs, and the chaining hash table 0.28 µs.
Every structure's measured growth curve matched its asymptotic class —
logarithmic (heap, AVL), constant (hash), linear (list) — and an
engineered all-collision worst case confirmed that a hash table without
a good hash function degrades to exactly the linear behavior theory
predicts.

## 2. Methodology

**Hardware & environment:** Apple Silicon Mac (arm64), macOS 15.7.3,
CPython 3.13.5 in the project virtual environment, single-threaded.

**Datasets:** input sizes 10³, 10⁴, 10⁵, 10⁶. Keys are unique random
integers drawn with a seeded RNG (seed 42) from a range 10× the input
size, so every structure at a given size receives identical data.
Lookup samples are 10,000 present keys per configuration (100 for the
linear list, whose scans are ~1,000× more expensive); deletions use
10,000 sampled keys.

**Measurement:** `time.perf_counter()` around bulk operation loops,
reported as time per operation (total ÷ operations), the standard
technique when a single operation is far below timer resolution.
Timing 10³–10⁶ operations per sample averages away scheduler noise;
the load-factor experiment additionally pins capacity (2¹⁷ slots,
rehashing disabled) so load factor is the only variable. Raw data:
`benchmarks/results/comparison_table.csv` (81 measurements) produced
by `benchmarks/week3_structures_benchmark.py`.

## 3. Results

### Heap performance (vs Python `heapq`)

| n | MinHeap insert | MinHeap extract | heapq insert | heapq extract |
|---:|---:|---:|---:|---:|
| 10³ | 0.220 µs | 1.080 µs | 0.041 µs | 0.082 µs |
| 10⁶ | 0.238 µs | 2.780 µs | 0.047 µs | 0.666 µs |

Two shapes stand out. **Insert is flat** (0.22 → 0.24 µs across 1,000×
more data) even though insert is O(log n) worst case: a random new
element usually belongs near the bottom, so the average sift-up is O(1)
— the worst case exists but random input rarely triggers it. **Extract
shows the logarithm** (1.08 → 2.78 µs): extraction always moves the
last leaf to the root and sifts it down the full height, so its cost
must grow with log n, and does. `heapq` is 4–7× faster with identical
asymptotics — it is the same algorithm compiled to C, a pure
constant-factor gap that log-log plots show as parallel lines
(`benchmarks/results/heap_performance.png`).

### AVL tree (vs Python `dict`)

| n | AVL insert | AVL search | AVL delete | dict insert | dict search |
|---:|---:|---:|---:|---:|---:|
| 10³ | 1.86 µs | 0.33 µs | 1.86 µs | 0.062 µs | 0.014 µs |
| 10⁶ | 5.22 µs | 1.39 µs | 5.93 µs | 0.092 µs | 0.097 µs |

The AVL tree held its O(log n) promise on all three operations —
search grew 4.3× while n grew 1,000× (a pure log-height model predicts
~1.9×; the remainder is memory hierarchy, as a million-node tree no
longer fits in cache). The tree's height after 10⁶ random inserts was
within the theoretical bound of 1.44·log₂(n+2) ≈ 29, and the balance
audit in the test suite verifies every node's balance factor stays in
{−1, 0, +1}. `dict` wins raw speed by 15–60× — it is a C hash table
doing O(1) work — but the comparison clarifies what the AVL tree buys
for its logarithm: **order**. `in_order_traversal()` yields sorted
output, and range/min/max queries follow tree paths; a dict can only
deliver those by sorting its keys at O(n log n) per query.

### Hash tables: load factor and collision strategy

| n | Chaining insert | Chaining search | Probing insert | Probing search |
|---:|---:|---:|---:|---:|
| 10³ | 0.61 µs | 0.098 µs | 0.78 µs | 0.209 µs |
| 10⁶ | 2.12 µs | 0.281 µs | 1.15 µs | 0.449 µs |

Search stayed near-constant for both strategies across 1,000× growth
(the ~3× drift again tracks cache misses, not algorithmic growth). The
fixed-capacity load-factor sweep (`hash_performance.png`, right panel)
separates the strategies: **chaining's** lookup cost rose gently from
0.086 to 0.119 µs between load 0.1 and 0.9 — chains just get slightly
longer — while **linear probing** rose from 0.191 to 0.471 µs with a
sharp bend after load 0.7, the classic clustering signature: occupied
runs merge into long clusters that every probe must traverse. This is
precisely why the implementation rehashes probing at load 0.6 but
chaining at 0.75.

**Engineered worst case:** with every key hashing to bucket 0 and
rehashing disabled, lookups measured 25.5 µs at n = 10³, 126 µs at
5×10³, and 255 µs at 10⁴ — a perfect 10× time for 10× n. The
average-case O(1) table became an O(n) linked-list scan, ~2,700×
slower than the healthy table at the same size.

### Comparison table: asymptotic vs empirical

| Structure | Operation | Theoretical | Growth 10³→10⁶ | Verdict |
|---|---|---|---:|---|
| list | search | O(n) | 1,084× | linear ✓ |
| MinHeap | extract | O(log n) | 2.6× | logarithmic ✓ |
| AVL tree | search | O(log n) | 4.3× | logarithmic ✓ |
| Chaining | search | O(1) | 2.9× | ~constant ✓ (cache drift) |
| Probing | search | O(1) | 2.1× | ~constant ✓ (cache drift) |
| dict | search | O(1) | 6.9× (0.014→0.097 µs) | ~constant ✓ |
| Chaining (all-collide) | get | O(n) worst | 10× per 10× n | linear ✓ |

## 4. Amortized Analysis

Hash-table insert is the textbook amortized case. Most inserts cost
O(1), but the insert that pushes the load factor past its threshold
triggers a rehash that touches all n entries — an O(n) spike. The
doubling schedule is what tames this: growing 8 → 16 → … → 2²¹ slots
on the way to 10⁶ entries costs about 2 million total re-insertions
across ~18 rehashes, i.e. ~2 extra units of work per insert *averaged
over the sequence* — O(1) amortized, by the aggregate method. The
measurements agree: chaining's *average* insert grew only 0.61 → 2.12
µs across 1,000× more data even though every measured run absorbed
multiple full rehashes, and probing's average was flat (0.78 → 1.15
µs). The same argument covers Python's `list.append` behind the heap:
its geometric over-allocation is why `MinHeap.insert` could stay at
~0.23 µs while the backing array grew to a million slots. The
practical caveat: amortized O(1) is an average, not a promise about
any single operation — the unlucky insert that lands on the 10⁶-entry
rehash pays for all of it, which matters for latency-sensitive
systems and motivates incremental-rehashing designs (as used by
Redis).

## 5. Practical Recommendations

- **Hash tables** are the default associative container: fastest
  measured lookups (0.1–0.4 µs) whenever only exact-key access is
  needed. Prefer chaining when deletions are frequent or the load
  factor may run high; probing's contiguous array is more
  cache-friendly at low load but needs tombstones and earlier
  rehashing.
- **AVL trees** earn their 3–15× slower lookups when order matters:
  sorted iteration, range queries, nearest-key lookups, or a hard
  O(log n) *worst-case* guarantee (hash tables can be driven to O(n)
  by adversarial keys, as the collision experiment showed).
- **Heaps** are unbeatable for the specific job of "repeatedly give me
  the extreme element": O(1) peek, O(log n) insert/extract, zero
  pointer overhead in an array. That is a priority queue — schedulers,
  Dijkstra's algorithm, top-k streams (demonstrated in
  `examples/week3_demo.py`).
- **Space/time trade-offs observed:** probing stores entries in two
  flat arrays (compact, cache-friendly); chaining pays one node object
  per entry but degrades more gracefully; the AVL tree pays two child
  pointers plus a height per node to maintain order; the heap pays
  nothing beyond the array. A rehash also transiently doubles memory.
- **Hybrid/specialized structures:** the measurements motivate real
  designs — Python's dict itself (open addressing with perturbed
  probing), Timsort's run-based merging (Week 2), incremental
  rehashing for latency, and B-trees, which take the AVL idea wide
  (many keys per node) to respect the cache effects visible in every
  large-n measurement here.

## 6. Conclusion

The week's central lesson is that data-structure choice *is* an
algorithmic decision: the same "find a key" workload spans 0.28 µs to
2,396 µs — four orders of magnitude — purely on structural grounds,
and each cost curve was predictable from first principles before any
code ran. The empirical work also sharpened three theory points: the
gap between worst-case and typical behavior (heap insert's flat
average), the meaning of amortized cost (rehashing's invisible O(n)
spikes), and the fragility of average-case guarantees (one bad hash
function re-creates the linked list). Scalable systems come from
matching the structure's invariant — heap order, AVL balance, hash
uniformity — to the access pattern the algorithm actually needs.

## References

1. Amakobe, M. *Advanced Algorithms: A Journey Through Computational
   Problem Solving* — Ch. 3 (Core Data Structures); §1.6 (Algorithm
   Laboratory).
2. Python Software Foundation: `heapq` and `time.perf_counter()`
   documentation, Python 3.13.
3. Project reports: `docs/performance_analysis.md` (Week 1),
   `analysis/week2_report.md` (Week 2).
