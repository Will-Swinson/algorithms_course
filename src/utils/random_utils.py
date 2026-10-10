"""
Seeded random-input generators for the Week 7 randomized algorithms.

Two kinds of randomness show up this week and they are kept apart:

- INPUT randomness: the arrays, numbers, and sets the algorithms are
  run on. Generated here from a fixed seed so every benchmark and test
  sees the same inputs on every run.
- ALGORITHM randomness: the coin flips an algorithm makes internally
  (pivot choice, Miller–Rabin bases, hash-function choice). Each
  algorithm takes an optional `rng` (a random.Random) so a caller can
  make those flips reproducible too, or leave it None to use the
  module-level `random` generator.

Separating the two lets the benchmark hold the input fixed and vary
only the algorithm's seed, which isolates the variance the algorithm's
own randomness causes.
"""
import random
from typing import List, Optional, Set, Tuple

ARRAY_KINDS = ("random", "sorted", "reverse", "few_unique")


def make_rng(seed: Optional[int] = None) -> random.Random:
    """An independent generator: seeded if `seed` is given."""
    return random.Random(seed)


def random_array(n: int, kind: str = "random",
                 seed: Optional[int] = None) -> List[int]:
    """
    n integers in one of four orders:

        random      distinct values 0..n-1, shuffled
        sorted      0, 1, ..., n-1          (worst case for first-pivot)
        reverse     n-1, ..., 1, 0          (also worst case)
        few_unique  values drawn from only 10 distinct keys

    Distinct keys in the first three let the measured comparison count
    be checked against the exact expected-value formulas, which assume
    no ties.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    rng = make_rng(seed)
    if kind == "random":
        arr = list(range(n))
        rng.shuffle(arr)
        return arr
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n - 1, -1, -1))
    if kind == "few_unique":
        return [rng.randrange(10) for _ in range(n)]
    raise ValueError(f"unknown kind {kind!r}; expected one of {ARRAY_KINDS}")


def random_strings(count: int, length: int = 12, prefix: str = "",
                   seed: Optional[int] = None) -> List[str]:
    """
    `count` distinct random lowercase strings.

    A `prefix` keeps two generated batches disjoint (e.g. "in-" for
    items inserted into a Bloom filter, "out-" for items that never
    were), so any "yes" for an "out-" item is a genuine false positive.
    """
    rng = make_rng(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    seen: Set[str] = set()
    out = []
    while len(out) < count:
        s = prefix + "".join(rng.choice(letters) for _ in range(length))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def random_set_pair(size: int, jaccard: float,
                    seed: Optional[int] = None) -> Tuple[Set[int], Set[int]]:
    """
    Two sets of `size` integers each whose Jaccard similarity is as
    close to `jaccard` as integer counts allow.

    With |A| = |B| = s and |A ∩ B| = i, the union has 2s − i elements,
    so J = i / (2s − i), which solves to i = 2sJ / (1 + J).
    """
    if not 0.0 <= jaccard <= 1.0:
        raise ValueError(f"jaccard must be in [0, 1], got {jaccard}")
    rng = make_rng(seed)
    shared = round(2 * size * jaccard / (1 + jaccard))
    universe = rng.sample(range(10 ** 9), 2 * size - shared)
    common = universe[:shared]
    a = set(common) | set(universe[shared:size])
    b = set(common) | set(universe[size:])
    return a, b

