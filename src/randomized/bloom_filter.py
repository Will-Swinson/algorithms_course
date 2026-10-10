"""
Bloom filter — probabilistic set membership in a few bits per item.

A Bloom filter is an m-bit array plus k hash functions. To add an
item, set the k bits it hashes to. To check an item, look at the same
k bits: if any is 0 the item was definitely never added; if all are 1
it PROBABLY was. The errors are one-sided, like Miller–Rabin's:

    check() == False   always correct (no false negatives)
    check() == True    wrong with probability ≈ (1 − e^(−kn/m))^k

because unrelated items may have set all k of those bits between them.
The filter never stores the items, so its size depends only on m, not
on how long the items are: ~9.6 bits per item gives 1% false
positives whether the items are 8-byte ints or 2-KB URLs. A Python set
of the same strings costs hundreds of bits per item.

Where the randomness is: the hash functions. Each filter draws a
random 16-byte salt for BLAKE2b, so its k hash functions are a random
member of a large family. An adversary who doesn't know the salt
cannot pick items that collide on purpose.

k hash functions from one hash (Kirsch & Mitzenmacher 2006): split
one 128-bit digest into h1 and h2 and use

    g_i(x) = (h1 + i·h2) mod m,   i = 0, ..., k − 1

This has the same asymptotic false-positive rate as k independent
hashes, for the cost of a single hash computation.

Complexity: add and check are O(k) — independent of how many items
the filter holds. Space O(m) bits; with m chosen for a target rate,
O(n).
"""
import hashlib
import math
import random
from typing import Any, Iterable, Optional

from src.utils.probability_analysis import (
    bloom_false_positive_rate, optimal_bloom_parameters,
)


def _to_bytes(item: Any) -> bytes:
    """Stable byte encoding: bytes as-is, str as UTF-8, else repr()."""
    if isinstance(item, bytes):
        return b"b" + item
    if isinstance(item, str):
        return b"s" + item.encode("utf-8")
    return b"r" + repr(item).encode("utf-8")


class BloomFilter:
    """
    Bloom filter over any items with a stable repr (str, bytes, int,
    tuples of those, ...).

        bf = BloomFilter.for_capacity(10_000, fp_rate=0.01)
        bf.add("alice")
        "alice" in bf      # True
        "bob" in bf        # almost certainly False
    """

    def __init__(self, num_bits: int, num_hashes: int,
                 seed: Optional[int] = None) -> None:
        """
        Args:
            num_bits: m, size of the bit array (> 0).
            num_hashes: k, bits set/probed per item (> 0).
            seed: Seeds the random hash salt; None for a fresh random
                salt. Two filters can only be compared bit-for-bit if
                they share a seed.
        """
        if num_bits <= 0:
            raise ValueError(f"num_bits must be positive, got {num_bits}")
        if num_hashes <= 0:
            raise ValueError(f"num_hashes must be positive, got {num_hashes}")
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self._bits = bytearray((num_bits + 7) // 8)
        self._salt = random.Random(seed).randbytes(16)
        self._count = 0

    @classmethod
    def for_capacity(cls, capacity: int, fp_rate: float = 0.01,
                     seed: Optional[int] = None) -> "BloomFilter":
        """A filter sized so `capacity` items give `fp_rate` false
        positives, using the memory-optimal m and k."""
        m, k = optimal_bloom_parameters(capacity, fp_rate)
        return cls(m, k, seed)

    def _positions(self, item: Any) -> Iterable[int]:
        digest = hashlib.blake2b(_to_bytes(item), digest_size=16,
                                 salt=self._salt).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") | 1   # odd: never 0
        m = self.num_bits
        return ((h1 + i * h2) % m for i in range(self.num_hashes))

    def add(self, item: Any) -> None:
        """Insert item: set its k bits. O(k)."""
        bits = self._bits
        for pos in self._positions(item):
            bits[pos >> 3] |= 1 << (pos & 7)
        self._count += 1

    def check(self, item: Any) -> bool:
        """
        False: item was definitely never added.
        True: item was probably added (false-positive rate
        ≈ expected_fp_rate()). O(k).
        """
        bits = self._bits
        return all(bits[pos >> 3] >> (pos & 7) & 1
                   for pos in self._positions(item))

    __contains__ = check

    def update(self, items: Iterable[Any]) -> None:
        """add() every item."""
        for item in items:
            self.add(item)

    def __len__(self) -> int:
        """Number of add() calls (duplicates counted each time)."""
        return self._count

    @property
    def load_factor(self) -> float:
        """Items added per bit, n / m."""
        return self._count / self.num_bits

    def bits_set(self) -> int:
        """How many of the m bits are 1."""
        return sum(bin(byte).count("1") for byte in self._bits)

    def fill_ratio(self) -> float:
        """Fraction of bits set. ≈ 1 − e^(−kn/m); 0.5 at optimal k."""
        return self.bits_set() / self.num_bits

    def expected_fp_rate(self) -> float:
        """Theoretical false-positive rate for the items added so far."""
        return bloom_false_positive_rate(self.num_bits, self._count,
                                         self.num_hashes)

    def estimated_fp_rate(self) -> float:
        """FP rate implied by the bits actually set: fill_ratio^k."""
        return self.fill_ratio() ** self.num_hashes

    def estimated_count(self) -> float:
        """
        Estimate of distinct items added, from the bits alone
        (Swamidass & Baldi): n ≈ −(m / k)·ln(1 − X/m).
        """
        x = self.bits_set()
        if x == self.num_bits:
            return math.inf
        return -self.num_bits / self.num_hashes * math.log(
            1 - x / self.num_bits)

    def memory_bytes(self) -> int:
        """Bytes used by the bit array."""
        return len(self._bits)

    def __repr__(self) -> str:
        return (f"BloomFilter(num_bits={self.num_bits}, "
                f"num_hashes={self.num_hashes}, items={self._count})")
