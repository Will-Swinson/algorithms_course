"""
Polynomial rolling hashes for Rabin–Karp.

A window x_0 … x_{m−1} hashes to

    H = (x_0·B^{m−1} + x_1·B^{m−2} + … + x_{m−1}) mod q

and sliding the window one step costs O(1):

    H' = ((H − x_0·B^{m−1})·B + x_m) mod q.

Two different strings of length m collide when their difference
polynomial (degree < m) vanishes at B mod q. For prime q and B chosen
uniformly at random, a polynomial of degree < m has at most m − 1
roots, so

    Pr[collision] ≤ (m − 1) / q      for any fixed pair of strings.

That is the probabilistic guarantee, and it is why the defaults are a
large prime (the Mersenne prime 2^61 − 1) and a random base. The bad
choices are easy to demonstrate (the benchmark does):

    small q           ≈ n/q spurious hits per scan
    B = 1             H is the character sum: every anagram collides
    B ≡ 0 (mod q)     H is the last character only
    q = 2^k, even B   high powers of B vanish: only the last few
                      characters count

Classes:
    RollingHash    one (base, modulus) pair: hash, roll, window hashes
    PrefixHash     O(n) prefix table → O(1) hash of any substring
    DoubleHash     two independent moduli (collision ≈ 1/(q1·q2))
"""
import random
from typing import Iterator, List, Optional, Sequence, Tuple, Union

MERSENNE_61 = (1 << 61) - 1
DEFAULT_MODULUS = MERSENNE_61

Text = Union[str, bytes, Sequence[int]]


def as_codes(text: Text) -> Sequence[int]:
    """Integer codes of a string: bytes for ASCII (fast indexing),
    code points otherwise; bytes and int sequences pass through."""
    if isinstance(text, str):
        return text.encode("ascii") if text.isascii() else [ord(c) for c in text]
    return text


def random_base(modulus: int = DEFAULT_MODULUS,
                rng: Optional[random.Random] = None,
                low: int = 256) -> int:
    """Uniform base in [low, modulus − 2] (above the alphabet size)."""
    rng = rng or random.Random()
    low = min(low, max(2, modulus - 2))
    return rng.randint(low, max(low, modulus - 2))


class RollingHash:
    """Polynomial hash with a fixed base and modulus."""

    def __init__(self, base: Optional[int] = None,
                 modulus: int = DEFAULT_MODULUS,
                 rng: Optional[random.Random] = None):
        if modulus < 2:
            raise ValueError("modulus must be >= 2")
        if base is not None and base < 1:
            raise ValueError("base must be >= 1")
        self.modulus = modulus
        self.base = random_base(modulus, rng) if base is None else base % modulus

    def hash(self, text: Text) -> int:
        h, b, q = 0, self.base, self.modulus
        for x in as_codes(text):
            h = (h * b + x) % q
        return h

    def power(self, m: int) -> int:
        """B^(m−1) mod q: the weight of the character leaving the window."""
        return pow(self.base, max(m - 1, 0), self.modulus)

    def roll(self, h: int, outgoing: int, incoming: int, high: int) -> int:
        """Slide one step: drop `outgoing` (weight `high`), add `incoming`."""
        return ((h - outgoing * high) * self.base + incoming) % self.modulus

    def window_hashes(self, text: Text, m: int) -> Iterator[Tuple[int, int]]:
        """(start, hash) of every length-m window, in O(n) total."""
        codes = as_codes(text)
        n = len(codes)
        if m <= 0 or m > n:
            return
        b, q = self.base, self.modulus
        high = self.power(m)
        h = 0
        for x in codes[:m]:
            h = (h * b + x) % q
        yield 0, h
        for i in range(m, n):
            h = ((h - codes[i - m] * high) * b + codes[i]) % q
            yield i - m + 1, h

    def __repr__(self) -> str:
        return f"RollingHash(base={self.base}, modulus={self.modulus})"


class PrefixHash:
    """
    pre[i] = hash of text[:i]; then any substring hash is
    hash(text[i:j]) = pre[j] − pre[i]·B^(j−i) (mod q), in O(1).
    """

    def __init__(self, text: Text, hasher: Optional[RollingHash] = None):
        self.hasher = hasher or RollingHash()
        b, q = self.hasher.base, self.hasher.modulus
        codes = as_codes(text)
        self.pre = [0] * (len(codes) + 1)
        self.pw = [1] * (len(codes) + 1)
        for i, x in enumerate(codes):
            self.pre[i + 1] = (self.pre[i] * b + x) % q
            self.pw[i + 1] = self.pw[i] * b % q

    def substring(self, i: int, j: int) -> int:
        if not 0 <= i <= j < len(self.pre):
            raise IndexError(f"substring [{i}, {j}) out of range")
        q = self.hasher.modulus
        return (self.pre[j] - self.pre[i] * self.pw[j - i]) % q


class DoubleHash:
    """Pair of independent rolling hashes; hash() returns a tuple."""

    def __init__(self, first: Optional[RollingHash] = None,
                 second: Optional[RollingHash] = None,
                 rng: Optional[random.Random] = None):
        rng = rng or random.Random()
        self.first = first or RollingHash(modulus=MERSENNE_61, rng=rng)
        self.second = second or RollingHash(modulus=1_000_000_007, rng=rng)

    def hash(self, text: Text) -> Tuple[int, int]:
        return self.first.hash(text), self.second.hash(text)

    def window_hashes(self, text: Text, m: int) -> Iterator[Tuple[int, Tuple[int, int]]]:
        for (i, a), (_, b) in zip(self.first.window_hashes(text, m),
                                  self.second.window_hashes(text, m)):
            yield i, (a, b)


def find_collision(length: int, hasher: RollingHash,
                   alphabet: str = "abcdefghijklmnopqrstuvwxyz",
                   rng: Optional[random.Random] = None,
                   max_tries: int = 2_000_000) -> Optional[Tuple[str, str]]:
    """
    Birthday search for two different strings of `length` with equal
    hashes: expected ≈ √(πq/2) tries, so feasible only for small q —
    which is exactly the point.
    """
    rng = rng or random.Random(0)
    seen = {}
    for _ in range(max_tries):
        s = "".join(rng.choice(alphabet) for _ in range(length))
        h = hasher.hash(s)
        other = seen.get(h)
        if other is not None and other != s:
            return other, s
        seen[h] = s
    return None


def collision_bound(m: int, modulus: int) -> float:
    """Upper bound (m − 1)/q on the collision probability of two fixed
    length-m strings under a random base and prime modulus."""
    return max(m - 1, 0) / modulus


def hash_tokens(tokens: Sequence[str], hasher: Optional[RollingHash] = None) -> List[int]:
    """Hash each token (e.g. word) to an integer code for token-level RK."""
    hasher = hasher or RollingHash(base=131, modulus=MERSENNE_61)
    return [hasher.hash(t) for t in tokens]
