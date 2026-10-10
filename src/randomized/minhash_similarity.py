"""
MinHash — estimating Jaccard similarity from short signatures.

Jaccard similarity of two sets is |A ∩ B| / |A ∪ B|. Computing it
exactly needs both full sets. MinHash compresses each set to a
signature of k numbers such that comparing two SIGNATURES estimates
the similarity of the SETS.

The key fact: pick a random hash function h and look at the element
of A ∪ B with the smallest hash. It is equally likely to be any
element of the union, and it is in BOTH sets exactly when it lies in
the intersection, so

    P[ min h(A) == min h(B) ] = |A ∩ B| / |A ∪ B| = J(A, B).

One hash gives a 0/1 coin with bias J. A signature of k independent
hashes gives k coins, and the fraction that agree estimates J with
standard error sqrt(J(1 − J) / k) — 0.125 at k = 16, 0.022 at
k = 512 for J = 0.5. The error does not depend on the set sizes.

Hash family: h_i(x) = (a_i·x + b_i) mod p with p = 2^61 − 1 (a
Mersenne prime) and random a_i, b_i — a universal family, so the
argmin is close to uniform over the union. Tokens are first mapped to
64-bit integers with BLAKE2b.

Complexity: signature O(k · |S|); comparing two signatures O(k),
independent of the set sizes.
"""
import hashlib
import random
import re
from typing import Any, Iterable, List, Optional, Set

MERSENNE_PRIME = (1 << 61) - 1
MAX_HASH = MERSENNE_PRIME           # larger than any h_i value


def jaccard_similarity(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Exact |A ∩ B| / |A ∪ B|; two empty sets count as identical (1.0)."""
    a, b = set(a), set(b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens (letters, digits, apostrophes)."""
    return re.findall(r"[a-z0-9']+", text.lower())


def shingles(text: str, k: int = 3, unit: str = "word") -> Set[str]:
    """
    The set of k-shingles (contiguous runs of k tokens) of a text.

    unit="word" uses word tokens; unit="char" uses characters of the
    whitespace-normalized lowercase text. Shingles capture word ORDER,
    so reordered documents share fewer shingles than bag-of-words sets.
    A text shorter than k yields one shingle of everything it has.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if unit == "word":
        tokens = tokenize(text)
        joiner = " "
    elif unit == "char":
        tokens = list(" ".join(text.lower().split()))
        joiner = ""
    else:
        raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")
    if not tokens:
        return set()
    if len(tokens) <= k:
        return {joiner.join(tokens)}
    return {joiner.join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _token_hash(token: Any) -> int:
    """Stable 64-bit integer for a token (same across runs, unlike hash())."""
    data = token if isinstance(token, bytes) else repr(token).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(),
                          "little")


class MinHash:
    """
    A family of k random hash functions and the signatures they produce.

    Signatures are only comparable when made by the SAME MinHash
    object (or two built with the same num_hashes and seed).

        mh = MinHash(128, seed=1)
        mh.similarity(mh.signature(doc_a), mh.signature(doc_b))
    """

    def __init__(self, num_hashes: int = 128,
                 seed: Optional[int] = None) -> None:
        if num_hashes < 1:
            raise ValueError(f"num_hashes must be >= 1, got {num_hashes}")
        rng = random.Random(seed)
        self.num_hashes = num_hashes
        # Drawn as (a, b) pairs so that MinHash(k, s) is exactly the
        # first k functions of MinHash(K, s) for any K > k
        coeffs = [(rng.randrange(1, MERSENNE_PRIME),
                   rng.randrange(0, MERSENNE_PRIME))
                  for _ in range(num_hashes)]
        self._a = [a for a, _ in coeffs]
        self._b = [b for _, b in coeffs]

    def signature(self, items: Iterable[Any]) -> List[int]:
        """
        The k minimum hash values of the set. An empty set gets all
        MAX_HASH. O(k · |S|).
        """
        xs = [_token_hash(x) % MERSENNE_PRIME for x in set(items)]
        if not xs:
            return [MAX_HASH] * self.num_hashes
        p = MERSENNE_PRIME
        return [min((a * x + b) % p for x in xs)
                for a, b in zip(self._a, self._b)]

    @staticmethod
    def similarity(sig_a: List[int], sig_b: List[int]) -> float:
        """Estimated Jaccard similarity: fraction of agreeing positions.
        O(k)."""
        if len(sig_a) != len(sig_b):
            raise ValueError("signatures have different lengths "
                             f"({len(sig_a)} vs {len(sig_b)})")
        if not sig_a:
            raise ValueError("signatures are empty")
        return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)

    def estimate(self, a: Iterable[Any], b: Iterable[Any]) -> float:
        """Signature both sets and compare them."""
        return self.similarity(self.signature(a), self.signature(b))


def minhash_similarity(a: Iterable[Any], b: Iterable[Any],
                       num_hashes: int = 128,
                       seed: Optional[int] = None) -> float:
    """One-shot MinHash estimate of J(a, b) with a fresh hash family."""
    return MinHash(num_hashes, seed).estimate(a, b)
