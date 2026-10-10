"""
Monte Carlo primality testing: Fermat and Miller–Rabin.

Monte Carlo: the running time is fixed (k rounds), but the ANSWER may
be wrong with small probability. Both tests here are one-sided:

    "composite"  is always correct — the test found a witness, an
                 explicit proof that n is not prime.
    "prime"      means "k random bases all failed to find a witness".
                 For a composite n that happens only if every base was
                 a LIAR, so the error probability is
                 (fraction of liars)^k.

Fermat test. Fermat's little theorem: if n is prime then
a^(n−1) ≡ 1 (mod n) for every a not divisible by n. A base with
a^(n−1) ≢ 1 is a Fermat witness. Weakness: Carmichael numbers (561,
1105, 1729, ...) satisfy the congruence for EVERY base coprime to n,
so the only witnesses are bases sharing a factor with n. For 561 that
leaves ~57% liars per round, and the error shrinks only like 0.57^k.

Miller–Rabin. Write n − 1 = 2^s · d with d odd. For prime n the
sequence a^d, a^(2d), ..., a^(2^s·d) ≡ 1 must either start at 1 or hit
−1 just before the first 1, because ±1 are the only square roots of 1
modulo a prime. A base that breaks this pattern is a strong witness.
Rabin proved at most 1/4 of bases are strong liars for ANY odd
composite — Carmichael numbers included — so k rounds err with
probability at most 4^(−k): 1e−3 at k = 5, 1e−6 at k = 10, 1e−12 at
k = 20.

Cost: each round is one modular exponentiation, O(log n) squarings of
O(log n)-bit numbers = O(log³ n) with schoolbook multiplication, so
O(k · log³ n) per test. Trial division needs O(√n) divisions — at
n ≈ 10^18 that is 10^9 operations versus a few hundred for k = 20.
"""
import math
import random
from typing import Optional, Tuple

from src.utils.timer import CallStats



def _check_n(n: int, name: str) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"{name} expects an int, got {type(n).__name__}")


def _check_rounds(rounds: int) -> None:
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")


def _trivial(n: int) -> Optional[bool]:
    """Decide n < 4 and even n directly; None when a real test is
    needed. Deliberately no small-prime trial division: that would
    catch every Carmichael number below ~10^5 before the randomized
    test ever ran, hiding exactly the behavior being measured."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    return None


# ---------------------------------------------------------------------------
# Deterministic baseline
# ---------------------------------------------------------------------------

def is_prime_trial_division(n: int, stats: Optional[CallStats] = None
                            ) -> bool:
    """
    Exact primality by trial division with the 6k ± 1 wheel: every
    prime > 3 is of the form 6k ± 1, so only those divisors up to √n
    are tried.

    Complexity: O(√n) divisions — fine to ~10^12, hopeless at 10^30.
    """
    _check_n(n, "is_prime_trial_division")
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    divisions = 2
    limit = math.isqrt(n)
    f = 5
    result = True
    while f <= limit:
        divisions += 2
        if n % f == 0 or n % (f + 2) == 0:
            result = False
            break
        f += 6
    if stats is not None:
        stats.count(divisions)
    return result


# ---------------------------------------------------------------------------
# Single-base witness checks
# ---------------------------------------------------------------------------

def is_fermat_witness(a: int, n: int) -> bool:
    """True if base a proves n composite via a^(n−1) ≢ 1 (mod n)."""
    return pow(a, n - 1, n) != 1


def decompose(n: int) -> Tuple[int, int]:
    """(s, d) with n − 1 = 2^s · d and d odd. Requires odd n >= 3."""
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return s, d


def is_strong_witness(a: int, n: int, s: Optional[int] = None,
                      d: Optional[int] = None) -> bool:
    """
    True if base a proves odd n > 2 composite in the Miller–Rabin
    sense: a^d ≢ ±1 and none of a^(2^r·d), 0 < r < s, is ≡ −1.
    """
    if s is None or d is None:
        s, d = decompose(n)
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return False
        if x == 1:          # reached 1 without passing −1
            return True
    return True


# ---------------------------------------------------------------------------
# Monte Carlo tests
# ---------------------------------------------------------------------------

def fermat_test(n: int, rounds: int = 20,
                rng: Optional[random.Random] = None,
                stats: Optional[CallStats] = None) -> bool:
    """
    Probabilistic primality via Fermat's little theorem.

    Args:
        n: Integer to test.
        rounds: Number of random bases a in [2, n − 2] to try (k).
        rng: Optional random.Random for reproducible bases.
        stats: Optional counter; receives one count per modular
            exponentiation performed.

    Returns:
        False if n is certainly composite; True if n is prime OR is a
        composite that fooled every round (Carmichael numbers fool
        most bases).

    Complexity: O(k · log³ n).
    """
    _check_n(n, "fermat_test")
    _check_rounds(rounds)
    trivial = _trivial(n)
    if trivial is not None:
        return trivial
    rng = rng or random
    for _ in range(rounds):
        if stats is not None:
            stats.count()
        if is_fermat_witness(rng.randint(2, n - 2), n):
            return False
    return True


def miller_rabin(n: int, rounds: int = 20,
                 rng: Optional[random.Random] = None,
                 stats: Optional[CallStats] = None) -> bool:
    """
    Probabilistic primality via the Miller–Rabin strong-witness test.

    Args:
        n: Integer to test.
        rounds: Number of random bases a in [2, n − 2] to try (k).
        rng: Optional random.Random for reproducible bases.
        stats: Optional counter; receives one count per round
            (each round is one modular exponentiation plus up to s − 1
            squarings).

    Returns:
        False if n is certainly composite; True if n is prime or (with
        probability at most 4^(−rounds)) a composite that fooled every
        round.

    Complexity: O(k · log³ n).
    """
    _check_n(n, "miller_rabin")
    _check_rounds(rounds)
    trivial = _trivial(n)
    if trivial is not None:
        return trivial
    rng = rng or random
    s, d = decompose(n)
    for _ in range(rounds):
        if stats is not None:
            stats.count()
        if is_strong_witness(rng.randint(2, n - 2), n, s, d):
            return False
    return True


# ---------------------------------------------------------------------------
# Exact liar counts (for measuring the per-round error)
# ---------------------------------------------------------------------------

def liar_fraction(n: int, test: str = "miller_rabin") -> float:
    """
    Exact fraction of bases a in [2, n − 2] that LIE about composite n
    — i.e. the probability one round of `test` wrongly says "prime".

    Enumerates every base, so O(n · log³ n): meant for n up to ~10^5.

    Args:
        n: Odd composite >= 9.
        test: "fermat" or "miller_rabin".
    """
    _check_n(n, "liar_fraction")
    if n < 9 or n % 2 == 0:
        raise ValueError(f"n must be an odd composite >= 9, got {n}")
    if test == "fermat":
        liars = sum(1 for a in range(2, n - 1)
                    if not is_fermat_witness(a, n))
    elif test == "miller_rabin":
        s, d = decompose(n)
        liars = sum(1 for a in range(2, n - 1)
                    if not is_strong_witness(a, n, s, d))
    else:
        raise ValueError(f"unknown test {test!r}")
    return liars / (n - 3)
