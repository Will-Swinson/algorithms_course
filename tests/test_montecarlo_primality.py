"""Test suite for Fermat and Miller–Rabin primality testing and the
trial-division baseline."""

import random

import pytest

from src.randomized.montecarlo_primality import (
    decompose, fermat_test, is_fermat_witness, is_prime_trial_division,
    is_strong_witness, liar_fraction, miller_rabin,
)
from src.utils.probability_analysis import miller_rabin_error_bound
from src.utils.timer import CallStats


def sieve(limit):
    is_p = [False, False] + [True] * (limit - 1)
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = [False] * len(is_p[i * i::i])
    return is_p


PRIMES_TO_5000 = sieve(5_000)
CARMICHAEL = [561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341]
LARGE_PRIMES = [
    2 ** 61 - 1,                      # Mersenne
    2 ** 89 - 1,
    2 ** 127 - 1,
    1_000_000_007,
    998_244_353,
    10 ** 18 + 9,
]
LARGE_COMPOSITES = [
    (2 ** 61 - 1) * (2 ** 31 - 1),
    1_000_000_007 * 998_244_353,
    3_215_031_751,                    # strong pseudoprime to bases 2,3,5,7
    2 ** 64 + 1,                      # = 274177 × 67280421310721
]


class TestTrialDivision:

    def test_matches_sieve(self):
        for n in range(-5, 5_001):
            assert is_prime_trial_division(n) == (n >= 0 and
                                                  PRIMES_TO_5000[n]), n

    def test_large_known_values(self):
        assert is_prime_trial_division(1_000_000_007)
        assert not is_prime_trial_division(1_000_000_007 * 3)

    def test_counts_divisions(self):
        stats = CallStats()
        is_prime_trial_division(1_000_003, stats)
        assert 300 < stats.calls < 400        # ≈ √n / 3

    def test_type_check(self):
        with pytest.raises(TypeError):
            is_prime_trial_division(7.0)
        with pytest.raises(TypeError):
            is_prime_trial_division(True)


@pytest.mark.parametrize("test", [fermat_test, miller_rabin],
                         ids=["fermat", "miller_rabin"])
class TestBothTests:

    def test_never_rejects_a_prime(self, test):
        # one-sided error: primes ALWAYS pass, for every base
        rng = random.Random(1)
        for n in range(2, 5_001):
            if PRIMES_TO_5000[n]:
                assert test(n, 10, rng), n

    def test_large_primes(self, test):
        rng = random.Random(2)
        for p in LARGE_PRIMES:
            assert test(p, 20, rng)

    def test_large_composites(self, test):
        rng = random.Random(3)
        for c in LARGE_COMPOSITES:
            assert not test(c, 20, rng)

    def test_small_cases(self, test):
        assert [n for n in range(-3, 4) if test(n)] == [2, 3]
        assert not test(4) and not test(100)

    def test_rounds_validation(self, test):
        with pytest.raises(ValueError):
            test(97, 0)

    def test_type_check(self, test):
        with pytest.raises(TypeError):
            test("97")

    def test_counts_rounds_on_prime(self, test):
        stats = CallStats()
        test(1_000_000_007, 12, random.Random(0), stats)
        assert stats.calls == 12

    def test_stops_at_first_witness(self, test):
        stats = CallStats()
        test(1_000_000_007 * 998_244_353, 20, random.Random(0), stats)
        assert stats.calls < 20


class TestMillerRabin:

    def test_odd_composites_rejected_with_few_rounds(self):
        rng = random.Random(4)
        wrong = [n for n in range(9, 5_001, 2)
                 if not PRIMES_TO_5000[n] and miller_rabin(n, 10, rng)]
        assert wrong == []

    def test_carmichael_numbers_rejected(self):
        rng = random.Random(5)
        for n in CARMICHAEL:
            assert not miller_rabin(n, 10, rng)

    def test_strong_witness_definition(self):
        # 2047 = 23·89 is a strong pseudoprime to base 2 but not base 3
        assert not is_strong_witness(2, 2047)
        assert is_strong_witness(3, 2047)

    def test_decompose(self):
        assert decompose(561) == (4, 35)        # 560 = 2^4 · 35
        assert decompose(13) == (2, 3)
        for n in [97, 1105, 2 ** 61 - 1]:
            s, d = decompose(n)
            assert d % 2 == 1 and (2 ** s) * d == n - 1

    def test_liar_fraction_below_quarter(self):
        for n in range(9, 1_500, 2):
            if not PRIMES_TO_5000[n]:
                assert liar_fraction(n, "miller_rabin") <= 0.25, n

    def test_error_rate_within_bound(self):
        # one round on a worst-ish composite; empirical ≈ liar fraction
        n = 703                               # 19·37, ~23% strong liars
        rng = random.Random(6)
        trials = 4_000
        errors = sum(miller_rabin(n, 1, rng) for _ in range(trials))
        assert errors / trials == pytest.approx(liar_fraction(n), abs=0.03)
        assert errors / trials < 0.25 + 0.03


class TestFermat:

    def test_fermat_witness(self):
        assert is_fermat_witness(2, 15)          # 2^14 mod 15 = 4
        assert not is_fermat_witness(2, 341)     # 341 = 11·31 fools base 2

    def test_carmichael_fools_coprime_bases(self):
        from math import gcd
        n = 561
        for a in range(2, 200):
            if gcd(a, n) == 1:
                assert not is_fermat_witness(a, n)

    def test_carmichael_liar_fraction(self):
        # φ(561) = 320 coprime residues; bases 1 and 560 excluded
        assert liar_fraction(561, "fermat") == pytest.approx(318 / 558)

    def test_fermat_fooled_far_more_than_miller_rabin(self):
        rng = random.Random(7)
        trials = 2_000
        f = sum(fermat_test(561, 3, rng) for _ in range(trials))
        m = sum(miller_rabin(561, 3, rng) for _ in range(trials))
        assert f / trials == pytest.approx((318 / 558) ** 3, abs=0.04)
        assert m < f / 10

    def test_more_rounds_fewer_errors(self):
        rng = random.Random(8)
        trials = 3_000
        rates = [sum(fermat_test(1105, k, rng) for _ in range(trials))
                 for k in [1, 3, 6]]
        assert rates[0] > rates[1] > rates[2]


class TestLiarFraction:

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            liar_fraction(7)                     # too small
        with pytest.raises(ValueError):
            liar_fraction(100)                   # even
        with pytest.raises(ValueError):
            liar_fraction(561, "lucas")

    def test_error_bound_formula(self):
        assert miller_rabin_error_bound(1) == 0.25
        assert miller_rabin_error_bound(10) == pytest.approx(9.54e-7,
                                                             rel=1e-3)
