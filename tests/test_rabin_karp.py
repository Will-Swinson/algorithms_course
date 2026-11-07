"""Test suite for Rabin-Karp: rolling hashes, verification, Monte
Carlo false positives, bad parameters, and multi-pattern search."""

import random

import pytest

from src.strings.core.kmp import kmp_search
from src.strings.core.rabin_karp import (
    expected_spurious_hits, rabin_karp, rabin_karp_multi,
)
from src.strings.utils.generators import english_like_text
from src.strings.utils.rolling_hash import (
    MERSENNE_61, DoubleHash, PrefixHash, RollingHash, as_codes,
    collision_bound, find_collision, hash_tokens,
)


def reference(text, pattern):
    return [i for i in range(len(text) - len(pattern) + 1)
            if text[i:i + len(pattern)] == pattern]


class TestRollingHash:

    def test_window_hashes_equal_direct_hashes(self):
        h = RollingHash(rng=random.Random(1))
        text = "the quick brown fox jumps"
        for i, value in h.window_hashes(text, 5):
            assert value == h.hash(text[i:i + 5])

    def test_roll(self):
        h = RollingHash(base=31, modulus=1_000_003)
        a = h.hash("abc")
        rolled = h.roll(a, ord("a"), ord("d"), h.power(3))
        assert rolled == h.hash("bcd")

    def test_window_edge_cases(self):
        h = RollingHash()
        assert list(h.window_hashes("abc", 4)) == []
        assert list(h.window_hashes("abc", 0)) == []

    def test_prefix_hash(self):
        h = RollingHash(rng=random.Random(2))
        text = "mississippi"
        ph = PrefixHash(text, h)
        for i in range(len(text)):
            for j in range(i, len(text) + 1):
                assert ph.substring(i, j) == h.hash(text[i:j])
        with pytest.raises(IndexError):
            ph.substring(3, 99)

    def test_double_hash(self):
        d = DoubleHash(rng=random.Random(3))
        assert d.hash("abc") == d.hash("abc") != d.hash("abd")
        windows = dict(d.window_hashes("abcabc", 3))
        assert windows[0] == windows[3]

    def test_validation(self):
        with pytest.raises(ValueError):
            RollingHash(modulus=1)
        with pytest.raises(ValueError):
            RollingHash(base=0)

    def test_as_codes(self):
        assert as_codes("ab") == b"ab"
        assert as_codes("é") == [233]
        assert as_codes([1, 2]) == [1, 2]

    def test_find_collision_small_modulus(self):
        h = RollingHash(modulus=101, rng=random.Random(4))
        a, b = find_collision(6, h)
        assert a != b and h.hash(a) == h.hash(b)

    def test_collision_bound(self):
        assert collision_bound(10, 101) == pytest.approx(9 / 101)
        assert collision_bound(1, 7) == 0

    def test_hash_tokens(self):
        codes = hash_tokens(["a", "b", "a"])
        assert codes[0] == codes[2] != codes[1]


class TestSearch:

    @pytest.mark.parametrize("seed", range(50))
    @pytest.mark.parametrize("modulus", [7, 101, MERSENNE_61])
    def test_exact_with_verification(self, seed, modulus):
        rng = random.Random(seed)
        text = "".join(rng.choices("abc", k=rng.randint(0, 150)))
        pattern = "".join(rng.choices("abc", k=rng.randint(1, 5)))
        r = rabin_karp(text, pattern, modulus=modulus, seed=seed)
        assert r.positions == reference(text, pattern)
        assert r.stats["hash_hits"] == len(r.positions) + r.stats["spurious"]

    @pytest.mark.parametrize("text,pattern,expected", [
        ("abc", "abcd", []), ("abc", "", [0, 1, 2, 3]), ("aaaa", "aa", [0, 1, 2]),
        ("héllo héllo", "llo", [2, 8]),
    ])
    def test_edge_cases(self, text, pattern, expected):
        assert rabin_karp(text, pattern, seed=1).positions == expected

    def test_small_modulus_causes_spurious_hits(self):
        text = english_like_text(20_000, seed=1)
        r = rabin_karp(text, "zzzzzzzz", modulus=31, seed=1)
        assert r.positions == [] and r.stats["spurious"] > 100

    def test_monte_carlo_false_positives(self):
        text = english_like_text(20_000, seed=2)
        r = rabin_karp(text, "zzzzzzzz", modulus=31, seed=1, verify=False)
        assert r.stats["false_positives"] == len(r.positions) > 100
        big = rabin_karp(text, "zzzzzzzz", seed=1, verify=False)
        assert big.positions == [] and big.stats["false_positives"] == 0

    def test_bad_bases(self):
        text = "abcbcacabxyz" * 50
        anagram = rabin_karp(text, "cba", base=1, modulus=MERSENNE_61)
        assert anagram.stats["spurious"] > 0        # abc, bca, cab all collide
        last_only = rabin_karp(text, "zzc", base=101, modulus=101)
        assert last_only.stats["spurious"] > 0
        assert anagram.positions == reference(text, "cba")

    def test_spurious_rate_near_theory(self):
        text = english_like_text(100_000, seed=3)
        q = 1009
        spurious = sum(rabin_karp(text, p, modulus=q, seed=k).stats["spurious"]
                       for k, p in enumerate(["zzqzzqzzq", "qqqqqxqqq",
                                              "xzxzxzxzx"]))
        expected = 3 * expected_spurious_hits(len(text), 9, q)
        assert 0.7 * expected < spurious < 1.3 * expected

    def test_worst_case_every_window_hits(self):
        r = rabin_karp("a" * 500, "a" * 10, seed=1)
        assert len(r.positions) == 491 and r.comparisons == 491 * 10

    def test_shared_hasher(self):
        h = RollingHash(rng=random.Random(9))
        assert rabin_karp("abcabc", "bc", hasher=h).positions == [1, 4]

    def test_bytes_and_lists(self):
        assert rabin_karp(b"abcab", b"ab", seed=1).positions == [0, 3]
        assert rabin_karp([1, 2, 1, 2], [1, 2], seed=1).positions == [0, 2]


class TestMulti:

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_single_searches(self, seed):
        rng = random.Random(seed)
        text = "".join(rng.choices("abcd", k=300))
        patterns = ["".join(rng.choices("abcd", k=rng.randint(1, 5)))
                    for _ in range(12)]
        multi = rabin_karp_multi(text, patterns, modulus=97, seed=seed)
        for p in patterns:
            assert multi[p].positions == kmp_search(text, p).positions

    def test_edge_patterns(self):
        res = rabin_karp_multi("abc", ["", "abcd", "b", "b"], seed=1)
        assert res[""].positions == [0, 1, 2, 3]
        assert res["abcd"].positions == []
        assert res["b"].positions == [1]
        assert len(res) == 3

    def test_unverified(self):
        res = rabin_karp_multi("aaaa", ["aa"], seed=1, verify=False)
        assert res["aa"].positions == [0, 1, 2]
