"""Test suite for KMP pattern matching, the prefix function, the
brute-force baseline, and the fallback trace."""

import itertools
import random

import pytest

from src.strings.core.kmp import (
    brute_force_search, format_trace, kmp_search, kmp_trace, prefix_function,
)


def reference(text, pattern):
    return [i for i in range(len(text) - len(pattern) + 1)
            if text[i:i + len(pattern)] == pattern]


class TestPrefixFunction:

    @pytest.mark.parametrize("pattern,lps", [
        ("", []), ("a", [0]), ("aaaa", [0, 1, 2, 3]),
        ("abcd", [0, 0, 0, 0]), ("ababaca", [0, 0, 1, 2, 3, 0, 1]),
        ("aabaaab", [0, 1, 0, 1, 2, 2, 3]), ("abab", [0, 0, 1, 2]),
    ])
    def test_known(self, pattern, lps):
        assert prefix_function(pattern) == lps

    @pytest.mark.parametrize("seed", range(30))
    def test_definition(self, seed):
        rng = random.Random(seed)
        p = "".join(rng.choices("ab", k=rng.randint(1, 20)))
        for i, v in enumerate(prefix_function(p)):
            s = p[:i + 1]
            best = max(k for k in range(len(s)) if s[:k] == s[len(s) - k:])
            assert v == best

    def test_works_on_sequences(self):
        assert prefix_function([1, 2, 1, 2]) == [0, 0, 1, 2]


class TestSearch:

    @pytest.mark.parametrize("search", [kmp_search, brute_force_search])
    @pytest.mark.parametrize("text,pattern,expected", [
        ("abracadabra", "abra", [0, 7]),
        ("aaaaa", "aa", [0, 1, 2, 3]),             # overlapping
        ("abc", "abcd", []),                        # pattern longer than text
        ("", "a", []),
        ("abc", "", [0, 1, 2, 3]),                  # empty pattern
        ("abababab", "abab", [0, 2, 4]),            # periodic
        ("aaaaaaab", "aaab", [4]),                  # repeated characters
        ("xyz", "xyz", [0]),
    ])
    def test_edge_cases(self, search, text, pattern, expected):
        assert search(text, pattern).positions == expected

    @pytest.mark.parametrize("seed", range(60))
    def test_matches_reference(self, seed):
        rng = random.Random(seed)
        alphabet = rng.choice(["ab", "abc", "acgt"])
        text = "".join(rng.choices(alphabet, k=rng.randint(0, 200)))
        pattern = "".join(rng.choices(alphabet, k=rng.randint(1, 6)))
        expected = reference(text, pattern)
        assert kmp_search(text, pattern).positions == expected
        assert brute_force_search(text, pattern).positions == expected

    def test_exhaustive_binary(self):
        for n in range(0, 9):
            for t in itertools.product("ab", repeat=n):
                text = "".join(t)
                for p in ("a", "ab", "aba", "bb", "abab"):
                    assert kmp_search(text, p).positions == reference(text, p)

    def test_precomputed_lps(self):
        lps = prefix_function("aba")
        assert kmp_search("ababa", "aba", lps).positions == [0, 2]

    def test_unicode(self):
        assert kmp_search("héllo wörld héllo", "héllo").positions == [0, 12]


class TestComplexity:

    def test_linear_comparisons_on_worst_case(self):
        text, pattern = "a" * 5000, "a" * 49 + "b"
        kmp = kmp_search(text, pattern)
        brute = brute_force_search(text, pattern)
        assert kmp.positions == brute.positions == []
        assert kmp.comparisons <= 2 * len(text)
        assert brute.comparisons > 20 * kmp.comparisons

    @pytest.mark.parametrize("seed", range(10))
    def test_comparisons_at_most_2n(self, seed):
        rng = random.Random(seed)
        text = "".join(rng.choices("ab", k=3000))
        r = kmp_search(text, "abaab")
        assert r.comparisons <= 2 * len(text)
        assert r.stats["fallbacks"] <= len(text)

    def test_count_property(self):
        assert kmp_search("aaaa", "a").count == 4


class TestTrace:

    def test_actions(self):
        steps = kmp_trace("abababd", "ababd")
        actions = [s["action"] for s in steps]
        assert "fallback" in actions and actions[-1] == "found"
        fb = next(s for s in steps if s["action"] == "fallback")
        assert (fb["j"], fb["next_j"]) == (4, 2)          # lps[3] = 2
        assert steps[-1]["start"] == 2

    def test_trace_positions_match_search(self):
        text, p = "aabaabaaab", "aab"
        found = [s["start"] for s in kmp_trace(text, p) if s["action"] == "found"]
        assert found == kmp_search(text, p).positions

    def test_advance_and_empty(self):
        assert kmp_trace("xyz", "a")[0]["action"] == "advance"
        assert kmp_trace("abc", "") == []

    def test_format(self):
        out = format_trace("abababd", "ababd")
        assert "lps[3] = 2" in out and "MATCH at 2" in out
