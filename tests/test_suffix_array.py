"""Test suite for suffix array construction (three methods), Kasai
LCP, binary-search queries, and LCP-derived results."""

import random

import numpy as np
import pytest

from src.strings.core.suffix_array import (
    SuffixArray, build_suffix_array, kasai_lcp, longest_common_substring,
    suffix_array_doubling, suffix_array_naive, suffix_array_python,
)


def brute_lcp(text, sa):
    out = [0]
    for a, b in zip(sa, sa[1:]):
        h = 0
        while a + h < len(text) and b + h < len(text) and text[a + h] == text[b + h]:
            h += 1
        out.append(h)
    return out


class TestConstruction:

    def test_banana(self):
        sa = SuffixArray("banana")
        assert sa.sa.tolist() == [5, 3, 1, 0, 4, 2]
        assert sa.lcp == [0, 1, 3, 0, 0, 2]

    @pytest.mark.parametrize("seed", range(80))
    def test_methods_agree(self, seed):
        rng = random.Random(seed)
        text = "".join(rng.choices(rng.choice(["ab", "acgt", "abcdefg"]),
                                   k=rng.randint(0, 120)))
        naive = suffix_array_naive(text)
        assert suffix_array_doubling(text)[0].tolist() == naive
        assert suffix_array_python(text) == naive
        assert kasai_lcp(text, naive) == brute_lcp(text, naive)

    @pytest.mark.parametrize("text", ["", "a", "aaaaaaa", "abababab", "zyxwv",
                                      "héllo wörld", "a\x00b"])
    def test_special_inputs(self, text):
        for method in ("doubling", "python", "naive"):
            assert build_suffix_array(text, method).tolist() == suffix_array_naive(text)

    def test_rounds_follow_longest_repeat(self):
        _, rounds = suffix_array_doubling("a" * 64)
        assert rounds == 6 or rounds == 7
        _, rounds = suffix_array_doubling("abcdefgh")
        assert rounds == 1

    def test_large_matches_python(self):
        text = "".join(random.Random(1).choices("acgt", k=5000))
        assert suffix_array_doubling(text)[0].tolist() == suffix_array_python(text)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            build_suffix_array("abc", "dc3")

    def test_methods_on_object(self):
        sa = SuffixArray("mississippi", method="python")
        assert sa.find("ssi") == [2, 5]
        assert SuffixArray("mississippi", method="naive", lcp=False).lcp is None


class TestQueries:

    @pytest.fixture
    def sa(self):
        return SuffixArray("mississippi")

    def test_find_and_count(self, sa):
        assert sa.find("issi") == [1, 4]
        assert sa.find("i") == [1, 4, 7, 10]
        assert sa.count("ss") == 2 and sa.count("x") == 0
        assert sa.find("mississippix") == []
        assert sa.contains("pp") and not sa.contains("pq")

    def test_empty_pattern(self, sa):
        assert sa.count("") == 12 and sa.find("") == list(range(12))

    def test_range_is_contiguous(self, sa):
        lo, hi = sa.range("s")
        assert all(sa.suffix(r).startswith("s") for r in range(lo, hi))
        assert hi - lo == 4

    @pytest.mark.parametrize("seed", range(40))
    def test_random_queries(self, seed):
        rng = random.Random(seed)
        text = "".join(rng.choices("abc", k=300))
        sa = SuffixArray(text)
        for _ in range(20):
            p = "".join(rng.choices("abc", k=rng.randint(1, 6)))
            ref = [i for i in range(len(text)) if text.startswith(p, i)]
            assert sa.find(p) == ref and sa.count(p) == len(ref)


class TestDerived:

    @pytest.mark.parametrize("text,lrs", [("banana", "ana"), ("mississippi", "issi"),
                                          ("abcd", ""), ("aaaa", "aaa"), ("", "")])
    def test_longest_repeated(self, text, lrs):
        assert SuffixArray(text).longest_repeated_substring() == lrs

    @pytest.mark.parametrize("seed", range(20))
    def test_distinct_substrings(self, seed):
        rng = random.Random(seed)
        text = "".join(rng.choices("ab", k=rng.randint(0, 25)))
        expected = len({text[i:j] for i in range(len(text))
                        for j in range(i + 1, len(text) + 1)})
        assert SuffixArray(text).distinct_substrings() == expected

    def test_stats_and_table(self):
        sa = SuffixArray("banana", lcp=False)
        st = sa.stats()                       # builds LCP on demand
        assert st.max_lcp == 3 and st.distinct_substrings == 15
        assert st.mean_lcp == pytest.approx(1.0)
        assert sa.table(3)[2] == (2, 1, 3, "ana")

    def test_longest_common_substring(self):
        assert longest_common_substring("xabcdey", "zzabcdq") == ("abcd", 1, 2)
        assert longest_common_substring("abc", "xyz") == ("", -1, -1)
        with pytest.raises(ValueError):
            longest_common_substring("a\x01", "b")

    @pytest.mark.parametrize("seed", range(15))
    def test_lcs_length_brute_force(self, seed):
        rng = random.Random(seed)
        a = "".join(rng.choices("ab", k=rng.randint(1, 20)))
        b = "".join(rng.choices("ab", k=rng.randint(1, 20)))
        best = max((j - i for i in range(len(a)) for j in range(i + 1, len(a) + 1)
                    if a[i:j] in b), default=0)
        s, ia, ib = longest_common_substring(a, b)
        assert len(s) == best
        if best:
            assert a[ia:ia + best] == s == b[ib:ib + best]

    def test_numpy_input(self):
        sa = SuffixArray("cab")
        assert isinstance(sa.sa, np.ndarray) and sa.sa.tolist() == [1, 2, 0]
