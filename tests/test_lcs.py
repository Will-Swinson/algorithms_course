"""Test suite for longest common subsequence solvers and reconstruction."""

import random

import pytest

from src.dp.lcs import (
    lcs_recursive, lcs_memo, lcs_tabulation, build_lcs_table,
    reconstruct_lcs, lcs,
)
from src.utils.timer import CallStats

ALL = [lcs_recursive, lcs_memo, lcs_tabulation]
DP = [lcs_memo, lcs_tabulation]


def is_subsequence(sub, s):
    it = iter(s)
    return all(ch in it for ch in sub)


def random_dna(length, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


@pytest.mark.parametrize("solve", ALL)
class TestCorrectness:

    @pytest.mark.parametrize("x, y, expected", [
        ("ABCBDAB", "BDCABA", 4),
        ("AGGTAB", "GXTXAYB", 4),
        ("ABCDGH", "AEDFHR", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("", "abc", 0),
        ("abc", "", 0),
        ("", "", 0),
        ("a", "a", 1),
        ("aaaa", "aa", 2),
    ])
    def test_known_lengths(self, solve, x, y, expected):
        assert solve(x, y) == expected

    def test_symmetric(self, solve):
        assert solve("ACCGGTCG", "GTCGTTCG") == solve("GTCGTTCG", "ACCGGTCG")

    def test_case_sensitive(self, solve):
        assert solve("abc", "ABC") == 0

    def test_subsequence_of_itself(self, solve):
        assert solve("ACE", "ABCDE") == 3

    @pytest.mark.parametrize("bad", [None, 123, ["a", "b"]])
    def test_non_string_raises(self, solve, bad):
        with pytest.raises(TypeError):
            solve(bad, "abc")


@pytest.mark.parametrize("seed", range(12))
def test_all_methods_agree_on_random_strings(seed):
    x, y = random_dna(9, seed), random_dna(8, seed + 100)
    assert lcs_recursive(x, y) == lcs_memo(x, y) == lcs_tabulation(x, y)


class TestDPAtScale:

    def test_memo_and_tabulation_agree_on_1000_chars(self):
        x, y = random_dna(1000, 1), random_dna(1000, 2)
        assert lcs_memo(x, y) == lcs_tabulation(x, y)

    def test_identical_long_strings(self):
        s = random_dna(1200, 3)
        # memo recurses 1,200 deep on the diagonal — past the default limit
        assert lcs_memo(s, s) == lcs_tabulation(s, s) == 1200

    def test_bounded_by_shorter_string(self):
        x, y = random_dna(300, 4), random_dna(50, 5)
        assert lcs_tabulation(x, y) <= 50


class TestReconstruction:

    def test_classic_example(self):
        length, sub = lcs("ABCBDAB", "BDCABA")
        assert length == len(sub) == 4
        assert is_subsequence(sub, "ABCBDAB")
        assert is_subsequence(sub, "BDCABA")

    def test_unique_answer(self):
        assert lcs("AGGTAB", "GXTXAYB") == (4, "GTAB")

    def test_empty_when_no_common_chars(self):
        assert lcs("abc", "xyz") == (0, "")

    def test_empty_input(self):
        assert lcs("", "abc") == (0, "")

    @pytest.mark.parametrize("seed", range(20))
    def test_reconstructed_is_common_subsequence(self, seed):
        x, y = random_dna(60, seed), random_dna(45, seed + 50)
        table = build_lcs_table(x, y)
        sub = reconstruct_lcs(table, x, y)
        assert len(sub) == table[-1][-1]
        assert is_subsequence(sub, x) and is_subsequence(sub, y)

    def test_table_dimensions(self):
        table = build_lcs_table("abcd", "xy")
        assert len(table) == 5 and all(len(r) == 3 for r in table)


class TestCallCounts:

    def test_recursive_exponential_without_matches(self):
        """No shared characters: every call branches twice."""
        small, large = CallStats(), CallStats()
        lcs_recursive("aaaa", "bbbb", stats=small)
        lcs_recursive("aaaaaa", "bbbbbb", stats=large)
        assert large.calls > 10 * small.calls

    def test_memo_bounded_by_table_size(self):
        x, y = random_dna(40, 1), random_dna(30, 2)
        stats = CallStats()
        lcs_memo(x, y, stats=stats)
        assert stats.calls <= 2 * (len(x) + 1) * (len(y) + 1)

    def test_tabulation_counts_every_cell(self):
        stats = CallStats()
        lcs_tabulation("abcde", "abc", stats=stats)
        assert stats.calls == 15
        assert stats.max_depth == 0

    def test_recursion_depth_at_most_m_plus_n(self):
        stats = CallStats()
        lcs_recursive("abcd", "wxyz", stats=stats)
        # longest chain: (4,4) -> (4,3) -> ... -> (1,1) -> (0,1) = 8 calls
        assert stats.max_depth == 8

    def test_memo_far_fewer_calls_than_recursive(self):
        x, y = "ACGTACGTAC", "TGCATGCATG"
        rec, memo = CallStats(), CallStats()
        lcs_recursive(x, y, stats=rec)
        lcs_memo(x, y, stats=memo)
        assert memo.calls * 10 < rec.calls
