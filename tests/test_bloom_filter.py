"""Test suite for the Bloom filter and its sizing formulas."""

import math

import pytest

from src.randomized.bloom_filter import BloomFilter
from src.utils.probability_analysis import (
    bloom_false_positive_rate, optimal_bloom_parameters,
)
from src.utils.random_utils import random_strings


def fp_rate(bf, probes, prefix="probe-"):
    return sum(bf.check(f"{prefix}{i}") for i in range(probes)) / probes


class TestBasics:

    def test_no_false_negatives(self):
        bf = BloomFilter(10_000, 5, seed=1)
        items = random_strings(1_000, prefix="in-", seed=2)
        bf.update(items)
        assert all(bf.check(x) for x in items)

    def test_empty_filter_rejects_everything(self):
        bf = BloomFilter(1_000, 4, seed=1)
        assert not any(bf.check(f"x{i}") for i in range(500))
        assert bf.bits_set() == 0

    def test_contains_operator(self):
        bf = BloomFilter(1_000, 3, seed=1)
        bf.add("alice")
        assert "alice" in bf

    def test_mixed_item_types(self):
        bf = BloomFilter(5_000, 4, seed=1)
        items = ["a", b"a", 1, 1.5, (2, "x"), None]
        bf.update(items)
        assert all(x in bf for x in items)

    def test_str_and_bytes_are_different_items(self):
        bf = BloomFilter(1 << 16, 6, seed=3)
        bf.add("key")
        # with 1 item in 65k bits, a collision is ~1e-25
        assert not bf.check(b"key")

    def test_len_counts_adds(self):
        bf = BloomFilter(1_000, 3, seed=1)
        bf.update(["a", "b", "a"])
        assert len(bf) == 3

    def test_sets_at_most_k_bits_per_item(self):
        bf = BloomFilter(10_000, 7, seed=1)
        bf.add("one")
        assert 1 <= bf.bits_set() <= 7

    def test_memory_is_m_bits(self):
        assert BloomFilter(8_000, 3).memory_bytes() == 1_000
        assert BloomFilter(8_001, 3).memory_bytes() == 1_001

    @pytest.mark.parametrize("m, k", [(0, 3), (-1, 3), (100, 0)])
    def test_invalid_parameters(self, m, k):
        with pytest.raises(ValueError):
            BloomFilter(m, k)

    def test_same_seed_same_bits(self):
        a, b = BloomFilter(2_000, 4, seed=9), BloomFilter(2_000, 4, seed=9)
        for bf in (a, b):
            bf.update(range(100))
        assert a._bits == b._bits

    def test_different_seed_different_hashes(self):
        a, b = BloomFilter(2_000, 4, seed=1), BloomFilter(2_000, 4, seed=2)
        for bf in (a, b):
            bf.update(range(100))
        assert a._bits != b._bits

    def test_repr(self):
        assert "num_bits=100" in repr(BloomFilter(100, 2))


class TestFalsePositiveRate:

    def test_matches_theory_at_optimal_size(self):
        n = 5_000
        bf = BloomFilter.for_capacity(n, 0.01, seed=4)
        bf.update(f"member-{i}" for i in range(n))
        measured = fp_rate(bf, 40_000)
        assert measured == pytest.approx(0.01, abs=0.004)
        assert measured == pytest.approx(bf.expected_fp_rate(), abs=0.004)

    def test_rate_grows_with_load(self):
        rates = []
        for n in [500, 2_000, 6_000]:
            bf = BloomFilter(20_000, 4, seed=5)
            bf.update(f"member-{i}" for i in range(n))
            rates.append(fp_rate(bf, 20_000))
        assert rates[0] < rates[1] < rates[2]

    def test_more_bits_per_item_fewer_false_positives(self):
        n = 2_000
        rates = []
        for bits in [4, 8, 16]:
            m, k = bits * n, max(1, round(bits * math.log(2)))
            bf = BloomFilter(m, k, seed=6)
            bf.update(f"member-{i}" for i in range(n))
            rates.append(fp_rate(bf, 20_000))
        assert rates[0] > rates[1] > rates[2]

    def test_fill_ratio_half_at_optimal_k(self):
        bf = BloomFilter.for_capacity(5_000, 0.01, seed=7)
        bf.update(range(5_000))
        assert bf.fill_ratio() == pytest.approx(0.5, abs=0.03)

    def test_estimated_fp_rate_close_to_expected(self):
        bf = BloomFilter(30_000, 5, seed=8)
        bf.update(range(3_000))
        assert bf.estimated_fp_rate() == pytest.approx(
            bf.expected_fp_rate(), rel=0.15)

    def test_estimated_count(self):
        bf = BloomFilter(50_000, 5, seed=9)
        bf.update(range(4_000))
        assert bf.estimated_count() == pytest.approx(4_000, rel=0.05)

    def test_estimated_count_saturated(self):
        bf = BloomFilter(8, 3, seed=1)
        bf.update(range(200))
        assert bf.estimated_count() == math.inf

    def test_load_factor(self):
        bf = BloomFilter(1_000, 3)
        bf.update(range(250))
        assert bf.load_factor == 0.25


class TestSizingFormulas:

    def test_optimal_parameters_one_percent(self):
        m, k = optimal_bloom_parameters(1_000, 0.01)
        assert 9_500 <= m <= 9_600            # ≈ 9.585 bits per item
        assert k == 7

    def test_optimal_parameters_tighter_rate_more_bits(self):
        m1, _ = optimal_bloom_parameters(1_000, 0.01)
        m2, k2 = optimal_bloom_parameters(1_000, 0.001)
        assert m2 == pytest.approx(1.5 * m1, rel=0.01)
        assert k2 == 10

    def test_fp_formula_values(self):
        assert bloom_false_positive_rate(1_000, 0, 3) == 0
        assert bloom_false_positive_rate(1_000, 100, 7) == pytest.approx(
            (1 - math.exp(-0.7)) ** 7)

    @pytest.mark.parametrize("n, p", [(0, 0.01), (10, 0), (10, 1)])
    def test_invalid_sizing(self, n, p):
        with pytest.raises(ValueError):
            optimal_bloom_parameters(n, p)

    def test_invalid_fp_formula(self):
        with pytest.raises(ValueError):
            bloom_false_positive_rate(0, 10, 3)
