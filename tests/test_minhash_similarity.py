"""Test suite for MinHash signatures, Jaccard similarity, and text
shingling."""

import math

import pytest

from src.randomized.minhash_similarity import (
    MAX_HASH, MinHash, jaccard_similarity, minhash_similarity, shingles,
    tokenize,
)
from src.utils.probability_analysis import minhash_standard_error
from src.utils.random_utils import random_set_pair


class TestJaccard:

    def test_known_values(self):
        assert jaccard_similarity({1, 2, 3}, {2, 3, 4}) == 0.5
        assert jaccard_similarity({1}, {2}) == 0.0
        assert jaccard_similarity({1, 2}, {1, 2}) == 1.0

    def test_empty_sets(self):
        assert jaccard_similarity(set(), set()) == 1.0
        assert jaccard_similarity(set(), {1}) == 0.0

    def test_accepts_iterables_with_duplicates(self):
        assert jaccard_similarity([1, 1, 2], (2, 2, 3)) == pytest.approx(1 / 3)


class TestShingles:

    def test_tokenize(self):
        assert tokenize("Hello, World! It's 2026.") == \
            ["hello", "world", "it's", "2026"]

    def test_word_shingles(self):
        assert shingles("a b c d", 2) == {"a b", "b c", "c d"}

    def test_char_shingles(self):
        assert shingles("abcd", 3, unit="char") == {"abc", "bcd"}

    def test_char_shingles_normalize_whitespace(self):
        assert shingles("A  b", 2, unit="char") == {"a ", " b"}

    def test_short_text_single_shingle(self):
        assert shingles("just two", 3) == {"just two"}

    def test_empty_text(self):
        assert shingles("", 3) == set()

    def test_order_matters(self):
        a = shingles("the cat sat on the mat", 2)
        b = shingles("the mat sat on the cat", 2)
        assert jaccard_similarity(a, b) < 1.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            shingles("text", 0)
        with pytest.raises(ValueError):
            shingles("text", 2, unit="line")


class TestSignatures:

    def test_signature_length(self):
        assert len(MinHash(64, seed=1).signature({1, 2, 3})) == 64

    def test_identical_sets_identical_signatures(self):
        mh = MinHash(32, seed=1)
        assert mh.signature({"a", "b"}) == mh.signature(["b", "a", "a"])
        assert mh.estimate({"a", "b"}, {"b", "a"}) == 1.0

    def test_disjoint_sets_estimate_near_zero(self):
        a, b = random_set_pair(300, 0.0, seed=2)
        assert MinHash(256, seed=3).estimate(a, b) < 0.02

    def test_empty_set_signature(self):
        assert MinHash(8, seed=1).signature(set()) == [MAX_HASH] * 8

    def test_signature_is_elementwise_min_over_union(self):
        # sig(A ∪ B) = min(sig(A), sig(B)) position by position
        mh = MinHash(50, seed=4)
        a, b = {1, 2, 3, 4}, {3, 4, 5, 6, 7}
        merged = [min(x, y) for x, y in zip(mh.signature(a),
                                            mh.signature(b))]
        assert mh.signature(a | b) == merged

    def test_same_seed_same_family(self):
        s = {"x", "y", "z"}
        assert MinHash(16, seed=5).signature(s) == \
            MinHash(16, seed=5).signature(s)

    def test_different_seed_different_family(self):
        s = set(range(100))
        assert MinHash(16, seed=5).signature(s) != \
            MinHash(16, seed=6).signature(s)

    def test_smaller_family_is_prefix_of_larger(self):
        # lets the benchmark take k = 16..512 from one 512-hash signature
        s = set(range(50))
        assert MinHash(16, seed=7).signature(s) == \
            MinHash(64, seed=7).signature(s)[:16]

    def test_similarity_validation(self):
        with pytest.raises(ValueError):
            MinHash.similarity([1, 2], [1])
        with pytest.raises(ValueError):
            MinHash.similarity([], [])
        with pytest.raises(ValueError):
            MinHash(0)


class TestEstimationAccuracy:

    @pytest.mark.parametrize("target", [0.2, 0.5, 0.8])
    def test_estimate_within_four_standard_errors(self, target):
        a, b = random_set_pair(400, target, seed=8)
        true_j = jaccard_similarity(a, b)
        k = 256
        est = minhash_similarity(a, b, num_hashes=k, seed=9)
        assert abs(est - true_j) < 4 * minhash_standard_error(true_j, k)

    def test_unbiased_over_many_families(self):
        a, b = random_set_pair(200, 0.4, seed=10)
        true_j = jaccard_similarity(a, b)
        trials, k = 60, 32
        ests = [minhash_similarity(a, b, k, seed=t) for t in range(trials)]
        mean = sum(ests) / trials
        se_of_mean = minhash_standard_error(true_j, k) / math.sqrt(trials)
        assert abs(mean - true_j) < 4 * se_of_mean

    def test_error_shrinks_with_k(self):
        a, b = random_set_pair(300, 0.5, seed=11)
        true_j = jaccard_similarity(a, b)

        def rmse(k):
            errs = [minhash_similarity(a, b, k, seed=t) - true_j
                    for t in range(30)]
            return math.sqrt(sum(e * e for e in errs) / len(errs))

        assert rmse(256) < rmse(16) / 2

    def test_text_near_duplicates_rank_correctly(self):
        base = ("randomized algorithms use coin flips to make decisions "
                "that are fast on average and simple to implement in "
                "practice across many different problem domains")
        near = base.replace("simple", "easy")
        far = "bloom filters store set membership in a compact bit array"
        mh = MinHash(256, seed=12)
        sb, sn, sf = (mh.signature(shingles(t, 2)) for t in (base, near, far))
        assert mh.similarity(sb, sn) > 0.6
        assert mh.similarity(sb, sf) < 0.1

    def test_standard_error_formula(self):
        assert minhash_standard_error(0.5, 16) == pytest.approx(0.125)
        assert minhash_standard_error(0.0, 64) == 0.0
