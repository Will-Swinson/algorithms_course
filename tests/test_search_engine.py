"""Test suite for the unified StringSearchEngine."""

import pytest

from src.strings.core.search_engine import (
    METHODS, Recommendation, SearchResult, StringSearchEngine,
)
from src.strings.utils.generators import english_like_text, sample_patterns


@pytest.fixture(scope="module")
def text():
    return english_like_text(30_000, seed=4)


class TestFind:

    def test_methods_agree(self, text):
        engine = StringSearchEngine(text)
        for p in sample_patterns(text, 10, 8, seed=1) + ["the", "zzzz"]:
            results = engine.compare(p)
            assert set(results) == set(METHODS)
            positions = results["kmp"].positions
            assert all(r.positions == positions for r in results.values())

    def test_result_fields(self, text):
        engine = StringSearchEngine(text)
        r = engine.find("the", method="sa")
        assert isinstance(r, SearchResult)
        assert r.count == len(r.positions) > 0
        assert r.search_time >= 0 and r.preprocess_time > 0
        assert engine.find("the", "kmp").preprocess_time < r.preprocess_time

    def test_lazy_build_once(self, text):
        engine = StringSearchEngine(text)
        assert engine.stats["sa"].preprocess_time == 0
        engine.find("water", "sa")
        first = engine.stats["sa"].preprocess_time
        engine.find("water", "sa")
        assert engine.stats["sa"].preprocess_time == first > 0
        assert engine.stats["sa"].queries == 2

    def test_count(self, text):
        engine = StringSearchEngine(text)
        n = text.count("the ")
        for m in METHODS:
            assert engine.count("the ", m) >= n

    def test_find_many(self, text):
        engine = StringSearchEngine(text)
        pats = ["the", "water", "zzz", "the"]
        batch = engine.find_many(pats, "rk")
        assert set(batch) == {"the", "water", "zzz"}
        assert batch["water"] == engine.find("water", "kmp").positions
        assert engine.find_many(["the"], "sa")["the"] == batch["the"]

    def test_unknown_method(self, text):
        with pytest.raises(ValueError):
            StringSearchEngine(text).find("x", "boyer-moore")

    def test_tree_limit(self):
        engine = StringSearchEngine("abc" * 100, tree_max_chars=100)
        with pytest.raises(ValueError):
            engine.find("abc", "tree")
        assert "tree" not in engine.compare("abc")

    def test_memory_measurement(self):
        engine = StringSearchEngine("abracadabra" * 50, measure_memory=True)
        engine.preprocess("sa")
        engine.preprocess("tree")
        assert engine.stats["tree"].index_memory > engine.stats["sa"].index_memory > 0
        assert engine.preprocess("kmp") == 0


class TestRecommendation:

    def test_few_vs_many_queries(self, text):
        engine = StringSearchEngine(text)
        few = engine.recommend(1)
        many = engine.recommend(10_000)
        assert isinstance(few, Recommendation)
        assert few.method in ("kmp", "rk")
        assert many.method in ("sa", "tree")
        assert "repeated queries" in str(many)
        assert many.break_even["sa"] and many.break_even["sa"] > 1
        assert set(many.totals) == set(METHODS)

    def test_validation(self, text):
        with pytest.raises(ValueError):
            StringSearchEngine(text).recommend(0)

    def test_summary(self, text):
        engine = StringSearchEngine(text)
        engine.find("the", "kmp")
        rows = {r["method"]: r for r in engine.summary()}
        assert rows["kmp"]["queries"] == 1 and rows["sa"]["queries"] == 0

    def test_restricted_methods(self, text):
        engine = StringSearchEngine(text)
        rec = engine.recommend(5, methods=["kmp", "rk"])
        assert rec.method in ("kmp", "rk") and "no preprocessing" in rec.reason
