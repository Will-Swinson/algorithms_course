"""Test suite for the radix-trie autocomplete engine."""

import random

import pytest

from src.strings.applications.autocomplete import (
    Autocomplete, SortedPrefixIndex, term_frequencies,
)

VOCAB = {"water": 120, "watch": 40, "way": 300, "was": 500, "wave": 40,
         "the": 1000, "then": 80, "there": 150, "this": 400, "w": 1}


@pytest.fixture
def ac():
    return Autocomplete(VOCAB)


class TestTrie:

    def test_complete_ranks_by_score(self, ac):
        assert [c.term for c in ac.complete("wa", 3)] == ["was", "way", "water"]
        assert [c.term for c in ac.complete("th", 10)] == \
            ["the", "this", "there", "then"]

    def test_ties_alphabetical(self, ac):
        assert [c.term for c in ac.complete("wa", 5)][-2:] == ["watch", "wave"]

    def test_prefix_is_a_term(self, ac):
        assert ac.complete("w", 1)[0].term == "was"
        assert "w" in ac.all_completions("w")

    def test_counts_and_membership(self, ac):
        assert ac.count_prefix("wa") == 5
        assert ac.count_prefix("w") == 6
        assert ac.count_prefix("") == len(VOCAB) == len(ac)
        assert ac.count_prefix("x") == 0 and not ac.has_prefix("thx")
        assert ac.has_prefix("ther") and "then" in ac

    def test_prefix_inside_edge(self, ac):
        assert ac.all_completions("wat") == ["watch", "water"]
        assert ac.all_completions("thea") == []
        assert ac.complete("zz") == [] and ac.complete("wa", 0) == []

    def test_insert_splits_edges(self):
        ac = Autocomplete()
        for t in ["romane", "romanus", "romulus", "rubens", "ruber", "rubicon",
                  "rubicundus", "rom"]:
            ac.insert(t, 1)
        assert ac.all_completions("rom") == ["rom", "romane", "romanus", "romulus"]
        assert ac.count_prefix("rub") == 4
        assert len(ac.root.children) == 1          # single edge "r"

    def test_insert_accumulates(self, ac):
        ac.insert("watch", 1000)
        assert ac.complete("wa", 1)[0].term == "watch"
        assert ac.frequency["watch"] == 1040
        with pytest.raises(ValueError):
            ac.insert("", 1)


class TestQueryWeighting:

    def test_record_query_promotes(self, ac):
        assert ac.complete("wat", 1)[0].term == "water"
        for _ in range(10):
            ac.record_query("watch")
        top = ac.complete("wat", 1)[0]
        assert top.term == "watch" and top.selections == 10
        assert top.score == 40 + 10 * 10

    def test_learn_new_terms(self, ac):
        ac.record_query("wasabi")
        assert "wasabi" in ac and ac.count_prefix("was") == 2
        with pytest.raises(KeyError):
            ac.record_query("zebra", learn=False)

    def test_weight_parameter(self):
        ac = Autocomplete({"aa": 10, "ab": 5}, query_weight=0.0)
        ac.record_query("ab")
        assert ac.complete("a", 1)[0].term == "aa"


class TestAgainstBaseline:

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_sorted_scan(self, seed):
        rng = random.Random(seed)
        vocab = {"".join(rng.choices("abcde", k=rng.randint(1, 6))): rng.randint(1, 50)
                 for _ in range(300)}
        ac, base = Autocomplete(vocab), SortedPrefixIndex(vocab)
        for _ in range(40):
            p = "".join(rng.choices("abcde", k=rng.randint(0, 3)))
            k = rng.randint(1, 8)
            assert [(c.term, c.score) for c in ac.complete(p, k)] == base.complete(p, k)

    def test_from_text(self):
        ac = Autocomplete.from_text("The water, the WATCH; the way. a", min_length=2)
        assert ac.frequency["the"] == 3 and "a" not in ac
        assert term_frequencies("Don't stop")["don't"] == 1
