"""Test suite for plagiarism detection: Rabin-Karp phrase
fingerprints, suffix-array common substrings, scores, highlighting."""

import pytest

from src.strings.applications.plagiarism import (
    compare_documents, highlight, longest_common_substrings, shared_phrases,
    tokenize,
)
from src.strings.utils.generators import english_like_text

SOURCE = ("The quick brown fox jumps over the lazy dog while the farmer "
          "watches from the old wooden fence near the river bank.")
COPY = ("In my essay I argue that the quick brown fox jumps over the lazy "
        "dog while nobody notices anything at all.")


class TestTokens:

    def test_tokenize_offsets(self):
        toks = tokenize("Hello, World!")
        assert [t.word for t in toks] == ["hello", "world"]
        assert "Hello, World!"[toks[1].start:toks[1].end] == "World"


class TestSharedPhrases:

    def test_finds_copied_run(self):
        matches, stats = shared_phrases(COPY, SOURCE, k=4)
        assert len(matches) == 1
        m = matches[0]
        assert m.text.lower() == "the quick brown fox jumps over the lazy dog while"
        assert SOURCE[m.b_span[0]:m.b_span[1]].lower() == m.text.lower()
        assert m.length == 10 and stats["spurious"] == 0

    def test_k_too_large(self):
        matches, _ = shared_phrases(COPY, SOURCE, k=11)
        assert matches == []

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            shared_phrases("a b", "a b", k=0)

    def test_case_and_punctuation_ignored(self):
        matches, _ = shared_phrases("ONE, two; THREE four", "one two three four", k=4)
        assert len(matches) == 1


class TestCommonSubstrings:

    def test_longest(self):
        common = longest_common_substrings(COPY, SOURCE, min_length=10)
        # character-level and case-sensitive: "The" vs "the" leaves "he quick..."
        assert common[0].text == "he quick brown fox jumps over the lazy dog while "
        assert COPY[common[0].a_start:].startswith(common[0].text)
        assert SOURCE[common[0].b_start:].startswith(common[0].text)

    def test_no_nested_duplicates(self):
        common = longest_common_substrings(COPY, SOURCE, min_length=5, top=10)
        spans = [(c.a_start, c.a_start + c.length) for c in common]
        for i, (s, e) in enumerate(spans):
            for s2, e2 in spans[:i]:
                assert not (s2 <= s and e <= e2)

    def test_separator_check(self):
        with pytest.raises(ValueError):
            longest_common_substrings("a\x01", "b")


class TestReport:

    def test_identical(self):
        r = compare_documents(SOURCE, SOURCE, k=3)
        assert r.containment_a == r.containment_b == r.jaccard == 1.0
        assert r.coverage_a == pytest.approx(1.0, abs=0.02)
        assert r.verdict() == "heavy overlap"

    def test_unrelated(self):
        r = compare_documents("alpha beta gamma delta epsilon",
                              "one two three four five six", k=2)
        assert r.similarity == 0 and r.matches == []
        assert r.verdict() == "no shared phrases"

    @pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5])
    def test_containment_tracks_copied_share(self, fraction):
        source = english_like_text(30_000, seed=1)
        own = english_like_text(10_000, seed=99)
        copied = int(fraction * 10_000)
        doc = own[:10_000 - copied] + " " + source[5_000:5_000 + copied]
        r = compare_documents(doc, source, k=6)
        assert r.containment_a == pytest.approx(fraction, abs=0.06)
        assert r.coverage_a == pytest.approx(fraction, abs=0.06)
        assert r.longest_common[0].length >= copied - 20

    def test_empty(self):
        r = compare_documents("", "something here", k=2)
        assert r.similarity == 0 and r.longest_common == []


class TestHighlight:

    def test_styles(self):
        text = "abc def ghi"
        assert highlight(text, [(4, 7)]) == "abc [[def]] ghi"
        assert highlight(text, [(0, 3), (2, 7)]) == "[[abc def]] ghi"
        assert highlight("a<b>", [(1, 3)], "html") == "a<mark>&lt;b</mark>&gt;"
        assert "\x1b[1;31m" in highlight(text, [(0, 1)], "ansi")
        assert highlight(text, []) == text
