"""Test suite for the string generators and the text loader."""

import os

import pytest

from src.strings.core.kmp import kmp_search
from src.strings.utils.generators import (
    DNA, dna_sequence, english_like_text, fibonacci_string, log_lines, mutate,
    periodic_string, plant, random_text, sample_patterns,
)
from src.strings.utils.text_loader import (
    iter_chunks, load_text, normalize, search_stream, write_text,
)


class TestGenerators:

    def test_lengths_and_seeds(self):
        for gen in (random_text, english_like_text, log_lines, dna_sequence):
            a, b = gen(5000, seed=3), gen(5000, seed=3)
            assert len(a) == 5000 and a == b
            assert gen(5000, seed=4) != a

    def test_dna(self):
        seq = dna_sequence(20_000, gc_content=0.7, seed=1)
        assert set(seq) <= set(DNA)
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        assert gc == pytest.approx(0.7, abs=0.02)
        with pytest.raises(ValueError):
            dna_sequence(10, gc_content=1.5)

    def test_dna_repeats_raise_lcp(self):
        from src.strings.core.suffix_array import SuffixArray
        plain = SuffixArray(dna_sequence(20_000, seed=2)).stats().max_lcp
        rep = SuffixArray(dna_sequence(20_000, repeat_fraction=0.2, seed=2)).stats().max_lcp
        assert rep > 3 * plain

    def test_english(self):
        text = english_like_text(10_000, seed=1)
        words = text.lower().split()
        assert words.count("the") > words.count("water") and text[0].isupper()

    def test_logs(self):
        lines = log_lines(3000, seed=2).splitlines()
        assert lines[0].startswith("2026-11-07T") and "status=" in lines[0]

    def test_periodic_and_fibonacci(self):
        assert periodic_string("ab", 3, "c") == "abababc"
        assert [fibonacci_string(k) for k in range(1, 6)] == \
            ["b", "a", "ab", "aba", "abaab"]
        with pytest.raises(ValueError):
            fibonacci_string(0)

    def test_plant(self):
        text, where = plant("x" * 200, ["abc", "hello"], seed=1)
        assert len(text) == 200
        for p, starts in where.items():
            assert kmp_search(text, p).positions == starts
        with pytest.raises(ValueError):
            plant("xx", ["abc"], seed=1)

    def test_mutate(self):
        seq, pos = mutate("AAAAAAAAAA", 3, seed=1)
        assert len(pos) == 3 and all(seq[i] != "A" for i in pos)
        assert sum(c != "A" for c in seq) == 3
        with pytest.raises(ValueError):
            mutate("AC", 3)

    def test_sample_patterns(self):
        text = english_like_text(5000, seed=1)
        pats = sample_patterns(text, 10, 8, present_fraction=1.0, seed=2)
        assert len(pats) == 10 and all(p in text and len(p) == 8 for p in pats)
        absent = sample_patterns(text, 10, 12, present_fraction=0.0, seed=2)
        assert sum(p in text for p in absent) <= 1


class TestTextLoader:

    def test_round_trip_and_cap(self, tmp_path):
        path = str(tmp_path / "t.txt")
        write_text("héllo wörld " * 100, path)
        assert load_text(path).startswith("héllo wörld")
        capped = load_text(path, max_bytes=7)
        assert capped == "héllo "[:len(capped)] and len(capped) <= 7

    def test_normalize(self):
        assert normalize("  Héllo   World ", lower=True, collapse_whitespace=True,
                         ascii_only=True) == "hello world"
        path_text = "Ab"
        assert normalize(path_text) == "Ab"

    def test_load_with_normalization(self, tmp_path):
        path = str(tmp_path / "n.txt")
        write_text("A  B\nC", path)
        assert load_text(path, lower=True, collapse_whitespace=True) == "a b c"

    def test_chunks_overlap(self, tmp_path):
        path = str(tmp_path / "c.txt")
        text = "".join(chr(97 + i % 26) for i in range(1000))
        write_text(text, path)
        chunks = list(iter_chunks(path, chunk_size=100, overlap=10))
        for offset, chunk in chunks:
            assert text[offset:offset + len(chunk)] == chunk
        assert chunks[-1][0] + len(chunks[-1][1]) == len(text)
        with pytest.raises(ValueError):
            list(iter_chunks(path, chunk_size=5, overlap=5))

    def test_search_stream_matches_whole_file(self, tmp_path):
        path = str(tmp_path / "s.txt")
        text = english_like_text(50_000, seed=5)
        write_text(text, path)
        for pattern in ("the ", "water", "zzz", "e"):
            expected = kmp_search(text, pattern).positions
            got = search_stream(path, pattern,
                                lambda t, p: kmp_search(t, p).positions,
                                chunk_size=777)
            assert got == expected
        with pytest.raises(ValueError):
            search_stream(path, "", lambda t, p: [])

    def test_file_missing(self):
        with pytest.raises(OSError):
            load_text(os.path.join("nope", "missing.txt"))
