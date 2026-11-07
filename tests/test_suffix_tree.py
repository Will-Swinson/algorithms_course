"""Test suite for the Ukkonen suffix tree: structure, queries, trace
of implicit/explicit extensions, and DOT export."""

import random

import pytest

from src.strings.core.suffix_array import SuffixArray
from src.strings.core.suffix_tree import ROOT, SuffixTree, choose_terminator


def all_suffixes_present(tree):
    """Every suffix of text$ is spelled by a root-to-leaf path."""
    leaves = {}
    stack = [(ROOT, "")]
    while stack:
        u, spelled = stack.pop()
        if tree.is_leaf(u):
            leaves[tree.suffix_index[u]] = spelled
        else:
            for v in tree.children[u].values():
                stack.append((v, spelled + tree.edge_label(v)))
    return all(leaves.get(i) == tree.s[i:] for i in range(len(tree.s)))


class TestStructure:

    @pytest.mark.parametrize("text", ["banana", "mississippi", "abcabxabcd",
                                      "aaaa", "abc", "", "a"])
    def test_leaves_spell_suffixes(self, text):
        tree = SuffixTree(text)
        assert tree.num_leaves == len(text) + 1
        assert all_suffixes_present(tree)

    @pytest.mark.parametrize("seed", range(60))
    def test_random_structure(self, seed):
        rng = random.Random(seed)
        text = "".join(rng.choices(rng.choice(["ab", "acgt", "abc"]),
                                   k=rng.randint(0, 60)))
        tree = SuffixTree(text)
        assert all_suffixes_present(tree)
        assert tree.num_nodes <= 2 * (len(text) + 1)        # O(n) nodes
        for u in range(1, tree.num_nodes):                  # internal nodes branch
            if not tree.is_leaf(u):
                assert len(tree.children[u]) >= 2

    def test_suffix_links(self):
        tree = SuffixTree("mississippi")
        depth = tree.string_depths()
        for u in range(1, tree.num_nodes):
            if not tree.is_leaf(u) and tree.link[u] != ROOT:
                # link from xα to α: one character shorter
                assert depth[tree.link[u]] == depth[u] - 1

    def test_terminator(self):
        assert choose_terminator("abc") == "$"
        assert choose_terminator("a$b") == "\x00"
        assert SuffixTree("a$b").terminator == "\x00"
        with pytest.raises(ValueError):
            SuffixTree("abc", terminator="a")
        with pytest.raises(ValueError):
            SuffixTree("abc", terminator="$$")

    def test_repr(self):
        assert repr(SuffixTree("banana")) == "SuffixTree(n=6, nodes=11, leaves=7)"


class TestQueries:

    @pytest.fixture
    def tree(self):
        return SuffixTree("mississippi")

    def test_find_count(self, tree):
        assert tree.find("issi") == [1, 4]
        assert tree.find("ssippi") == [5]
        assert tree.count("s") == 4 and tree.count("x") == 0
        assert tree.find("mississippis") == []
        assert tree.contains("sip") and not tree.contains("spi")
        assert tree.count("") == 12 and tree.find("")[:3] == [0, 1, 2]

    def test_pattern_with_terminator_char_absent(self, tree):
        assert tree.find("i$") == []

    @pytest.mark.parametrize("seed", range(40))
    def test_matches_suffix_array(self, seed):
        rng = random.Random(seed)
        text = "".join(rng.choices("abc", k=rng.randint(1, 200)))
        tree, sa = SuffixTree(text), SuffixArray(text)
        for _ in range(20):
            p = "".join(rng.choices("abc", k=rng.randint(1, 7)))
            assert tree.find(p) == sa.find(p)
            assert tree.count(p) == sa.count(p)
        assert tree.distinct_substrings() == sa.distinct_substrings()
        assert len(tree.longest_repeated_substring()) == \
            len(sa.longest_repeated_substring())

    def test_longest_prefix_match(self, tree):
        assert tree.longest_prefix_match("ssippixyz") == "ssippi"
        assert tree.longest_prefix_match("zzz") == ""
        assert tree.longest_prefix_match("issis") == "issis"

    def test_derived(self, tree):
        assert tree.longest_repeated_substring() == "issi"
        assert SuffixTree("abcd").longest_repeated_substring() == ""
        assert SuffixTree("banana").distinct_substrings() == 15


class TestTrace:

    def test_phases_and_rules(self):
        tree = SuffixTree("abcabxabcd", record=True)
        assert len(tree.trace) == 11                       # 10 chars + $
        phase3 = tree.trace[3]                             # 'a' already present
        assert phase3["explicit"] == 0 and phase3["implicit"] == 1
        assert phase3["after"][2] == 1                     # active_length 1
        phase5 = tree.trace[5]                             # 'x' forces splits
        assert phase5["explicit"] == 3 and phase5["splits"] == 2
        assert tree.trace[-1]["implicit"] == 0             # $ makes all explicit

    def test_explicit_extensions_total_leaves(self):
        tree = SuffixTree("abracadabra", record=True)
        assert sum(p["explicit"] for p in tree.trace) == tree.num_leaves

    def test_snapshots_grow(self):
        tree = SuffixTree("abab", record=True)
        sizes = [len(p["snapshot"]) for p in tree.trace]
        assert sizes == sorted(sizes) and sizes[-1] == tree.num_nodes - 1


def test_to_dot():
    dot = SuffixTree("banana").to_dot()
    assert dot.startswith("digraph SuffixTree {") and dot.endswith("}")
    assert 'label="ana"' in dot or 'label="na"' in dot
    assert "style=dashed" in dot
    assert "style=dashed" not in SuffixTree("banana").to_dot(show_links=False)
    long_label = SuffixTree("abcdefghijklmnop").to_dot(max_label=5)
    assert "…" in long_label
