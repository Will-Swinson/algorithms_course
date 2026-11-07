"""
Autocomplete over a compressed prefix trie (radix tree).

Every term is stored along a root-to-node path whose edge labels are
compressed runs of characters (one edge per branch-free run), the
prefix-tree counterpart of a suffix tree. Each node keeps

    best   the highest score of any term in its subtree
    terms  how many terms the subtree holds

so a prefix query walks down in O(|prefix|) and then pulls the top-k
completions with a best-first search ordered by `best`: it only opens
subtrees that can still beat the k-th result, instead of listing every
completion and sorting.

Scoring with query-frequency weighting:

    score(term) = frequency(term) + query_weight · times_selected(term)

record_query(term) bumps a term (a user picked it); since a score only
grows, the `best` values on its path are raised in O(depth). Terms
never typed before can be added the same way (learn=True).

SortedPrefixIndex is the baseline: a sorted term list, bisect to the
prefix range, then scan the whole range for the top k.
"""
import bisect
import heapq
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

WORD = re.compile(r"[a-z][a-z'\-]*")


@dataclass
class Completion:
    term: str
    score: float
    frequency: float
    selections: int


class _Node:
    __slots__ = ("children", "term", "best", "count")

    def __init__(self):
        self.children: Dict[str, Tuple[str, "_Node"]] = {}   # first char -> (label, child)
        self.term: Optional[str] = None                      # term ending here
        self.best = float("-inf")
        self.count = 0


def term_frequencies(text: str, min_length: int = 1) -> Counter:
    """Word counts of a text (lower-cased)."""
    return Counter(w for w in WORD.findall(text.lower()) if len(w) >= min_length)


class Autocomplete:
    """
    ac = Autocomplete({"water": 120, "watch": 40, "way": 300})
    ac.complete("wa", k=2)      -> way, water
    ac.record_query("watch"); ...
    """

    def __init__(self, frequencies: Optional[Mapping[str, float]] = None,
                 query_weight: float = 10.0):
        self.root = _Node()
        self.query_weight = query_weight
        self.frequency: Dict[str, float] = {}
        self.selections: Counter = Counter()
        for term, freq in (frequencies or {}).items():
            self.insert(term, freq)

    @classmethod
    def from_text(cls, text: str, min_length: int = 2, **kw) -> "Autocomplete":
        return cls(term_frequencies(text, min_length), **kw)

    def __len__(self) -> int:
        return len(self.frequency)

    def __contains__(self, term: str) -> bool:
        return term in self.frequency

    def score(self, term: str) -> float:
        return self.frequency[term] + self.query_weight * self.selections[term]

    # ------------------------------------------------------------------
    # updates
    # ------------------------------------------------------------------
    def insert(self, term: str, frequency: float = 1.0) -> None:
        """Add a term, or add `frequency` to an existing one."""
        if not term:
            raise ValueError("empty term")
        new = term not in self.frequency
        self.frequency[term] = self.frequency.get(term, 0.0) + frequency
        node, rest = self.root, term
        path = [node]
        while rest:
            entry = node.children.get(rest[0])
            if entry is None:
                child = _Node()
                node.children[rest[0]] = (rest, child)
                node = child
                path.append(node)
                rest = ""
                break
            label, child = entry
            common = _common_prefix(label, rest)
            if common < len(label):
                # split the edge: label[:common] → mid → label[common:] → child
                mid = _Node()
                mid.children[label[common]] = (label[common:], child)
                mid.best, mid.count = child.best, child.count
                node.children[rest[0]] = (label[:common], mid)
                child = mid
            node = child
            path.append(node)
            rest = rest[common:]
        node.term = term
        if new:
            for p in path:
                p.count += 1
        s = self.score(term)
        for p in path:
            if s > p.best:
                p.best = s

    def record_query(self, term: str, learn: bool = True) -> None:
        """A user selected `term`: raise its score (adds it if learn)."""
        if term not in self.frequency:
            if not learn:
                raise KeyError(term)
            self.insert(term, 0.0)
        self.selections[term] += 1
        s = self.score(term)
        for node in self._path(term):
            if s > node.best:
                node.best = s

    def _path(self, term: str) -> List[_Node]:
        node, rest, path = self.root, term, [self.root]
        while rest:
            label, node = node.children[rest[0]]
            path.append(node)
            rest = rest[len(label):]
        return path

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def _prefix_node(self, prefix: str) -> Optional[Tuple[_Node, str]]:
        """Node whose subtree holds exactly the terms starting with
        prefix, and the string spelled from the root to it."""
        node, rest, spelled = self.root, prefix, ""
        while rest:
            entry = node.children.get(rest[0])
            if entry is None:
                return None
            label, child = entry
            if len(rest) <= len(label):
                return (child, spelled + label) if label.startswith(rest) else None
            if not rest.startswith(label):
                return None
            spelled += label
            rest = rest[len(label):]
            node = child
        return node, spelled

    def count_prefix(self, prefix: str) -> int:
        found = self._prefix_node(prefix)
        return found[0].count if found else 0

    def has_prefix(self, prefix: str) -> bool:
        return self.count_prefix(prefix) > 0

    def all_completions(self, prefix: str) -> List[str]:
        found = self._prefix_node(prefix)
        if not found:
            return []
        out, stack = [], [found[0]]
        while stack:
            node = stack.pop()
            if node.term is not None:
                out.append(node.term)
            stack.extend(child for _, child in node.children.values())
        return sorted(out)

    def complete(self, prefix: str, k: int = 5) -> List[Completion]:
        """Top-k completions of `prefix` by score (ties: alphabetical)."""
        if k <= 0:
            return []
        found = self._prefix_node(prefix)
        if not found:
            return []
        start = found[0]
        # best-first search. Keys (−value, kind, name): subtrees (kind 0)
        # whose best ties a term's score are opened before that term
        # (kind 1) is emitted, so equal scores come out alphabetically.
        heap: List[Tuple[float, int, str, int, object]] = [(-start.best, 0, "", 0, start)]
        counter = 1
        results: List[Completion] = []
        while heap and len(results) < k:
            neg, kind, name, _, item = heapq.heappop(heap)
            if kind == 1:
                results.append(Completion(name, -neg, self.frequency[name],
                                          self.selections[name]))
                continue
            node = item
            if node.term is not None:
                heapq.heappush(heap, (-self.score(node.term), 1, node.term,
                                      counter, None))
                counter += 1
            for _, child in node.children.values():
                heapq.heappush(heap, (-child.best, 0, "", counter, child))
                counter += 1
        return results


class SortedPrefixIndex:
    """Baseline: sorted terms + bisect; scans the whole prefix range."""

    def __init__(self, frequencies: Mapping[str, float]):
        self.terms = sorted(frequencies)
        self.frequency = dict(frequencies)

    def complete(self, prefix: str, k: int = 5) -> List[Tuple[str, float]]:
        lo = bisect.bisect_left(self.terms, prefix)
        hi = bisect.bisect_left(self.terms, prefix + "\U0010ffff")
        return heapq.nsmallest(k, ((t, self.frequency[t]) for t in self.terms[lo:hi]),
                               key=lambda x: (-x[1], x[0]))


def _common_prefix(a: str, b: str) -> int:
    i, n = 0, min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return i
