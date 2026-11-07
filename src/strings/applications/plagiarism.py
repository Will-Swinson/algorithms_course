"""
Plagiarism detection: shared phrases and longest common substrings.

Two complementary signals:

1. Shared phrases (Rabin–Karp over words). Tokenize both documents
   into lower-case words, hash each word to an integer, and roll a
   polynomial hash over every window of k consecutive words (k-gram
   "shingles"). Matching fingerprints are candidate copied phrases;
   each is verified word-by-word, so a hash collision can never
   produce a false match. Overlapping matches are merged into maximal
   copied regions. Scores:
       jaccard      |A ∩ B| / |A ∪ B| over the k-gram sets
       containment  |A ∩ B| / |A|  — how much of A appears in B
                    (the right score when a short essay copies from a
                    long source)
       coverage     fraction of A's characters inside copied regions

2. Longest common substrings (suffix array). Build the suffix array of
   A + sep + B; adjacent suffixes from different documents with a long
   LCP are long verbatim overlaps — character-level, so it also
   catches copying that tokenization would blur (code, DNA, URLs).

compare_documents() combines both into a PlagiarismReport and
highlight() marks the matched regions for display.
"""
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from src.strings.core.suffix_array import SuffixArray
from src.strings.utils.rolling_hash import MERSENNE_61, RollingHash, hash_tokens

WORD = re.compile(r"\w+", re.UNICODE)

Span = Tuple[int, int]                  # [start, end) character offsets


@dataclass
class Token:
    word: str
    start: int
    end: int


@dataclass
class PhraseMatch:
    """A maximal run of shared words: word ranges and character spans."""
    a_words: Tuple[int, int]
    b_words: Tuple[int, int]
    a_span: Span
    b_span: Span
    text: str

    @property
    def length(self) -> int:
        return self.a_words[1] - self.a_words[0]


@dataclass
class CommonSubstring:
    text: str
    a_start: int
    b_start: int

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class PlagiarismReport:
    k: int
    jaccard: float
    containment_a: float            # share of A's k-grams found in B
    containment_b: float
    coverage_a: float               # share of A's characters in copied regions
    coverage_b: float
    matches: List[PhraseMatch]
    longest_common: List[CommonSubstring]
    spurious_hash_hits: int = 0
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def similarity(self) -> float:
        """Headline score: the larger containment (catches copying in
        either direction, regardless of document lengths)."""
        return max(self.containment_a, self.containment_b)

    def verdict(self, threshold: float = 0.25) -> str:
        s = self.similarity
        if s >= 0.6:
            return "heavy overlap"
        if s >= threshold:
            return "substantial overlap"
        if s > 0:
            return "minor overlap"
        return "no shared phrases"


def tokenize(text: str) -> List[Token]:
    return [Token(m.group().lower(), m.start(), m.end()) for m in WORD.finditer(text)]


def kgram_fingerprints(codes: Sequence[int], k: int,
                       hasher: RollingHash) -> Dict[int, List[int]]:
    """{rolling hash of codes[i:i+k]: [i, ...]} for every k-gram."""
    table: Dict[int, List[int]] = defaultdict(list)
    for i, h in hasher.window_hashes(codes, k):
        table[h].append(i)
    return table


def shared_phrases(a: str, b: str, k: int = 5, seed: int = 0) -> Tuple[List[PhraseMatch], Dict]:
    """
    Maximal copied word runs (≥ k words) shared by a and b, found by
    Rabin–Karp fingerprints of word k-grams and verified exactly.
    Returns (matches, stats).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    ta, tb = tokenize(a), tokenize(b)
    wa, wb = [t.word for t in ta], [t.word for t in tb]
    word_hash = RollingHash(base=257, modulus=MERSENNE_61)
    ca, cb = hash_tokens(wa, word_hash), hash_tokens(wb, word_hash)
    hasher = RollingHash(modulus=MERSENNE_61, rng=random.Random(seed))
    fa = kgram_fingerprints(ca, k, hasher)
    stats = {"kgrams_a": max(len(wa) - k + 1, 0), "kgrams_b": max(len(wb) - k + 1, 0),
             "hash_hits": 0, "spurious": 0, "shared_a": 0, "shared_b": 0}
    pairs = set()
    shared_a, shared_b = set(), set()
    for j, h in hasher.window_hashes(cb, k):
        for i in fa.get(h, ()):
            stats["hash_hits"] += 1
            if wa[i:i + k] == wb[j:j + k]:
                pairs.add((i, j))
                shared_a.add(i)
                shared_b.add(j)
            else:
                stats["spurious"] += 1
    stats["shared_a"], stats["shared_b"] = len(shared_a), len(shared_b)
    stats["distinct_shared"] = len({tuple(wa[i:i + k]) for i in shared_a})
    stats["distinct_a"] = len({tuple(wa[i:i + k]) for i in range(stats["kgrams_a"])})
    stats["distinct_b"] = len({tuple(wb[j:j + k]) for j in range(stats["kgrams_b"])})

    # merge diagonal runs (i, j), (i+1, j+1), ... into maximal matches
    matches = []
    for i, j in sorted(pairs):
        if (i - 1, j - 1) in pairs:
            continue
        end = 0
        while (i + end + 1, j + end + 1) in pairs:
            end += 1
        a_last, b_last = i + end + k - 1, j + end + k - 1
        span_a = (ta[i].start, ta[a_last].end)
        span_b = (tb[j].start, tb[b_last].end)
        matches.append(PhraseMatch((i, a_last + 1), (j, b_last + 1), span_a, span_b,
                                   a[span_a[0]:span_a[1]]))
    matches.sort(key=lambda m: (-m.length, m.a_span))
    return matches, stats


def longest_common_substrings(a: str, b: str, min_length: int = 20,
                              top: int = 5, separator: str = "\x01") -> List[CommonSubstring]:
    """
    Up to `top` longest verbatim overlaps (≥ min_length characters)
    between a and b, from the suffix array of a + separator + b.
    Overlaps that sit inside an already-reported one are skipped.
    """
    if separator in a or separator in b:
        raise ValueError("separator occurs in the input")
    joined = a + separator + b
    index = SuffixArray(joined)
    sa, lcp, cut = index.sa, index.lcp, len(a)
    candidates = []
    for r in range(1, len(joined)):
        if lcp[r] < min_length:
            continue
        i, j = int(sa[r - 1]), int(sa[r])
        if (i < cut) == (j < cut):
            continue
        ia, jb = (i, j) if i < cut else (j, i)
        candidates.append((lcp[r], ia, jb - cut - 1))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    chosen: List[CommonSubstring] = []
    for length, ia, ib in candidates:
        if any(c.a_start <= ia and ia + length <= c.a_start + c.length
               for c in chosen):
            continue
        chosen.append(CommonSubstring(a[ia:ia + length], ia, ib))
        if len(chosen) == top:
            break
    return chosen


def _coverage(spans: Sequence[Span], length: int) -> float:
    if length == 0:
        return 0.0
    covered, last = 0, 0
    for s, e in sorted(spans):
        s = max(s, last)
        if e > s:
            covered += e - s
            last = e
    return covered / length


def compare_documents(a: str, b: str, k: int = 5, min_common: int = 20,
                      top: int = 5, seed: int = 0) -> PlagiarismReport:
    """Similarity scores, copied phrases and longest common substrings."""
    matches, stats = shared_phrases(a, b, k, seed)
    union = stats["distinct_a"] + stats["distinct_b"] - stats["distinct_shared"]
    jaccard = stats["distinct_shared"] / union if union else 0.0
    cont_a = stats["shared_a"] / stats["kgrams_a"] if stats["kgrams_a"] else 0.0
    cont_b = stats["shared_b"] / stats["kgrams_b"] if stats["kgrams_b"] else 0.0
    common = longest_common_substrings(a, b, min_common, top) if a and b else []
    return PlagiarismReport(
        k=k, jaccard=jaccard, containment_a=cont_a, containment_b=cont_b,
        coverage_a=_coverage([m.a_span for m in matches], len(a)),
        coverage_b=_coverage([m.b_span for m in matches], len(b)),
        matches=matches, longest_common=common,
        spurious_hash_hits=stats["spurious"], stats=stats)


def highlight(text: str, spans: Sequence[Span], style: str = "brackets") -> str:
    """
    Mark spans in text. style: "brackets" ([[...]]), "ansi" (red,
    for terminals) or "html" (<mark>...</mark>, text escaped).
    Overlapping spans are merged.
    """
    open_tag, close_tag = {"brackets": ("[[", "]]"),
                           "ansi": ("\x1b[1;31m", "\x1b[0m"),
                           "html": ("<mark>", "</mark>")}[style]
    merged: List[List[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    esc = (lambda x: x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")) \
        if style == "html" else (lambda x: x)
    out, last = [], 0
    for s, e in merged:
        out.append(esc(text[last:s]))
        out.append(open_tag + esc(text[s:e]) + close_tag)
        last = e
    out.append(esc(text[last:]))
    return "".join(out)
