"""
Week 11 demo: string search, from linear scans to suffix indexes.

Run from the project root:

    python examples/week11_demo.py

Shows (1) KMP's failure function and its fallbacks on a mismatch,
(2) Rabin–Karp collisions with a small modulus and a bad base, and
batch matching, (3) the suffix array + LCP of "banana" and the
queries they answer, (4) Ukkonen's suffix tree phase by phase,
(5) the StringSearchEngine comparing all four methods and
recommending one, and (6) the three applications: plagiarism
detection, DNA search with mutations, and autocomplete.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strings.applications import (
    Autocomplete, GenomeIndex, compare_documents, highlight, read_fasta,
)
from src.strings.core import (
    StringSearchEngine, SuffixArray, SuffixTree, brute_force_search,
    kmp_search, prefix_function, rabin_karp, rabin_karp_multi,
)
from src.strings.core.kmp import format_trace
from src.strings.utils.generators import (
    dna_sequence, english_like_text, mutate, sample_patterns,
)
from src.strings.utils.rolling_hash import RollingHash


def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_kmp():
    header("1. KMP: the failure function turns mismatches into jumps")
    pattern = "ababd"
    print(f"  pattern {pattern!r}: lps = {prefix_function(pattern)}")
    print(format_trace("abababd", pattern, limit=9))
    text, bad = "a" * 20_000, "a" * 99 + "b"
    kmp, brute = kmp_search(text, bad), brute_force_search(text, bad)
    print(f"\n  worst case a^20000 vs a^99 b: KMP {kmp.comparisons:,} comparisons, "
          f"brute force {brute.comparisons:,} ({brute.comparisons / kmp.comparisons:.0f}x)")


def demo_rabin_karp():
    header("2. Rabin-Karp: rolling hashes, collisions, batches")
    text = english_like_text(200_000, seed=1)
    absent = "qzqzqzqzqzqz"
    for label, hasher in (("q = 101, random base", RollingHash(modulus=101, rng=random.Random(1))),
                          ("q = 2^61 - 1, random base", RollingHash(rng=random.Random(1))),
                          ("base = 1 (sum of characters)", RollingHash(base=1))):
        r = rabin_karp(text, absent, hasher=hasher)
        print(f"  {label:30s} spurious hits {r.stats['spurious']:6,}  "
              f"(expected ~{(len(text) - 11) / hasher.modulus:,.2f} for a random base)")
    mc = rabin_karp(text, absent, modulus=101, seed=1, verify=False)
    print(f"  without verification (Monte Carlo), q = 101: "
          f"{mc.stats['false_positives']:,} false matches reported")
    patterns = sample_patterns(text, 50, 10, seed=2)
    start = time.perf_counter()
    rabin_karp_multi(text, patterns, seed=1)
    multi = time.perf_counter() - start
    start = time.perf_counter()
    for p in patterns:
        kmp_search(text, p)
    print(f"  50 patterns: one Rabin-Karp pass {multi:.2f} s vs 50 KMP scans "
          f"{time.perf_counter() - start:.2f} s")


def demo_suffix_array():
    header("3. Suffix array + LCP of 'banana'")
    sa = SuffixArray("banana")
    print("  rank  SA  LCP  suffix")
    for rank, start, lcp, suffix in sa.table():
        print(f"  {rank:4d} {start:3d} {lcp:4d}  {suffix}")
    print(f"  find('ana') = {sa.find('ana')} via two binary searches; "
          f"count('a') = {sa.count('a')}")
    print(f"  longest repeat = {sa.longest_repeated_substring()!r} (max LCP); "
          f"distinct substrings = {sa.distinct_substrings()} = 21 - sum(LCP)")


def demo_suffix_tree():
    header("4. Ukkonen's suffix tree on 'abcabxabcd'")
    tree = SuffixTree("abcabxabcd", record=True)
    for ph in tree.trace:
        node, edge, length, remainder = ph["after"]
        c = tree.s[ph["phase"]].replace(tree.terminator, "$")
        kind = "implicit (rule 3)" if ph["implicit"] else "all explicit"
        print(f"  phase {ph['phase']:2d} '{c}': {ph['explicit']} explicit "
              f"extension(s), {ph['splits']} split(s); {ph['implicit']} suffix(es) "
              f"left implicit -> {kind}; active length {length}")
    print(f"  {tree}: find('abc') = {tree.find('abc')}, longest repeat "
          f"{tree.longest_repeated_substring()!r}")


def demo_engine():
    header("5. StringSearchEngine: one API, four methods")
    text = english_like_text(1_000_000, seed=3)
    engine = StringSearchEngine(text)
    for method in ("kmp", "rk", "sa", "tree"):
        r = engine.find("the water", method=method)
        print(f"  {method:4s} {r.count:3d} matches  search {r.search_time * 1e3:8.3f} ms  "
              f"preprocessing {r.preprocess_time:6.2f} s")
    for q in (1, 20, 1000):
        print(f"  {q:>4} queries -> {engine.recommend(q)}")


def demo_applications():
    header("6. Applications")
    source = english_like_text(6_000, seed=4)
    essay = (english_like_text(1_200, seed=5) + " " + source[2_000:2_600] + " "
             + english_like_text(800, seed=6))
    report = compare_documents(essay, source, k=5)
    print(f"  plagiarism: {report.verdict()}, similarity {report.similarity:.2f}, "
          f"{len(report.matches)} copied region(s), longest common substring "
          f"{report.longest_common[0].length} chars")
    first = report.matches[0]
    snippet = highlight(essay, [first.a_span])
    start = max(0, first.a_span[0] - 40)
    print(f"    ...{snippet[start:start + 160]}...")

    fasta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "string_inputs", "sample_genome.fasta")
    records = read_fasta(fasta)
    index = GenomeIndex(records)
    chr_b = records[1].sequence
    gene = chr_b[4_321:4_351]
    mutated, where = mutate(gene, 2, seed=8)
    exact = index.find_gene(gene)
    fuzzy = index.find_mutations(mutated, 2, both_strands=False)
    print(f"  DNA ({len(records)} FASTA records, {len(index):,} bp): gene found at "
          f"{[(h.record, h.position, h.strand) for h in exact]}")
    print(f"    with 2 mutations at gene offsets {where}: "
          f"{[(h.record, h.position, [m[0] for m in h.mismatches]) for h in fuzzy]}")
    genome = dna_sequence(1_000_000, repeat_fraction=0.05, seed=7)
    big = GenomeIndex(genome)
    probe = mutate(genome[500_000:500_030], 2, seed=9)[0]
    start = time.perf_counter()
    hits = big.find_mutations(probe, 2)
    print(f"  1 Mb genome: {len(hits)} hit(s) with <= 2 mismatches in "
          f"{(time.perf_counter() - start) * 1e3:.2f} ms (index built once)")

    ac = Autocomplete.from_text(english_like_text(300_000, seed=9))
    print(f"  autocomplete ({len(ac):,} words): 'wa' -> "
          f"{[c.term for c in ac.complete('wa', 4)]}")
    for _ in range(30):
        ac.record_query("watch")
    print(f"  after 30 users pick 'watch': 'wa' -> "
          f"{[c.term for c in ac.complete('wa', 4)]}")


if __name__ == "__main__":
    demo_kmp()
    demo_rabin_karp()
    demo_suffix_array()
    demo_suffix_tree()
    demo_engine()
    demo_applications()
