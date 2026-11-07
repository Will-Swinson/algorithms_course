"""Test suite for DNA search: FASTA I/O, reverse complements, exact
gene search on both strands, and mismatch-tolerant search."""

import io

import pytest

from src.strings.applications.dna_search import (
    FastaRecord, GenomeIndex, brute_force_mutations, hamming_mismatches,
    read_fasta, reverse_complement, write_fasta,
)
from src.strings.utils.generators import dna_sequence, mutate

FASTA = """>chr1 first test chromosome
ACGTACGTTT
GGCCA
; a comment
>chr2
ttttgcacgt
"""


class TestFasta:

    def test_parse_string(self):
        recs = read_fasta(FASTA)
        assert [r.name for r in recs] == ["chr1", "chr2"]
        assert recs[0].description == "first test chromosome"
        assert recs[0].sequence == "ACGTACGTTTGGCCA"
        assert recs[1].sequence == "TTTTGCACGT"

    def test_round_trip(self, tmp_path):
        recs = read_fasta(FASTA)
        path = str(tmp_path / "g.fasta")
        write_fasta(recs, path, width=4)
        assert read_fasta(path) == recs
        assert read_fasta(io.StringIO(FASTA)) == recs

    @pytest.mark.parametrize("bad", ["ACGT\n>x\nAC", ">\nACGT"])
    def test_errors(self, bad):
        with pytest.raises(ValueError):
            read_fasta(bad if bad.startswith(">") else io.StringIO(bad))


def test_reverse_complement():
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("acgN") == "Ncgt"
    assert hamming_mismatches("ACGT", "AGGA") == [(1, "C", "G"), (3, "T", "A")]


@pytest.mark.parametrize("method", ["sa", "tree"])
class TestIndex:

    def test_exact_both_strands(self, method):
        index = GenomeIndex(read_fasta(FASTA), method)
        # chr1 = ACGTACGTTTGGCCA: TACG at 3, its reverse complement CGTA at 1
        hits = index.find_gene("TACG")
        assert [(h.record, h.position, h.strand) for h in hits] == \
            [("chr1", 1, "-"), ("chr1", 3, "+")]
        fwd = index.find_gene("TACG", both_strands=False)
        assert [(h.record, h.position) for h in fwd] == [("chr1", 3)]

    def test_palindrome_reported_once(self, method):
        index = GenomeIndex("TTACGTAA", method)        # ACGT is its own RC
        assert len(index.find_gene("ACGT")) == 1

    def test_records_do_not_bleed(self, method):
        index = GenomeIndex(read_fasta(FASTA), method)
        # chr1 ends "...GCCA", chr2 starts "TTTT": no hit across the boundary
        assert index.find_gene("CCATTTT", both_strands=False) == []
        assert [h.record for h in index.find_gene("GCAC", both_strands=False)] == ["chr2"]

    def test_mutations_match_brute_force(self, method):
        genome = dna_sequence(5000, seed=7)
        index = GenomeIndex(genome, method)
        gene, _ = mutate(genome[1000:1024], 2, seed=3)
        for k in range(4):
            hits = index.find_mutations(gene, k, both_strands=False)
            assert [h.position for h in sorted(hits, key=lambda h: h.position)] == \
                brute_force_mutations(genome, gene, k)

    def test_mutation_details(self, method):
        genome = "AAAACCCCGGGGTTTT"
        index = GenomeIndex(genome, method)
        hits = index.find_mutations("CCACGG", 1, both_strands=False)
        assert len(hits) == 1
        h = hits[0]
        assert h.position == 4 and h.distance == 1      # CCCCGG vs CCACGG
        assert h.mismatches == [(2, "A", "C")]

    def test_count_and_len(self, method):
        index = GenomeIndex("ACGTACGT", method)
        assert index.count("acg") == 2 and len(index) == 8


class TestValidation:

    def test_bad_inputs(self):
        index = GenomeIndex("ACGT")
        with pytest.raises(ValueError):
            index.find_gene("")
        with pytest.raises(ValueError):
            index.find_mutations("AC", 2)
        with pytest.raises(ValueError):
            index.find_mutations("ACGT", -1)
        with pytest.raises(ValueError):
            GenomeIndex([])
        with pytest.raises(ValueError):
            GenomeIndex("ACGT", method="kmp")
        with pytest.raises(ValueError):
            GenomeIndex([FastaRecord("x", "", "AC#GT")])

    def test_gc_content(self):
        assert GenomeIndex("GGCCAATT").gc_content() == 0.5

    def test_find_genes(self):
        index = GenomeIndex("ACGTTTACG")
        res = index.find_genes({"g1": "ACG", "g2": "GGG"}, both_strands=False)
        assert [h.position for h in res["g1"]] == [0, 6] and res["g2"] == []
