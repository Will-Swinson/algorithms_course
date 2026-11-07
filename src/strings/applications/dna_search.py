"""
DNA sequence search over an indexed genome.

A genome is indexed once (suffix array or suffix tree) and then
queried many times — the access pattern that makes indexes win
(a read aligner maps millions of reads against one reference).

    read_fasta / write_fasta   FASTA records: ">name description" then
                               sequence lines; case folded to upper
    reverse_complement         DNA is double-stranded: a gene can lie
                               on either strand, so searches can cover
                               both
    GenomeIndex                one index over all records (joined with
                               a separator), positions mapped back to
                               (record, offset, strand)
    find_gene                  exact occurrences
    find_mutations             approximate matches with ≤ k
                               substitutions (Hamming distance)

Approximate search by the pigeonhole principle: split the pattern into
k + 1 pieces. An occurrence with ≤ k mismatches leaves at least one
piece intact, so exact hits of the pieces (from the index) give every
candidate alignment; each candidate is then verified by counting
mismatches. The returned hits name the mutated positions and bases.
"""
import io
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Union

from src.strings.core.suffix_array import SuffixArray
from src.strings.core.suffix_tree import SuffixTree

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
SEPARATOR = "#"


@dataclass
class FastaRecord:
    name: str
    description: str
    sequence: str


@dataclass
class Hit:
    record: str
    position: int                  # 0-based offset in the record (forward strand)
    strand: str                    # "+" or "-"
    matched: str                   # the genome substring
    mismatches: List[Tuple[int, str, str]] = field(default_factory=list)
    # (offset within the pattern, expected base, genome base)

    @property
    def distance(self) -> int:
        return len(self.mismatches)


def read_fasta(source: Union[str, io.TextIOBase]) -> List[FastaRecord]:
    """Parse FASTA from a path, file object, or a string that starts with '>'."""
    if isinstance(source, str):
        if source.lstrip().startswith(">"):
            lines = source.splitlines()
        else:
            with open(source) as fh:
                lines = fh.read().splitlines()
    else:
        lines = source.read().splitlines()
    records: List[FastaRecord] = []
    name = desc = None
    chunks: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if name is not None:
                records.append(FastaRecord(name, desc, "".join(chunks).upper()))
            header = line[1:].split(None, 1)
            if not header:
                raise ValueError("FASTA header without a name")
            name, desc = header[0], header[1] if len(header) > 1 else ""
            chunks = []
        else:
            if name is None:
                raise ValueError("sequence data before the first '>' header")
            chunks.append(line.replace(" ", ""))
    if name is not None:
        records.append(FastaRecord(name, desc, "".join(chunks).upper()))
    return records


def write_fasta(records: Iterable[FastaRecord], path: str, width: int = 70) -> None:
    with open(path, "w") as fh:
        for r in records:
            fh.write(f">{r.name}{' ' + r.description if r.description else ''}\n")
            for i in range(0, len(r.sequence), width):
                fh.write(r.sequence[i:i + width] + "\n")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def hamming_mismatches(a: str, b: str) -> List[Tuple[int, str, str]]:
    return [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]


class GenomeIndex:
    """
    Suffix-array or suffix-tree index over one or more sequences.

        index = GenomeIndex(read_fasta("genome.fasta"), method="sa")
        index.find_gene("ATGGCC")                 # exact, both strands
        index.find_mutations("ATGGCCTTA", 1)      # ≤ 1 substitution
    """

    def __init__(self, records: Union[str, Iterable[FastaRecord]], method: str = "sa"):
        if isinstance(records, str):
            records = [FastaRecord("seq", "", records.upper())]
        self.records = list(records)
        if not self.records:
            raise ValueError("no sequences to index")
        if method not in ("sa", "tree"):
            raise ValueError("method must be 'sa' or 'tree'")
        self.method = method
        self.offsets: List[int] = []
        parts = []
        pos = 0
        for r in self.records:
            if SEPARATOR in r.sequence:
                raise ValueError(f"record {r.name} contains the separator {SEPARATOR!r}")
            self.offsets.append(pos)
            parts.append(r.sequence)
            pos += len(r.sequence) + 1
        self.genome = SEPARATOR.join(parts)
        self.index = SuffixArray(self.genome, lcp=False) if method == "sa" \
            else SuffixTree(self.genome)

    def __len__(self) -> int:
        return sum(len(r.sequence) for r in self.records)

    def _locate(self, g: int) -> Tuple[int, int]:
        """Global position -> (record number, offset)."""
        lo, hi = 0, len(self.offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.offsets[mid] <= g:
                lo = mid
            else:
                hi = mid - 1
        return lo, g - self.offsets[lo]

    def _exact(self, pattern: str) -> List[int]:
        return self.index.find(pattern)

    def count(self, pattern: str) -> int:
        return self.index.count(pattern.upper())

    def find_gene(self, gene: str, both_strands: bool = True) -> List[Hit]:
        """Exact occurrences, sorted by (record, position, strand)."""
        gene = gene.upper()
        if not gene or SEPARATOR in gene:
            raise ValueError("gene must be a non-empty sequence")
        hits = []
        strands = [("+", gene)]
        rc = reverse_complement(gene)
        if both_strands and rc != gene:
            strands.append(("-", rc))
        for strand, query in strands:
            for g in self._exact(query):
                rec, off = self._locate(g)
                hits.append(Hit(self.records[rec].name, off, strand, query))
        hits.sort(key=lambda h: (h.record, h.position, h.strand))
        return hits

    def find_genes(self, genes: Dict[str, str], both_strands: bool = True) -> Dict[str, List[Hit]]:
        return {name: self.find_gene(seq, both_strands) for name, seq in genes.items()}

    def find_mutations(self, gene: str, max_mismatches: int = 1,
                       both_strands: bool = True) -> List[Hit]:
        """
        Occurrences with at most `max_mismatches` substitutions
        (exact ones included, with distance 0).
        """
        gene = gene.upper()
        k = max_mismatches
        m = len(gene)
        if k < 0:
            raise ValueError("max_mismatches must be >= 0")
        if m == 0 or SEPARATOR in gene:
            raise ValueError("gene must be a non-empty sequence")
        if k + 1 > m:
            raise ValueError("pattern too short for that many mismatches")
        hits: Dict[Tuple[int, str], Hit] = {}
        strands = [("+", gene)]
        rc = reverse_complement(gene)
        if both_strands and rc != gene:
            strands.append(("-", rc))
        bounds = [round(i * m / (k + 1)) for i in range(k + 2)]
        n = len(self.genome)
        for strand, query in strands:
            for a, b in zip(bounds, bounds[1:]):
                for g in self._exact(query[a:b]):
                    start = g - a
                    if start < 0 or start + m > n or (start, strand) in hits:
                        continue
                    window = self.genome[start:start + m]
                    if SEPARATOR in window:
                        continue
                    diff = hamming_mismatches(query, window)
                    if len(diff) <= k:
                        rec, off = self._locate(start)
                        hits[(start, strand)] = Hit(self.records[rec].name, off,
                                                    strand, window, diff)
        return sorted(hits.values(), key=lambda h: (h.distance, h.record, h.position))

    def gc_content(self) -> float:
        seq = self.genome.replace(SEPARATOR, "")
        return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def brute_force_mutations(genome: str, gene: str, max_mismatches: int) -> List[int]:
    """Reference implementation (forward strand, one sequence)."""
    m = len(gene)
    return [i for i in range(len(genome) - m + 1)
            if len(hamming_mismatches(gene, genome[i:i + m])) <= max_mismatches]
