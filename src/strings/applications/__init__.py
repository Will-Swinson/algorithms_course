"""String applications: plagiarism detection (Rabin-Karp phrase
fingerprints + suffix-array common substrings), DNA search over an
indexed genome (exact and approximate), and top-k autocomplete over a
compressed prefix trie with query-frequency weighting."""

from src.strings.applications.plagiarism import (
    PlagiarismReport, compare_documents, highlight,
    longest_common_substrings, shared_phrases,
)
from src.strings.applications.dna_search import (
    GenomeIndex, Hit, read_fasta, reverse_complement, write_fasta,
)
from src.strings.applications.autocomplete import (
    Autocomplete, SortedPrefixIndex, term_frequencies,
)
