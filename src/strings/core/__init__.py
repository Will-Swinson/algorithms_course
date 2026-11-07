"""Pattern-matching core: KMP, Rabin-Karp, suffix array + LCP,
Ukkonen suffix tree, and the StringSearchEngine that unifies them."""

from src.strings.core.kmp import (
    MatchResult, brute_force_search, kmp_search, kmp_trace, prefix_function,
)
from src.strings.core.rabin_karp import rabin_karp, rabin_karp_multi
from src.strings.core.suffix_array import (
    SuffixArray, build_suffix_array, kasai_lcp, longest_common_substring,
)
from src.strings.core.suffix_tree import SuffixTree
from src.strings.core.search_engine import (
    Recommendation, SearchResult, StringSearchEngine,
)
