"""
Suffix tree by Ukkonen's online algorithm: O(n) construction.

A suffix tree of T$ is the compressed trie of all its suffixes: every
internal node branches, every edge is labelled by a substring stored
as an index pair [start, end), and leaf i spells suffix T[i:]$. The
unique terminator $ makes every suffix end at a leaf.

Ukkonen adds one character per phase i, extending every suffix that
is still "in progress" by T[i]. Three tricks make the whole thing
linear:

  1. Once a leaf, always a leaf. Leaf edges end at a shared global end
     (here: they store no end at all), so extending every leaf by one
     character costs nothing.
  2. Active point (active_node, active_edge, active_length) marks where
     the next extension starts, and `remainder` counts the suffixes
     still waiting to be inserted explicitly.
  3. Suffix links: from the internal node for xα to the node for α,
     so after an insertion the next, one-shorter suffix is reached in
     O(1) amortized instead of from the root.

Each extension in a phase is one of
    rule 2 (explicit)  the next character is missing: add a leaf,
                       splitting the edge first if the active point is
                       inside one; remainder decreases
    rule 3 (implicit)  the character is already there: the suffix is
                       implicitly present, move the active point one
                       character down and end the phase (showstopper)
so each phase ends with `remainder` suffixes left implicit. With
record=True every phase is logged (active point before and after,
explicit extensions, leaves and splits created) — the "implicit vs
explicit suffix" bookkeeping, and the source of the step-by-step
drawings.

Storage: flat lists indexed by node id (start, end, suffix link,
suffix index, children dict). Internal nodes only get a children
dict; leaves keep None. Even so, a Python suffix tree costs a few
hundred bytes per text character — the memory side of the speed /
memory trade-off measured in the benchmark.

Queries, after locating the pattern by walking down from the root
in O(m):
    find / count          O(m + occ) / O(m) (leaf counts precomputed)
    longest_prefix_match  longest prefix of a query that occurs in T
    longest_repeated_substring  deepest internal node
    distinct_substrings   Σ edge lengths (terminator excluded)
to_dot() writes Graphviz DOT; visualization/draw_suffix_tree.py draws
the same tree with matplotlib.
"""
from typing import Dict, List, Optional, Tuple

ROOT = 0


def choose_terminator(text: str) -> str:
    """A character absent from `text` ('$' when possible)."""
    for c in ("$", "\x00", "#", "\x01"):
        if c not in text:
            return c
    used = set(text)
    for code in range(0xE000, 0xF8FF):          # private-use area
        if chr(code) not in used:
            return chr(code)
    raise ValueError("no unused character available for the terminator")


class SuffixTree:
    """Ukkonen suffix tree over `text` (a terminator is appended)."""

    def __init__(self, text: str, terminator: Optional[str] = None,
                 record: bool = False):
        if terminator is None:
            terminator = choose_terminator(text)
        elif terminator in text or len(terminator) != 1:
            raise ValueError("terminator must be one character absent from text")
        self.text = text
        self.terminator = terminator
        self.s = text + terminator
        self.start: List[int] = []
        self.end: List[Optional[int]] = []      # None = leaf (runs to the end)
        self.link: List[int] = []
        self.suffix_index: List[int] = []      # leaves: start of their suffix
        self.children: List[Optional[Dict[str, int]]] = []
        self.trace: List[Dict] = []
        self._leaf_count: Optional[List[int]] = None
        self._depth: Optional[List[int]] = None
        self._new_node(-1, -1, internal=True)  # root
        self._build(record)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _new_node(self, start: int, end: Optional[int], internal: bool,
                  suffix: int = -1) -> int:
        self.start.append(start)
        self.end.append(end)
        self.link.append(ROOT)
        self.suffix_index.append(suffix)
        self.children.append({} if internal else None)
        return len(self.start) - 1

    def _build(self, record: bool) -> None:
        s = self.s
        start, end, link, children = self.start, self.end, self.link, self.children
        new_node = self._new_node
        active_node, active_edge, active_length = ROOT, 0, 0
        remainder = 0
        for i, c in enumerate(s):
            remainder += 1
            last_new = None
            if record:
                before = (active_node, active_edge, active_length, remainder)
                leaves = splits = 0
            while remainder:
                if active_length == 0:
                    active_edge = i
                nxt = children[active_node].get(s[active_edge])
                if nxt is None:
                    # rule 2 at a node: hang a new leaf
                    children[active_node][s[active_edge]] = \
                        new_node(i, None, False, i - remainder + 1)
                    if record:
                        leaves += 1
                    if last_new is not None:
                        link[last_new] = active_node
                        last_new = None
                else:
                    e = end[nxt]
                    length = (i + 1 if e is None else e) - start[nxt]
                    if active_length >= length:  # walk down (skip/count)
                        active_edge += length
                        active_length -= length
                        active_node = nxt
                        continue
                    if s[start[nxt] + active_length] == c:
                        # rule 3: already present, phase ends (implicit)
                        if last_new is not None and active_node != ROOT:
                            link[last_new] = active_node
                            last_new = None
                        active_length += 1
                        break
                    # rule 2 inside an edge: split it
                    split = new_node(start[nxt], start[nxt] + active_length, True)
                    children[active_node][s[active_edge]] = split
                    children[split][c] = new_node(i, None, False, i - remainder + 1)
                    start[nxt] += active_length
                    children[split][s[start[nxt]]] = nxt
                    if last_new is not None:
                        link[last_new] = split
                    last_new = split
                    if record:
                        leaves += 1
                        splits += 1
                remainder -= 1
                if active_node == ROOT and active_length > 0:
                    active_length -= 1
                    active_edge = i - remainder + 1
                elif active_node != ROOT:
                    active_node = link[active_node]
            if record:
                self.trace.append({
                    "phase": i, "char": c, "before": before,
                    "after": (active_node, active_edge, active_length, remainder),
                    "explicit": leaves, "splits": splits,
                    "implicit": remainder, "nodes": len(start),
                    "snapshot": self._snapshot(i),
                })
        self.size = len(s)

    def _snapshot(self, i: int) -> List[Tuple[int, int, int, int]]:
        """(parent, child, start, end) of every edge after phase i."""
        edges = []
        for parent, kids in enumerate(self.children):
            if kids:
                for child in kids.values():
                    e = self.end[child]
                    edges.append((parent, child, self.start[child],
                                  i + 1 if e is None else e))
        return edges

    # ------------------------------------------------------------------
    # structure
    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.start)

    @property
    def num_leaves(self) -> int:
        return sum(1 for c in self.children if c is None)

    def is_leaf(self, node: int) -> bool:
        return self.children[node] is None

    def edge_end(self, node: int) -> int:
        e = self.end[node]
        return self.size if e is None else e

    def edge_label(self, node: int) -> str:
        return self.s[self.start[node]:self.edge_end(node)] if node != ROOT else ""

    def edges(self):
        """(parent, child) pairs in DFS order (children sorted by label)."""
        stack = [ROOT]
        while stack:
            u = stack.pop()
            kids = self.children[u]
            if kids:
                ordered = [kids[k] for k in sorted(kids)]
                for v in ordered:
                    yield u, v
                stack.extend(reversed(ordered))

    def _post_order(self) -> List[int]:
        order, stack = [], [ROOT]
        while stack:
            u = stack.pop()
            order.append(u)
            kids = self.children[u]
            if kids:
                stack.extend(kids.values())
        order.reverse()
        return order

    def leaf_counts(self) -> List[int]:
        if self._leaf_count is None:
            counts = [0] * self.num_nodes
            for u in self._post_order():
                kids = self.children[u]
                if kids is None:
                    counts[u] = 1
                else:
                    counts[u] = sum(counts[v] for v in kids.values())
            self._leaf_count = counts
        return self._leaf_count

    def string_depths(self) -> List[int]:
        if self._depth is None:
            depth = [0] * self.num_nodes
            stack = [ROOT]
            while stack:
                u = stack.pop()
                kids = self.children[u]
                if kids:
                    for v in kids.values():
                        depth[v] = depth[u] + self.edge_end(v) - self.start[v]
                        stack.append(v)
            self._depth = depth
        return self._depth

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def _locate(self, pattern: str) -> Tuple[int, int]:
        """
        Walk `pattern` down from the root. Returns (node, matched): the
        node whose edge the walk ended on (or at) and how many pattern
        characters matched.
        """
        s, start, children = self.s, self.start, self.children
        node, j, m = ROOT, 0, len(pattern)
        # the terminator is not part of the text: a query may not match it
        cut = pattern.find(self.terminator)
        if cut >= 0:
            m = cut
        while j < m:
            kids = children[node]
            if not kids:
                return node, j
            child = kids.get(pattern[j])
            if child is None:
                return node, j
            a, b = start[child], self.edge_end(child)
            k = 0
            while k < b - a and j < m:
                if s[a + k] != pattern[j]:
                    return child, j
                k += 1
                j += 1
            node = child
        return node, j

    def contains(self, pattern: str) -> bool:
        return self._locate(pattern)[1] == len(pattern)

    def count(self, pattern: str) -> int:
        if not pattern:
            return len(self.text) + 1
        node, j = self._locate(pattern)
        if j < len(pattern):
            return 0
        return self.leaf_counts()[node]

    def find(self, pattern: str) -> List[int]:
        """Sorted start positions of every occurrence."""
        if not pattern:
            return list(range(len(self.text) + 1))
        node, j = self._locate(pattern)
        if j < len(pattern):
            return []
        out, stack = [], [node]
        children, idx = self.children, self.suffix_index
        while stack:
            u = stack.pop()
            kids = children[u]
            if kids is None:
                out.append(idx[u])
            else:
                stack.extend(kids.values())
        out.sort()
        return out

    def longest_prefix_match(self, query: str) -> str:
        """Longest prefix of `query` that occurs somewhere in the text."""
        return query[:self._locate(query)[1]]

    def longest_repeated_substring(self) -> str:
        depth = self.string_depths()
        best, node = 0, ROOT
        for u in range(1, self.num_nodes):
            if self.children[u] is not None and depth[u] > best:
                best, node = depth[u], u
        if best == 0:
            return ""
        leaf = node
        while self.children[leaf] is not None:
            leaf = next(iter(self.children[leaf].values()))
        i = self.suffix_index[leaf]
        return self.text[i:i + best]

    def distinct_substrings(self) -> int:
        """Number of distinct non-empty substrings of the text."""
        total = 0
        for u in range(1, self.num_nodes):
            length = self.edge_end(u) - self.start[u]
            if self.children[u] is None:
                length -= 1                      # drop the terminator
            total += length
        return total

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def to_dot(self, show_links: bool = True, max_label: int = 12) -> str:
        """Graphviz DOT source (render with `dot -Tpng`)."""
        def label(u):
            text = self.edge_label(u).replace(self.terminator, "$")
            if len(text) > max_label:
                text = text[:max_label - 1] + "…"
            return text.replace("\\", "\\\\").replace('"', '\\"')

        lines = ["digraph SuffixTree {", "  node [shape=circle, label=\"\"];",
                 "  edge [fontname=\"Courier\"];"]
        lines.append(f"  n{ROOT} [shape=point, width=0.15];")
        for u in range(1, self.num_nodes):
            if self.children[u] is None:
                lines.append(f"  n{u} [shape=box, label=\"{self.suffix_index[u]}\"];")
            else:
                lines.append(f"  n{u} [shape=point, width=0.1];")
        for u, v in self.edges():
            lines.append(f"  n{u} -> n{v} [label=\"{label(v)}\"];")
        if show_links:
            for u in range(1, self.num_nodes):
                if self.children[u] is not None and self.link[u] != u:
                    lines.append(f"  n{u} -> n{self.link[u]} [style=dashed, "
                                 f"color=red, constraint=false];")
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"SuffixTree(n={len(self.text)}, nodes={self.num_nodes}, "
                f"leaves={self.num_leaves})")
