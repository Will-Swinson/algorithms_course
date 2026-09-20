# File: src/graphs/graph.py
"""
Generic graph class supporting both classic representations.

A graph is a set of NODES connected by EDGES. Two storage strategies
exist, and which one is right depends on the graph's density:

    Adjacency LIST   {node: {neighbor: weight}}. Space O(V + E) —
                     only edges that exist are stored. Ideal for
                     sparse graphs (E << V²), which most real graphs
                     are (road networks, social graphs, the web).

    Adjacency MATRIX V×V grid where cell [i][j] holds the weight of
                     edge i→j (or None). Space O(V²) regardless of how
                     many edges exist. Ideal for dense graphs and for
                     O(1) "is u connected to v?" checks.

Both representations sit behind one API, so every algorithm in this
package (BFS, DFS, Dijkstra) runs unchanged on either.

Complexity summary (V = nodes, E = edges):
    Operation        List            Matrix
    add_node         O(1)            O(V) (grow row + column)
    add_edge         O(1)            O(1)
    remove_node      O(V + E)        O(V²) (rebuild grid)
    remove_edge      O(1)            O(1)
    has_edge         O(1)            O(1)
    get_neighbors    O(deg(v))       O(V) (scan the whole row)
    space            O(V + E)        O(V²)
"""
from typing import Any, Dict, Iterator, List, Optional, Tuple

ADJACENCY_LIST = "list"
ADJACENCY_MATRIX = "matrix"


class Graph:
    """
    Graph supporting directed/undirected edges, optional weights, and
    a choice of adjacency-list or adjacency-matrix storage.

    Args:
        directed: If False (default), add_edge(u, v) also creates the
            reverse edge v→u, and removals mirror likewise.
        representation: "list" (default) or "matrix".

    Nodes may be any hashable value (ints, strings, tuples...).
    """

    def __init__(self, directed: bool = False,
                 representation: str = ADJACENCY_LIST):
        if representation not in (ADJACENCY_LIST, ADJACENCY_MATRIX):
            raise ValueError(
                f"representation must be '{ADJACENCY_LIST}' or "
                f"'{ADJACENCY_MATRIX}', got {representation!r}"
            )
        self.directed = directed
        self.representation = representation
        # Adjacency list: {node: {neighbor: weight}}
        self._adj: Dict[Any, Dict[Any, float]] = {}
        # Adjacency matrix: node <-> row/column index, grid of weights
        self._index: Dict[Any, int] = {}
        self._nodes_by_index: List[Any] = []
        self._matrix: List[List[Optional[float]]] = []

    # -- node operations ---------------------------------------------------

    def add_node(self, node: Any) -> None:
        """Add a node (no-op if already present). O(1) list, O(V) matrix."""
        if self.representation == ADJACENCY_LIST:
            self._adj.setdefault(node, {})
        elif node not in self._index:
            self._index[node] = len(self._nodes_by_index)
            self._nodes_by_index.append(node)
            for row in self._matrix:       # new column in every row
                row.append(None)
            self._matrix.append([None] * len(self._nodes_by_index))

    def remove_node(self, node: Any) -> None:
        """
        Remove a node and every edge touching it.

        Raises:
            KeyError: If the node is not in the graph.
        """
        self._require_node(node)
        if self.representation == ADJACENCY_LIST:
            del self._adj[node]
            for neighbors in self._adj.values():
                neighbors.pop(node, None)
        else:
            removed = self._index.pop(node)
            self._nodes_by_index.pop(removed)
            self._matrix.pop(removed)
            for row in self._matrix:
                row.pop(removed)
            # Every node after the removed one shifts down one index
            for i, n in enumerate(self._nodes_by_index):
                self._index[n] = i

    def has_node(self, node: Any) -> bool:
        """Return True if the node exists. O(1)."""
        if self.representation == ADJACENCY_LIST:
            return node in self._adj
        return node in self._index

    def nodes(self) -> List[Any]:
        """All nodes, in insertion order. O(V)."""
        if self.representation == ADJACENCY_LIST:
            return list(self._adj)
        return list(self._nodes_by_index)

    def __len__(self) -> int:
        """Number of nodes."""
        if self.representation == ADJACENCY_LIST:
            return len(self._adj)
        return len(self._index)

    def __contains__(self, node: Any) -> bool:
        return self.has_node(node)

    # -- edge operations ------------------------------------------------------

    def add_edge(self, u: Any, v: Any, weight: float = 1) -> None:
        """
        Add edge u→v with the given weight (default 1 = unweighted).
        Unknown endpoints are added automatically. In an undirected
        graph the reverse edge v→u is created too. Re-adding an edge
        updates its weight. O(1) after any node insertion.
        """
        self.add_node(u)
        self.add_node(v)
        if self.representation == ADJACENCY_LIST:
            self._adj[u][v] = weight
            if not self.directed:
                self._adj[v][u] = weight
        else:
            self._matrix[self._index[u]][self._index[v]] = weight
            if not self.directed:
                self._matrix[self._index[v]][self._index[u]] = weight

    def remove_edge(self, u: Any, v: Any) -> None:
        """
        Remove edge u→v (and v→u when undirected).

        Raises:
            KeyError: If either node or the edge does not exist.
        """
        self._require_node(u)
        self._require_node(v)
        if not self.has_edge(u, v):
            raise KeyError(f"no edge {u!r} -> {v!r}")
        if self.representation == ADJACENCY_LIST:
            del self._adj[u][v]
            if not self.directed:
                del self._adj[v][u]
        else:
            self._matrix[self._index[u]][self._index[v]] = None
            if not self.directed:
                self._matrix[self._index[v]][self._index[u]] = None

    def has_edge(self, u: Any, v: Any) -> bool:
        """Return True if edge u→v exists. O(1) in both representations."""
        if self.representation == ADJACENCY_LIST:
            return u in self._adj and v in self._adj[u]
        if u not in self._index or v not in self._index:
            return False
        return self._matrix[self._index[u]][self._index[v]] is not None

    def get_weight(self, u: Any, v: Any) -> float:
        """
        Return the weight of edge u→v.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.has_edge(u, v):
            raise KeyError(f"no edge {u!r} -> {v!r}")
        if self.representation == ADJACENCY_LIST:
            return self._adj[u][v]
        return self._matrix[self._index[u]][self._index[v]]

    def get_neighbors(self, node: Any) -> List[Tuple[Any, float]]:
        """
        Return [(neighbor, weight), ...] for every edge leaving `node`.

        This is THE performance-defining operation: O(deg(node)) for
        the adjacency list, but O(V) for the matrix because the whole
        row must be scanned — the reason traversals run slower on
        matrix-backed sparse graphs.

        Raises:
            KeyError: If the node is not in the graph.
        """
        self._require_node(node)
        if self.representation == ADJACENCY_LIST:
            return list(self._adj[node].items())
        row = self._matrix[self._index[node]]
        return [(self._nodes_by_index[j], w)
                for j, w in enumerate(row) if w is not None]

    def num_edges(self) -> int:
        """Number of edges (each undirected edge counted once). O(V + E)."""
        if self.representation == ADJACENCY_LIST:
            total = sum(len(nbrs) for nbrs in self._adj.values())
        else:
            total = sum(1 for row in self._matrix
                        for w in row if w is not None)
        return total if self.directed else total // 2

    # -- misc ------------------------------------------------------------------

    def _require_node(self, node: Any) -> None:
        if not self.has_node(node):
            raise KeyError(f"node {node!r} not in graph")

    def __str__(self) -> str:
        """Human-readable adjacency listing, one node per line."""
        kind = "directed" if self.directed else "undirected"
        arrow = "->" if self.directed else "--"
        lines = [f"Graph ({kind}, {self.representation}, "
                 f"{len(self)} nodes, {self.num_edges()} edges)"]
        for node in self.nodes():
            edges = ", ".join(
                f"{node}{arrow}{nbr} (w={w:g})"
                for nbr, w in self.get_neighbors(node)
            )
            lines.append(f"  {node}: {edges if edges else '(no edges)'}")
        return "\n".join(lines)
