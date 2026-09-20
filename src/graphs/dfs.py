# File: src/graphs/dfs.py
"""
Depth-First Search — follow one path as deep as possible, then backtrack.

Where BFS uses a queue (FIFO), DFS uses a STACK (LIFO): the most
recently discovered node is explored next, so the traversal plunges
down a single path until it dead-ends, then backs up to the latest
branch point. The stack can be explicit (iterative version) or the
Python call stack itself (recursive version) — same algorithm, same
O(V + E) time, different bookkeeping.

The recursive version reads closest to the definition but inherits
Python's recursion limit (~1000 frames), so a long path graph
overflows it; the iterative version has no such limit and is what the
benchmarks use.
"""
import sys
from typing import Any, List, Set

from src.graphs.graph import Graph


def dfs_iterative(graph: Graph, start: Any) -> List[Any]:
    """
    Depth-first traversal using an explicit stack, returning nodes in
    visit order.

    Neighbors are pushed in REVERSE order so they are popped — and
    therefore visited — in the same left-to-right order the recursive
    version produces, making the two versions' outputs comparable.

    Args:
        graph: The graph to traverse.
        start: Node to begin from.

    Returns:
        List of nodes in DFS visit order (starts with `start`).

    Raises:
        KeyError: If `start` is not in the graph.

    Complexity: O(V + E) time, O(V) space for stack + visited set.
    """
    if not graph.has_node(start):
        raise KeyError(f"start node {start!r} not in graph")

    order: List[Any] = []
    visited: Set[Any] = set()
    stack = [start]
    while stack:
        node = stack.pop()                 # LIFO: newest discovery first
        if node in visited:
            # A node can be pushed more than once before its first
            # visit (reached via several neighbors); later pops skip it
            continue
        visited.add(node)
        order.append(node)
        neighbors = [n for n, _w in graph.get_neighbors(node)]
        for neighbor in reversed(neighbors):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def dfs_recursive(graph: Graph, start: Any) -> List[Any]:
    """
    Depth-first traversal using recursion (the call stack is the
    stack), returning nodes in visit order.

    Produces the same order as dfs_iterative. Depth is bounded by
    Python's recursion limit, so prefer dfs_iterative for graphs whose
    paths may exceed ~1000 nodes.

    Raises:
        KeyError: If `start` is not in the graph.
        RecursionError: If a path exceeds the interpreter's limit.

    Complexity: O(V + E) time, O(V) space (call stack + visited set).
    """
    if not graph.has_node(start):
        raise KeyError(f"start node {start!r} not in graph")

    order: List[Any] = []
    visited: Set[Any] = set()

    def visit(node: Any) -> None:
        visited.add(node)
        order.append(node)
        for neighbor, _weight in graph.get_neighbors(node):
            if neighbor not in visited:
                visit(neighbor)

    visit(start)
    return order


def dfs_all(graph: Graph) -> List[Any]:
    """
    Depth-first traversal of EVERY node of a possibly disconnected
    graph: iterative DFS from each unvisited node in insertion order.
    O(V + E) overall.
    """
    order: List[Any] = []
    visited: Set[Any] = set()
    for start in graph.nodes():
        if start in visited:
            continue
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            neighbors = [n for n, _w in graph.get_neighbors(node)]
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append(neighbor)
    return order
