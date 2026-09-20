# File: src/graphs/bfs.py
"""
Breadth-First Search — explore a graph level by level.

BFS visits the start node, then everything 1 edge away, then
everything 2 edges away, and so on. The engine behind that order is a
FIFO QUEUE: newly discovered nodes go to the BACK, the next node to
process comes from the FRONT — so nearer nodes are always finished
before farther ones. That property makes BFS find shortest paths (by
edge count) in unweighted graphs.

Every node is enqueued at most once (guarded by the visited set) and
every edge is examined at most once (twice when undirected), giving
O(V + E) time and O(V) space for the queue + visited set.
"""
from collections import deque
from typing import Any, Dict, List, Optional

from src.graphs.graph import Graph


def bfs(graph: Graph, start: Any) -> List[Any]:
    """
    Traverse the component containing `start`, returning nodes in the
    order they were visited (level by level).

    Nodes in other components are NOT visited — use bfs_all() to cover
    a disconnected graph.

    Args:
        graph: The graph to traverse (directed or undirected).
        start: Node to begin from.

    Returns:
        List of nodes in BFS visit order (starts with `start`).

    Raises:
        KeyError: If `start` is not in the graph.

    Complexity: O(V + E) time, O(V) space.
    """
    if not graph.has_node(start):
        raise KeyError(f"start node {start!r} not in graph")

    order: List[Any] = []
    visited = {start}          # marked when ENQUEUED, so no duplicates
    queue = deque([start])
    while queue:
        node = queue.popleft()             # FIFO: oldest discovery first
        order.append(node)
        for neighbor, _weight in graph.get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def bfs_all(graph: Graph) -> List[Any]:
    """
    Traverse EVERY node of a possibly disconnected graph: run BFS from
    each not-yet-visited node in insertion order, concatenating the
    component orders. Still O(V + E) overall.
    """
    order: List[Any] = []
    visited = set()
    for start in graph.nodes():
        if start in visited:
            continue
        visited.add(start)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _weight in graph.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return order


def bfs_shortest_paths(graph: Graph, start: Any) -> Dict[Any, Optional[Any]]:
    """
    BFS variant recording each node's predecessor on a fewest-edges
    path from `start` — the unweighted analogue of Dijkstra's output.

    Returns:
        {node: predecessor} for every reachable node (start maps to
        None). Unreachable nodes are absent.

    Raises:
        KeyError: If `start` is not in the graph.
    """
    if not graph.has_node(start):
        raise KeyError(f"start node {start!r} not in graph")

    predecessors: Dict[Any, Optional[Any]] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor, _weight in graph.get_neighbors(node):
            if neighbor not in predecessors:
                predecessors[neighbor] = node
                queue.append(neighbor)
    return predecessors
