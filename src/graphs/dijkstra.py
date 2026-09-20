# File: src/graphs/dijkstra.py
"""
Dijkstra's algorithm — single-source shortest paths on weighted graphs.

BFS finds fewest-EDGE paths; Dijkstra generalizes to fewest total
WEIGHT (all weights must be non-negative). The idea: grow a "settled"
region outward from the source, always settling the unsettled node
with the smallest known distance next. Greedy works here because with
non-negative weights, a node's smallest tentative distance can never
be improved by routing through a farther node.

"Pick the closest unsettled node" is a priority-queue job — this
implementation uses the Week 3 MinHeap, giving O((V + E) log V), the
assignment's O(E log V) for connected graphs. A list-based variant
(linear scan for the minimum, O(V²)) is included for the benchmark
comparison the assignment asks for.

Stale-entry handling: rather than decrease-key (which binary heaps
don't support directly), a node is re-pushed whenever its distance
improves, and outdated pops are skipped ("lazy deletion").
"""
from typing import Any, Dict, List, Optional, Tuple

from src.graphs.graph import Graph
from src.structures.heap import MinHeap

INFINITY = float("inf")


def dijkstra(graph: Graph, source: Any) -> Tuple[Dict[Any, float],
                                                 Dict[Any, Optional[Any]]]:
    """
    Compute shortest-path distances from `source` to every node, using
    a min-heap priority queue.

    Args:
        graph: Weighted graph (directed or undirected). Weights must
            be non-negative.
        source: Start node.

    Returns:
        (distances, predecessors):
        - distances: {node: total weight of the shortest path from
          source}. Unreachable nodes map to float("inf") — this is the
          graceful handling of disconnected graphs.
        - predecessors: {node: previous node on its shortest path}.
          The source and unreachable nodes map to None. Feed this to
          reconstruct_path() to recover an actual route.

    Raises:
        KeyError: If `source` is not in the graph.
        ValueError: If a negative edge weight is encountered.

    Complexity: O((V + E) log V) time — every node is popped once with
        its final distance, and each edge relaxation costs one O(log V)
        heap push. O(V) space (plus stale heap entries, at most E).
    """
    if not graph.has_node(source):
        raise KeyError(f"source node {source!r} not in graph")

    distances: Dict[Any, float] = {node: INFINITY for node in graph.nodes()}
    predecessors: Dict[Any, Optional[Any]] = {node: None
                                              for node in graph.nodes()}
    distances[source] = 0

    heap = MinHeap()
    counter = 0                    # tie-breaker: nodes never compared
    heap.insert((0, counter, source))
    settled = set()

    while not heap.is_empty():
        dist, _, node = heap.extract_min()
        if node in settled:
            continue               # stale entry: node improved earlier
        settled.add(node)

        for neighbor, weight in graph.get_neighbors(node):
            if weight < 0:
                raise ValueError(
                    f"negative weight {weight} on edge "
                    f"{node!r} -> {neighbor!r}: Dijkstra requires "
                    f"non-negative weights"
                )
            candidate = dist + weight
            if candidate < distances[neighbor]:     # relaxation
                distances[neighbor] = candidate
                predecessors[neighbor] = node
                counter += 1
                heap.insert((candidate, counter, neighbor))

    return distances, predecessors


def dijkstra_list_based(graph: Graph, source: Any
                        ) -> Tuple[Dict[Any, float],
                                   Dict[Any, Optional[Any]]]:
    """
    Dijkstra with a plain-list "priority queue": each round scans all
    unsettled nodes for the minimum distance — O(V²) total.

    Same inputs, outputs, and error behavior as dijkstra(). Exists for
    the Week 4 benchmark: on sparse graphs the heap version wins by a
    growing margin; on very dense graphs (E ~ V²) the two converge,
    because O(V²) == O(E) there and the list version pays no log
    factor.
    """
    if not graph.has_node(source):
        raise KeyError(f"source node {source!r} not in graph")

    distances: Dict[Any, float] = {node: INFINITY for node in graph.nodes()}
    predecessors: Dict[Any, Optional[Any]] = {node: None
                                              for node in graph.nodes()}
    distances[source] = 0
    unsettled = set(graph.nodes())

    while unsettled:
        # Linear scan replaces the heap's extract_min
        node = min(unsettled, key=lambda n: distances[n])
        if distances[node] == INFINITY:
            break                  # remaining nodes are unreachable
        unsettled.remove(node)

        for neighbor, weight in graph.get_neighbors(node):
            if weight < 0:
                raise ValueError(
                    f"negative weight {weight} on edge "
                    f"{node!r} -> {neighbor!r}: Dijkstra requires "
                    f"non-negative weights"
                )
            candidate = distances[node] + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = node

    return distances, predecessors


def reconstruct_path(predecessors: Dict[Any, Optional[Any]], source: Any,
                     target: Any) -> Optional[List[Any]]:
    """
    Rebuild the shortest path source→target by walking the predecessor
    map backwards from the target.

    Returns:
        [source, ..., target], or None if the target is unreachable
        (it has no predecessor chain leading back to the source).
    """
    if target == source:
        return [source]
    if predecessors.get(target) is None:
        return None                # unreachable (or unknown) target

    path = [target]
    node = predecessors[target]
    while node is not None:
        path.append(node)
        node = predecessors[node]
    path.reverse()
    return path if path[0] == source else None
