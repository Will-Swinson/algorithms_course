# File: src/utils/graph_generator.py
"""
Seeded random graph generators for testing and benchmarking.

Density vocabulary used throughout Week 4 (V nodes):
    sparse   E ≈ 2V        (average degree ~4 — road networks, meshes)
    dense    E ≈ 0.5·V²    (half of all possible edges — near-cliques)
    random   E from an edge probability p you choose

`connected=True` first links all nodes with a random spanning tree
(V − 1 edges), guaranteeing one component, then sprinkles the
remaining edges randomly — so traversals from any start node reach
everything, which keeps benchmark comparisons fair.
"""
import random
from typing import Optional, Tuple

from src.graphs.graph import Graph, ADJACENCY_LIST


def generate_graph(num_nodes: int, num_edges: int, directed: bool = False,
                   weighted: bool = False,
                   weight_range: Tuple[float, float] = (1, 100),
                   connected: bool = True,
                   representation: str = ADJACENCY_LIST,
                   seed: Optional[int] = None) -> Graph:
    """
    Generate a random graph with exactly `num_nodes` nodes (labeled
    0..n-1) and approximately `num_edges` distinct edges.

    Args:
        num_nodes: Node count (>= 1).
        num_edges: Target edge count; clamped to the maximum possible
            for the node count and directedness.
        directed: Build a directed graph.
        weighted: Assign uniform random weights from weight_range;
            unweighted graphs get weight 1 on every edge.
        weight_range: (low, high) for random weights.
        connected: Guarantee a single component via a random spanning
            tree before adding random edges (undirected meaning; for
            directed graphs the tree edges point away from the root,
            guaranteeing reachability from node 0).
        representation: "list" or "matrix" storage.
        seed: RNG seed for reproducibility.

    Returns:
        The generated Graph.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")

    rng = random.Random(seed)
    graph = Graph(directed=directed, representation=representation)
    for node in range(num_nodes):
        graph.add_node(node)

    max_edges = num_nodes * (num_nodes - 1)
    if not directed:
        max_edges //= 2
    num_edges = min(num_edges, max_edges)

    def pick_weight() -> float:
        if not weighted:
            return 1
        return round(rng.uniform(*weight_range), 3)

    edges_added = 0
    if connected and num_nodes > 1:
        # Random spanning tree: attach each node (in shuffled order)
        # to a uniformly chosen earlier node — V-1 edges, one component.
        # Node 0 is pinned as the root so that in DIRECTED graphs,
        # where tree edges point parent -> child, every node is
        # reachable from node 0 (benchmarks start traversals there).
        order = [0] + rng.sample(range(1, num_nodes), num_nodes - 1)
        for i in range(1, num_nodes):
            parent = order[rng.randrange(i)]
            graph.add_edge(parent, order[i], pick_weight())
            edges_added += 1

    # Sprinkle the remaining edges uniformly at random
    attempts_left = max(num_edges * 20, 1000)   # guard against stall
    while edges_added < num_edges and attempts_left > 0:
        attempts_left -= 1
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v or graph.has_edge(u, v):
            continue
        graph.add_edge(u, v, pick_weight())
        edges_added += 1

    return graph


def generate_sparse_graph(num_nodes: int, **kwargs) -> Graph:
    """Sparse graph: E ≈ 2V (average degree ~4)."""
    return generate_graph(num_nodes, num_edges=2 * num_nodes, **kwargs)


def generate_dense_graph(num_nodes: int, **kwargs) -> Graph:
    """Dense graph: E ≈ half of all possible edges."""
    directed = kwargs.get("directed", False)
    max_edges = num_nodes * (num_nodes - 1)
    if not directed:
        max_edges //= 2
    return generate_graph(num_nodes, num_edges=max_edges // 2, **kwargs)


def generate_random_graph(num_nodes: int, edge_probability: float = 0.1,
                          **kwargs) -> Graph:
    """Erdős–Rényi-style graph: each possible edge exists with
    probability `edge_probability` (approximated by edge count)."""
    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be in [0, 1]")
    directed = kwargs.get("directed", False)
    max_edges = num_nodes * (num_nodes - 1)
    if not directed:
        max_edges //= 2
    return generate_graph(num_nodes,
                          num_edges=round(max_edges * edge_probability),
                          **kwargs)
