"""Test suite for Floyd–Warshall all-pairs shortest paths."""

import math
import random

import pytest

from src.dp_advanced.floyd_warshall import (
    AllPairsShortestPaths, NegativeCycleError, dijkstra_all_pairs,
    floyd_warshall, floyd_warshall_matrix, reconstruct_path,
)
from src.graphs.graph import Graph, ADJACENCY_LIST, ADJACENCY_MATRIX
from src.utils.graph_generator import generate_graph
from src.utils.matrix_utils import INF, graph_to_matrix
from src.utils.timer import CallStats


def bellman_ford_all_pairs(nodes, edges):
    """Independent oracle: V-1 rounds of relaxation from every source."""
    dist = {}
    for s in nodes:
        d = {v: INF for v in nodes}
        d[s] = 0
        for _ in range(len(nodes) - 1):
            for u, v, w in edges:
                if d[u] + w < d[v]:
                    d[v] = d[u] + w
        dist[s] = d
    return dist


def directed(edges, nodes=None, rep=ADJACENCY_LIST):
    g = Graph(directed=True, representation=rep)
    for node in nodes or []:
        g.add_node(node)
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


CLRS_EDGES = [  # CLRS Figure 25.4 — has negative edges, no negative cycle
    (1, 2, 3), (1, 3, 8), (1, 5, -4), (2, 4, 1), (2, 5, 7),
    (3, 2, 4), (4, 1, 2), (4, 3, -5), (5, 4, 6),
]
CLRS_DIST = [
    [0, 1, -3, 2, -4],
    [3, 0, -4, 1, -1],
    [7, 4, 0, 5, 3],
    [2, -1, -5, 0, -2],
    [8, 5, 1, 6, 0],
]


@pytest.mark.parametrize("rep", [ADJACENCY_LIST, ADJACENCY_MATRIX])
class TestDistances:

    def test_clrs_example_with_negative_edges(self, rep):
        result = floyd_warshall(directed(CLRS_EDGES, [1, 2, 3, 4, 5], rep))
        assert result.dist == CLRS_DIST

    def test_unreachable_is_infinite(self, rep):
        g = directed([("a", "b", 2)], ["a", "b", "c"], rep)
        result = floyd_warshall(g)
        assert result.distance("a", "c") == INF
        assert result.distance("b", "a") == INF
        assert result.path("a", "c") is None

    def test_diagonal_zero(self, rep):
        result = floyd_warshall(directed(CLRS_EDGES, [1, 2, 3, 4, 5], rep))
        assert all(result.dist[i][i] == 0 for i in range(5))

    def test_indirect_route_beats_direct_edge(self, rep):
        g = directed([("a", "c", 10), ("a", "b", 2), ("b", "c", 3)],
                     rep=rep)
        result = floyd_warshall(g)
        assert result.distance("a", "c") == 5
        assert result.path("a", "c") == ["a", "b", "c"]

    def test_undirected_graph(self, rep):
        g = Graph(directed=False, representation=rep)
        g.add_edge("x", "y", 4)
        g.add_edge("y", "z", 1)
        result = floyd_warshall(g)
        assert result.distance("z", "x") == 5
        assert result.path("z", "x") == ["z", "y", "x"]

    def test_single_node(self, rep):
        g = Graph(representation=rep)
        g.add_node("solo")
        result = floyd_warshall(g)
        assert result.dist == [[0]]
        assert result.path("solo", "solo") == ["solo"]


class TestNegativeWeights:

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_bellman_ford_with_negative_edges(self, seed):
        rng = random.Random(seed)
        nodes = list(range(8))
        # DAG edges (u < v) may be negative without creating a cycle;
        # back edges stay positive and heavy enough to keep cycles >= 0
        edges = []
        for u in nodes:
            for v in nodes:
                if u < v and rng.random() < 0.4:
                    edges.append((u, v, rng.randint(-10, 10)))
                elif u > v and rng.random() < 0.2:
                    edges.append((u, v, rng.randint(80, 100)))
        g = directed(edges, nodes)
        result = floyd_warshall(g)
        oracle = bellman_ford_all_pairs(nodes, edges)
        for i in nodes:
            for j in nodes:
                assert result.dist[i][j] == oracle[i][j]

    def test_negative_cycle_raises(self):
        g = directed([("a", "b", 1), ("b", "c", -3), ("c", "a", 1)])
        with pytest.raises(NegativeCycleError):
            floyd_warshall(g)

    def test_zero_weight_cycle_is_fine(self):
        g = directed([("a", "b", 2), ("b", "a", -2)])
        assert floyd_warshall(g).distance("a", "a") == 0

    def test_negative_self_loop_is_negative_cycle(self):
        g = directed([("a", "a", -1), ("a", "b", 1)])
        with pytest.raises(NegativeCycleError):
            floyd_warshall(g)

    def test_undirected_negative_edge_is_negative_cycle(self):
        g = Graph(directed=False)
        g.add_edge("u", "v", -1)
        with pytest.raises(NegativeCycleError):
            floyd_warshall(g)

    def test_negative_cycle_error_is_value_error(self):
        assert issubclass(NegativeCycleError, ValueError)

    def test_dijkstra_rejects_what_floyd_warshall_handles(self):
        g = directed(CLRS_EDGES, [1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            dijkstra_all_pairs(g)
        assert floyd_warshall(g).dist == CLRS_DIST


class TestPaths:

    def test_every_path_weight_equals_distance(self):
        g = directed(CLRS_EDGES, [1, 2, 3, 4, 5])
        result = floyd_warshall(g)
        for u in g.nodes():
            for v in g.nodes():
                path = result.path(u, v)
                weight = sum(g.get_weight(a, b)
                             for a, b in zip(path, path[1:]))
                assert weight == result.distance(u, v)

    def test_clrs_path(self):
        result = floyd_warshall(directed(CLRS_EDGES, [1, 2, 3, 4, 5]))
        assert result.path(1, 2) == [1, 5, 4, 3, 2]

    def test_reconstruct_path_indices(self):
        _, pred = floyd_warshall_matrix([[0, 1, INF], [INF, 0, 1],
                                         [INF, INF, 0]])
        assert reconstruct_path(pred, 0, 2) == [0, 1, 2]
        assert reconstruct_path(pred, 2, 0) is None
        assert reconstruct_path(pred, 1, 1) == [1]

    def test_predecessor_matrix_initial_edges(self):
        _, pred = floyd_warshall_matrix([[0, 5], [INF, 0]])
        assert pred == [[None, 0], [None, None]]


class TestAgainstDijkstra:

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("edges_per_node", [2, 8])
    def test_same_distances_on_random_graphs(self, seed, edges_per_node):
        g = generate_graph(25, 25 * edges_per_node, directed=True,
                           weighted=True, seed=seed)
        fw = floyd_warshall(g)
        dj = dijkstra_all_pairs(g)
        for row_fw, row_dj in zip(fw.dist, dj.dist):
            for a, b in zip(row_fw, row_dj):
                assert a == b or math.isclose(a, b)

    def test_dijkstra_all_pairs_result_type(self):
        g = generate_graph(10, 20, weighted=True, seed=1)
        result = dijkstra_all_pairs(g)
        assert isinstance(result, AllPairsShortestPaths)
        assert result.path(0, 0) == [0]


class TestMatrixInterface:

    def test_input_not_modified(self):
        weights = [[0, 4, INF], [INF, 0, 1], [2, INF, 0]]
        snapshot = [row[:] for row in weights]
        floyd_warshall_matrix(weights)
        assert weights == snapshot

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            floyd_warshall_matrix([[0, 1], [1]])

    def test_relaxation_count_is_v_cubed_when_connected(self):
        n = 6
        weights = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
        stats = CallStats()
        floyd_warshall_matrix(weights, stats=stats)
        assert stats.calls == n ** 3

    def test_unreachable_rows_skipped(self):
        weights = [[0, INF], [INF, 0]]  # nothing connects
        stats = CallStats()
        floyd_warshall_matrix(weights, stats=stats)
        assert stats.calls == 2 * 2  # only i == k rows are reachable

    def test_graph_to_matrix(self):
        g = directed([("a", "b", 3)], ["a", "b"])
        nodes, matrix = graph_to_matrix(g)
        assert nodes == ["a", "b"]
        assert matrix == [[0, 3], [INF, 0]]
