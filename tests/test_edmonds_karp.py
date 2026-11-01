"""Test suite for Edmonds-Karp and capacity-scaling max-flow."""

import random

import networkx as nx
import pytest

from src.flow.core import capacity_scaling_max_flow, edmonds_karp
from src.flow.core.network import FlowNetwork
from src.flow.utils.generators import grid_network, sparse_network
from src.flow.utils.validators import validate_flow

CLRS = [(0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (3, 2, 9),
        (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
SOLVERS = [edmonds_karp, capacity_scaling_max_flow]


def random_edges(seed, n, m, floats=False):
    rng = random.Random(seed)
    edges = []
    for _ in range(m):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.append((u, v, rng.random() * 9 if floats else rng.randint(0, 9)))
    return edges


def nx_value(n, edges, s, t):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u, v, c in edges:
        old = graph[u][v]["capacity"] if graph.has_edge(u, v) else 0
        graph.add_edge(u, v, capacity=old + c)
    return nx.maximum_flow_value(graph, s, t)


@pytest.mark.parametrize("solver", SOLVERS)
class TestKnownNetworks:

    def test_clrs(self, solver):
        net = FlowNetwork.from_edges(6, CLRS)
        result = solver(net, 0, 5)
        assert result.value == 23
        assert result.min_cut == frozenset({0, 1, 2, 4})
        assert validate_flow(net, 0, 5, value=23, max_flow=True)

    def test_no_path(self, solver):
        net = FlowNetwork.from_edges(3, [(0, 1, 5)])
        result = solver(net, 0, 2)
        assert result.value == 0 and result.iterations == 0

    def test_single_edge_and_parallel(self, solver):
        net = FlowNetwork.from_edges(2, [(0, 1, 3), (0, 1, 4)])
        assert solver(net, 0, 1).value == 7

    def test_antiparallel(self, solver):
        net = FlowNetwork.from_edges(3, [(0, 1, 5), (1, 0, 5), (1, 2, 3)])
        assert solver(net, 0, 2).value == 3

    def test_zero_capacities(self, solver):
        net = FlowNetwork.from_edges(3, [(0, 1, 0), (1, 2, 5)])
        assert solver(net, 0, 2).value == 0

    def test_float_capacities(self, solver):
        net = FlowNetwork.from_edges(4, [(0, 1, 0.5), (0, 2, 1.25),
                                         (1, 3, 1.0), (2, 3, 0.75)])
        assert solver(net, 0, 3).value == pytest.approx(1.25)

    def test_flow_matrix(self, solver):
        net = FlowNetwork.from_edges(6, CLRS)
        m = solver(net, 0, 5).flow_matrix()
        assert m[0].sum() == 23 and m[:, 5].sum() == 23

    def test_bad_terminals(self, solver):
        net = FlowNetwork.from_edges(2, [(0, 1, 1)])
        with pytest.raises(ValueError):
            solver(net, 0, 0)
        with pytest.raises(ValueError):
            solver(net, 0, 7)

    def test_reruns_from_scratch(self, solver):
        net = FlowNetwork.from_edges(6, CLRS)
        solver(net, 0, 5)
        assert solver(net, 0, 5).value == 23


class TestEdmondsKarpInstrumentation:

    def test_counters(self):
        net = FlowNetwork.from_edges(6, CLRS)
        r = edmonds_karp(net, 0, 5)
        assert r.iterations == len(r.stats["bottlenecks"]) == 3
        assert sum(r.stats["bottlenecks"]) == 23
        assert r.stats["residual_updates"] == 2 * sum(r.stats["path_lengths"])
        assert r.stats["nodes_scanned"] > 0

    @pytest.mark.parametrize("seed", range(5))
    def test_path_lengths_never_decrease(self, seed):
        net, s, t = sparse_network(800, seed=seed)
        lengths = edmonds_karp(net, s, t).stats["path_lengths"]
        assert all(a <= b for a, b in zip(lengths, lengths[1:]))

    def test_record_trace(self):
        net = FlowNetwork.from_edges(6, CLRS)
        r = edmonds_karp(net, 0, 5, record=True)
        assert len(r.trace) == 3
        assert r.trace[-1]["value"] == 23
        assert len(r.trace[0]["flows"]) == net.num_edges
        first = r.trace[0]["path"]
        assert net.tail(first[0]) == 0 and net.to[first[-1]] == 5

    def test_max_iterations_stops_early(self):
        net = FlowNetwork.from_edges(6, CLRS)
        r = edmonds_karp(net, 0, 5, max_iterations=1)
        assert r.iterations == 1 and r.value == 12
        assert validate_flow(net, 0, 5)


class TestCapacityScaling:

    def test_phases_recorded(self):
        net = FlowNetwork.from_edges(6, CLRS)
        phases = capacity_scaling_max_flow(net, 0, 5).stats["phases"]
        assert [d for d, _ in phases] == [16, 8, 4, 2, 1]

    def test_fewer_augmentations_on_wide_capacities(self):
        net, s, t = sparse_network(2_000, capacity="exponential",
                                   cap_range=(1, 10_000), seed=4)
        ek = edmonds_karp(net, s, t)
        sc = capacity_scaling_max_flow(net, s, t)
        assert sc.value == ek.value
        assert sc.iterations < ek.iterations

    def test_empty_network(self):
        net = FlowNetwork(2)
        assert capacity_scaling_max_flow(net, 0, 1).value == 0


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("seed", range(40))
def test_matches_networkx(solver, seed):
    n = 2 + seed % 11
    edges = random_edges(seed, n, 4 * n, floats=seed % 4 == 0)
    net = FlowNetwork.from_edges(n, edges)
    result = solver(net, 0, n - 1)
    assert result.value == pytest.approx(nx_value(n, edges, 0, n - 1))
    assert validate_flow(net, 0, n - 1, value=result.value, max_flow=True)


@pytest.mark.parametrize("solver", SOLVERS)
def test_grid(solver):
    net, s, t = grid_network(12, 12, seed=1)
    result = solver(net, s, t)
    assert validate_flow(net, s, t, value=result.value, max_flow=True)
