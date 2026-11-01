"""Test suite for min-cost max-flow (successive shortest paths with
Johnson potentials)."""

import itertools
import random

import networkx as nx
import pytest

from src.flow.core import edmonds_karp, min_cost_flow, min_cost_max_flow
from src.flow.core.network import FlowNetwork
from src.flow.utils.generators import sparse_network
from src.flow.utils.validators import check_min_cost, validate_flow

# two disjoint routes: cheap one (cost 2/unit, cap 3), dear one (5/unit, cap 4)
ROUTES = [(0, 1, 3, 1), (1, 3, 3, 1), (0, 2, 4, 2), (2, 3, 4, 3)]


def nx_min_cost(n, edges, s, t):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u, v, c, w in edges:
        graph.add_edge(u, v, capacity=c, weight=w)
    flow = nx.max_flow_min_cost(graph, s, t)
    return nx.maximum_flow_value(graph, s, t), nx.cost_of_flow(graph, flow)


class TestKnown:

    def test_two_routes(self):
        net = FlowNetwork.from_edges(4, ROUTES)
        r = min_cost_max_flow(net, 0, 3)
        assert r.value == 7 and r.cost == 3 * 2 + 4 * 5
        assert r.stats["path_costs"] == [2, 5]

    def test_flow_limit_uses_cheapest_first(self):
        net = FlowNetwork.from_edges(4, ROUTES)
        r = min_cost_flow(net, 0, 3, 4)
        assert r.value == 4 and r.cost == 3 * 2 + 1 * 5
        assert r.stats["feasible"]

    def test_infeasible_amount(self):
        net = FlowNetwork.from_edges(4, ROUTES)
        r = min_cost_flow(net, 0, 3, 100)
        assert r.value == 7 and not r.stats["feasible"]

    def test_zero_amount(self):
        net = FlowNetwork.from_edges(4, ROUTES)
        r = min_cost_flow(net, 0, 3, 0)
        assert r.value == 0 and r.cost == 0

    def test_needs_cancellation(self):
        # the first cheapest path uses 1->2, the second must cancel it
        edges = [(0, 1, 1, 1), (0, 2, 1, 5), (1, 2, 1, 1), (1, 3, 1, 5),
                 (2, 3, 1, 1)]
        net = FlowNetwork.from_edges(4, edges)
        r = min_cost_max_flow(net, 0, 3)
        assert r.value == 2 and r.cost == nx_min_cost(4, edges, 0, 3)[1]
        assert net.flow_on(4) == 0

    def test_path_costs_non_decreasing(self):
        net, s, t = sparse_network(600, cost_range=(1, 30), seed=2)
        costs = min_cost_max_flow(net, s, t).stats["path_costs"]
        assert all(a <= b for a, b in zip(costs, costs[1:]))

    def test_value_matches_max_flow(self):
        net, s, t = sparse_network(800, cost_range=(1, 30), seed=3)
        assert min_cost_max_flow(net, s, t).value == edmonds_karp(net, s, t).value

    def test_negative_costs_acyclic(self):
        edges = [(0, 1, 2, -4), (1, 2, 2, 1), (0, 2, 3, 1), (2, 3, 4, -1)]
        net = FlowNetwork.from_edges(4, edges)
        r = min_cost_max_flow(net, 0, 3)
        assert r.value == 4 and r.cost == 2 * (-4 + 1 - 1) + 2 * (1 - 1)

    def test_negative_cycle_rejected(self):
        net = FlowNetwork.from_edges(4, [(0, 1, 1, 1), (1, 2, 1, -5),
                                         (2, 1, 1, 1), (2, 3, 1, 1)])
        with pytest.raises(ValueError, match="negative-cost cycle"):
            min_cost_max_flow(net, 0, 3)

    def test_negative_flow_limit(self):
        with pytest.raises(ValueError):
            min_cost_max_flow(FlowNetwork.from_edges(4, ROUTES), 0, 3, -1)

    def test_record_trace(self):
        net = FlowNetwork.from_edges(4, ROUTES)
        r = min_cost_max_flow(net, 0, 3, record=True)
        assert [step["unit_cost"] for step in r.trace] == [2, 5]
        assert r.trace[-1]["cost"] == r.cost


class TestOptimality:

    @pytest.mark.parametrize("seed", range(40))
    def test_matches_networkx(self, seed):
        rng = random.Random(seed)
        n = rng.randint(2, 10)
        pairs = {(rng.randrange(n), rng.randrange(n)) for _ in range(3 * n)}
        edges = [(u, v, rng.randint(0, 8), rng.randint(0, 9))
                 for u, v in pairs if u != v]
        net = FlowNetwork.from_edges(n, edges)
        r = min_cost_max_flow(net, 0, n - 1)
        assert (r.value, r.cost) == nx_min_cost(n, edges, 0, n - 1)

    @pytest.mark.parametrize("seed", range(10))
    def test_potentials_certify_optimality(self, seed):
        net, s, t = sparse_network(400, cost_range=(1, 20), seed=seed)
        r = min_cost_max_flow(net, s, t)
        assert check_min_cost(net, r.stats["potentials"]) == []
        assert check_min_cost(net) == []
        assert validate_flow(net, s, t, value=r.value, max_flow=True,
                             min_cost=True, potentials=r.stats["potentials"])

    def test_brute_force_small(self):
        # enumerate every integral flow on a 4-node network
        edges = [(0, 1, 2, 3), (0, 2, 2, 1), (1, 2, 1, 1), (1, 3, 2, 1),
                 (2, 3, 2, 4)]
        best = None
        for flows in itertools.product(*(range(c + 1) for _, _, c, _ in edges)):
            excess = [0] * 4
            for (u, v, _, _), f in zip(edges, flows):
                excess[u] -= f
                excess[v] += f
            if excess[1] or excess[2]:
                continue
            key = (-excess[3], sum(f * w for (_, _, _, w), f in zip(edges, flows)))
            best = key if best is None or key < best else best
        r = min_cost_max_flow(FlowNetwork.from_edges(4, edges), 0, 3)
        assert (-r.value, r.cost) == best
