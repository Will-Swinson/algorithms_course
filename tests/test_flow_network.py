"""Test suite for the FlowNetwork residual-graph data structure."""

import numpy as np
import pytest

from src.flow.core.network import FlowNetwork, flow_matrix_from

CLRS = [(0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (3, 2, 9),
        (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]


class TestConstruction:

    def test_add_edge_creates_twin(self):
        net = FlowNetwork(2)
        e = net.add_edge(0, 1, 5, cost=3)
        assert e == 0 and net.num_edges == 1
        assert net.to[e] == 1 and net.to[e ^ 1] == 0
        assert net.cap[e] == 5 and net.cap[e ^ 1] == 0
        assert net.cost[e] == 3 and net.cost[e ^ 1] == -3
        assert net.adj[0] == [0] and net.adj[1] == [1]
        assert net.tail(e) == 0 and net.tail(e ^ 1) == 1

    def test_add_nodes(self):
        net = FlowNetwork()
        assert net.add_node() == 0
        assert list(net.add_nodes(3)) == [1, 2, 3]
        assert net.n == 4

    def test_parallel_and_antiparallel_edges(self):
        net = FlowNetwork.from_edges(2, [(0, 1, 2), (0, 1, 3), (1, 0, 4)])
        assert net.num_edges == 3
        assert len(net.adj[0]) == 3 and len(net.adj[1]) == 3

    @pytest.mark.parametrize("edge", [(0, 0, 1), (0, 5, 1), (-1, 1, 1),
                                      (0, 1, -1), (0, 1, float("inf")),
                                      (0, 1, float("nan"))])
    def test_rejects_bad_edges(self, edge):
        with pytest.raises(ValueError):
            FlowNetwork(3).add_edge(*edge)

    def test_rejects_bad_cost(self):
        with pytest.raises(ValueError):
            FlowNetwork(2).add_edge(0, 1, 1, float("inf"))

    def test_negative_size(self):
        with pytest.raises(ValueError):
            FlowNetwork(-1)


class TestResidualOperations:

    def test_push_and_cancel(self):
        net = FlowNetwork.from_edges(2, [(0, 1, 5)])
        net.push(0, 3)
        assert net.flow_on(0) == 3 and net.cap[1] == 3
        net.push(1, 2)                     # cancel two units
        assert net.flow_on(0) == 1

    def test_push_over_capacity(self):
        net = FlowNetwork.from_edges(2, [(0, 1, 5)])
        with pytest.raises(ValueError):
            net.push(0, 6)
        with pytest.raises(ValueError):
            net.push(0, -1)

    def test_flow_on_requires_forward(self):
        net = FlowNetwork.from_edges(2, [(0, 1, 5)])
        with pytest.raises(ValueError):
            net.flow_on(1)

    def test_reset(self):
        net = FlowNetwork.from_edges(2, [(0, 1, 5)])
        net.push(0, 5)
        net.reset()
        assert net.cap == [5, 0]

    def test_residual_graph(self):
        net = FlowNetwork.from_edges(3, [(0, 1, 4), (1, 2, 2)])
        net.push(0, 2)
        net.push(2, 2)
        residual = net.get_residual_graph()
        assert [(a.v, a.capacity, a.forward) for a in residual[0]] == [(1, 2, True)]
        assert sorted((a.v, a.capacity, a.forward) for a in residual[1]) == \
            [(0, 2, False)]
        assert [(a.v, a.forward) for a in residual[2]] == [(1, False)]

    def test_copy_is_independent(self):
        net = FlowNetwork.from_edges(2, [(0, 1, 5)])
        other = net.copy()
        other.push(0, 5)
        other.add_edge(1, 0, 1)
        assert net.cap == [5, 0] and net.num_edges == 1


class TestInspection:

    @pytest.fixture
    def flowed(self):
        net = FlowNetwork.from_edges(4, [(0, 1, 3, 1), (0, 2, 2, 2),
                                         (1, 3, 2, 1), (2, 3, 3, 1),
                                         (1, 2, 1, 5)])
        for e, f in [(0, 3), (2, 2), (8, 1), (4, 2), (6, 3)]:
            net.push(e, f)
        return net

    def test_excess_and_value(self, flowed):
        assert flowed.excess() == [-5, 0, 0, 5]
        assert flowed.flow_value(0) == 5

    def test_flow_dict_and_matrix(self, flowed):
        assert flowed.flow_dict() == {(0, 1): 3, (0, 2): 2, (1, 3): 2,
                                      (2, 3): 3, (1, 2): 1}
        m = flowed.flow_matrix()
        assert m.shape == (4, 4) and m[0, 1] == 3 and m.sum() == 11

    def test_total_cost(self, flowed):
        assert flowed.total_cost() == 3 + 4 + 2 + 3 + 5

    def test_min_cut(self, flowed):
        side = flowed.min_cut(0)
        assert side == frozenset({0})
        assert flowed.cut_capacity(side) == 5
        assert len(flowed.cut_edges(side)) == 2

    def test_edges_iterator(self, flowed):
        edges = list(flowed.edges())
        assert len(edges) == 5
        assert (edges[0].u, edges[0].v, edges[0].capacity, edges[0].flow) == \
            (0, 1, 3, 3)

    def test_flow_matrix_sums_parallel(self):
        m = flow_matrix_from({(0, 1): 2.0}, 2)
        assert np.array_equal(m, [[0, 2], [0, 0]])

    def test_validate_delegates(self, flowed):
        assert flowed.validate(0, 3)
        assert flowed.validate(0, 3, max_flow=True)

    def test_repr(self):
        assert repr(FlowNetwork.from_edges(6, CLRS)) == "FlowNetwork(n=6, edges=9)"
