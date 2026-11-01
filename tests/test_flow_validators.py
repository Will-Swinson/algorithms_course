"""Test suite for flow validation: each check must accept valid flows
and catch the violation it targets."""

import pytest

from src.flow.core import min_cost_max_flow, push_relabel
from src.flow.core.network import FlowNetwork
from src.flow.utils.validators import (
    ValidationReport, assert_valid, check_capacity, check_conservation,
    check_max_flow, check_min_cost, check_residual, validate_flow,
)

PATH = [(0, 1, 5, 1), (1, 2, 3, 1)]


def path_net():
    return FlowNetwork.from_edges(3, PATH)


class TestValidFlows:

    def test_zero_flow_is_feasible(self):
        net = path_net()
        assert check_capacity(net) == []
        assert check_conservation(net, 0, 2) == []
        assert check_residual(net) == []

    def test_max_flow_certificate(self):
        net = path_net()
        push_relabel(net, 0, 2)
        assert check_max_flow(net, 0, 2, value=3) == []
        report = validate_flow(net, 0, 2, value=3, max_flow=True)
        assert report.ok and bool(report)
        assert "max-flow certificate" in report.checks
        assert str(report).startswith("valid")

    def test_min_cost(self):
        net = FlowNetwork.from_edges(4, [(0, 1, 3, 1), (1, 3, 3, 1),
                                         (0, 2, 4, 2), (2, 3, 4, 3)])
        r = min_cost_max_flow(net, 0, 3)
        assert validate_flow(net, 0, 3, max_flow=True, min_cost=True,
                             potentials=r.stats["potentials"])
        assert check_min_cost(net) == []


class TestViolations:

    def test_capacity(self):
        net = path_net()
        net.cap[0] = -1                     # flow 6 on a 5-capacity edge
        assert check_capacity(net)
        net.cap[0] = 7                      # negative flow
        assert check_capacity(net)

    def test_conservation(self):
        net = path_net()
        net.push(0, 2)                      # 2 units stuck at node 1
        errors = check_conservation(net, 0, 2)
        assert any("node 1" in e for e in errors)
        assert not validate_flow(net, 0, 2)

    def test_residual_pair_mismatch(self):
        net = path_net()
        net.cap[1] = 4                      # twin no longer matches
        assert check_residual(net)

    def test_residual_bad_adjacency(self):
        net = path_net()
        net.adj[0].append(3)                # edge 3 starts at node 2
        assert check_residual(net)

    def test_not_maximum(self):
        net = path_net()
        net.push(0, 1)
        net.push(2, 1)
        errors = check_max_flow(net, 0, 2)
        assert any("augmenting path" in e for e in errors)

    def test_wrong_reported_value(self):
        net = path_net()
        push_relabel(net, 0, 2)
        assert check_max_flow(net, 0, 2, value=4)

    def test_not_min_cost(self):
        # one unit down the expensive route while the cheap one is idle:
        # a feasible flow, but residual cycle 0->1->3->2->0 costs
        # 1 + 1 - 9 - 9 < 0, so it is not the cheapest flow of value 1
        net = FlowNetwork.from_edges(4, [(0, 1, 1, 1), (1, 3, 1, 1),
                                         (0, 2, 1, 9), (2, 3, 1, 9)])
        for e in (4, 6):
            net.push(e, 1)
        assert validate_flow(net, 0, 3)
        assert check_min_cost(net)
        assert not validate_flow(net, 0, 3, min_cost=True)
        net.reset()
        for e in (0, 2):                    # the cheap route instead
            net.push(e, 1)
        assert check_min_cost(net) == []

    def test_bad_potentials(self):
        net = path_net()
        assert check_min_cost(net, potentials=[0, 5, 0])

    def test_assert_valid(self):
        net = path_net()
        assert_valid(net, 0, 2)
        net.push(0, 1)
        with pytest.raises(AssertionError, match="INVALID"):
            assert_valid(net, 0, 2)

    def test_error_list_truncated(self):
        net = FlowNetwork(102)
        for v in range(1, 101):
            net.add_edge(0, v, 1)
            net.push(2 * (v - 1), 1)        # every middle node keeps excess
        errors = check_conservation(net, 0, 101)
        assert len(errors) == 21 and errors[-1].startswith("...")

    def test_report_str(self):
        report = ValidationReport(errors=["x"], checks=["capacity"])
        assert not report and "INVALID" in str(report)
