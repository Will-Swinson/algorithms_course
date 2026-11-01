"""Test suite for push-relabel max-flow: FIFO and highest-label
variants, the gap and global-relabel heuristics, counters, traces."""

import random

import networkx as nx
import pytest

from src.flow.core import edmonds_karp, push_relabel
from src.flow.core.network import FlowNetwork
from src.flow.utils.generators import dense_network, grid_network, sparse_network
from src.flow.utils.validators import validate_flow

CLRS = [(0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (3, 2, 9),
        (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
CONFIGS = [dict(variant=v, gap=g, global_relabel=r)
           for v in ("fifo", "highest") for g in (True, False)
           for r in (True, False)]


def config_id(c):
    return f"{c['variant']}-gap{int(c['gap'])}-global{int(c['global_relabel'])}"


def nx_value(n, edges, s, t):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u, v, c in edges:
        old = graph[u][v]["capacity"] if graph.has_edge(u, v) else 0
        graph.add_edge(u, v, capacity=old + c)
    return nx.maximum_flow_value(graph, s, t)


@pytest.mark.parametrize("config", CONFIGS, ids=config_id)
class TestAllConfigurations:

    def test_clrs(self, config):
        net = FlowNetwork.from_edges(6, CLRS)
        result = push_relabel(net, 0, 5, **config)
        assert result.value == 23
        assert result.min_cut == frozenset({0, 1, 2, 4})
        assert validate_flow(net, 0, 5, value=23, max_flow=True)

    def test_result_is_a_flow_not_a_preflow(self, config):
        # node 1 receives 10 but can forward only 1: 9 must return to s
        net = FlowNetwork.from_edges(3, [(0, 1, 10), (1, 2, 1)])
        result = push_relabel(net, 0, 2, **config)
        assert result.value == 1
        assert net.excess() == [-1, 0, 1]

    @pytest.mark.parametrize("seed", range(25))
    def test_matches_networkx(self, config, seed):
        rng = random.Random(seed)
        n = rng.randint(2, 13)
        edges = [(rng.randrange(n), rng.randrange(n),
                  rng.random() * 5 if seed % 5 == 0 else rng.randint(0, 9))
                 for _ in range(rng.randint(0, 45))]
        edges = [e for e in edges if e[0] != e[1]]
        net = FlowNetwork.from_edges(n, edges)
        result = push_relabel(net, 0, n - 1, **config)
        assert result.value == pytest.approx(nx_value(n, edges, 0, n - 1))
        assert validate_flow(net, 0, n - 1, value=result.value, max_flow=True)

    def test_disconnected(self, config):
        net = FlowNetwork.from_edges(4, [(0, 1, 3), (2, 3, 3)])
        assert push_relabel(net, 0, 3, **config).value == 0


class TestLargeNetworks:

    @pytest.mark.parametrize("variant", ["fifo", "highest"])
    def test_sparse_10k_matches_edmonds_karp(self, variant):
        net, s, t = sparse_network(10_000, seed=3)
        ek = edmonds_karp(net, s, t).value
        result = push_relabel(net, s, t, variant=variant)
        assert result.value == ek
        assert validate_flow(net, s, t, value=ek, max_flow=True)

    def test_sparse_50k(self):
        net, s, t = sparse_network(50_000, seed=1)
        result = push_relabel(net, s, t)
        assert validate_flow(net, s, t, value=result.value, max_flow=True)

    @pytest.mark.parametrize("variant", ["fifo", "highest"])
    def test_grid_and_dense(self, variant):
        for net, s, t in (grid_network(40, 40, seed=2),
                          dense_network(150, 0.3, seed=2)):
            result = push_relabel(net, s, t, variant=variant)
            assert validate_flow(net, s, t, value=result.value, max_flow=True)


class TestInstrumentation:

    def test_counters(self):
        net, s, t = sparse_network(500, seed=5)
        st = push_relabel(net, s, t).stats
        assert st["pushes"] >= st["saturating_pushes"] > 0
        assert st["relabels"] > 0 and st["discharges"] > 0
        assert st["global_relabels"] >= 0 and st["gap_nodes"] >= 0

    def test_iterations_are_discharges(self):
        net = FlowNetwork.from_edges(6, CLRS)
        r = push_relabel(net, 0, 5)
        assert r.iterations == r.stats["discharges"]
        assert r.algorithm == "push_relabel_fifo"

    def test_profile(self):
        net, s, t = sparse_network(1_000, seed=6)
        r = push_relabel(net, s, t, profile_every=10)
        prof = r.stats["profile"]
        assert len(prof) > 2
        ops = [p[0] for p in prof]
        assert ops == sorted(ops)
        assert prof[-1][1] == r.value and prof[-1][2] == 0

    def test_record_trace(self):
        net = FlowNetwork.from_edges(6, CLRS)
        r = push_relabel(net, 0, 5, global_relabel=False, record=True)
        ops = [step["op"] for step in r.trace]
        assert ops[0] == "init" and "push" in ops and "relabel" in ops
        assert r.trace[0]["heights"][0] == 6
        assert r.trace[-1]["value"] == 23
        for step in r.trace:
            assert all(h <= 2 * 6 for h in step["heights"])

    def test_gap_heuristic_lifts_nodes(self):
        net, s, t = sparse_network(3_000, seed=7)
        r = push_relabel(net, s, t, global_relabel=False)
        assert r.stats["gap_nodes"] > 0

    def test_global_relabel_frequency(self):
        net, s, t = grid_network(30, 30, seed=3)
        often = push_relabel(net, s, t, global_relabel_freq=0.05)
        rare = push_relabel(net, s, t, global_relabel_freq=5.0)
        assert often.value == rare.value
        assert often.stats["global_relabels"] > rare.stats["global_relabels"]

    def test_bad_variant(self):
        with pytest.raises(ValueError):
            push_relabel(FlowNetwork.from_edges(2, [(0, 1, 1)]), 0, 1,
                         variant="lifo")
