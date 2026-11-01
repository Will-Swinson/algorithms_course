"""Test suite for the flow-network and application input generators."""

import numpy as np
import pytest

from src.flow.utils.generators import (
    CAPACITY_DISTRIBUTIONS, bipartite_graph, dense_network, grid_edges,
    grid_network, random_jobs, sample_capacities, sparse_network,
    synthetic_image,
)


class TestCapacities:

    @pytest.mark.parametrize("dist", CAPACITY_DISTRIBUTIONS)
    def test_range_and_type(self, dist):
        caps = sample_capacities(2_000, dist, 1, 500, np.random.default_rng(1))
        assert caps.min() >= 1 and caps.max() <= 500
        assert caps.dtype.kind == "i"

    def test_unit(self):
        assert set(sample_capacities(50, "unit").tolist()) == {1}

    def test_shapes_differ(self):
        rng = np.random.default_rng(0)
        uniform = sample_capacities(5_000, "uniform", 1, 1_000, rng)
        heavy = sample_capacities(5_000, "powerlaw", 1, 1_000, rng)
        assert np.median(heavy) < np.median(uniform) / 10

    @pytest.mark.parametrize("args", [("gaussian", 1, 2), ("uniform", 0, 5),
                                      ("uniform", 5, 2)])
    def test_rejects(self, args):
        with pytest.raises(ValueError):
            sample_capacities(3, *args)


class TestNetworks:

    def test_sparse(self):
        net, s, t = sparse_network(1_000, 4.0, seed=1)
        assert (s, t) == (0, 999) and net.n == 1_000
        inner = [e for e in net.edges() if s not in (e.u, e.v)
                 and t not in (e.u, e.v)]
        assert len(inner) == 4 * 998
        assert len({(e.u, e.v) for e in inner}) == len(inner)
        assert all(e.u != e.v for e in net.edges())
        assert len(net.adj[s]) == int(np.sqrt(1_000)) + 1

    def test_sparse_seeded(self):
        a, _, _ = sparse_network(300, seed=5)
        b, _, _ = sparse_network(300, seed=5)
        assert a.to == b.to and a.base == b.base

    def test_sparse_costs(self):
        net, _, _ = sparse_network(200, cost_range=(2, 9), seed=1)
        costs = [e.cost for e in net.edges()]
        assert min(costs) >= 2 and max(costs) <= 9

    def test_dense(self):
        net, s, t = dense_network(60, 0.5, seed=2)
        possible = 60 * 59 - 59 - 59 + 1     # nothing into s or out of t
        assert 0.4 * possible < net.num_edges < 0.6 * possible
        assert not any(e.v == s or e.u == t for e in net.edges())

    def test_grid(self):
        net, s, t = grid_network(5, 7, seed=3)
        assert net.n == 37
        assert net.num_edges == 2 * len(grid_edges(5, 7)) + 35
        assert all(len([e for e in net.adj[v] if not e & 1 and
                        net.to[e] == t]) + len([e for e in net.adj[v] if e & 1
                                                and net.to[e] == s]) == 1
                   for v in range(1, 36))

    def test_grid_edges(self):
        assert grid_edges(2, 2) == [(0, 1), (0, 2), (1, 3), (2, 3)]
        assert len(grid_edges(10, 10)) == 180

    @pytest.mark.parametrize("call", [lambda: sparse_network(2),
                                      lambda: dense_network(1),
                                      lambda: dense_network(5, 0),
                                      lambda: grid_network(0, 3)])
    def test_rejects(self, call):
        with pytest.raises(ValueError):
            call()


class TestApplicationInputs:

    def test_bipartite(self):
        edges = bipartite_graph(100, 80, 3.0, seed=1)
        assert all(0 <= a < 100 and 0 <= b < 80 for a, b in edges)
        assert len(set(edges)) == len(edges)
        assert {a for a, _ in edges} == set(range(100))
        assert bipartite_graph(0, 0) == []
        with pytest.raises(ValueError):
            bipartite_graph(3, 0)

    @pytest.mark.parametrize("shape", ["circle", "rect", "shapes"])
    def test_synthetic_image(self, shape):
        image, mask = synthetic_image(40, 50, shape, noise=0.1, seed=2)
        assert image.shape == mask.shape == (40, 50)
        assert 0 <= image.min() and image.max() <= 1
        assert 0 < mask.mean() < 1
        assert image[mask].mean() > image[~mask].mean() + 0.3

    def test_synthetic_image_bad_shape(self):
        with pytest.raises(ValueError):
            synthetic_image(10, 10, "star")

    def test_random_jobs(self):
        jobs = random_jobs(40, 12, max_length=4, seed=3)
        assert len(jobs) == 40
        for p, r, d in jobs.values():
            assert 1 <= p <= 4 and 0 <= r and r + p <= d <= 12
