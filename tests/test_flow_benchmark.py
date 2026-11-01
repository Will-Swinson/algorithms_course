"""Test suite for the Week 10 flow benchmark pipeline, run on tiny
sizes into a temporary directory."""

import csv
import os

import pytest

from benchmarks import week10_flow_benchmark as bench


@pytest.fixture(scope="module")
def results():
    samples = {}
    limits = {"ek": 400, "mcf": 300}
    rows = (bench.bench_scaling("sparse", [200, 400], 2, limits=limits,
                                samples=samples)
            + bench.bench_scaling("dense", [40, 60], 1, samples=samples)
            + bench.bench_scaling("grid", [8, 12], 1, limits={"ek": 100},
                                  samples=samples)
            + bench.bench_memory([200, 400])
            + bench.bench_capacity_dists(n=300, instances=1)
            + bench.bench_convergence([("sparse", 400), ("grid", 10)], samples)
            + bench.bench_ablation([("sparse", 300)], 1, cap=5.0)
            + bench.bench_matching([30, 60], 1)
            + bench.bench_segmentation([12, 16], [0.1, 0.4], 1, samples)
            + bench.bench_scheduling(8, 6, [1, 3], 3))
    return rows, samples


def by(rows, experiment, **match):
    return [r for r in rows if r["experiment"] == experiment
            and all(r[k] == v for k, v in match.items())]


class TestRows:

    def test_every_flow_validated(self, results):
        rows, _ = results
        assert all(r["valid"] is True for r in rows)

    def test_runtime_rows(self, results):
        rows, _ = results
        sparse = by(rows, "runtime", family="sparse")
        assert {r["method"] for r in sparse} == set(bench.METHODS)
        assert {r["n"] for r in sparse if r["method"] == "mcf"} == {200}
        for r in sparse:
            assert r["mean_time_s"] > 0 and r["instances"] == 2

    def test_methods_agree_on_flow_value(self, results):
        rows, _ = results
        for family in ("sparse", "dense", "grid"):
            for n in {r["n"] for r in by(rows, "runtime", family=family)}:
                values = {r["flow_value"] for r in
                          by(rows, "runtime", family=family, n=n)}
                assert len(values) == 1

    def test_limits_respected(self, results):
        rows, _ = results
        grid_ek = by(rows, "runtime", family="grid", method="ek")
        assert [r["n"] for r in grid_ek] == [66]       # 8x8 + 2 only

    def test_counters_present(self, results):
        rows, _ = results
        ek = by(rows, "runtime", family="sparse", method="ek")[0]
        pr = by(rows, "runtime", family="sparse", method="pr-fifo")[0]
        assert ek["augmentations"] > 0 and pr["pushes"] > 0

    def test_memory(self, results):
        rows, _ = results
        mem = by(rows, "memory")
        assert {r["method"] for r in mem} >= {"network", "ek", "pr-fifo", "mcf"}
        assert all(r["peak_mem_mb"] > 0 for r in mem)

    def test_capacity_and_ablation(self, results):
        rows, _ = results
        assert len(by(rows, "capacity_dist")) == 12
        abl = by(rows, "ablation")
        assert [r["parameter"] for r in abl] == [a for a, _ in bench.ABLATIONS]

    def test_applications(self, results):
        rows, _ = results
        assert len(by(rows, "matching")) == 6
        noise = by(rows, "segmentation_noise")
        assert {r["method"] for r in noise} == {"graph-cut", "threshold"}
        sched = by(rows, "scheduling")
        assert sched[0]["flow_value"] <= sched[1]["flow_value"]

    def test_samples(self, results):
        _, samples = results
        assert ("aug", "sparse", "ek") in samples
        assert ("profile", "grid", "highest") in samples


class TestHelpers:

    def test_run_capped(self):
        import time
        assert bench.run_capped(lambda: 7, 1.0)[1] == 7
        elapsed, result = bench.run_capped(lambda: time.sleep(2), 0.2)
        assert elapsed is None and result is None

    def test_with_costs_keeps_structure(self):
        net, s, t = bench.build("sparse", 100, 1)
        costed = bench.with_costs(net, 1)
        assert costed.to == net.to and costed.base == net.base
        assert all(c >= 1 for c in costed.cost[::2])
        assert net.cost == [0] * len(net.cost)

    def test_to_networkx_merges_parallel(self):
        from src.flow.core import FlowNetwork
        g = bench.to_networkx(FlowNetwork.from_edges(2, [(0, 1, 2), (0, 1, 3)]))
        assert g[0][1]["capacity"] == 5


def test_plots_and_csv(results, tmp_path):
    rows, samples = results
    bench.write_csv(rows, str(tmp_path / bench.CSV_NAME))
    with open(tmp_path / bench.CSV_NAME) as fh:
        assert len(list(csv.DictReader(fh))) == len(rows)
    bench.plot_runtime(rows, str(tmp_path))
    bench.plot_memory(rows, str(tmp_path))
    bench.plot_augmentations(rows, samples, str(tmp_path))
    bench.plot_convergence(rows, samples, str(tmp_path))
    bench.plot_applications(rows, str(tmp_path))
    bench.plot_examples(str(tmp_path))
    for name in ("flow_runtime.png", "flow_memory.png", "augmentations.png",
                 "pr_convergence.png", "flow_applications.png",
                 "network_example.png", "matching_example.png",
                 "segmentation_example.png", "flow_storyboard.png",
                 "ek_animation.gif", "pr_animation.gif"):
        assert os.path.getsize(tmp_path / name) > 1_000, name
