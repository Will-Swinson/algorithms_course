"""Test suite for the Week 9 benchmark pipeline, run on tiny sizes.

(Named test_approx_benchmark.py because Week 6 already has
tests/test_benchmark_comparison.py.)"""

import csv
import os

import pytest

from benchmarks import week9_approx_benchmark as bench


@pytest.fixture(scope="module")
def results():
    samples = {}
    rows = (bench.bench_vertex_cover(sizes=[10, 20], instances=2,
                                     fast_sizes=[100], lp_sizes=[20],
                                     bipartite=[2, 3], samples=samples)
            + bench.bench_set_cover(sizes=[20, 30], instances=2, trials=4,
                                    large_sizes=[60], rounds_sweep=[1, 3],
                                    rounds_n=30, bad_k=[2, 3],
                                    samples=samples)
            + bench.bench_facility(sizes=[(4, 10), (5, 12)], instances=2,
                                   fast_sizes=[(5, 20)], samples=samples)
            + bench.bench_tsp(exact_sizes=[5, 6], instances=2,
                              large_sizes=[20, 30], large_instances=1,
                              christofides_max=20, tour_n=6,
                              samples=samples)
            + bench.bench_maxcut(sizes=[6, 8], instances=2, trials=50,
                                 ls_trials=5, large_sizes=[50], hist_n=8,
                                 samples=samples))
    return rows, samples


def by(rows, experiment, method=None):
    return [r for r in rows if r["experiment"] == experiment
            and (method is None or r["method"] == method)]


class TestExperiments:

    def test_vertex_cover(self, results):
        rows, _ = results
        for r in by(rows, "vc", "greedy_matching"):
            assert 1 <= r["worst_ratio"] <= 2
        assert all(r["mean_ratio"] == 2 for r in by(rows, "vc_tight"))
        assert all(r["mean_ratio"] == pytest.approx(2) for r in
                   by(rows, "vc_certified", "greedy_matching"))

    def test_set_cover(self, results):
        rows, samples = results
        for r in by(rows, "sc", "deterministic_rounding"):
            assert r["worst_ratio"] <= r["guarantee"] + 1e-9
        gaps = by(rows, "sc", "lp_relaxation")
        assert all(r["worst_ratio"] <= 1 + 1e-9 for r in gaps)
        assert by(rows, "sc_large")[0]["ratio_basis"] == "LP"
        assert [r["mean_ratio"] for r in by(rows, "sc_bad", "greedy")] == \
            [1.0, 1.5]
        assert len(samples["sc_repairs"]) == 2

    def test_facility(self, results):
        rows, _ = results
        for r in by(rows, "fl", "greedy"):
            assert 1 - 1e-9 <= r["worst_ratio"] <= 1.861

    def test_tsp(self, results):
        rows, samples = results
        for r in by(rows, "tsp", "christofides"):
            assert r["worst_ratio"] <= 1.5 + 1e-9
        assert not by(rows, "tsp_large", "christofides") or max(
            r["input_size"] for r in by(rows, "tsp_large", "christofides")
        ) <= 20
        assert len(samples["tours"]["tours"]) == 3

    def test_maxcut(self, results):
        rows, samples = results
        for r in by(rows, "mc", "derandomized"):
            assert r["worst_ratio"] >= 0.5 - 1e-9
        assert len(samples["mc_theory"]) == 2 * 2 * 2
        assert samples["mc_hist"]["opt"] >= samples["mc_hist"]["W"] / 2


class TestOutput:

    def test_csv(self, results, tmp_path):
        rows, _ = results
        path = str(tmp_path / "table.csv")
        bench.write_csv(rows, path)
        with open(path) as f:
            read = list(csv.DictReader(f))
        assert len(read) == len(rows)
        assert list(read[0].keys()) == bench.FIELDS

    def test_summary(self, results, capsys):
        rows, _ = results
        bench.print_summary(rows)
        assert "christofides" in capsys.readouterr().out

    def test_plots(self, results, tmp_path):
        rows, samples = results
        bench.make_plots(rows, samples, str(tmp_path))
        for name in ["approx_ratios.png", "cost_vs_optimal.png",
                     "randomized_maxcut_results.png",
                     "set_cover_rounding.png", "approx_runtime.png",
                     "tsp_tours.png"]:
            assert os.path.getsize(tmp_path / name) > 0
