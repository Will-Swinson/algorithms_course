"""Test suite for the Week 6 benchmark pipeline (run on tiny sizes),
its plotting helpers, and the matrix/graph utilities it relies on."""

import csv
import math
import os

import pytest

from benchmarks import week6_dp_advanced_benchmark as bench
from src.utils.matrix_utils import (
    INF, euclidean_distance_matrix, format_matrix, random_points,
)
from src.utils.visualization import plot_metric_panels, plot_table_heatmap


class TestKnapsackBenchmark:

    def test_all_methods_measured_for_both_sweeps(self):
        rows = bench.bench_knapsack(n_sizes=[10, 20], fixed_w=50,
                                    w_sizes=[30, 60], fixed_n=10)
        methods = {m for m, _ in bench.KNAPSACK_METHODS}
        for experiment in ["knapsack_n", "knapsack_w"]:
            got = {r["method"] for r in rows if r["experiment"] == experiment}
            assert got == methods

    def test_capacity_sweep_keyed_by_w(self):
        rows = bench.bench_knapsack(n_sizes=[10], fixed_w=50,
                                    w_sizes=[30, 60], fixed_n=10)
        sizes = {r["input_size"] for r in rows
                 if r["experiment"] == "knapsack_w"}
        assert sizes == {30, 60}

    def test_cell_counts(self):
        rows = bench.bench_knapsack(n_sizes=[10], fixed_w=50, w_sizes=[],
                                    fixed_n=10)
        standard = next(r for r in rows if r["method"] == "standard_2d")
        assert standard["work"] == 10 * 51

    def test_1d_uses_less_memory(self):
        rows = bench.bench_knapsack(n_sizes=[100], fixed_w=500, w_sizes=[],
                                    fixed_n=10)
        mem = {r["method"]: r["peak_memory_bytes"] for r in rows}
        assert mem["space_optimized_1d"] * 10 < mem["standard_2d"]
        assert mem["1d_with_items"] * 5 < mem["2d_with_items"]

    def test_make_items_seeded(self):
        assert bench.make_items(20) == bench.make_items(20)


class TestOtherExperiments:

    def test_mcm_rows_and_costs(self):
        rows = bench.bench_mcm(recursive_sizes=[4], dp_sizes=[4, 8])
        methods = {(r["method"], r["input_size"]) for r in rows
                   if r["experiment"] == "mcm"}
        assert methods == {("recursive", 4), ("memoization", 4),
                           ("bottom_up", 4), ("memoization", 8),
                           ("bottom_up", 8)}
        costs = {(r["method"], r["input_size"]): r["work"] for r in rows
                 if r["experiment"] == "mcm_cost"}
        for n in [4, 8]:
            assert costs[("optimal", n)] <= costs[("left_to_right", n)]

    def test_floyd_warshall_rows(self):
        rows = bench.bench_floyd_warshall([10, 20])
        assert {r["experiment"] for r in rows} == {"fw_sparse", "fw_dense"}
        fw = next(r for r in rows if r["method"] == "floyd_warshall"
                  and r["experiment"] == "fw_dense" and r["input_size"] == 20)
        assert fw["work"] <= 20 ** 3

    def test_make_graph_densities(self):
        assert bench.make_graph(20, "sparse").num_edges() == 80
        assert bench.make_graph(20, "dense").num_edges() == 20 * 19 // 4

    def test_tsp_rows(self):
        rows = bench.bench_tsp(brute_sizes=[4, 5], bitmask_sizes=[4, 5, 6])
        brute = [r for r in rows if r["method"] == "brute_force"]
        assert [r["work"] for r in brute] == [6, 24]  # (n-1)!
        assert len([r for r in rows if r["method"] == "bitmask"]) == 3


class TestOutput:

    @pytest.fixture
    def rows(self):
        rows = (bench.bench_knapsack(n_sizes=[10, 20], fixed_w=40,
                                     w_sizes=[20, 40], fixed_n=10)
                + bench.bench_mcm(recursive_sizes=[3, 4], dp_sizes=[3, 4, 6])
                + bench.bench_floyd_warshall([8, 12])
                + bench.bench_tsp(brute_sizes=[4, 5], bitmask_sizes=[4, 5]))
        bench.add_speedups(rows)
        return rows

    def test_speedups_use_baselines(self, rows):
        measured = {(r["experiment"], r["method"], r["input_size"],
                     r["parameter"]) for r in rows}
        for r in rows:
            base = bench.BASELINES.get(r["method"])
            if base and (r["experiment"], base, r["input_size"],
                         r["parameter"]) in measured:
                assert r["speedup_vs_baseline"] is not None, r
            elif base:
                assert r["speedup_vs_baseline"] is None  # no baseline run
            if r["method"] in {"standard_2d", "recursive", "brute_force",
                               "dijkstra_all_pairs"}:
                assert r["speedup_vs_baseline"] is None

    def test_csv(self, rows, tmp_path):
        path = str(tmp_path / "out.csv")
        bench.write_csv(rows, path)
        with open(path) as f:
            read = list(csv.DictReader(f))
        assert len(read) == len(rows)
        assert list(read[0].keys()) == bench.FIELDS

    def test_summary_prints(self, rows, capsys):
        bench.print_summary(rows)
        out = capsys.readouterr().out
        for experiment in ["knapsack_n", "knapsack_w", "mcm", "mcm_cost",
                           "fw_sparse", "fw_dense", "tsp"]:
            assert f"--- {experiment}" in out

    def test_make_plots_writes_all_pngs(self, rows, tmp_path):
        bench.make_plots(rows, str(tmp_path))
        for name in ["knapsack_space_comparison.png", "mcm_performance.png",
                     "mcm_dp_table.png", "floyd_warshall_scaling.png",
                     "tsp_bitmask_runtime.png"]:
            assert os.path.getsize(tmp_path / name) > 0


class TestPlotHelpers:

    def test_metric_panels_skip_missing_points(self, tmp_path):
        out = str(tmp_path / "panels.png")
        plot_metric_panels([
            {"title": "a", "xlabel": "x", "ylabel": "y",
             "series": {"s": ([1, 2, 3], [1.0, None, 0.0])},
             "refs": {"r": ([1, 3], [1, 9])}, "logx": True, "logy": True},
            {"title": "b", "xlabel": "x", "ylabel": "y",
             "series": {"s": ([1, 2], [3, 4])}},
            {"title": "c", "xlabel": "x", "ylabel": "y", "series": {}},
        ], out, "test")
        assert os.path.getsize(out) > 0

    def test_heatmap_with_blank_cells(self, tmp_path):
        out = str(tmp_path / "heat.png")
        plot_table_heatmap([[0, 5], [None, 0]], out, "t",
                           row_labels=["r1", "r2"], col_labels=["c1", "c2"])
        assert os.path.getsize(out) > 0


class TestMatrixUtils:

    def test_euclidean_matrix_symmetric_zero_diagonal(self):
        d = euclidean_distance_matrix(random_points(6, seed=1))
        for i in range(6):
            assert d[i][i] == 0
            for j in range(6):
                assert d[i][j] == d[j][i]

    def test_euclidean_known_distance(self):
        d = euclidean_distance_matrix([(0, 0), (3, 4)])
        assert math.isclose(d[0][1], 5.0)

    def test_format_matrix(self):
        text = format_matrix([[0, INF], [None, 2]], labels=["a", "b"],
                             width=4)
        lines = text.splitlines()
        assert len(lines) == 3
        assert "∞" in lines[1] and "·" in lines[2]
