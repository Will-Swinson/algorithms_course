"""Test suite for the Week 8 benchmark pipeline, run on tiny sizes."""

import csv
import os

import pytest

from benchmarks import week8_complexity_benchmark as bench
from src.utils.visualization import draw_highlighted_graph


@pytest.fixture(scope="module")
def rows():
    return (bench.bench_subset_sum(hard_sizes=[4, 8, 12], instances=2,
                                   dp_sizes=[10, 30], greedy_sizes=[100],
                                   ratio_sizes=[5, 20], ratio_instances=4)
            + bench.bench_sat(bf_sizes=[4, 8], plain_sizes=[5, 10],
                              dpll_sizes=[10, 20], instances=4,
                              small_instances=2, two_sat_sizes=[100],
                              budgets=[10, 1_000], budget_n=20,
                              budget_seeds=1)
            + bench.bench_vertex_cover(exact_sizes=[20, 40], instances=2,
                                       fast_sizes=[100], bipartite=[2, 4])
            + bench.bench_tsp(exact_sizes=[5, 7], instances=2, brute_max=6,
                              large_sizes=[20, 30], large_instances=2,
                              pass_n=20, passes=[1, None])
            + bench.bench_reductions(clause_counts=[10, 30], validation=20))


def by(rows, experiment, method=None):
    return [r for r in rows if r["experiment"] == experiment
            and (method is None or r["method"] == method)]


class TestExperiments:

    def test_subset_sum(self, rows):
        bt = by(rows, "ss_exact", "backtracking")
        assert {r["parameter"] for r in bt} == {"no-instance", "planted"}
        worst = [r for r in bt if r["parameter"] == "no-instance"]
        assert all(r["mean_work"] < r["bound"] for r in worst)
        assert worst[-1]["mean_work"] > worst[0]["mean_work"]
        for r in by(rows, "ss_ratio"):
            assert 0.5 <= r["worst_quality"] <= r["mean_quality"] <= 1

    def test_sat(self, rows):
        methods = {r["method"] for r in by(rows, "sat_exact")}
        assert {"brute_force", "backtracking", "dpll"} <= methods
        for r in by(rows, "sat_exact", "walksat"):
            assert 0 <= r["success_rate"] <= 1
        rand = by(rows, "sat_quality", "random_assignment")
        assert all(0.6 < r["mean_quality"] <= 1 for r in rand)
        assert all(r["mean_time_s"] > 0 for r in by(rows, "sat_quality"))
        budget = by(rows, "sat_budget", "walksat")
        assert budget[0]["success_rate"] <= budget[-1]["success_rate"]

    def test_vertex_cover(self, rows):
        for r in by(rows, "vc_exact", "2approx_matching"):
            assert 1 <= r["worst_quality"] <= 2
        assert all(r["mean_quality"] == 2 for r in by(rows, "vc_tight"))

    def test_tsp(self, rows):
        for r in by(rows, "tsp_exact"):
            assert r["mean_quality"] >= 1 - 1e-9
        for r in by(rows, "tsp_exact", "mst_double_tree"):
            assert r["worst_quality"] <= 2
        brute = by(rows, "tsp_exact", "brute_force")
        assert [r["input_size"] for r in brute] == [5]     # capped at 6
        assert len(by(rows, "tsp_passes")) == 2

    def test_reductions(self, rows):
        checks = by(rows, "reduction_check")
        assert len(checks) == 3
        assert all(r["success_rate"] == 1.0 for r in checks)
        sizes = by(rows, "reduction_size")
        assert all(r["mean_work"] > 0 for r in sizes)


class TestOutput:

    def test_csv(self, rows, tmp_path):
        path = str(tmp_path / "table.csv")
        bench.write_csv(rows, path)
        with open(path) as f:
            read = list(csv.DictReader(f))
        assert len(read) == len(rows)
        assert list(read[0].keys()) == bench.FIELDS
        assert {r["approach"] for r in read} == {
            "exact", "heuristic", "approximation", "polynomial",
            "reduction"}

    def test_summary(self, rows, capsys):
        bench.print_summary(rows)
        out = capsys.readouterr().out
        assert "Exact solvers" in out and "consistent: 100%" in out

    def test_plots(self, rows, tmp_path):
        bench.make_plots(rows, str(tmp_path))
        for name in ["runtime_growth.png", "approximation_quality.png",
                     "tradeoff_curves.png",
                     "reduction_3sat_vertex_cover.png"]:
            assert os.path.getsize(tmp_path / name) > 0

    def test_ref_curve(self):
        xs, ys = bench._ref([1, 2, 3], [1, 4, 9], lambda x: x * x)
        assert ys == pytest.approx([1, 4, 9])
        assert bench._ref([], [], lambda x: x) == ([], [])

    def test_draw_highlighted_graph(self, tmp_path):
        out = tmp_path / "g.png"
        draw_highlighted_graph([("a", "b"), ("b", "c")],
                               {"a": (0, 0), "b": (1, 0), "c": (2, 0)},
                               ["a"], str(out), "partial cover", "caption")
        assert os.path.getsize(out) > 0
