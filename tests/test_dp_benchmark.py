"""Test suite for the Week 5 measurement utilities, DP plot helper,
and benchmark pipeline (run on tiny sizes)."""

import csv
import os
import sys
import time

import pytest

from src.dp import fib_naive, fib_tabulation, knapsack_tabulation, lcs
from src.dp._recursion import recursion_depth
from src.utils.timer import (
    CallStats, Timer, count_calls, peak_memory, time_call,
)
from src.utils.visualization import plot_dp_comparison
from benchmarks import week5_dp_benchmark as bench


class TestCallStats:

    def test_enter_exit_tracks_depth(self):
        s = CallStats()
        s.enter(); s.enter(); s.exit(); s.enter(); s.exit(); s.exit()
        assert s.calls == 3
        assert s.max_depth == 2

    def test_count_adds_calls_without_depth(self):
        s = CallStats()
        s.count(10)
        assert s.calls == 10 and s.max_depth == 0

    def test_count_calls_helper(self):
        stats = count_calls(fib_naive, 5)
        assert stats.calls == 15 and stats.max_depth == 5


class TestTiming:

    def test_timer_measures_elapsed(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.009

    def test_time_call_positive(self):
        assert time_call(lambda: fib_tabulation(100)) > 0

    def test_time_call_averages_fast_calls(self):
        calls = []
        time_call(lambda: calls.append(1), min_time=0.01)
        assert len(calls) > 1  # re-run because a single call is too fast

    def test_time_call_runs_slow_call_once(self):
        calls = []
        time_call(lambda: (calls.append(1), time.sleep(0.02)),
                  min_time=0.01)
        assert len(calls) == 1


class TestPeakMemory:

    def test_returns_result(self):
        _, result = peak_memory(lambda: 42)
        assert result == 42

    def test_larger_allocation_larger_peak(self):
        small, _ = peak_memory(lambda: [0] * 1_000)
        large, _ = peak_memory(lambda: [0] * 100_000)
        assert large > small

    def test_dp_table_memory_grows_with_capacity(self):
        w, v = [3, 4, 5] * 10, [4, 5, 6] * 10
        small, _ = peak_memory(lambda: knapsack_tabulation(w, v, 100))
        large, _ = peak_memory(lambda: knapsack_tabulation(w, v, 1000))
        assert large > 5 * small


class TestRecursionDepthGuard:

    def test_raises_then_restores_limit(self):
        before = sys.getrecursionlimit()
        with recursion_depth(before + 5000):
            assert sys.getrecursionlimit() >= before + 5000
        assert sys.getrecursionlimit() == before

    def test_never_lowers_limit(self):
        before = sys.getrecursionlimit()
        with recursion_depth(10):
            assert sys.getrecursionlimit() == before

    def test_restores_on_exception(self):
        before = sys.getrecursionlimit()
        with pytest.raises(RuntimeError):
            with recursion_depth(before + 5000):
                raise RuntimeError
        assert sys.getrecursionlimit() == before


class TestPlot:

    def test_writes_png(self, tmp_path):
        out = str(tmp_path / "dp.png")
        plot_dp_comparison({
            "recursive": {"sizes": [1, 2], "time": [1e-3, 1e-2],
                          "calls": [3, 9], "memory": [100, None]},
            "tabulation": {"sizes": [1, 2, 4], "time": [1e-5, 2e-5, 4e-5],
                           "calls": [2, 3, 5], "memory": [50, 60, 80]},
        }, out, title="test")
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_without_baseline_or_memory(self, tmp_path):
        out = str(tmp_path / "dp.png")
        plot_dp_comparison({
            "memoization": {"sizes": [10, 100], "time": [1e-4, 1e-3],
                            "calls": [19, 199]},
        }, out, title="test", log_x=True)
        assert os.path.getsize(out) > 0


class TestBenchmarkPipeline:

    def test_fibonacci_rows(self):
        rows = bench.bench_fibonacci(sizes=[5, 12], count_limit=5,
                                     memory_limit=5)
        naive = {r["input_size"]: r for r in rows
                 if r["method"] == "recursive"}
        assert naive[5]["calls_source"] == "measured"
        assert naive[12]["calls_source"] == "formula"
        assert naive[12]["calls"] == 2 * 233 - 1
        assert naive[12]["peak_memory_bytes"] is None
        assert {r["method"] for r in rows} == \
            {"recursive", "memoization", "tabulation"}

    def test_naive_limit_skips_large_n(self):
        rows = bench.bench_fibonacci(sizes=[5, 12], naive_limit=5)
        assert not any(r["method"] == "recursive" and r["input_size"] == 12
                       for r in rows)

    def test_knapsack_memo_limit(self):
        rows = bench.bench_knapsack(recursive_sizes=[4], dp_sizes=[4, 30],
                                    memo_limit=10)
        methods_at_30 = {r["method"] for r in rows if r["input_size"] == 30}
        assert methods_at_30 == {"tabulation"}

    def test_knapsack_methods_agree(self):
        w, v, c = bench.make_items(12)
        from src.dp import knapsack_recursive, knapsack_memo
        assert knapsack_recursive(w, v, c) == knapsack_memo(w, v, c) \
            == knapsack_tabulation(w, v, c)

    def test_capacity_sweep_cells_scale_with_w(self):
        rows = bench.bench_knapsack_capacity(n=10, capacities=[10, 100])
        cells = [r["calls"] for r in rows if r["method"] == "tabulation"]
        assert cells == [10 * 11, 10 * 101]

    def test_lcs_rows(self):
        rows = bench.bench_lcs(recursive_lengths=[4], dp_lengths=[4, 20])
        assert len(rows) == 5

    def test_make_strings_seeded(self):
        assert bench.make_strings(10) == bench.make_strings(10)
        assert set("".join(bench.make_strings(50))) <= set("ACGT")

    def test_speedups_filled_against_recursive(self):
        rows = bench.bench_fibonacci(sizes=[15], count_limit=15,
                                     memory_limit=15)
        bench.add_speedups(rows)
        for r in rows:
            if r["method"] == "recursive":
                assert r["speedup_vs_recursive"] is None
            else:
                assert r["speedup_vs_recursive"] > 1

    def test_csv_and_series(self, tmp_path):
        rows = bench.bench_lcs(recursive_lengths=[4], dp_lengths=[4])
        bench.add_speedups(rows)
        path = str(tmp_path / "out.csv")
        bench.write_csv(rows, path)
        with open(path) as f:
            read = list(csv.DictReader(f))
        assert len(read) == 3
        assert list(read[0].keys()) == bench.FIELDS

        series = bench.series_for(rows, "lcs")
        assert set(series) == {"recursive", "memoization", "tabulation"}
        assert series["tabulation"]["sizes"] == [4]

    def test_make_plots_writes_three_pngs(self, tmp_path):
        rows = (bench.bench_fibonacci(sizes=[5, 10], memory_limit=10)
                + bench.bench_knapsack(recursive_sizes=[4], dp_sizes=[4, 8])
                + bench.bench_lcs(recursive_lengths=[4], dp_lengths=[4, 8]))
        bench.add_speedups(rows)
        bench.make_plots(rows, str(tmp_path))
        for name in ["fibonacci_comparison.png", "knapsack_performance.png",
                     "lcs_performance.png"]:
            assert os.path.getsize(tmp_path / name) > 0
