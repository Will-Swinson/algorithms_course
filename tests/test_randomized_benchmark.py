"""Test suite for the Week 7 benchmark pipeline (run on tiny sizes),
the random-input generators, the probability helpers, and the mixed
panel plot."""

import csv
import os

import pytest

from benchmarks import week7_randomized_benchmark as bench
from src.randomized.minhash_similarity import jaccard_similarity
from src.utils.probability_analysis import (
    harmonic, summarize, wilson_interval,
)
from src.utils.random_utils import (
    ARRAY_KINDS, random_array, random_set_pair, random_strings,
)
from src.utils.visualization import plot_mixed_panels


class TestRandomUtils:

    @pytest.mark.parametrize("kind", ARRAY_KINDS)
    def test_array_kinds(self, kind):
        arr = random_array(50, kind, seed=1)
        assert len(arr) == 50
        if kind != "few_unique":
            assert sorted(arr) == list(range(50))
        else:
            assert len(set(arr)) <= 10

    def test_sorted_and_reverse(self):
        assert random_array(4, "sorted") == [0, 1, 2, 3]
        assert random_array(4, "reverse") == [3, 2, 1, 0]

    def test_array_seeded(self):
        assert random_array(30, seed=5) == random_array(30, seed=5)
        assert random_array(30, seed=5) != random_array(30, seed=6)

    def test_array_invalid(self):
        with pytest.raises(ValueError):
            random_array(-1)
        with pytest.raises(ValueError):
            random_array(5, "zigzag")

    def test_random_strings_distinct_with_prefix(self):
        s = random_strings(200, length=4, prefix="in-", seed=1)
        assert len(set(s)) == 200
        assert all(x.startswith("in-") and len(x) == 7 for x in s)

    @pytest.mark.parametrize("target", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_set_pair_jaccard(self, target):
        a, b = random_set_pair(200, target, seed=2)
        assert len(a) == len(b) == 200
        assert jaccard_similarity(a, b) == pytest.approx(target, abs=0.01)

    def test_set_pair_invalid(self):
        with pytest.raises(ValueError):
            random_set_pair(10, 1.5)


class TestProbabilityHelpers:

    def test_harmonic(self):
        assert harmonic(0) == 0
        assert harmonic(4) == pytest.approx(25 / 12)
        exact = sum(1 / i for i in range(1, 5_001))
        assert harmonic(5_000) == pytest.approx(exact, rel=1e-12)
        with pytest.raises(ValueError):
            harmonic(-1)

    def test_summarize(self):
        s = summarize([2, 4, 4, 4, 5, 5, 7, 9])
        assert s.mean == 5
        assert s.variance == pytest.approx(32 / 7)
        assert s.minimum == 2 and s.maximum == 9
        assert s.ci95_low < 5 < s.ci95_high
        assert s.cv == pytest.approx(s.std / 5)

    def test_summarize_single_and_empty(self):
        assert summarize([3.0]).variance == 0
        with pytest.raises(ValueError):
            summarize([])

    def test_wilson_zero_errors(self):
        lo, hi = wilson_interval(0, 100_000)
        assert lo == 0
        assert 3e-5 < hi < 5e-5             # ≈ 3.84 / n

    def test_wilson_contains_observed_rate(self):
        lo, hi = wilson_interval(30, 100)
        assert lo < 0.3 < hi

    def test_wilson_invalid(self):
        with pytest.raises(ValueError):
            wilson_interval(1, 0)
        with pytest.raises(ValueError):
            wilson_interval(5, 4)


@pytest.fixture(scope="module")
def results():
    """Every experiment on tiny sizes, with samples for the plots."""
    samples = {}
    rows = (bench.bench_quicksort([50, 200], trials=4, worst_sizes=[50, 200],
                                  dist_n=200, dist_trials=20,
                                  samples=samples)
            + bench.bench_selection([50, 200], trials=4, dist_n=200,
                                    dist_trials=30, samples=samples)
            + bench.bench_primality(digits_trial=[6, 8],
                                    digits_mc=[6, 8, 30], rounds=[1, 3],
                                    carmichael=[561, 1105],
                                    carmichael_trials=100, liar_limit=200,
                                    samples=samples)
            + bench.bench_bloom(loads=[0.05, 0.2], load_ks=[3],
                                num_bits=4_096, num_items=500,
                                bits_per_item=[4, 8], k_sweep=[2, 5],
                                probes=2_000)
            + bench.bench_minhash(ks=[16, 64], jaccards=[0.3, 0.7],
                                  set_size=60, trials=3, text_ks=[16, 64],
                                  samples=samples))
    return rows, samples


class TestExperiments:

    def test_quicksort_methods(self, results):
        rows, _ = results
        got = {(r["method"], r["input_size"]) for r in rows
               if r["experiment"] == "quicksort_random"}
        for method in ["randomized_quicksort", "deterministic_quicksort",
                       "week2_quick_sort", "python_sorted"]:
            assert {(method, 50), (method, 200)} <= got

    def test_quicksort_worst_case_counts(self, results):
        rows, _ = results
        det = next(r for r in rows if r["experiment"] == "quicksort_sorted"
                   and r["method"] == "deterministic_quicksort"
                   and r["input_size"] == 200)
        assert det["mean_work"] == det["expected_work"] == 200 * 199 / 2
        assert det["std_work"] == 0

    def test_quicksort_trial_statistics(self, results):
        rows, _ = results
        r = next(r for r in rows if r["experiment"] == "quicksort_random"
                 and r["method"] == "randomized_quicksort")
        assert r["trials"] == 4
        assert r["var_time_s2"] == pytest.approx(r["std_time_s"] ** 2)
        assert r["mean_depth"] > 0

    def test_selection_rows(self, results):
        rows, samples = results
        params = {r["parameter"] for r in rows
                  if r["experiment"] == "selection"}
        assert params == {"k=min", "k=median", "k=max"}
        budgets = [r for r in rows if r["experiment"] == "selection_budget"]
        assert len(budgets) == len(bench.SEL_BUDGETS)
        rates = [r["success_rate"] for r in budgets]
        assert rates == sorted(rates)          # bigger budget, more success
        assert len(samples["sel_dist"]["comparisons"]) == 30

    def test_primality_rows(self, results):
        rows, samples = results
        speed = {(r["method"], r["input_size"]) for r in rows
                 if r["experiment"] == "primality_speed"}
        assert ("trial_division", 8) in speed
        assert ("trial_division", 30) not in speed
        assert ("miller_rabin", 30) in speed
        fermat_k1 = next(r for r in rows if r["experiment"] ==
                         "primality_error" and r["method"] == "fermat"
                         and r["parameter"] == "k=1")
        assert fermat_k1["error"] > 0.3          # Carmichael numbers lie
        assert fermat_k1["error_ci_high"] >= fermat_k1["error"]
        assert len(samples["liars"]["n"]) == len(
            samples["liars"]["miller_rabin"])

    def test_next_prime(self):
        assert bench.next_prime(90) == 97
        assert bench.next_prime(10 ** 6) == 1_000_003

    def test_odd_composites(self):
        assert bench.odd_composites(30) == [9, 15, 21, 25, 27]

    def test_bloom_rows(self, results):
        rows, _ = results
        load = [r for r in rows if r["experiment"] == "bloom_load"]
        assert len(load) == 2
        assert load[0]["error"] < load[1]["error"]
        mem = {r["method"]: r for r in rows if r["experiment"] ==
               "bloom_memory"}
        assert mem["python_set"]["mean_work"] > \
            mem["bloom_filter"]["mean_work"]

    def test_minhash_rows(self, results):
        rows, samples = results
        syn = [r for r in rows if r["experiment"] == "minhash_synthetic"]
        assert len(syn) == 4
        assert all(r["error"] is not None for r in syn)
        assert set(samples["minhash"]["text"]) == {16, 64}

    def test_mutate_text(self):
        text = "one two three four five six"
        assert bench.mutate_text(text, 0.0, seed=1) == text
        assert bench.mutate_text(text, 1.0, seed=1) != text
        assert len(bench.mutate_text(text, 0.5, seed=1).split()) == 6

    def test_text_corpus_size(self):
        docs = bench.text_corpus()
        assert len(docs) == len(bench.BASE_TEXTS) * \
            len(bench.TEXT_MUTATION_RATES)
        assert len({name for name, _ in docs}) == len(docs)

    def test_param_value(self):
        assert bench._param_value("k=7 load=0.1", "load") == 0.1
        with pytest.raises(KeyError):
            bench._param_value("k=7", "J")


class TestOutput:

    def test_csv(self, results, tmp_path):
        rows, _ = results
        path = str(tmp_path / "summary.csv")
        bench.write_csv(rows, path)
        with open(path) as f:
            read = list(csv.DictReader(f))
        assert len(read) == len(rows)
        assert list(read[0].keys()) == bench.FIELDS
        paradigms = {r["paradigm"] for r in read}
        assert paradigms == {"las_vegas", "monte_carlo", "deterministic"}

    def test_summary_prints(self, results, capsys):
        rows, _ = results
        bench.print_summary(rows)
        out = capsys.readouterr().out
        assert "QuickSort" in out and "Selection" in out
        assert "false_positive_rate" in out

    def test_make_plots_writes_all_pngs(self, results, tmp_path):
        rows, samples = results
        bench.make_plots(rows, samples, str(tmp_path))
        for name in ["quicksort_comparison.png", "selection_runtime.png",
                     "primality_accuracy.png", "bloom_filter_fp_rate.png",
                     "minhash_similarity_plot.png"]:
            assert os.path.getsize(tmp_path / name) > 0

    def test_plots_without_samples(self, results, tmp_path):
        rows, _ = results
        bench.make_plots(rows, {}, str(tmp_path))
        assert os.path.getsize(tmp_path / "quicksort_comparison.png") > 0


class TestMixedPanels:

    def test_all_kinds(self, tmp_path):
        out = str(tmp_path / "mixed.png")
        plot_mixed_panels([
            {"title": "line", "xlabel": "x", "ylabel": "y",
             "series": {"s": ([1, 2, 4], [1, None, 3])},
             "bands": {"s": ([1, 2, 4], [0.5, 1, 2], [1.5, 3, 4])},
             "refs": {"r": ([1, 4], [1, 4])}, "logx": True, "logy": True},
            {"kind": "hist", "title": "hist", "xlabel": "x", "ylabel": "n",
             "samples": {"a": [1, 2, 2, 3], "empty": []},
             "vlines": {"mean": 2}},
            {"kind": "scatter", "title": "scatter", "xlabel": "x",
             "ylabel": "y", "points": {"p": ([0.1, 0.5], [0.2, 0.4])},
             "refs": {"y = x": ([0, 1], [0, 1])}},
        ], out, "test")
        assert os.path.getsize(out) > 0

    def test_unknown_kind(self, tmp_path):
        with pytest.raises(ValueError):
            plot_mixed_panels([{"kind": "pie", "title": "", "xlabel": "",
                                "ylabel": ""}], str(tmp_path / "x.png"), "t")
