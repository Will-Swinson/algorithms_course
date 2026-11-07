"""Test suite for the Week 11 string benchmark pipeline, run on tiny
sizes into a temporary directory."""

import csv
import os

import pytest

from benchmarks import week11_string_benchmark as bench


@pytest.fixture(scope="module")
def results():
    samples = {}
    rows = (bench.bench_search([2_000, 8_000], queries=4,
                               limits={"naive": 2_000, "tree": 8_000},
                               samples=samples)
            + bench.bench_pattern_length(n=5_000, lengths=(4, 16), queries=3)
            + bench.bench_worst_case(n=2_000, lengths=(2, 8))
            + bench.bench_collisions(n=5_000, moduli=(31, 1_009, bench.MERSENNE_61),
                                     patterns=2)
            + bench.bench_birthday(moduli=(101, 1_009), trials=3)
            + bench.bench_batch(n=5_000, counts=(1, 5))
            + bench.bench_construction(sizes=(500, 2_000))
            + bench.bench_lcp(n=5_000, samples=samples)
            + bench.bench_plagiarism(fractions=(0.0, 0.5), length=3_000)
            + bench.bench_dna(sizes=(5_000,), genes=4, max_mismatches=(0, 2))
            + bench.bench_autocomplete(sizes=(500, 2_000), queries=20))
    return rows, samples


def by(rows, experiment, **match):
    return [r for r in rows if r["experiment"] == experiment
            and all(r[k] == v for k, v in match.items())]


class TestRows:

    def test_all_valid(self, results):
        rows, _ = results
        assert all(r["valid"] is True for r in rows)

    def test_search_rows(self, results):
        rows, _ = results
        search = by(rows, "search", dataset="english")
        assert {r["method"] for r in search} == set(bench.METHODS)
        assert {r["n"] for r in search if r["method"] == "naive"} == {2_000}
        for r in by(rows, "search"):
            assert r["mean_search_s"] > 0
            if r["method"] in ("sa", "tree"):
                assert r["preprocess_s"] > 0 and r["memory_mb"] > 0

    def test_worst_case(self, results):
        rows, _ = results
        naive = {r["pattern_length"]: r["value"] for r in
                 by(rows, "worst_case", method="naive")}
        kmp = {r["pattern_length"]: r["value"] for r in
               by(rows, "worst_case", method="kmp")}
        assert naive[8] > 3 * kmp[8]
        rk_all = by(rows, "worst_case", parameter="all windows match")
        assert all(r["value"] > 0 for r in rk_all)

    def test_collisions_follow_theory(self, results):
        rows, _ = results
        small = by(rows, "collision_modulus", parameter="q=31")[0]
        assert small["value"] == pytest.approx(small["value3"], rel=0.3)
        assert small["value2"] >= small["value"] * 0.5
        assert by(rows, "collision_modulus",
                  parameter=f"q={bench.MERSENNE_61}")[0]["value"] == 0
        params = {r["parameter"]: r["value"] for r in by(rows, "collision_params")}
        assert params["base 1 (character sum)"] > params["q=2^61-1, random base"]

    def test_birthday_and_batch(self, results):
        rows, _ = results
        assert all(r["value"] > 1 for r in by(rows, "birthday"))
        assert {r["method"] for r in by(rows, "batch")} == {"rk-multi", "kmp", "rk"}

    def test_construction_and_lcp(self, results):
        rows, samples = results
        methods = {r["method"] for r in by(rows, "construction")}
        assert methods == {"doubling", "python", "naive", "tree", "kasai"}
        assert {r["dataset"] for r in by(rows, "lcp")} == \
            {"english", "dna", "logs", "random"}
        assert ("lcp", "dna") in samples

    def test_applications(self, results):
        rows, _ = results
        plag = {r["parameter"]: r["value"] for r in by(rows, "plagiarism")}
        assert plag["copied=0.0"] == 0 and plag["copied=0.5"] > 0.35
        dna = {r["parameter"]: r["value"] for r in by(rows, "dna", method="sa")}
        assert dna["k=0"] == 0 and dna["k=2"] == 1
        assert len(by(rows, "autocomplete")) == 4


def test_zipf_vocabulary():
    vocab = bench.zipf_vocabulary(200)
    assert len(vocab) == 200
    freqs = list(vocab.values())
    assert freqs == sorted(freqs, reverse=True)


def test_plots_and_csv(results, tmp_path):
    rows, samples = results
    bench.write_csv(rows, str(tmp_path / bench.CSV_NAME))
    with open(tmp_path / bench.CSV_NAME) as fh:
        assert len(list(csv.DictReader(fh))) == len(rows)
    bench.plot_runtime(rows, str(tmp_path))
    bench.plot_collisions(rows, str(tmp_path))
    bench.plot_sa_lcp(rows, samples, str(tmp_path))
    bench.plot_heatmaps(rows, str(tmp_path))
    bench.plot_applications(rows, str(tmp_path))
    bench.plot_examples(str(tmp_path))
    for name in ("runtime_comparison.png", "collision_analysis.png",
                 "sa_lcp_stats.png", "heatmap_dataset_sizes.png",
                 "string_applications.png", "suffix_array_example.png",
                 "suffix_tree_example.png", "suffix_tree_example.dot",
                 "ukkonen_steps.png", "kmp_trace.png"):
        assert os.path.getsize(tmp_path / name) > 500, name
