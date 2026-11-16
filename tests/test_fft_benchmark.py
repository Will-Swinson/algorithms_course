"""Test suite for the Week 12 benchmark pipeline, run on tiny sizes."""

import csv
import os

import pytest

from benchmarks import week12_fft_benchmark as bench


@pytest.fixture(scope="module")
def rows():
    return (bench.bench_fft(range(4, 9))
            + bench.bench_convolution(n=1 << 10, kernel_sizes=(3, 65))
            + bench.bench_matmul((8, 16), classical_max=16, cutoff=4)
            + bench.bench_strassen_error((16, 32), cutoff=4)
            + bench.bench_decompositions((10, 20))
            + bench.bench_conditioning(range(2, 7))
            + bench.bench_qr_orthogonality((1e2, 1e8), n=10)
            + bench.bench_lu_growth(n=10, trials=3)
            + bench.bench_cancellation()
            + bench.bench_summation((100, 1000))
            + bench.bench_power_method((0.5, 0.9), n=10)
            + bench.bench_windows()
            + bench.bench_peak_accuracy((0, 20), trials=3, n=1024)
            + bench.bench_psd_variance(n=4096, trials=3)
            + bench.bench_filters(n=4096)
            + bench.bench_compression((10, 90), size=32)
            + bench.bench_conv2d(size=32, kernels=(3, 7))
            + bench.bench_denoise(size=64)
            + bench.bench_audio(duration=0.5)
            + bench.bench_time_series(trials=2))


def by(rows, experiment, **match):
    return [r for r in rows if r["experiment"] == experiment
            and all(r[k] == v for k, v in match.items())]


def test_all_checks_pass(rows):
    bad = [(r["experiment"], r["method"], r["n"]) for r in rows if r["valid"] is False]
    assert bad == []


def test_fft_rows(rows):
    methods = {r["method"] for r in by(rows, "fft")}
    assert {"direct DFT O(n²)", "iterative (numpy)", "numpy.fft (C reference)",
            "Bluestein (n = 2^k + 1)"} <= methods
    assert all(r["value"] < 1e-9 for r in by(rows, "fft"))
    assert len(by(rows, "fft_roundtrip")) == 5


def test_matrix_rows(rows):
    strassen = by(rows, "matmul", method="Strassen (cutoff 4)")
    assert strassen and strassen[-1]["value2"] < 16 ** 3
    errs = {r["method"]: r["value"] for r in by(rows, "strassen_error", n=32)}
    assert len(errs) == 2
    assert {r["method"] for r in by(rows, "decomposition")} >= {"LU (ours)", "log-determinant (ours)"}


def test_stability_rows(rows):
    hil = sorted((r["n"], r["value"]) for r in by(rows, "conditioning", method="Hilbert"))
    assert [k for _, k in hil] == sorted(k for _, k in hil)
    qr = {(r["method"], r["parameter"]): r["value"] for r in by(rows, "qr_orthogonality")}
    assert qr[("Householder", 1e8)] < qr[("modified Gram-Schmidt", 1e8)] < \
        qr[("classical Gram-Schmidt", 1e8)]
    growth = {r["method"]: r["value"] for r in by(rows, "lu_growth")}
    assert growth["no pivoting"] > growth["partial pivoting"]
    assert len(by(rows, "cancellation")) == 6
    pm = sorted((r["parameter"], r["value"]) for r in by(rows, "power_method"))
    assert pm[0][1] < pm[1][1]


def test_signal_image_app_rows(rows):
    assert {r["method"] for r in by(rows, "window")} == {"rectangular", "hann", "hamming",
                                                          "blackman"}
    var = {r["method"]: r["value"] for r in by(rows, "psd_variance")}
    assert var["Welch (256, 50%)"] < var["periodogram"]
    comp = sorted((r["parameter"], r["value"]) for r in by(rows, "compression", method="shapes"))
    assert comp[0][1] < comp[1][1]
    hum = {r["method"]: r["value"] for r in by(rows, "hum_removal")}
    assert hum["least-squares subtraction"] < hum["FFT band-stop notches"]
    assert by(rows, "period_detection", method="7-day cycle")[0]["value"] == 1.0


def test_plots_and_csv(rows, tmp_path):
    bench.write_csv(rows, str(tmp_path / bench.CSV_NAME))
    with open(tmp_path / bench.CSV_NAME) as fh:
        assert len(list(csv.DictReader(fh))) == len(rows)
    out = str(tmp_path)
    bench.plot_fft(rows, out)
    bench.plot_matrix(rows, out)
    bench.plot_stability(rows, out)
    bench.plot_signals(rows, out)
    bench.plot_images(rows, out)
    bench.plot_time_series(rows, out)
    bench.make_animations(out)
    for name in ("fft_performance.png", "matrix_performance.png", "numerical_stability.png",
                 "signal_analysis.png", "image_processing.png", "image_examples.png",
                 "time_series_analysis.png", "fft_stages.gif", "filter_animation.gif"):
        assert os.path.getsize(tmp_path / name) > 3000, name
