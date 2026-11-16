"""
Week 12 benchmark: FFT, matrix algorithms, numerical stability, and
the signal / image / time-series applications.

Every computed result is checked against numpy (or an exact reference)
and the CSV records the error, so speed is never reported for a wrong
answer.

1. FFT: direct DFT, textbook recursion (Python lists), recursive and
   iterative Cooley–Tukey (numpy-vectorized), Bluestein for
   non-power-of-two n, and numpy.fft as the C reference; n = 2^4 … 2^20.
   Accuracy against numpy and round-trip error. Convolution: direct
   O(nm) vs FFT O((n+m) log) vs numpy.convolve.
2. Matrices: classical triple loop vs Strassen (pure Python base) vs
   numpy for multiplication, multiplication counts, float32 error;
   LU / QR / solve timing.
3. Stability: κ of Hilbert / Vandermonde / random matrices; LU solve
   error vs κ (with and without pivoting); QR orthogonality loss for
   classical Gram–Schmidt, modified Gram–Schmidt and Householder;
   catastrophic cancellation cases; summation error; power-method
   convergence vs eigenvalue gap.
4. Signals: window leakage, peak-frequency accuracy vs SNR, Welch vs
   periodogram variance, FFT filter attenuation and ringing, FIR stopband
   by window.
5. Images: DCT compression quality / rate sweep, FFT vs direct 2-D
   convolution, denoising and periodic-noise removal.
6. Applications: hum removal (subtract vs FFT notch), spectral gating
   gain vs input SNR, period detection, FFT vs direct autocorrelation.

Outputs (benchmarks/results/):
    fft_performance.png         runtime, accuracy, convolution crossover
    matrix_performance.png      multiplication, Strassen, decompositions
    numerical_stability.png     conditioning, QR orthogonality, cancellation
    signal_analysis.png         windows, spectrogram, peaks, filters
    image_processing.png        compression rate–distortion, filters
    image_examples.png          compression grid + filter gallery
    time_series_analysis.png    periods, ACF, smoothing, audio cleanup
    fft_stages.gif              iterative FFT stage by stage
    filter_animation.gif        low-pass cutoff sweeping down
    fft_comparison_table.csv    one row per configuration

Usage:
    python benchmarks/week12_fft_benchmark.py          # ~40 s
    python benchmarks/week12_fft_benchmark.py --quick  # ~15 s
"""
import argparse
import csv
import importlib
import math
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.fft_analyzer.applications import (
    AudioProcessor, autocorrelation, detect_periodicities, synthetic_series,
)
from src.fft_analyzer.applications.data_analyzer import autocorrelation_direct, smooth
# core/__init__ re-exports the function fft, which shadows the submodule
# of the same name, so the modules are loaded by their full names
F = importlib.import_module("src.fft_analyzer.core.fft")
M = importlib.import_module("src.fft_analyzer.core.matrix")
N = importlib.import_module("src.fft_analyzer.core.numerical")
from src.fft_analyzer.image import compression as C
from src.fft_analyzer.image import filters as IF
from src.fft_analyzer.signal import analysis as SA
from src.fft_analyzer.signal import filters as SF
from src.fft_analyzer.signal import generators as G
from src.fft_analyzer.visualization import (
    animate_filtering, animate_fft_stages, plot_spectrogram, plot_windows,
)
from src.utils.timer import time_call
from src.utils.visualization import plot_mixed_panels

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_NAME = "fft_comparison_table.csv"
SEED = 12

FIELDS = ["experiment", "method", "n", "parameter", "mean_time_s", "value",
          "value2", "valid", "note"]

FFT_METHODS = {
    "dft": ("direct DFT O(n²)", lambda x: F.dft(x), 1 << 12),
    "python": ("recursive, Python lists", lambda x: F.fft_recursive_python(x), 1 << 13),
    "recursive": ("recursive (numpy)", lambda x: F.fft_recursive(x, 16), 1 << 20),
    "iterative": ("iterative (numpy)", lambda x: F.fft_iterative(x), 1 << 20),
    "numpy": ("numpy.fft (C reference)", lambda x: np.fft.fft(x), 1 << 20),
}
COLORS = {"direct DFT O(n²)": "#7f7f7f", "recursive, Python lists": "#bcbd22",
          "recursive (numpy)": "#ff7f0e", "iterative (numpy)": "#1f77b4",
          "numpy.fft (C reference)": "#2ca02c", "Bluestein (n = 2^k + 1)": "#9467bd"}


def make_row(**kw) -> Dict:
    row = {f: "" for f in FIELDS}
    row.update(kw)
    return row


def timed(func: Callable[[], object], min_time: float = 0.05) -> float:
    return time_call(func, min_time=min_time, max_runs=50)


# ----------------------------------------------------------------------
# 1. FFT
# ----------------------------------------------------------------------
def bench_fft(exponents: Sequence[int] = range(4, 21)) -> List[Dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for e in exponents:
        n = 1 << e
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        ref = np.fft.fft(x)
        scale = np.abs(ref).max()
        for key, (label, func, limit) in FFT_METHODS.items():
            if n > limit:
                continue
            out = np.asarray(func(x))
            err = float(np.abs(out - ref).max() / scale)
            t = timed(lambda: func(x), 0.02 if n < 4096 else 0.0)
            rows.append(make_row(experiment="fft", method=label, n=n, mean_time_s=t,
                                 value=err, valid=err < 1e-9,
                                 note="value = max error / max |X| vs numpy"))
        # round-trip error of our FFT, and a non-power-of-two length
        rt = float(np.abs(F.ifft(F.fft(x)) - x).max() / np.abs(x).max())
        rows.append(make_row(experiment="fft_roundtrip", method="iterative", n=n,
                             value=rt, valid=rt < 1e-12,
                             note="max |ifft(fft(x)) - x| / max |x|"))
        if n <= 1 << 19:
            m = n + 1
            y = rng.normal(size=m)
            err = float(np.abs(F.fft(y) - np.fft.fft(y)).max() / np.abs(np.fft.fft(y)).max())
            t = timed(lambda: F.fft(y), 0.02 if m < 4096 else 0.0)
            rows.append(make_row(experiment="fft", method="Bluestein (n = 2^k + 1)",
                                 n=m, mean_time_s=t, value=err, valid=err < 1e-9))
        print(f"  n=2^{e:<2d} " + "  ".join(
            f"{r['method'].split()[0]} {r['mean_time_s'] * 1e3:.3g}ms"
            for r in rows if r["n"] in (n, n + 1) and r["experiment"] == "fft"))
    return rows


def bench_convolution(n: int = 1 << 16, kernel_sizes=(3, 9, 33, 129, 513, 2049, 8193)) -> List[Dict]:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=n)
    rows = []
    for m in kernel_sizes:
        k = rng.normal(size=m)
        ref = np.convolve(x, k)
        for label, func in (("FFT (ours)", lambda: F.linear_convolution(x, k)),
                            ("direct shift-add", lambda: F.direct_convolution(x, k)),
                            ("numpy.convolve (C, direct)", lambda: np.convolve(x, k))):
            out = func()
            err = float(np.abs(out - ref).max() / np.abs(ref).max())
            rows.append(make_row(experiment="convolution", method=label, n=n,
                                 parameter=m, mean_time_s=timed(func, 0.02),
                                 value=err, valid=err < 1e-9,
                                 note="parameter = kernel length"))
        print(f"  conv kernel {m:5d}: " + "  ".join(
            f"{r['method'].split()[0]} {r['mean_time_s'] * 1e3:.2f}ms" for r in rows[-3:]))
    return rows


# ----------------------------------------------------------------------
# 2. matrices
# ----------------------------------------------------------------------
def bench_matmul(sizes=(16, 32, 64, 128, 256), classical_max: int = 256,
                 cutoff: int = 32) -> List[Dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in sizes:
        A, B = rng.normal(size=(n, n)), rng.normal(size=(n, n))
        ref = A @ B
        methods = [("numpy @ (BLAS)", lambda: A @ B)]
        if n <= classical_max:
            methods.append(("classical triple loop", lambda: M.matmul_classical(A, B)))
            methods.append((f"Strassen (cutoff {cutoff})",
                            lambda: M.strassen(A, B, cutoff=cutoff)))
        for label, func in methods:
            out = func()
            err = float(np.abs(out - ref).max() / np.abs(ref).max())
            rows.append(make_row(experiment="matmul", method=label, n=n,
                                 mean_time_s=timed(func, 0.05 if n <= 64 else 0.0),
                                 value=err, valid=err < 1e-10,
                                 value2=(M.strassen_multiplications(n, cutoff)
                                         if label.startswith("Strassen") else n ** 3),
                                 note="value2 = scalar multiplications"))
        print(f"  matmul n={n:4d}: " + "  ".join(
            f"{r['method'].split()[0]} {r['mean_time_s']:.4f}s"
            for r in rows if r["n"] == n))
    return rows


def bench_strassen_error(sizes=(32, 64, 128, 256, 512), cutoff: int = 16) -> List[Dict]:
    """float32 products vs a float64 reference: Strassen's extra
    additions cost accuracy."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for n in sizes:
        A, B = rng.random((n, n)), rng.random((n, n))
        ref = A @ B
        A32, B32 = A.astype(np.float32), B.astype(np.float32)
        numpy32 = (A32 @ B32).astype(float)
        # M.strassen works in float64; this mirror keeps every block
        # addition in float32 so the extra rounding is visible
        stras32 = _strassen32(A32, B32, cutoff)
        for label, out in (("classical (float32 BLAS)", numpy32),
                           (f"Strassen float32 (cutoff {cutoff})", stras32)):
            err = float(np.abs(out - ref).max() / np.abs(ref).max())
            rows.append(make_row(experiment="strassen_error", method=label, n=n,
                                 value=err, valid=True,
                                 note="max relative error vs float64"))
    return rows


def _strassen32(A, B, cutoff):
    n = A.shape[0]
    if n <= cutoff:
        return (A @ B).astype(np.float32)
    h = n // 2
    a11, a12, a21, a22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    b11, b12, b21, b22 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]
    m1 = _strassen32(a11 + a22, b11 + b22, cutoff)
    m2 = _strassen32(a21 + a22, b11, cutoff)
    m3 = _strassen32(a11, b12 - b22, cutoff)
    m4 = _strassen32(a22, b21 - b11, cutoff)
    m5 = _strassen32(a11 + a12, b22, cutoff)
    m6 = _strassen32(a21 - a11, b11 + b12, cutoff)
    m7 = _strassen32(a12 - a22, b21 + b22, cutoff)
    out = np.empty((n, n), dtype=np.float32)
    out[:h, :h] = m1 + m4 - m5 + m7
    out[:h, h:] = m3 + m5
    out[h:, :h] = m2 + m4
    out[h:, h:] = m1 - m2 + m3 + m6
    return out.astype(float)


def bench_decompositions(sizes=(25, 50, 100, 200, 400)) -> List[Dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for n in sizes:
        A = rng.normal(size=(n, n))
        b = rng.normal(size=n)
        x_ref = np.linalg.solve(A, b)
        checks = [
            ("LU (ours)", lambda: M.lu_decompose(A),
             lambda r: np.abs(r.P @ A - r.L @ r.U).max()),
            ("solve via LU (ours)", lambda: M.solve(A, b),
             lambda x: np.abs(x - x_ref).max() / np.abs(x_ref).max()),
            ("numpy.linalg.solve", lambda: np.linalg.solve(A, b),
             lambda x: np.abs(x - x_ref).max() / np.abs(x_ref).max()),
            ("Householder QR (ours)", lambda: M.qr_householder(A),
             lambda qr: np.abs(qr[0] @ qr[1] - A).max()),
            ("modified Gram-Schmidt (ours)", lambda: M.qr_modified_gram_schmidt(A),
             lambda qr: np.abs(qr[0] @ qr[1] - A).max()),
            ("numpy.linalg.qr", lambda: np.linalg.qr(A),
             lambda qr: np.abs(qr[0] @ qr[1] - A).max()),
            ("log-determinant (ours)", lambda: M.log_determinant(A),
             lambda sd: abs(sd[1] - np.linalg.slogdet(A)[1]) / abs(np.linalg.slogdet(A)[1])
             + (sd[0] != np.linalg.slogdet(A)[0])),
        ]
        for label, run, err_of in checks:
            out = run()
            err = float(err_of(out))
            rows.append(make_row(experiment="decomposition", method=label, n=n,
                                 mean_time_s=timed(run, 0.02), value=err,
                                 valid=err < 1e-6, note="value = reconstruction/solution error"))
        print(f"  decompositions n={n}: " + "  ".join(
            f"{r['method'].split()[0]} {r['mean_time_s'] * 1e3:.1f}ms"
            for r in rows if r["n"] == n))
    return rows


# ----------------------------------------------------------------------
# 3. stability
# ----------------------------------------------------------------------
def bench_conditioning(sizes=range(2, 15)) -> List[Dict]:
    rows = []
    rng = np.random.default_rng(SEED)
    for n in sizes:
        mats = {"Hilbert": N.hilbert(n),
                "Vandermonde (equispaced)": N.vandermonde(np.linspace(0, 1, n)),
                "random Gaussian": rng.normal(size=(n, n))}
        for family, A in mats.items():
            kappa = N.condition_number(A)
            ref = float(np.linalg.cond(A))
            x_true = np.ones(n)
            b = A @ x_true
            pivot_err = N.relative_error(M.solve(A, b), x_true)
            try:
                nopivot_err = N.relative_error(M.solve(A, b, pivot=False), x_true)
            except ValueError:
                nopivot_err = math.nan
            rows.append(make_row(experiment="conditioning", method=family, n=n,
                                 value=kappa, value2=pivot_err,
                                 parameter=nopivot_err,
                                 valid=abs(math.log10(kappa) - math.log10(ref)) < 0.5
                                 if ref < 1e15 else True,
                                 note=f"value = κ₂ (ours; numpy {ref:.3g}); value2 = "
                                      "LU solve error; parameter = no-pivot error"))
    return rows


def bench_qr_orthogonality(kappas=(1e1, 1e3, 1e5, 1e7, 1e9, 1e11, 1e13, 1e15),
                           n: int = 40) -> List[Dict]:
    rows = []
    for kappa in kappas:
        A = N.matrix_with_condition(n, kappa, seed=SEED)
        for label, f in (("classical Gram-Schmidt", M.qr_gram_schmidt),
                         ("modified Gram-Schmidt", M.qr_modified_gram_schmidt),
                         ("Householder", M.qr_householder)):
            Q, R = f(A)
            loss = float(np.linalg.norm(Q.T @ Q - np.eye(n)))
            resid = float(np.linalg.norm(Q @ R - A) / np.linalg.norm(A))
            rows.append(make_row(experiment="qr_orthogonality", method=label, n=n,
                                 parameter=kappa, value=loss, value2=resid, valid=True,
                                 note="value = ‖QᵀQ − I‖_F; value2 = ‖QR − A‖/‖A‖"))
    return rows


def bench_lu_growth(n: int = 60, trials: int = 20) -> List[Dict]:
    """Element growth with and without pivoting on random matrices, and
    on the 2×2 tiny-pivot example."""
    rng = np.random.default_rng(SEED)
    rows = []
    for pivot in (True, False):
        growth, errs = [], []
        for _ in range(trials):
            A = rng.normal(size=(n, n))
            r = M.lu_decompose(A, pivot=pivot)
            growth.append(r.growth)
            x = np.ones(n)
            errs.append(N.relative_error(M.lu_solve(r, A @ x), x))
        rows.append(make_row(experiment="lu_growth", method="partial pivoting" if pivot
                             else "no pivoting", n=n, value=statistics.mean(growth),
                             value2=statistics.mean(errs), valid=True,
                             note="value = mean growth factor; value2 = mean solve error"))
    A = np.array([[1e-20, 1.0], [1.0, 1.0]])
    for pivot in (True, False):
        x = M.solve(A, [1.0, 2.0], pivot=pivot)
        exact = np.array([1.0, 1.0]) / (1 - 1e-20)
        rows.append(make_row(experiment="lu_tiny_pivot", method="partial pivoting"
                             if pivot else "no pivoting", n=2,
                             value=N.relative_error(x, exact), valid=True,
                             note=f"solution {np.round(x, 6).tolist()}"))
    return rows


def bench_cancellation() -> List[Dict]:
    rows = []
    for case in N.cancellation_examples():
        rows.append(make_row(experiment="cancellation", method=case.name,
                             value=case.naive_error, value2=case.stable_error,
                             valid=case.stable_error <= case.naive_error,
                             note="value = naive relative error; value2 = stable"))
    return rows


def bench_summation(sizes=(1_000, 10_000, 100_000, 1_000_000)) -> List[Dict]:
    from fractions import Fraction
    rng = np.random.default_rng(SEED)
    rows = []
    for n in sizes:
        vals = (rng.random(n) * 10 ** rng.uniform(-3, 3, n)).tolist()
        exact = float(sum(Fraction(v) for v in vals)) if n <= 100_000 else math.fsum(vals)
        for label, func in (("naive loop", N.naive_sum), ("pairwise", N.pairwise_sum),
                            ("Kahan compensated", N.kahan_sum)):
            got = func(vals)
            rows.append(make_row(experiment="summation", method=label, n=n,
                                 value=max(abs(got - exact) / abs(exact), 1e-18),
                                 valid=True, note="relative error (1e-18 = exact)"))
    return rows


def bench_power_method(ratios=(0.5, 0.8, 0.9, 0.95, 0.99), n: int = 50) -> List[Dict]:
    rng = np.random.default_rng(SEED)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    rows = []
    for r in ratios:
        eig = np.concatenate([[1.0, r], rng.uniform(0, r * 0.9, n - 2)])
        A = (Q * eig) @ Q.T
        res = M.power_method(A, tol=1e-10, max_iter=100_000)
        rows.append(make_row(experiment="power_method", method="power method", n=n,
                             parameter=r, value=res.iterations,
                             value2=abs(res.value - 1.0), valid=res.converged,
                             note=f"iterations; theory ≈ log(1e-10)/log(r) = "
                                  f"{math.log(1e-10) / math.log(r):.0f}"))
    return rows


# ----------------------------------------------------------------------
# 4. signals
# ----------------------------------------------------------------------
def bench_windows() -> List[Dict]:
    return [make_row(experiment="window", method=w, value=p["sidelobe_db"],
                     value2=p["mainlobe_halfwidth_bins"], valid=True,
                     note="value = highest sidelobe dB; value2 = main-lobe half width (bins)")
            for w, p in ((w, SF.window_properties(w)) for w in SF.WINDOWS)]


def bench_peak_accuracy(snrs=(-10, -5, 0, 5, 10, 20, 40), trials: int = 30,
                        fs: float = 8000.0, n: int = 4096) -> List[Dict]:
    rows = []
    for snr in snrs:
        for win in ("rectangular", "hann", "blackman"):
            errs = []
            for k in range(trials):
                rng = np.random.default_rng(SEED * 100 + k)
                f0 = rng.uniform(500, 3000)
                t = np.arange(n) / fs
                x = G.add_noise(np.sin(2 * np.pi * f0 * t), snr, seed=k)
                peaks = SA.peak_frequencies(x, fs, 1, window_name=win)
                errs.append(abs(peaks[0].frequency - f0) if peaks else fs)
            rows.append(make_row(experiment="peak_accuracy", method=win, n=n,
                                 parameter=snr, value=statistics.median(errs),
                                 value2=fs / n, valid=True,
                                 note="value = median |error| Hz; value2 = bin width Hz"))
    return rows


def bench_psd_variance(n: int = 1 << 14, trials: int = 40) -> List[Dict]:
    rows = []
    for label, func in (("periodogram", lambda x: SA.periodogram(x, 1.0, "hann")),
                        ("Welch (256, 50%)", lambda x: SA.welch_psd(x, 1.0, 256, 0.5))):
        rel_std = []
        for k in range(trials):
            _, p = func(G.white_noise(n, 1.0, seed=k))
            core = p[5:-5]
            rel_std.append(core.std() / core.mean())
        rows.append(make_row(experiment="psd_variance", method=label, n=n,
                             value=statistics.mean(rel_std), valid=True,
                             note="relative std of the PSD of white noise (ideal 0)"))
    return rows


def bench_filters(fs: float = 8000.0, n: int = 1 << 14) -> List[Dict]:
    rows = []
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 2500 * t)
    for label, tr in (("brick-wall FFT low-pass", 0.0),
                      ("raised-cosine 600 Hz transition", 600.0),
                      ("raised-cosine 1200 Hz transition", 1200.0)):
        y = SF.lowpass(x, fs, 1000, transition=tr)
        att = SF.measure_attenuation(y, x, fs, (2450, 2550))
        keep = SF.measure_attenuation(y, x, fs, (250, 350))
        step = np.r_[np.zeros(n // 2), np.ones(n // 2)]
        ring = SF.ringing(SF.lowpass(step, fs, 1000, transition=tr))
        rows.append(make_row(experiment="fft_filter", method=label, n=n,
                             value=att, value2=ring, parameter=keep, valid=keep > -0.1,
                             note="value = stopband tone change dB; value2 = step "
                                  "overshoot; parameter = passband tone change dB"))
    for win in SF.WINDOWS:
        h = SF.fir_lowpass(101, 1000, fs, win)
        rows.append(make_row(experiment="fir_window", method=win, n=101,
                             value=SF.stopband_attenuation(h, 1000, 300, fs), valid=True,
                             note="worst stopband gain (dB) above 1300 Hz, 101 taps"))
    return rows


# ----------------------------------------------------------------------
# 5. images
# ----------------------------------------------------------------------
def bench_compression(qualities=(5, 10, 20, 30, 50, 70, 85, 95), size: int = 256) -> List[Dict]:
    rows = []
    for kind in ("shapes", "texture"):
        img = IF.sample_image(size, kind)
        for q in qualities:
            start = time.perf_counter()
            r = C.quality_report(img, q)
            elapsed = time.perf_counter() - start
            rows.append(make_row(experiment="compression", method=kind, n=size,
                                 parameter=q, mean_time_s=elapsed, value=r["psnr"],
                                 value2=r["bits_per_pixel"], valid=True,
                                 note=f"value = PSNR dB; value2 = bits/pixel; SSIM "
                                      f"{r['ssim']:.4f}; file ratio {r['ratio_file']:.2f}; "
                                      f"zeros {r['zero_fraction']:.3f}"))
    return rows


def bench_conv2d(size: int = 256, kernels=(3, 5, 9, 15, 25, 41, 61, 81)) -> List[Dict]:
    rng = np.random.default_rng(SEED)
    img = rng.random((size, size))
    rows = []
    for k in kernels:
        ker = rng.random((k, k))
        ref = IF.direct_convolve2d(img, ker)
        for label, func in (("FFT 2-D convolution", lambda: IF.fft_convolve2d(img, ker)),
                            ("direct (shift-add)", lambda: IF.direct_convolve2d(img, ker))):
            out = func()
            err = float(np.abs(out - ref).max() / np.abs(ref).max())
            rows.append(make_row(experiment="conv2d", method=label, n=size, parameter=k,
                                 mean_time_s=timed(func, 0.02), value=err,
                                 valid=err < 1e-9, note="parameter = kernel side"))
    return rows


def bench_denoise(size: int = 256) -> List[Dict]:
    rows = []
    img = IF.sample_image(size, "shapes")
    rng = np.random.default_rng(SEED)
    for sigma in (0.05, 0.1, 0.2):
        noisy = img + rng.normal(0, sigma, img.shape)
        for label, out in (("noisy input", noisy),
                           ("Gaussian blur σ=1", IF.gaussian_blur(noisy, 1.0)),
                           ("FFT low-pass 0.15", IF.lowpass_denoise(noisy, 0.15))):
            rows.append(make_row(experiment="denoise", method=label, n=size,
                                 parameter=sigma, value=IF.psnr(img, out), valid=True,
                                 note="value = PSNR dB; parameter = noise σ"))
    noisy = IF.add_periodic_noise(img, (0.0, 0.2), 0.2)
    cleaned, notched = IF.remove_periodic_noise(noisy)
    for label, out in (("stripes added", noisy), ("spikes notched", cleaned)):
        rows.append(make_row(experiment="periodic_noise", method=label, n=size,
                             value=IF.psnr(img, out), valid=True,
                             note=f"PSNR dB; {notched} spectral bins notched"))
    return rows


# ----------------------------------------------------------------------
# 6. applications
# ----------------------------------------------------------------------
def bench_audio(fs: int = 16000, duration: float = 2.0) -> List[Dict]:
    rows = []
    t, music = G.multitone([440, 554.37, 659.25], duration, fs, [0.4, 0.3, 0.25])
    hum = 0.3 * np.sin(2 * np.pi * 60.1 * t + 0.4) + 0.12 * np.sin(2 * np.pi * 180.3 * t)
    noise = G.white_noise(len(t), 0.03, seed=SEED)
    a = AudioProcessor.from_signal(music + hum + noise, fs)
    hum_rms = np.sqrt(np.mean(hum ** 2))
    for method in ("subtract", "fft"):
        from src.fft_analyzer.signal.filters import notch_harmonics
        y = notch_harmonics(a.samples, fs, 60, 5, method=method)
        left = np.sqrt(np.mean((y - music - noise) ** 2))
        rows.append(make_row(experiment="hum_removal",
                             method="least-squares subtraction" if method == "subtract"
                             else "FFT band-stop notches",
                             n=len(t), value=20 * math.log10(left / hum_rms), valid=True,
                             note="residual hum relative to original hum (dB)"))
    for snr in (0, 5, 10, 15, 20):
        x = 0.5 * G.chirp(200, 3000, duration, fs)[1] + music * 0.5
        noisy = G.add_noise(x, snr, seed=SEED + snr)
        clean = AudioProcessor.from_signal(noisy, fs).denoise().samples
        rows.append(make_row(experiment="spectral_gating", method="spectral gating", n=len(t),
                             parameter=snr, value=G.snr_db(x, clean), valid=True,
                             note="value = output SNR dB; parameter = input SNR dB"))
    return rows


def bench_time_series(trials: int = 10) -> List[Dict]:
    rows = []
    found = {7: 0, 365: 0}
    for k in range(trials):
        series = synthetic_series(1460, seed=SEED + k)
        periods = [p.period for p in detect_periodicities(series, n=3)]
        for target in found:
            found[target] += any(abs(p - (365.25 if target == 365 else 7)) /
                                 (365.25 if target == 365 else 7) < 0.05 for p in periods)
    for target, hits in found.items():
        rows.append(make_row(experiment="period_detection", method=f"{target}-day cycle",
                             n=1460, value=hits / trials, valid=True,
                             note="fraction of synthetic series where it was found (±5%)"))
    rng = np.random.default_rng(SEED)
    for n in (1_000, 4_000, 16_000, 64_000):
        x = rng.normal(size=n)
        lag = n - 1                                   # the full ACF
        fft_t = timed(lambda: autocorrelation(x, lag), 0.02)
        err = float(np.abs(autocorrelation(x, 100) - autocorrelation_direct(x, 100)).max())
        rows.append(make_row(experiment="acf", method="FFT (Wiener-Khinchin)", n=n,
                             parameter=lag, mean_time_s=fft_t, value=err, valid=err < 1e-9))
        if n <= 16_000:                               # O(n²): too slow beyond
            rows.append(make_row(experiment="acf", method="direct O(n²)", n=n,
                                 parameter=lag, valid=True,
                                 mean_time_s=timed(lambda: autocorrelation_direct(x, lag),
                                                   0.02)))
    return rows


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
def _series(rows, experiment, key="n", value="mean_time_s", **match):
    out: Dict[str, tuple] = {}
    for r in rows:
        if r["experiment"] != experiment or r[value] == "" or \
                any(r[k] != v for k, v in match.items()):
            continue
        out.setdefault(r["method"], []).append((r[key], r[value]))
    return {m: ([p[0] for p in sorted(v)], [p[1] for p in sorted(v)]) for m, v in out.items()}


def plot_fft(rows, results_dir: str) -> None:
    rt = _series(rows, "fft")
    acc = _series(rows, "fft", value="value")
    roundtrip = _series(rows, "fft_roundtrip", value="value")
    ns = [r["n"] for r in rows if r["experiment"] == "fft_roundtrip"]
    if roundtrip and ns:
        ns = sorted(ns)
        roundtrip["ε·log₂n guide"] = (ns, [N.EPS * math.log2(n) for n in ns])
    conv = _series(rows, "convolution", key="parameter")
    panels = [
        {"title": "FFT runtime", "xlabel": "n", "ylabel": "seconds", "series": rt,
         "colors": COLORS, "logx": True, "logy": True, "measured_ticks": False, "log_labels": True,
         "refs": {"∝ n log n (iterative)": _nlogn_guide(rt.get("iterative (numpy)"))}},
        {"title": "Accuracy vs numpy (max error / max |X|)", "xlabel": "n",
         "ylabel": "relative error", "series": acc, "colors": COLORS, "logx": True,
         "logy": True, "measured_ticks": False, "log_labels": True},
        {"title": "Round trip ifft(fft(x)) error", "xlabel": "n",
         "ylabel": "max error / max |x|", "series": roundtrip, "logx": True, "logy": True,
         "measured_ticks": False, "log_labels": True},
        {"title": "Convolution, signal 2^16: direct vs FFT", "xlabel": "kernel length",
         "ylabel": "seconds", "series": conv, "logx": True, "logy": True},
    ]
    plot_mixed_panels(panels, os.path.join(results_dir, "fft_performance.png"),
                      "FFT: runtime, accuracy and convolution", ncols=2)


def _nlogn_guide(series):
    if not series:
        return ([], [])
    xs, ys = series
    c = ys[-1] / (xs[-1] * math.log2(xs[-1]))
    return xs, [c * x * math.log2(x) for x in xs]


def plot_matrix(rows, results_dir: str) -> None:
    mm = _series(rows, "matmul")
    mults = _series(rows, "matmul", value="value2")
    mults = {k: v for k, v in mults.items() if not k.startswith("numpy")}
    err = _series(rows, "strassen_error", value="value")
    dec = _series(rows, "decomposition")
    panels = [
        {"title": "Matrix multiplication time", "xlabel": "n", "ylabel": "seconds",
         "series": mm, "logx": True, "logy": True},
        {"title": "Scalar multiplications", "xlabel": "n", "ylabel": "count",
         "series": mults, "logx": True, "logy": True},
        {"title": "float32 error vs float64 reference", "xlabel": "n",
         "ylabel": "max relative error", "series": err, "logx": True, "logy": True},
        {"title": "Decompositions and solves", "xlabel": "n", "ylabel": "seconds",
         "series": dec, "logx": True, "logy": True},
    ]
    plot_mixed_panels(panels, os.path.join(results_dir, "matrix_performance.png"),
                      "Matrix algorithms: speed vs operation count vs accuracy", ncols=2)


def plot_stability(rows, results_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cond = [r for r in rows if r["experiment"] == "conditioning"]
    ax = axes[0, 0]
    for fam in dict.fromkeys(r["method"] for r in cond):
        pts = sorted((r["n"], r["value"]) for r in cond if r["method"] == fam)
        ax.semilogy(*zip(*pts), "o-", label=fam)
    ax.axhline(1 / N.EPS, color="k", ls=":", lw=1, label="1/ε")
    ax.set_title("Condition number κ₂ vs size")
    ax.set_xlabel("n")
    ax.set_ylabel("κ₂")

    ax = axes[0, 1]
    for fam, marker in zip(dict.fromkeys(r["method"] for r in cond), "os^"):
        pts = [(r["value"], r["value2"]) for r in cond if r["method"] == fam]
        ax.loglog(*zip(*pts), marker, label=f"{fam} (pivoting)")
    hil = [(r["value"], r["parameter"]) for r in cond if r["method"] == "Hilbert"
           and isinstance(r["parameter"], float) and r["parameter"] == r["parameter"]]
    if hil:
        ax.loglog(*zip(*hil), "x", color="#d62728", label="Hilbert, no pivoting")
    ks = np.logspace(0, 19, 50)
    ax.loglog(ks, np.minimum(ks * N.EPS, 10), "k:", label="κ·ε")
    ax.set_ylim(1e-17, 1e3)
    ax.set_title("LU solve error tracks κ·ε")
    ax.set_xlabel("κ₂(A)")
    ax.set_ylabel("relative error of x")

    ax = axes[0, 2]
    qr_rows = [r for r in rows if r["experiment"] == "qr_orthogonality"]
    for label in dict.fromkeys(r["method"] for r in qr_rows):
        pts = sorted((r["parameter"], r["value"]) for r in qr_rows if r["method"] == label)
        ax.loglog(*zip(*pts), "o-", label=label)
    if qr_rows:
        ks = np.array(sorted({r["parameter"] for r in qr_rows}))
        ax.loglog(ks, np.minimum(ks * N.EPS, 10), "k:", lw=1, label="κ·ε")
        ax.loglog(ks, np.minimum(ks ** 2 * N.EPS, 10), "k--", lw=1, label="κ²·ε")
    ax.set_title("QR: loss of orthogonality ‖QᵀQ − I‖")
    ax.set_xlabel("κ₂(A)")

    ax = axes[1, 0]
    canc = [r for r in rows if r["experiment"] == "cancellation"]
    if canc:
        y = np.arange(len(canc))
        ax.barh(y - 0.2, [max(r["value"], 1e-17) for r in canc], 0.4,
                color="#d62728", label="naive")
        ax.barh(y + 0.2, [max(r["value2"], 1e-17) for r in canc], 0.4,
                color="#2ca02c", label="stable rewrite")
        ax.set_yticks(y)
        ax.set_yticklabels([r["method"] for r in canc], fontsize=8)
        ax.set_xscale("log")
        ax.invert_yaxis()
    ax.set_title("Catastrophic cancellation (relative error; 1e-17 = exact)")

    ax = axes[1, 1]
    for label in ("naive loop", "pairwise", "Kahan compensated"):
        pts = sorted((r["n"], r["value"]) for r in rows if r["experiment"] == "summation"
                     and r["method"] == label)
        if pts:
            ax.loglog(*zip(*pts), "o-", label=label)
    ax.set_title("Summation error of n positive terms")
    ax.set_xlabel("n")
    ax.set_ylabel("relative error")

    ax = axes[1, 2]
    pm = sorted((r["parameter"], r["value"]) for r in rows if r["experiment"] == "power_method")
    if pm:
        ax.semilogy(*zip(*pm), "o-", label="measured iterations")
        ratios = np.array([p[0] for p in pm])
        ax.semilogy(ratios, np.log(1e-10) / np.log(ratios), "k:",
                    label="log(tol)/log|λ₂/λ₁|")
    ax.set_title("Power method: iterations vs eigenvalue ratio")
    ax.set_xlabel("|λ₂ / λ₁|")
    for a in axes.flat:
        a.grid(True, which="both", alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.suptitle("Numerical stability: conditioning, orthogonality, cancellation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(results_dir, "numerical_stability.png"), dpi=150)
    plt.close(fig)


def plot_signals(rows, results_dir: str) -> None:
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3)
    plot_windows(64, axes=[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])])
    fs = 8000
    t, c = G.chirp(100, 3500, 2.0, fs)
    _, tones = G.multitone([440, 1800], 2.0, fs, [0.4, 0.2])
    x = 0.6 * c + tones + G.white_noise(len(t), 0.05, seed=SEED)
    plot_spectrogram(x, fs, nperseg=512, ax=fig.add_subplot(gs[0, 2]),
                     title="Spectrogram: chirp 100→3500 Hz + 440 / 1800 Hz tones")
    ax = fig.add_subplot(gs[1, 0])
    for win in ("rectangular", "hann", "blackman"):
        pts = sorted((r["parameter"], r["value"]) for r in rows
                     if r["experiment"] == "peak_accuracy" and r["method"] == win)
        if pts:
            ax.semilogy(*zip(*pts), "o-", label=win)
    bw = [r["value2"] for r in rows if r["experiment"] == "peak_accuracy"]
    if bw:
        ax.axhline(bw[0], color="k", ls=":", label=f"one bin ({bw[0]:.2f} Hz)")
    ax.set_title("Peak-frequency error vs SNR (parabolic interpolation)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("median |error| (Hz)")
    ax = fig.add_subplot(gs[1, 1])
    for win in SF.WINDOWS:
        f, db = SF.frequency_response(SF.fir_lowpass(101, 1000, fs, win), fs)
        ax.plot(f, np.maximum(db, -140), lw=1, label=win)
    ax.set_title("101-tap windowed-sinc low-pass (cutoff 1 kHz)")
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB")
    ax = fig.add_subplot(gs[1, 2])
    n = 2048
    step = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    tt = np.arange(n) / fs
    for label, tr in (("brick-wall", 0.0), ("raised-cosine 600 Hz", 600.0),
                      ("raised-cosine 1200 Hz", 1200.0)):
        ax.plot(tt[n // 2 - 40: n // 2 + 80] * 1e3,
                SF.lowpass(step, fs, 1000, tr)[n // 2 - 40: n // 2 + 80], label=label)
    ax.set_title("Step response of the FFT low-pass (Gibbs ringing)")
    ax.set_xlabel("ms")
    for a in fig.axes:
        if a.get_label() != "<colorbar>":
            a.grid(True, alpha=0.3)
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8)
    fig.suptitle("Signal analysis: leakage, spectrograms, peaks, filters",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(results_dir, "signal_analysis.png"), dpi=150)
    plt.close(fig)


def plot_images(rows, results_dir: str) -> None:
    comp = [r for r in rows if r["experiment"] == "compression"]
    rd = {}
    q_psnr = {}
    for kind in dict.fromkeys(r["method"] for r in comp):
        pts = sorted((r["value2"], r["value"]) for r in comp if r["method"] == kind)
        rd[kind] = ([p[0] for p in pts], [p[1] for p in pts])
        qp = sorted((r["parameter"], r["value"]) for r in comp if r["method"] == kind)
        q_psnr[kind] = ([p[0] for p in qp], [p[1] for p in qp])
    panels = [
        {"title": "Rate–distortion: PSNR vs bits per pixel", "xlabel": "bits/pixel",
         "ylabel": "PSNR (dB)", "series": rd},
        {"title": "PSNR vs JPEG quality setting", "xlabel": "quality", "ylabel": "PSNR (dB)",
         "series": q_psnr},
        {"title": "2-D convolution of a 256×256 image", "xlabel": "kernel side",
         "ylabel": "seconds", "series": _series(rows, "conv2d", key="parameter"),
         "logy": True},
        {"title": "Denoising (Gaussian noise)", "xlabel": "noise σ", "ylabel": "PSNR (dB)",
         "series": _series(rows, "denoise", key="parameter", value="value")},
    ]
    plot_mixed_panels(panels, os.path.join(results_dir, "image_processing.png"),
                      "Image processing: DCT compression and FFT filtering", ncols=2)

    img = IF.sample_image(256, "shapes")
    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    for ax, q in zip(axes[0, 1:], (5, 15, 40, 85)):
        recon, c = C.round_trip(img, q)
        ax.imshow(recon[64:192, 64:192], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"quality {q}: {C.psnr(img, recon):.1f} dB, "
                     f"{c.bits_per_pixel:.2f} bpp", fontsize=9)
    axes[0, 0].imshow(img[64:192, 64:192], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("original (crop), 8 bpp", fontsize=9)
    noisy = IF.add_periodic_noise(img, (0.0, 0.2), 0.2)
    cleaned, _ = IF.remove_periodic_noise(noisy)
    gallery = [("Gaussian blur σ=3 (FFT)", IF.gaussian_blur(img, 3)),
               ("unsharp mask", np.clip(IF.sharpen(img, 1.5, 1.5), 0, 1)),
               ("Sobel edge magnitude", IF.sobel_edges(img, 1.0)[0]),
               ("stripe noise added", noisy),
               ("stripes notched in frequency", cleaned)]
    for ax, (title, im) in zip(axes[1], gallery):
        ax.imshow(im, cmap="gray", vmin=0, vmax=None if "Sobel" in title else 1)
        ax.set_title(title, fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("DCT compression (top) and FFT filters (bottom)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)
    fig.savefig(os.path.join(results_dir, "image_examples.png"), dpi=130)
    plt.close(fig)


def plot_time_series(rows, results_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    series = synthetic_series(1460, seed=SEED)
    days = np.arange(len(series))
    ax = axes[0, 0]
    ax.plot(days, series, color="#bbbbbb", lw=0.7, label="daily values")
    ax.plot(days, smooth(series, 29, "gaussian"), color="#1f77b4", lw=1.5,
            label="Gaussian smoothing (29 days, FFT convolution)")
    ax.set_title("Synthetic 4-year series: trend + weekly + yearly cycles")
    ax.set_xlabel("day")
    ax = axes[0, 1]
    acf = autocorrelation(series - np.polyval(np.polyfit(days, series, 1), days), 800)
    ax.plot(acf, lw=1)
    for p in detect_periodicities(series, n=2):
        ax.axvline(p.period, color="#d62728", ls="--", lw=0.8)
        ax.text(p.period, 0.9, f" {p.period:.1f} d", color="#d62728", fontsize=8)
    ax.set_title("Autocorrelation (FFT) with detected periods")
    ax.set_xlabel("lag (days)")
    ax = axes[1, 0]
    acf_rows = _series(rows, "acf")
    for label, (xs, ys) in acf_rows.items():
        ax.loglog(xs, ys, "o-", label=label)
    ax.set_title("Autocorrelation runtime")
    ax.set_xlabel("series length")
    ax.set_ylabel("seconds")
    ax = axes[1, 1]
    gate = sorted((r["parameter"], r["value"]) for r in rows
                  if r["experiment"] == "spectral_gating")
    if gate:
        ax.plot(*zip(*gate), "o-", label="after spectral gating")
        ax.plot([g[0] for g in gate], [g[0] for g in gate], "k:", label="input SNR")
    hum = [r for r in rows if r["experiment"] == "hum_removal"]
    for r in hum:
        ax.text(0.02, 0.92 - 0.07 * hum.index(r), f"hum removal, {r['method']}: "
                f"{r['value']:.1f} dB", transform=ax.transAxes, fontsize=8)
    ax.set_title("Audio clean-up")
    ax.set_xlabel("input SNR (dB)")
    ax.set_ylabel("output SNR (dB)")
    for a in axes.flat:
        a.grid(True, which="both", alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.suptitle("Time-series and audio applications", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(results_dir, "time_series_analysis.png"), dpi=150)
    plt.close(fig)


def make_animations(results_dir: str) -> None:
    animate_fft_stages(n=16, save_path=os.path.join(results_dir, "fft_stages.gif"))
    fs = 8000
    t, x = G.multitone([60, 440, 1500, 3200], 0.25, fs, [1.0, 0.6, 0.4, 0.3])
    animate_filtering(x, fs, np.geomspace(3900, 80, 30), max_time=0.03,
                      save_path=os.path.join(results_dir, "filter_animation.gif"))


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return str(v)


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def run(results_dir: str = RESULTS_DIR, quick: bool = False) -> List[Dict]:
    os.makedirs(results_dir, exist_ok=True)
    rows: List[Dict] = []
    print("1. FFT and convolution")
    rows += bench_fft(range(4, 17) if quick else range(4, 21))
    rows += bench_convolution(n=1 << 14 if quick else 1 << 16,
                              kernel_sizes=(3, 33, 513, 2049) if quick else
                              (3, 9, 33, 129, 513, 2049, 8193))
    print("2. Matrices")
    rows += bench_matmul((16, 32, 64, 128) if quick else (16, 32, 64, 128, 256, 512),
                         classical_max=128 if quick else 256)
    rows += bench_strassen_error((32, 64, 128) if quick else (32, 64, 128, 256, 512))
    rows += bench_decompositions((25, 50, 100) if quick else (25, 50, 100, 200, 400))
    print("3. Stability")
    rows += bench_conditioning(range(2, 15))
    rows += bench_qr_orthogonality(n=20 if quick else 40)
    rows += bench_lu_growth(trials=5 if quick else 20)
    rows += bench_cancellation()
    rows += bench_summation((1_000, 10_000) if quick else (1_000, 10_000, 100_000, 1_000_000))
    rows += bench_power_method()
    print("4. Signals")
    rows += bench_windows()
    rows += bench_peak_accuracy(trials=8 if quick else 30)
    rows += bench_psd_variance(trials=10 if quick else 40)
    rows += bench_filters()
    print("5. Images")
    rows += bench_compression(size=128 if quick else 256)
    rows += bench_conv2d(size=128 if quick else 256)
    rows += bench_denoise(size=128 if quick else 256)
    print("6. Applications")
    rows += bench_audio(duration=1.0 if quick else 2.0)
    rows += bench_time_series(trials=3 if quick else 10)

    path = os.path.join(results_dir, CSV_NAME)
    write_csv(rows, path)
    print(f"\nSaved {path} ({len(rows)} rows)")
    bad = [r for r in rows if r["valid"] is False]
    print(f"Validation: {len(bad)} configuration(s) failed: "
          f"{[(r['experiment'], r['method'], r['n']) for r in bad]}" if bad
          else "Validation: every checked result matched its reference")
    plot_fft(rows, results_dir)
    plot_matrix(rows, results_dir)
    plot_stability(rows, results_dir)
    plot_signals(rows, results_dir)
    plot_images(rows, results_dir)
    plot_time_series(rows, results_dir)
    make_animations(results_dir)
    print(f"Saved plots and animations to {results_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true", help="smaller sizes, fewer trials")
    args = parser.parse_args()
    start = time.perf_counter()
    run(quick=args.quick)
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
