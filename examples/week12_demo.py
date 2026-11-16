"""
Week 12 demo: the FFTAnalyzer toolkit — signal, audio, and image
workflows, plus the numerical foundations underneath.

Run from the project root:

    python examples/week12_demo.py

Sections: (1) the FFT itself — speed at 2^20 points and accuracy,
(2) signal analysis — peaks, leakage, filtering, (3) audio — a WAV with
mains hum and noise cleaned up and written back out, (4) images —
FFT filters and JPEG-style compression at several qualities,
(5) time series — periodicities and seasonality, (6) numerical
stability — conditioning, QR orthogonality, cancellation.
Output files go to a temporary directory (printed at the end).
"""
import importlib
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.fft_analyzer.applications import (
    AudioProcessor, ImageProcessor, TimeSeriesAnalyzer, synthetic_series, write_wav,
)
from src.fft_analyzer.image.filters import add_periodic_noise, sample_image
from src.fft_analyzer.signal import analysis as SA
from src.fft_analyzer.signal import filters as SF
from src.fft_analyzer.signal import generators as G

F = importlib.import_module("src.fft_analyzer.core.fft")
M = importlib.import_module("src.fft_analyzer.core.matrix")
N = importlib.import_module("src.fft_analyzer.core.numerical")

OUT = tempfile.mkdtemp(prefix="fft_analyzer_demo_")


def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_fft():
    header("1. Cooley-Tukey FFT")
    x = np.random.default_rng(0).normal(size=1 << 20)
    for name, f in (("iterative (bit reversal + vectorized butterflies)", F.fft_iterative),
                    ("recursive (numpy, DFT below 16 points)", lambda v: F.fft_recursive(v, 16)),
                    ("numpy.fft (C reference)", np.fft.fft)):
        start = time.perf_counter()
        X = f(x)
        print(f"  2^20 points, {name:50s} {time.perf_counter() - start:6.3f} s")
    ref = np.fft.fft(x)
    print(f"  max error vs numpy: {np.abs(F.fft_iterative(x) - ref).max() / np.abs(ref).max():.1e}"
          f"   round trip: {np.abs(F.ifft(F.fft(x)) - x).max():.1e}")
    y = np.random.default_rng(1).normal(size=1_000_003)
    start = time.perf_counter()
    F.fft(y)
    print(f"  n = 1,000,003 (prime-ish length) via Bluestein: {time.perf_counter() - start:.2f} s")
    a, b = np.random.default_rng(2).normal(size=50_000), np.ones(2001) / 2001
    start = time.perf_counter()
    fast = F.linear_convolution(a, b)
    t_fast = time.perf_counter() - start
    start = time.perf_counter()
    slow = F.direct_convolution(a, b)
    print(f"  convolution 50,000 x 2,001: FFT {t_fast * 1e3:.1f} ms vs direct "
          f"{(time.perf_counter() - start) * 1e3:.1f} ms (diff {np.abs(fast - slow).max():.1e})")


def demo_signal():
    header("2. Spectral analysis and filtering")
    fs = 8000
    t, x = G.multitone([440, 1250.6, 3000], 1.0, fs, [1.0, 0.5, 0.05])
    for win in ("rectangular", "hann", "blackman"):
        peaks = SA.peak_frequencies(x, fs, 3, window_name=win, threshold_db=-60)
        print(f"  {win:12s} peaks: " + ", ".join(f"{p.frequency:7.2f} Hz ({p.magnitude:.3f})"
                                                 for p in peaks))
    print("  window sidelobes: " + ", ".join(
        f"{w} {SF.window_properties(w)['sidelobe_db']:.0f} dB" for w in SF.WINDOWS))
    y = SF.lowpass(x, fs, 2000, transition=200)
    print(f"  low-pass 2 kHz: 3 kHz tone {SF.measure_attenuation(y, x, fs, (2990, 3010)):.0f} dB, "
          f"440 Hz tone {SF.measure_attenuation(y, x, fs, (435, 445)):.3f} dB")


def demo_audio():
    header("3. Audio: hum + noise in, clean WAV out")
    fs = 16000
    t, music = G.multitone([261.63, 329.63, 392.0], 3.0, fs, [0.3, 0.25, 0.2])
    hum = 0.25 * np.sin(2 * np.pi * 60.1 * t) + 0.1 * np.sin(2 * np.pi * 180.3 * t)
    noisy = music + hum + G.white_noise(len(t), 0.04, seed=3)
    path = os.path.join(OUT, "noisy.wav")
    write_wav(path, noisy, fs)
    audio = AudioProcessor.from_wav(path)
    print(f"  loaded {audio.duration:.1f} s at {audio.fs} Hz, SNR {G.snr_db(music, audio.samples):.1f} dB")
    print(f"  strongest peaks: {[round(p.frequency, 1) for p in audio.peaks(5)]}")
    dehum = audio.remove_hum(60, 5)
    clean = dehum.denoise()
    print(f"  after hum removal: SNR {G.snr_db(music, dehum.samples):.1f} dB; "
          f"after spectral gating: {G.snr_db(music, clean.samples):.1f} dB")
    clean.normalize().to_wav(os.path.join(OUT, "clean.wav"))
    clean.plot_spectrogram(os.path.join(OUT, "clean_spectrogram.png"), max_freq=1000)
    print(f"  pipeline: {' -> '.join(clean.history[1:])}")


def demo_image():
    header("4. Images: FFT filters and DCT compression")
    img = ImageProcessor.from_array(sample_image(256))
    noisy = ImageProcessor.from_array(add_periodic_noise(sample_image(256), (0.0, 0.2), 0.2))
    fixed = noisy.remove_periodic_noise()
    print(f"  stripe noise: PSNR {img.compare(noisy)['psnr']:.1f} dB -> "
          f"{img.compare(fixed)['psnr']:.1f} dB after notching ({fixed.history[-1]})")
    img.edges().save(os.path.join(OUT, "edges.png"))
    print("  quality  PSNR    SSIM   bits/px  ratio(entropy)  ratio(file)")
    for r in img.quality_sweep((10, 30, 50, 75, 95)):
        print(f"  {r['quality']:6d} {r['psnr']:6.1f}  {r['ssim']:.3f}  {r['bits_per_pixel']:6.2f}"
              f"  {r['ratio_entropy']:12.1f}  {r['ratio_file']:10.1f}")
    size = img.save_compressed(os.path.join(OUT, "image_q50.npz"), 50)
    print(f"  saved quality-50 coefficients: {size:,} bytes vs {256 * 256:,} raw")


def demo_time_series():
    header("5. Time series: periodicities and seasonality")
    sales = TimeSeriesAnalyzer(synthetic_series(1460, seed=7), name="daily sales")
    for p in sales.periodicities(3):
        print(f"  cycle of {p.period:7.2f} days  (ACF at lag {p.acf:+.2f}, "
              f"{100 * p.power:.1f}% of spectral power)")
    print(f"  seasonal strength: { {k: round(v, 2) for k, v in sales.seasonality([7, 30, 365]).items()} }")


def demo_stability():
    header("6. Numerical stability")
    for n in (4, 8, 12):
        H = N.hilbert(n)
        x = np.ones(n)
        err = N.relative_error(M.solve(H, H @ x), x)
        print(f"  Hilbert {n:2d}: {N.check_conditioning(H)}; solve error {err:.1e}")
    A = N.matrix_with_condition(40, 1e10, seed=1)
    for name, f in M.QR_METHODS.items():
        Q, _ = f(A)
        print(f"  QR {name:22s} ‖QᵀQ − I‖ = {np.linalg.norm(Q.T @ Q - np.eye(40)):.1e}  (κ = 1e10)")
    for case in N.cancellation_examples()[:4]:
        print(f"  {case.name:36s} naive error {case.naive_error:.1e}, stable {case.stable_error:.1e}")
    tiny = [[1e-20, 1.0], [1.0, 1.0]]
    print(f"  LU on [[1e-20, 1], [1, 1]] x = [1, 2]: with pivoting {M.solve(tiny, [1, 2])}, "
          f"without {M.solve(tiny, [1, 2], pivot=False)}")


if __name__ == "__main__":
    demo_fft()
    demo_signal()
    demo_audio()
    demo_image()
    demo_time_series()
    demo_stability()
    print(f"\nFiles written to {OUT}")
