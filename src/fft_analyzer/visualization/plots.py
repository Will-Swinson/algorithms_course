"""
Static plots for the toolkit (matplotlib, headless).

    plot_signal_and_spectrum  waveform above its one-sided magnitude
                              spectrum (dB), detected peaks marked
    plot_spectrum             magnitude spectrum of one or more signals
    plot_spectrogram          STFT power (dB) as a time × frequency image
    plot_fft2_heatmap         image next to its centred log-magnitude 2-D
                              spectrum
    plot_filter_response      FFT-filter mask or FIR response, in dB
    plot_windows              window shapes and their spectra (leakage)
    plot_condition_numbers    κ(A) and observed solve error vs size for
                              several matrix families, with the
                              "error ≈ κ·ε" guide line
    plot_compression_grid     original vs reconstructions at several
                              qualities with PSNR / bits-per-pixel labels
"""
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

from src.fft_analyzer.core.fft import fft, rfftfreq
from src.fft_analyzer.signal.analysis import peak_frequencies, spectrogram
from src.fft_analyzer.signal.filters import WINDOWS, window, window_spectrum
from src.fft_analyzer.image.filters import log_magnitude_spectrum


def _finish(fig, save_path: Optional[str], tight: bool = True):
    if tight:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def magnitude_db(x, fs: float = 1.0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.abs(fft(x * window("hann", n, symmetric=False)))[: n // 2 + 1]
    return rfftfreq(n, 1 / fs), 20 * np.log10(X / max(X.max(), 1e-300) + 1e-12)


def plot_spectrum(signals: Dict[str, np.ndarray], fs: float = 1.0, ax=None,
                  title: str = "Magnitude spectrum", max_freq: Optional[float] = None,
                  floor_db: float = -100):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 4))
    for label, x in signals.items():
        f, db = magnitude_db(x, fs)
        ax.plot(f, np.maximum(db, floor_db), lw=1, label=label)
    ax.set_xlim(0, max_freq or fs / 2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("dB (re. peak)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(signals) > 1:
        ax.legend(fontsize=8)
    return ax.figure


def plot_signal_and_spectrum(x, fs: float, title: str = "", n_peaks: int = 5,
                             max_time: Optional[float] = None,
                             max_freq: Optional[float] = None,
                             save_path: Optional[str] = None):
    x = np.asarray(x, dtype=float)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6))
    t = np.arange(len(x)) / fs
    limit = max_time if max_time is not None else (t[-1] if len(t) else 0.0)
    show = t <= limit
    a1.plot(t[show], x[show], lw=0.8)
    a1.set_xlabel("time (s)")
    a1.set_ylabel("amplitude")
    a1.set_title(title or "signal")
    a1.grid(True, alpha=0.3)
    plot_spectrum({"": x}, fs, ax=a2, max_freq=max_freq, title="spectrum (Hann window)")
    for p in peak_frequencies(x, fs, n_peaks):
        a2.axvline(p.frequency, color="#d62728", lw=0.8, ls="--")
        a2.text(p.frequency, -5, f"{p.frequency:.1f} Hz", rotation=90, fontsize=7,
                va="top", ha="right", color="#d62728")
    return _finish(fig, save_path)


def plot_spectrogram(x, fs: float, nperseg: int = 512, overlap: float = 0.75,
                     title: str = "Spectrogram", ax=None, max_freq: Optional[float] = None,
                     dynamic_range: float = 80.0, save_path: Optional[str] = None):
    times, freqs, power = spectrogram(x, fs, nperseg, overlap)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    top = power.max()
    im = ax.pcolormesh(times, freqs, power.T, shading="auto", cmap="magma",
                       vmin=top - dynamic_range, vmax=top)
    ax.figure.colorbar(im, ax=ax, label="power (dB)")
    ax.set_ylim(0, max_freq or fs / 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(title)
    return _finish(ax.figure, save_path) if own else ax.figure


def plot_fft2_heatmap(image, title: str = "", save_path: Optional[str] = None, axes=None):
    img = np.asarray(image, dtype=float)
    spec = log_magnitude_spectrum(img)
    own = axes is None
    if own:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(title or "image")
    h, w = img.shape
    im = axes[1].imshow(spec, cmap="inferno",
                        extent=(-0.5, 0.5, 0.5, -0.5))
    axes[1].set_title("log |FFT2| (centred)")
    axes[1].set_xlabel("horizontal frequency (cycles/px)")
    axes[1].set_ylabel("vertical frequency (cycles/px)")
    axes[1].figure.colorbar(im, ax=axes[1], fraction=0.046)
    axes[0].axis("off")
    return _finish(axes[0].figure, save_path) if own else axes[0].figure


def plot_filter_response(responses: Dict[str, tuple], title: str = "Filter response",
                         ax=None, floor_db: float = -120, save_path: Optional[str] = None):
    """responses: {label: (freqs, gain_db)}."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 4))
    for label, (f, db) in responses.items():
        ax.plot(f, np.maximum(db, floor_db), lw=1.2, label=label)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("gain (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(ax.figure, save_path) if own else ax.figure


def plot_windows(n: int = 64, save_path: Optional[str] = None, axes=None):
    own = axes is None
    if own:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in WINDOWS:
        axes[0].plot(window(name, n), label=name)
        f, db = window_spectrum(name, n)
        axes[1].plot(f, db, lw=1, label=name)
    axes[0].set_title(f"Window shapes (n = {n})")
    axes[0].set_xlabel("sample")
    axes[1].set_xlim(-12, 12)
    axes[1].set_ylim(-120, 5)
    axes[1].set_title("Spectral leakage: window spectra")
    axes[1].set_xlabel("frequency offset (bins)")
    axes[1].set_ylabel("dB")
    for a in axes:
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    return _finish(axes[0].figure, save_path) if own else axes[0].figure


def plot_condition_numbers(sizes: Sequence[int], kappas: Dict[str, Sequence[float]],
                           errors: Optional[Dict[str, Sequence[float]]] = None,
                           title: str = "Condition number and solve error",
                           save_path: Optional[str] = None, axes=None):
    """κ(A) per family (left) and observed relative error vs κ (right)."""
    own = axes is None
    if own:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, ks in kappas.items():
        axes[0].semilogy(sizes, ks, "o-", label=label)
    axes[0].axhline(1 / np.finfo(float).eps, color="k", ls=":", lw=1,
                    label="1/ε (numerically singular)")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("κ₂(A)")
    axes[0].set_title("Condition number vs size")
    if errors:
        for label, errs in errors.items():
            axes[1].loglog(kappas[label], errs, "o", label=label)
        allk = np.concatenate([np.asarray(v, float) for v in kappas.values()])
        ks = np.logspace(0, np.log10(max(allk.max(), 10)), 50)
        axes[1].loglog(ks, np.minimum(ks * np.finfo(float).eps, 10), "k:",
                       label="κ·ε")
        axes[1].set_xlabel("κ₂(A)")
        axes[1].set_ylabel("relative error of x in Ax = b")
        axes[1].set_title("Error grows with κ")
    for a in axes:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    axes[0].figure.suptitle(title, fontweight="bold")
    return _finish(axes[0].figure, save_path)


def plot_compression_grid(image, results: Sequence[Dict], recon: Sequence[np.ndarray],
                          save_path: Optional[str] = None, crop: Optional[slice] = None):
    """Original + one panel per quality setting."""
    n = len(recon) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    crop = crop or (slice(None), slice(None))
    axes[0].imshow(np.asarray(image)[crop], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original (8 bpp)")
    for ax, r, img in zip(axes[1:], results, recon):
        ax.imshow(np.asarray(img)[crop], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"q={r['quality']}: {r['psnr']:.1f} dB\n"
                     f"{r['bits_per_pixel']:.2f} bpp, SSIM {r['ssim']:.3f}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    return _finish(fig, save_path)
