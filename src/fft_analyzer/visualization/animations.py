"""
Animations (GIF via matplotlib's PillowWriter).

    animate_fft_stages   the iterative Cooley–Tukey FFT stage by stage:
                         left, the butterfly wiring of the current stage
                         (which pairs of elements combine, twiddle
                         exponents labelled); right, the magnitudes of the
                         working array — from the bit-reversed input to
                         the final spectrum, checked against numpy
    animate_filtering    a low-pass cutoff sweeping downward: the spectrum
                         with the removed band shaded, and the time signal
                         losing its high-frequency detail frame by frame
    fft_stage_frames / filtering_frames return the per-frame data, so the
    same content can be drawn as static panels
"""
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.fft_analyzer.core.fft import bit_reverse_indices, fft, fft_iterative, rfftfreq
from src.fft_analyzer.signal.filters import lowpass


def fft_stage_frames(x) -> List[Dict]:
    """One dict per stage: the working array and the butterflies
    (top, bottom, twiddle exponent k of ω_span^k) applied to reach it."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    _, stages = fft_iterative(x, record=True)
    frames = [{"stage": 0, "values": stages[0], "butterflies": [],
               "title": f"input in bit-reversed order {bit_reverse_indices(n).tolist()}"
               if n <= 16 else "input in bit-reversed order"}]
    span = 2
    for s, values in enumerate(stages[1:], start=1):
        half = span // 2
        flies = [(start + j, start + j + half, j) for start in range(0, n, span)
                 for j in range(half)]
        frames.append({"stage": s, "values": values, "butterflies": flies,
                       "span": span,
                       "title": f"stage {s}: {n // span} block(s) of {span}, "
                                f"twiddles ω_{span}^k"})
        span *= 2
    return frames


def _draw_stage(ax_w, ax_v, frame: Dict, n: int, final: np.ndarray, top: float):
    ax_w.clear()
    ax_v.clear()
    ys = np.arange(n)
    ax_w.scatter(np.zeros(n), ys, s=18, color="#555555", zorder=3)
    ax_w.scatter(np.ones(n), ys, s=18, color="#1f77b4", zorder=3)
    for a, b, k in frame["butterflies"]:
        for src in (a, b):
            for dst in (a, b):
                ax_w.plot([0, 1], [src, dst], color="#ff7f0e" if src == b else "#1f77b4",
                          lw=0.9, alpha=0.8)
        if n <= 16:
            ax_w.text(-0.06, b, f"×ω^{k}", fontsize=7, ha="right", va="center",
                      color="#ff7f0e")
    ax_w.set_ylim(n - 0.5, -0.5)
    ax_w.set_xlim(-0.35, 1.1)
    ax_w.axis("off")
    ax_w.set_title("butterflies (orange: multiplied by twiddle)", fontsize=9)
    ax_v.bar(ys, np.abs(frame["values"]), color="#1f77b4", label="|working array|")
    if frame["stage"] == len(np.binary_repr(n)) - 1:
        ax_v.plot(ys, np.abs(final), "r_", markersize=12, mew=2, label="|numpy FFT|")
    ax_v.set_ylim(0, top * 1.1)
    ax_v.set_xlabel("index")
    ax_v.legend(fontsize=8, loc="upper right")
    ax_v.grid(True, alpha=0.3)


def animate_fft_stages(x=None, n: int = 16, save_path: Optional[str] = None,
                       interval: int = 1400):
    """Animate the iterative FFT of x (default: two tones, n points)."""
    if x is None:
        t = np.arange(n)
        x = np.sin(2 * np.pi * 2 * t / n) + 0.5 * np.cos(2 * np.pi * 5 * t / n)
    x = np.asarray(x, dtype=complex)
    n = len(x)
    frames = fft_stage_frames(x)
    final = np.fft.fft(x)
    top = max(np.abs(f["values"]).max() for f in frames)
    fig, (ax_w, ax_v) = plt.subplots(1, 2, figsize=(11, 5),
                                     gridspec_kw={"width_ratios": [1, 1.4]})

    def draw(i):
        _draw_stage(ax_w, ax_v, frames[i], n, final, top)
        fig.suptitle(f"Iterative Cooley–Tukey FFT, n = {n} — {frames[i]['title']}",
                     fontsize=10)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=interval)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=max(1, round(1000 / interval))))
        plt.close(fig)
    return anim


def filtering_frames(x, fs: float, cutoffs: Sequence[float]) -> List[Dict]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    freqs = rfftfreq(n, 1 / fs)
    frames = []
    for fc in cutoffs:
        y = lowpass(x, fs, fc, transition=fc * 0.05)
        Y = np.abs(fft(y))[: n // 2 + 1]
        frames.append({"cutoff": fc, "signal": y, "spectrum": Y, "freqs": freqs,
                       "energy_kept": float(np.sum(y ** 2) / max(np.sum(x ** 2), 1e-300))})
    return frames


def animate_filtering(x, fs: float, cutoffs: Optional[Sequence[float]] = None,
                      max_time: Optional[float] = None, save_path: Optional[str] = None,
                      interval: int = 250):
    """Low-pass with a cutoff sweeping from fs/2 down: which frequencies
    are removed, and what that does to the waveform."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    cutoffs = list(cutoffs) if cutoffs is not None else \
        list(np.geomspace(fs / 2 * 0.98, fs / 100, 40))
    frames = filtering_frames(x, fs, cutoffs)
    X = np.abs(fft(x))[: n // 2 + 1]
    t = np.arange(n) / fs
    show = t <= (max_time or min(t[-1], 0.05))
    fig, (ax_s, ax_t) = plt.subplots(2, 1, figsize=(9, 6.5))
    peak = X.max()

    def draw(i):
        fr = frames[i]
        ax_s.clear()
        ax_t.clear()
        ax_s.semilogy(fr["freqs"], X / peak + 1e-6, color="#bbbbbb", lw=0.8,
                      label="original")
        ax_s.semilogy(fr["freqs"], fr["spectrum"] / peak + 1e-6, color="#1f77b4",
                      lw=1, label="filtered")
        ax_s.axvspan(fr["cutoff"], fs / 2, color="#d62728", alpha=0.12,
                     label="removed band")
        ax_s.set_xlim(0, fs / 2)
        ax_s.set_ylim(1e-6, 2)
        ax_s.set_xlabel("frequency (Hz)")
        ax_s.set_ylabel("|X| (re. peak)")
        ax_s.legend(fontsize=8, loc="upper right")
        ax_s.set_title(f"low-pass cutoff {fr['cutoff']:.0f} Hz — "
                       f"{100 * fr['energy_kept']:.1f}% of the energy kept")
        ax_t.plot(t[show], x[show], color="#bbbbbb", lw=0.8, label="original")
        ax_t.plot(t[show], fr["signal"][show], color="#1f77b4", lw=1.1, label="filtered")
        ax_t.set_xlabel("time (s)")
        ax_t.legend(fontsize=8, loc="upper right")
        for a in (ax_s, ax_t):
            a.grid(True, alpha=0.3)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=interval)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=max(1, round(1000 / interval))))
        plt.close(fig)
    return anim
