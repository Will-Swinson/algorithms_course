"""
Test-signal generators. Every function takes a duration (seconds) and
a sampling rate fs (Hz) and returns (t, x) unless noted.

    sine          A·sin(2πft + φ)
    multitone     sum of sines (frequencies, amplitudes, phases)
    chirp         frequency sweeping from f0 to f1, linearly or
                  exponentially (phase = 2π∫f(t)dt, so the instantaneous
                  frequency really is f(t))
    square, sawtooth  band-unlimited shapes (harmonics at every odd /
                  every multiple of f — useful for spotting aliasing)
    white_noise   flat spectrum (Gaussian samples)
    pink_noise    power ∝ 1/f: white noise shaped by 1/√f in the
                  frequency domain (our FFT), so every octave carries equal
                  power — the spectrum of many natural signals
    add_noise     signal + white noise at a target SNR in dB
"""
from typing import Optional, Sequence, Tuple

import numpy as np

from src.fft_analyzer.core.fft import fft, fftfreq, ifft, rfftfreq


def time_axis(duration: float, fs: float) -> np.ndarray:
    if duration < 0 or fs <= 0:
        raise ValueError("duration must be >= 0 and fs > 0")
    return np.arange(int(round(duration * fs))) / fs


def sine(freq: float, duration: float, fs: float, amplitude: float = 1.0,
         phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    t = time_axis(duration, fs)
    return t, amplitude * np.sin(2 * np.pi * freq * t + phase)


def multitone(freqs: Sequence[float], duration: float, fs: float,
              amplitudes: Optional[Sequence[float]] = None,
              phases: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    t = time_axis(duration, fs)
    amplitudes = [1.0] * len(freqs) if amplitudes is None else amplitudes
    phases = [0.0] * len(freqs) if phases is None else phases
    if not len(freqs) == len(amplitudes) == len(phases):
        raise ValueError("freqs, amplitudes and phases must have equal length")
    x = np.zeros_like(t)
    for f, a, p in zip(freqs, amplitudes, phases):
        x += a * np.sin(2 * np.pi * f * t + p)
    return t, x


def chirp(f0: float, f1: float, duration: float, fs: float,
          method: str = "linear", amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous frequency f0 → f1 over `duration`."""
    t = time_axis(duration, fs)
    if method == "linear":
        k = (f1 - f0) / duration
        phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    elif method == "exponential":
        if f0 <= 0 or f1 <= 0:
            raise ValueError("exponential chirp needs positive frequencies")
        k = (f1 / f0) ** (1 / duration)
        phase = 2 * np.pi * f0 * (k ** t - 1) / np.log(k) if k != 1 else 2 * np.pi * f0 * t
    else:
        raise ValueError("method must be 'linear' or 'exponential'")
    return t, amplitude * np.sin(phase)


def instantaneous_frequency(f0: float, f1: float, duration: float, t,
                            method: str = "linear") -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if method == "linear":
        return f0 + (f1 - f0) * t / duration
    return f0 * (f1 / f0) ** (t / duration)


def square(freq: float, duration: float, fs: float,
           amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    t = time_axis(duration, fs)
    return t, amplitude * np.where((freq * t) % 1.0 < 0.5, 1.0, -1.0)


def sawtooth(freq: float, duration: float, fs: float,
             amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    t = time_axis(duration, fs)
    return t, amplitude * (2 * ((freq * t) % 1.0) - 1)


def white_noise(n: int, std: float = 1.0, seed=None) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, std, n)


def pink_noise(n: int, std: float = 1.0, seed=None) -> np.ndarray:
    """1/f noise: shape white noise's spectrum by 1/√f (DC removed),
    then rescale to the requested standard deviation."""
    if n == 0:
        return np.zeros(0)
    white = white_noise(n, 1.0, seed)
    spectrum = fft(white)
    f = np.abs(fftfreq(n))                # bin frequencies, cycles/sample
    scale = np.zeros(n)
    scale[f > 0] = 1 / np.sqrt(f[f > 0])
    pink = ifft(spectrum * scale).real
    sd = pink.std()
    return pink * (std / sd) if sd > 0 else pink


def add_noise(x, snr_db: float, seed=None, kind: str = "white") -> np.ndarray:
    """x plus noise scaled so 10·log₁₀(P_signal / P_noise) = snr_db."""
    x = np.asarray(x, dtype=float)
    power = np.mean(x ** 2)
    noise_power = power / 10 ** (snr_db / 10)
    gen = white_noise if kind == "white" else pink_noise
    return x + gen(len(x), np.sqrt(noise_power), seed)


def snr_db(clean, noisy) -> float:
    clean, noisy = np.asarray(clean, dtype=float), np.asarray(noisy, dtype=float)
    noise = np.mean((noisy - clean) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(np.mean(clean ** 2) / noise))


def spectral_slope(x, fs: float = 1.0) -> float:
    """Slope of log power vs log frequency (−1 for pink, 0 for white)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    p = np.abs(fft(x)[: n // 2 + 1]) ** 2
    f = rfftfreq(n, 1 / fs)
    keep = f > 0
    return float(np.polyfit(np.log(f[keep]), np.log(p[keep] + 1e-300), 1)[0])
