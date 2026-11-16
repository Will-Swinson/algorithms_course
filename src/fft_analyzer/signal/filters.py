"""
Frequency-domain filters and window functions.

FFT filtering. Transform, multiply by a gain mask H(f), transform back:

    y = IFFT( FFT(x) · H(f) )

    lowpass   H = 1 for |f| ≤ fc
    highpass  H = 1 for |f| ≥ fc
    bandpass  H = 1 for f_lo ≤ |f| ≤ f_hi
    bandstop  the complement (e.g. removing 50/60 Hz mains hum)

The mask is symmetric in ±f, so a real signal stays real. A brick-wall
mask (transition=0) is "ideal" but its impulse response is a sinc, so
sharp edges ring (Gibbs phenomenon) and FFT filtering is circular —
energy from one end of the signal can wrap to the other. A raised-
cosine transition band of width `transition` Hz softens the mask and
the ringing; `pad` (on by default) zero-pads to suppress wrap-around.

Windows. Before an FFT, a finite signal is implicitly multiplied by a
rectangle; its sidelobes (−13 dB) leak energy from strong tones into
every other bin. Tapered windows trade a wider main lobe for much lower
sidelobes:

    window       first sidelobe   main-lobe width (bins)
    rectangular     −13 dB             2
    hann            −31 dB             4
    hamming         −43 dB             4
    blackman        −58 dB             6

FIR design by the window method: truncate the ideal sinc low-pass
impulse response to `numtaps` and multiply by a window — the same
trade-off, now between transition width and stopband attenuation.
apply_fir convolves with our FFT-based linear convolution.
"""
from typing import Tuple

import numpy as np

from src.fft_analyzer.core.fft import fft, fftfreq, ifft, linear_convolution, next_power_of_two

WINDOWS = ("rectangular", "hann", "hamming", "blackman")


# ----------------------------------------------------------------------
# windows
# ----------------------------------------------------------------------
def window(name: str, n: int, symmetric: bool = True) -> np.ndarray:
    """
    Window of length n. symmetric=True is the filter-design form
    (denominator n − 1); False is the periodic form for spectral
    analysis (denominator n).
    """
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    m = n - 1 if symmetric else n
    k = np.arange(n)
    if name == "rectangular":
        return np.ones(n)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / m)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * k / m)
    if name == "blackman":
        return (0.42 - 0.5 * np.cos(2 * np.pi * k / m)
                + 0.08 * np.cos(4 * np.pi * k / m))
    raise ValueError(f"unknown window {name!r}; choose from {WINDOWS}")


def hann(n: int) -> np.ndarray:
    return window("hann", n)


def hamming(n: int) -> np.ndarray:
    return window("hamming", n)


def blackman(n: int) -> np.ndarray:
    return window("blackman", n)


def window_spectrum(name: str, n: int = 64, oversample: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """(frequency in bins, magnitude in dB re. peak) of a window."""
    w = window(name, n, symmetric=False)
    size = next_power_of_two(n * oversample)
    W = np.abs(fft(np.concatenate([w, np.zeros(size - n)])))
    W = np.fft.fftshift(W)
    freq = (np.arange(size) - size // 2) * n / size
    db = 20 * np.log10(np.maximum(W / W.max(), 1e-12))
    return freq, db


def window_properties(name: str, n: int = 64) -> dict:
    """Highest sidelobe (dB), main-lobe half width (bins), coherent gain."""
    freq, db = window_spectrum(name, n)
    centre = len(db) // 2
    right = db[centre:]
    # main lobe ends at the first local minimum to the right of the peak
    i = 1
    while i < len(right) - 1 and right[i + 1] < right[i]:
        i += 1
    return {"window": name, "sidelobe_db": float(right[i:].max()),
            "mainlobe_halfwidth_bins": float(freq[centre + i]),
            "coherent_gain": float(window(name, n, False).mean())}


# ----------------------------------------------------------------------
# FFT filters
# ----------------------------------------------------------------------
def _taper(f_abs: np.ndarray, edge: float, width: float, rising: bool) -> np.ndarray:
    """Raised-cosine step at `edge` (0 → 1 if rising) over `width` Hz."""
    if width <= 0:
        return (f_abs >= edge).astype(float) if rising else (f_abs <= edge).astype(float)
    z = np.clip((f_abs - (edge - width / 2)) / width, 0.0, 1.0)
    up = 0.5 - 0.5 * np.cos(np.pi * z)
    return up if rising else 1.0 - up


def frequency_mask(n: int, fs: float, kind: str, cutoff, transition: float = 0.0) -> np.ndarray:
    """Gain H(f) for the n FFT bins of a signal sampled at fs."""
    f = np.abs(fftfreq(n, 1 / fs))
    nyq = fs / 2
    if kind in ("lowpass", "highpass"):
        fc = float(cutoff)
        if not 0 < fc < nyq:
            raise ValueError(f"cutoff must be in (0, {nyq}) Hz")
        return _taper(f, fc, transition, rising=kind == "highpass")
    if kind in ("bandpass", "bandstop"):
        lo, hi = map(float, cutoff)
        if not 0 <= lo < hi <= nyq:
            raise ValueError(f"need 0 <= low < high <= {nyq} Hz")
        band = _taper(f, lo, transition, True) * _taper(f, hi, transition, False)
        return band if kind == "bandpass" else 1.0 - band
    raise ValueError("kind must be lowpass, highpass, bandpass or bandstop")


def fft_filter(x, fs: float, kind: str, cutoff, transition: float = 0.0,
               pad: bool = True) -> np.ndarray:
    """Apply an FFT-domain filter to a 1-D real signal (or each row)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if n == 0:
        return x.copy()
    size = next_power_of_two(2 * n) if pad else n
    padded = np.zeros(x.shape[:-1] + (size,))
    padded[..., :n] = x
    H = frequency_mask(size, fs, kind, cutoff, transition)
    return ifft(fft(padded) * H).real[..., :n]


def lowpass(x, fs: float, cutoff: float, transition: float = 0.0, pad: bool = True):
    return fft_filter(x, fs, "lowpass", cutoff, transition, pad)


def highpass(x, fs: float, cutoff: float, transition: float = 0.0, pad: bool = True):
    return fft_filter(x, fs, "highpass", cutoff, transition, pad)


def bandpass(x, fs: float, low: float, high: float, transition: float = 0.0,
             pad: bool = True):
    return fft_filter(x, fs, "bandpass", (low, high), transition, pad)


def bandstop(x, fs: float, low: float, high: float, transition: float = 0.0,
             pad: bool = True):
    return fft_filter(x, fs, "bandstop", (low, high), transition, pad)


def refine_frequency(x, fs: float, guess: float, search: float = 2.0,
                     zero_pad: int = 8) -> float:
    """Frequency of the strongest component within ±search Hz of guess:
    peak of the zero-padded Hann spectrum, refined by a parabola through
    the log magnitudes."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    size = next_power_of_two(n * zero_pad)
    seg = np.zeros(size)
    seg[:n] = (x - x.mean()) * window("hann", n, symmetric=False)
    mag = np.abs(fft(seg)[: size // 2 + 1])
    df = fs / size
    lo = max(1, int((guess - search) / df))
    hi = min(len(mag) - 2, int((guess + search) / df) + 1)
    if hi <= lo:
        return guess
    k = lo + int(np.argmax(mag[lo:hi + 1]))
    a, b, c = np.log(mag[k - 1:k + 2] + 1e-300)
    denom = a - 2 * b + c
    return float((k + (0.5 * (a - c) / denom if denom else 0.0)) * df)


def subtract_sinusoids(x, fs: float, freqs) -> np.ndarray:
    """Remove sinusoids of the given frequencies by a least-squares fit
    of a·cos + b·sin for each (one QR solve for all of them)."""
    from src.fft_analyzer.core.matrix import least_squares
    x = np.asarray(x, dtype=float)
    if not len(freqs):
        return x.copy()
    t = np.arange(len(x)) / fs
    cols = []
    for f in freqs:
        cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    M = np.stack(cols, axis=1)
    return x - M @ least_squares(M, x)


def notch_harmonics(x, fs: float, base: float, harmonics: int = 5,
                    width: float = 2.0, method: str = "subtract") -> np.ndarray:
    """
    Remove a tone and its harmonics (mains hum at 50/60 Hz).

    method="subtract" (default): refine each harmonic's frequency from
    the spectrum, fit its amplitude and phase by least squares and
    subtract it. A steady hum is removed almost exactly, with no edge
    transients.
    method="fft": band-stop each harmonic (±width/2 Hz) with the FFT
    filter. Simple, but on a clip whose length is not a whole number of
    hum periods the abrupt start and end of the tone leak through as
    transients at the edges.
    """
    y = np.asarray(x, dtype=float)
    targets = [base * h for h in range(1, harmonics + 1)
               if base * h + width / 2 < fs / 2]
    if method == "subtract":
        freqs = [refine_frequency(y, fs, f, search=width) for f in targets]
        return subtract_sinusoids(y, fs, freqs)
    if method != "fft":
        raise ValueError("method must be 'subtract' or 'fft'")
    for f in targets:
        y = bandstop(y, fs, max(0.0, f - width / 2), f + width / 2)
    return y


# ----------------------------------------------------------------------
# FIR filters by the window method
# ----------------------------------------------------------------------
def fir_lowpass(numtaps: int, cutoff: float, fs: float,
                window_name: str = "hamming") -> np.ndarray:
    """Windowed-sinc low-pass FIR taps (odd numtaps → linear phase,
    unit DC gain)."""
    if numtaps < 1:
        raise ValueError("numtaps must be >= 1")
    fc = cutoff / fs
    m = np.arange(numtaps) - (numtaps - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * m)
    h *= window(window_name, numtaps)
    return h / h.sum()


def fir_highpass(numtaps: int, cutoff: float, fs: float,
                 window_name: str = "hamming") -> np.ndarray:
    """Spectral inversion of a low-pass (numtaps must be odd)."""
    if numtaps % 2 == 0:
        raise ValueError("a high-pass FIR needs an odd number of taps")
    h = -fir_lowpass(numtaps, cutoff, fs, window_name)
    h[numtaps // 2] += 1.0
    return h


def apply_fir(x, taps) -> np.ndarray:
    """Filter with FIR taps; output aligned with the input (delay
    (numtaps − 1)/2 removed)."""
    return linear_convolution(np.asarray(x, dtype=float), np.asarray(taps), mode="same")


def frequency_response(taps, fs: float = 1.0, n: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """(frequencies 0…fs/2, |H| in dB) of FIR taps."""
    taps = np.asarray(taps, dtype=float)
    size = next_power_of_two(max(n, len(taps)))
    H = fft(np.concatenate([taps, np.zeros(size - len(taps))]))[: size // 2 + 1]
    f = np.arange(size // 2 + 1) * fs / size
    return f, 20 * np.log10(np.maximum(np.abs(H), 1e-12))


def stopband_attenuation(taps, cutoff: float, transition: float,
                         fs: float = 1.0) -> float:
    """Worst gain (dB) above cutoff + transition."""
    f, db = frequency_response(taps, fs)
    stop = f >= cutoff + transition
    return float(db[stop].max()) if stop.any() else float("-inf")


def ringing(step_response) -> float:
    """Overshoot of a step response above its final value (Gibbs)."""
    y = np.asarray(step_response, dtype=float)
    final = np.median(y[-max(1, len(y) // 10):])
    return float(y.max() - final) if final else float(y.max())


def measure_attenuation(x_filtered, x_original, fs: float, band: Tuple[float, float]) -> float:
    """Power reduction (dB) of a frequency band after filtering."""
    def band_power(x):
        X = np.abs(fft(np.asarray(x, dtype=float))) ** 2
        f = np.abs(fftfreq(len(x), 1 / fs))
        sel = (f >= band[0]) & (f <= band[1])
        return X[sel].sum()
    before, after = band_power(x_original), band_power(x_filtered)
    if before == 0:
        return 0.0
    return float(10 * np.log10(max(after, 1e-300) / before))
