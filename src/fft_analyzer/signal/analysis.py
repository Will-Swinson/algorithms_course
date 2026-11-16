"""
Spectral analysis: periodograms, power spectral density, spectrograms
and peak frequencies.

    periodogram     |FFT(w·x)|² scaled to a one-sided PSD (units²/Hz):
                    Σ PSD·Δf equals the signal's mean power (Parseval),
                    whatever the window
    welch_psd       average periodograms of overlapping, windowed
                    segments: the variance of a single periodogram does
                    not shrink as n grows (every bin is a χ²₂ variable),
                    averaging K segments divides it by ~K at the cost of
                    frequency resolution fs/nperseg
    stft            short-time Fourier transform: FFTs of overlapping
                    windowed frames → time × frequency matrix
    istft           overlap-add inverse (exact for windows satisfying the
                    COLA condition, e.g. periodic Hann at 50% / 75%
                    overlap), used for spectral denoising
    spectrogram     |STFT|² in dB
    peak_frequencies  local maxima of the spectrum above a threshold,
                    refined by fitting a parabola through the log
                    magnitude of the peak bin and its neighbours — the
                    estimate is accurate to a fraction of a bin
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.fft_analyzer.core.fft import fft, ifft, next_power_of_two, rfftfreq
from src.fft_analyzer.signal.filters import window


def _one_sided(X: np.ndarray, n: int) -> np.ndarray:
    """|X|² of the non-negative bins, doubling those that have a
    negative-frequency twin."""
    P = np.abs(X[..., : n // 2 + 1]) ** 2
    if n % 2 == 0:
        P[..., 1:-1] *= 2
    else:
        P[..., 1:] *= 2
    return P


def periodogram(x, fs: float = 1.0, window_name: str = "rectangular",
                nfft: Optional[int] = None, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density: (frequencies, PSD)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("empty signal")
    if detrend:
        x = x - x.mean()
    w = window(window_name, n, symmetric=False)
    nfft = nfft or n
    seg = np.zeros(nfft)
    seg[:n] = x * w
    psd = _one_sided(fft(seg), nfft) / (fs * np.sum(w * w))
    return rfftfreq(nfft, 1 / fs), psd


def welch_psd(x, fs: float = 1.0, nperseg: int = 256, overlap: float = 0.5,
              window_name: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Welch's averaged periodogram."""
    x = np.asarray(x, dtype=float)
    nperseg = min(nperseg, len(x))
    if nperseg < 1:
        raise ValueError("empty signal")
    frames = _frames(x - x.mean(), nperseg, _hop(nperseg, overlap))
    w = window(window_name, nperseg, symmetric=False)
    P = _one_sided(fft(frames * w), nperseg) / (fs * np.sum(w * w))
    return rfftfreq(nperseg, 1 / fs), P.mean(axis=0)


def _hop(nperseg: int, overlap: float) -> int:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    return max(1, int(round(nperseg * (1 - overlap))))


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    count = 1 + max(0, (len(x) - size) // hop)
    idx = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


@dataclass
class STFTResult:
    times: np.ndarray            # frame centres (s)
    freqs: np.ndarray            # Hz, 0 … fs/2
    Z: np.ndarray                # complex, shape (frames, freqs)
    fs: float
    nperseg: int
    hop: int
    window_name: str
    length: int                  # original signal length

    @property
    def power_db(self) -> np.ndarray:
        return 10 * np.log10(np.abs(self.Z) ** 2 + 1e-20)


def stft(x, fs: float = 1.0, nperseg: int = 256, overlap: float = 0.75,
         window_name: str = "hann", pad: bool = True) -> STFTResult:
    """Short-time FFT. With pad=True the signal is zero-padded by half a
    frame at both ends so every sample is covered by full windows."""
    x = np.asarray(x, dtype=float)
    hop = _hop(nperseg, overlap)
    half = nperseg // 2 if pad else 0
    padded = np.concatenate([np.zeros(half), x, np.zeros(half + nperseg)])
    frames = _frames(padded, nperseg, hop)
    w = window(window_name, nperseg, symmetric=False)
    Z = fft(frames * w)[:, : nperseg // 2 + 1]
    times = (np.arange(len(frames)) * hop + nperseg / 2 - half) / fs
    return STFTResult(times, rfftfreq(nperseg, 1 / fs), Z, fs, nperseg, hop,
                      window_name, len(x))


def istft(result: STFTResult) -> np.ndarray:
    """Weighted overlap-add inverse of stft(pad=True)."""
    n, hop = result.nperseg, result.hop
    w = window(result.window_name, n, symmetric=False)
    full = np.zeros((result.Z.shape[0], n), dtype=complex)
    full[:, : n // 2 + 1] = result.Z
    # rebuild the negative frequencies from conjugate symmetry
    tail = result.Z[:, 1: (n + 1) // 2][:, ::-1]
    full[:, n // 2 + 1:] = np.conj(tail)
    frames = ifft(full).real
    total = (len(frames) - 1) * hop + n
    out = np.zeros(total)
    norm = np.zeros(total)
    for k, frame in enumerate(frames):
        out[k * hop: k * hop + n] += frame * w
        norm[k * hop: k * hop + n] += w * w
    out = np.divide(out, norm, out=np.zeros_like(out), where=norm > 1e-10)
    half = n // 2
    return out[half: half + result.length]


def spectrogram(x, fs: float = 1.0, nperseg: int = 256, overlap: float = 0.75,
                window_name: str = "hann") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, freqs, power in dB) — rows are time frames."""
    r = stft(x, fs, nperseg, overlap, window_name)
    return r.times, r.freqs, r.power_db


@dataclass
class Peak:
    frequency: float             # Hz, interpolated
    magnitude: float             # amplitude estimate of a sinusoid
    bin: int
    power_db: float


def peak_frequencies(x, fs: float = 1.0, n_peaks: int = 5,
                     window_name: str = "hann", threshold_db: float = -40.0,
                     min_separation: float = 0.0, zero_pad: int = 1) -> List[Peak]:
    """
    The strongest spectral peaks: local maxima within threshold_db of the
    largest, at least min_separation Hz apart, each refined by parabolic
    interpolation of the log magnitude.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return []
    w = window(window_name, n, symmetric=False)
    nfft = next_power_of_two(n * max(1, zero_pad))
    seg = np.zeros(nfft)
    seg[:n] = (x - x.mean()) * w
    mag = np.abs(fft(seg)[: nfft // 2 + 1])
    db = 20 * np.log10(mag + 1e-300)
    top = db.max()
    candidates = [k for k in range(1, len(mag) - 1)
                  if mag[k] >= mag[k - 1] and mag[k] > mag[k + 1]
                  and db[k] >= top + threshold_db]
    candidates.sort(key=lambda k: -mag[k])
    peaks: List[Peak] = []
    df = fs / nfft
    for k in candidates:
        a, b, c = db[k - 1], db[k], db[k + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom else 0.0
        freq = (k + delta) * df
        if any(abs(freq - p.frequency) < min_separation for p in peaks):
            continue
        peak_db = b - 0.25 * (a - c) * delta
        amplitude = 2 * 10 ** (peak_db / 20) / w.sum()
        peaks.append(Peak(float(freq), float(amplitude), k, float(peak_db)))
        if len(peaks) == n_peaks:
            break
    return peaks


def dominant_frequency(x, fs: float = 1.0) -> float:
    peaks = peak_frequencies(x, fs, n_peaks=1)
    return peaks[0].frequency if peaks else 0.0


def band_power(x, fs: float, low: float, high: float) -> float:
    """Mean power in [low, high] Hz from the periodogram."""
    f, psd = periodogram(x, fs)
    sel = (f >= low) & (f <= high)
    return float(np.sum(psd[sel]) * (f[1] - f[0])) if len(f) > 1 else 0.0


def spectral_centroid(x, fs: float = 1.0) -> float:
    f, psd = periodogram(x, fs)
    total = psd.sum()
    return float((f * psd).sum() / total) if total else 0.0
