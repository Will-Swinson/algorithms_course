"""
Time-series analytics in the frequency domain.

    autocorrelation      Wiener–Khinchin: the autocorrelation is the
                         inverse FFT of the power spectrum. Zero-padding
                         to ≥ 2n makes it the linear (not circular) ACF in
                         O(n log n); autocorrelation_direct is the O(n²)
                         reference
    detect_periodicities periods from the strongest periodogram peaks
                         (after removing a linear trend), each confirmed by
                         the ACF value at that lag
    seasonal_strength    share of detrended variance explained by the
                         average seasonal profile of a given period
                         (0 = none, 1 = perfectly periodic)
    smooth               convolution-based smoothing (moving average,
                         Gaussian, or Hann window) via FFT convolution,
                         with edge handling by reflection
    detrend              least-squares polynomial trend via our
                         Householder QR (well-conditioned on a scaled time
                         axis)
    synthetic_series     trend + weekly + yearly seasonality + AR(1) noise
                         (+ optional random-walk "price" form)

TimeSeriesAnalyzer bundles them for one series with a sampling interval.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.fft_analyzer.core.fft import fft, ifft, linear_convolution, next_power_of_two
from src.fft_analyzer.core.matrix import least_squares
from src.fft_analyzer.signal.analysis import periodogram
from src.fft_analyzer.signal.filters import window


def autocorrelation(x, max_lag: Optional[int] = None, normalize: bool = True) -> np.ndarray:
    """ACF r_k for k = 0..max_lag via |FFT|² (biased estimator)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros(0)
    x = x - x.mean()
    size = next_power_of_two(2 * n)
    X = fft(np.concatenate([x, np.zeros(size - n)]))
    r = ifft(np.abs(X) ** 2).real[:n] / n
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    r = r[: max_lag + 1]
    return r / r[0] if normalize and r[0] > 0 else r


def autocorrelation_direct(x, max_lag: Optional[int] = None, normalize: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    r = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(max_lag + 1)])
    return r / r[0] if normalize and r[0] > 0 else r


def detrend(x, degree: int = 1) -> np.ndarray:
    """Remove a least-squares polynomial trend (QR-based)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= degree:
        return x - x.mean()
    t = np.linspace(-1, 1, n)                      # scaled axis: small κ
    V = t[:, None] ** np.arange(degree + 1)[None, :]
    coef = least_squares(V, x)
    return x - V @ coef


def smooth(x, width: int, kind: str = "gaussian") -> np.ndarray:
    """Convolution smoothing with a normalized kernel of `width` samples."""
    x = np.asarray(x, dtype=float)
    if width <= 1 or len(x) == 0:
        return x.copy()
    if kind == "moving_average":
        k = np.ones(width)
    elif kind == "gaussian":
        sigma = width / 6
        ax = np.arange(width) - (width - 1) / 2
        k = np.exp(-ax ** 2 / (2 * sigma ** 2))
    elif kind == "hann":
        k = window("hann", width + 2)[1:-1]
    else:
        raise ValueError("kind must be moving_average, gaussian or hann")
    k /= k.sum()
    pad = min(width, len(x) - 1)
    padded = np.pad(x, (pad, pad), mode="reflect") if pad > 0 else x
    out = linear_convolution(padded, k, mode="same")
    return out[pad: pad + len(x)]


@dataclass
class Periodicity:
    period: float              # in units of the sampling interval × dt
    frequency: float           # cycles per unit time
    power: float               # share of total spectral power in the peak
    acf: float                 # autocorrelation at the nearest lag


def detect_periodicities(x, dt: float = 1.0, n: int = 3, min_period: float = 2.0,
                         max_period: Optional[float] = None,
                         detrend_degree: int = 1) -> List[Periodicity]:
    """Dominant cycles of a series sampled every dt time units."""
    x = np.asarray(x, dtype=float)
    y = detrend(x, detrend_degree)
    f, psd = periodogram(y, fs=1.0 / dt, window_name="hann", nfft=next_power_of_two(4 * len(y)))
    max_period = max_period or len(y) * dt / 2
    sel = (f > 0) & (1 / np.maximum(f, 1e-300) >= min_period * dt) & \
        (1 / np.maximum(f, 1e-300) <= max_period)
    total = psd[f > 0].sum()
    acf = autocorrelation(y)
    found: List[Periodicity] = []
    idx = np.nonzero(sel)[0]
    # local maxima only
    peaks = [i for i in idx if 0 < i < len(psd) - 1 and psd[i] >= psd[i - 1]
             and psd[i] > psd[i + 1]]
    peaks.sort(key=lambda i: -psd[i])
    for i in peaks:
        a, b, c = np.log(psd[i - 1] + 1e-300), np.log(psd[i] + 1e-300), np.log(psd[i + 1] + 1e-300)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom else 0.0
        freq = f[i] + delta * (f[1] - f[0])
        period = 1 / freq
        if any(abs(period - p.period) / p.period < 0.1 or
               abs(period / p.period - round(period / p.period)) < 0.05
               and period > p.period for p in found):
            continue                                   # near-duplicate or harmonic
        lag = int(round(period / dt))
        share = float(psd[max(i - 2, 0): i + 3].sum() / total) if total else 0.0
        found.append(Periodicity(float(period), float(freq), share,
                                 float(acf[lag]) if lag < len(acf) else 0.0))
        if len(found) == n:
            break
    return found


def seasonal_strength(x, period: int, detrend_degree: int = 1) -> float:
    """1 − Var(residual)/Var(detrended) after subtracting the mean
    seasonal profile (phase-averaged) of an integer period."""
    y = detrend(np.asarray(x, dtype=float), detrend_degree)
    if period < 2 or period > len(y) // 2:
        raise ValueError("period must be in 2 .. len(x)/2")
    phase = np.arange(len(y)) % period
    profile = np.array([y[phase == p].mean() for p in range(period)])
    resid = y - profile[phase]
    var = y.var()
    return float(max(0.0, 1 - resid.var() / var)) if var else 0.0


def synthetic_series(n: int = 730, trend: float = 0.02, weekly: float = 2.0,
                     yearly: float = 6.0, noise: float = 1.0, ar: float = 0.6,
                     seed=None, price: bool = False) -> np.ndarray:
    """Daily series: linear trend + 7-day + 365.25-day cycles + AR(1)
    noise. price=True returns a positive price path whose log-returns
    carry the same seasonal pattern."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e = rng.normal(0, noise, n)
    eps = np.zeros(n)
    for i in range(n):
        eps[i] = (ar * eps[i - 1] if i else 0.0) + e[i]
    series = (trend * t + weekly * np.sin(2 * np.pi * t / 7)
              + yearly * np.sin(2 * np.pi * t / 365.25) + eps)
    if price:
        returns = 0.001 * series / max(np.abs(series).max(), 1e-12) + rng.normal(0, 0.01, n)
        return 100 * np.exp(np.cumsum(returns))
    return series


class TimeSeriesAnalyzer:
    def __init__(self, values: Sequence[float], dt: float = 1.0, name: str = "series"):
        self.values = np.asarray(values, dtype=float)
        self.dt = dt
        self.name = name

    def periodicities(self, n: int = 3, **kw) -> List[Periodicity]:
        return detect_periodicities(self.values, self.dt, n, **kw)

    def seasonality(self, candidate_periods: Sequence[int]) -> Dict[int, float]:
        return {p: seasonal_strength(self.values, p) for p in candidate_periods
                if 2 <= p <= len(self.values) // 2}

    def acf(self, max_lag: Optional[int] = None) -> np.ndarray:
        return autocorrelation(self.values, max_lag)

    def smoothed(self, width: int, kind: str = "gaussian") -> np.ndarray:
        return smooth(self.values, width, kind)

    def trend(self, degree: int = 1) -> np.ndarray:
        return self.values - detrend(self.values, degree)

    def summary(self) -> Dict[str, object]:
        cycles = self.periodicities()
        return {"name": self.name, "length": len(self.values),
                "periods": [round(c.period, 2) for c in cycles],
                "seasonal_strength": {round(c.period): seasonal_strength(
                    self.values, int(round(c.period / self.dt)))
                    for c in cycles if 2 <= c.period / self.dt <= len(self.values) // 2}}
