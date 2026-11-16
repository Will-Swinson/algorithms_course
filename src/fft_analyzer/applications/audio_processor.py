"""
Audio processing workflow: WAV in → FFT filters / denoising → WAV out.

WAV files are read and written with the standard-library `wave` module
(PCM 8/16/24/32-bit, any channel count); samples become floats in
[−1, 1], one column per channel.

    AudioProcessor.from_wav(path) / from_signal(x, fs)
        .lowpass(cutoff) .highpass(cutoff) .bandpass(lo, hi)
        .bandstop(lo, hi) .remove_hum(50 or 60) .denoise(...)
        .normalize(peak)                 — each returns a new processor
        .spectrogram() .peaks()          — analysis
        .plot_spectrogram(path)          — visualization
        .to_wav(path)                    — output

denoise() is spectral gating: estimate each frequency's noise floor from
the quietest frames of the STFT, attenuate time-frequency cells that do
not rise clearly above it, and resynthesize with the inverse STFT.
"""
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.fft_analyzer.signal import analysis, filters


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """(samples as float in [−1, 1] with shape (n,) or (n, channels), fs)."""
    with wave.open(path, "rb") as w:
        channels, width, fs = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(float) - 128) / 128
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(float) / 32768
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        data = v.astype(float) / (1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(float) / 2 ** 31
    else:
        raise ValueError(f"unsupported sample width {width} bytes")
    data = data.reshape(-1, channels)
    return (data[:, 0] if channels == 1 else data), fs


def write_wav(path: str, samples, fs: int, sample_width: int = 2) -> None:
    """Write float samples in [−1, 1] (clipped) as PCM."""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    channels = x.shape[1]
    x = np.clip(x, -1.0, 1.0)
    if sample_width == 2:
        raw = np.round(x * 32767).astype("<i2").tobytes()
    elif sample_width == 1:
        raw = np.round(x * 127 + 128).astype(np.uint8).tobytes()
    elif sample_width == 4:
        raw = np.round(x * (2 ** 31 - 1)).astype("<i4").tobytes()
    else:
        raise ValueError("sample_width must be 1, 2 or 4")
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(int(fs))
        w.writeframes(raw)


@dataclass
class AudioProcessor:
    samples: np.ndarray                       # (n,) or (n, channels)
    fs: int
    history: List[str] = field(default_factory=list)

    @classmethod
    def from_wav(cls, path: str) -> "AudioProcessor":
        x, fs = read_wav(path)
        return cls(x, fs, [f"loaded {path}"])

    @classmethod
    def from_signal(cls, x, fs: int) -> "AudioProcessor":
        return cls(np.asarray(x, dtype=float), int(fs), ["from signal"])

    @property
    def duration(self) -> float:
        return len(self.samples) / self.fs

    @property
    def mono(self) -> np.ndarray:
        return self.samples if self.samples.ndim == 1 else self.samples.mean(axis=1)

    def _apply(self, func, label: str) -> "AudioProcessor":
        x = self.samples
        y = func(x) if x.ndim == 1 else np.stack([func(c) for c in x.T], axis=1)
        return AudioProcessor(y, self.fs, self.history + [label])

    # ---- filters -------------------------------------------------------
    def lowpass(self, cutoff: float, transition: float = 50.0) -> "AudioProcessor":
        return self._apply(lambda c: filters.lowpass(c, self.fs, cutoff, transition),
                           f"low-pass {cutoff:g} Hz")

    def highpass(self, cutoff: float, transition: float = 50.0) -> "AudioProcessor":
        return self._apply(lambda c: filters.highpass(c, self.fs, cutoff, transition),
                           f"high-pass {cutoff:g} Hz")

    def bandpass(self, low: float, high: float, transition: float = 50.0) -> "AudioProcessor":
        return self._apply(lambda c: filters.bandpass(c, self.fs, low, high, transition),
                           f"band-pass {low:g}-{high:g} Hz")

    def bandstop(self, low: float, high: float, transition: float = 10.0) -> "AudioProcessor":
        return self._apply(lambda c: filters.bandstop(c, self.fs, low, high, transition),
                           f"band-stop {low:g}-{high:g} Hz")

    def remove_hum(self, base: float = 60.0, harmonics: int = 5) -> "AudioProcessor":
        return self._apply(lambda c: filters.notch_harmonics(c, self.fs, base, harmonics),
                           f"hum removal {base:g} Hz x{harmonics}")

    def denoise(self, reduction_db: float = 18.0, threshold: float = 2.0,
                nperseg: int = 1024, noise_clip=None,
                floor_cap: float = 4.0) -> "AudioProcessor":
        """
        Spectral gating. The noise floor per frequency is taken from
        `noise_clip` (a noise-only recording, the reliable way) or, if
        none is given, from the median magnitude over time capped at
        floor_cap × the global median — without the cap a steady tone
        would count as its own noise floor and be gated away. Cells
        below threshold × floor are attenuated by reduction_db.
        """
        def floor_from(c):
            mag = np.abs(analysis.stft(c, self.fs, nperseg, overlap=0.75).Z)
            per_freq = np.median(mag, axis=0)
            return np.minimum(per_freq, floor_cap * np.median(mag))

        clip_floor = None
        if noise_clip is not None:
            clip = np.asarray(noise_clip, dtype=float)
            clip_floor = floor_from(clip if clip.ndim == 1 else clip.mean(axis=1))

        def gate(c):
            r = analysis.stft(c, self.fs, nperseg, overlap=0.75)
            mag = np.abs(r.Z)
            floor = clip_floor if clip_floor is not None else floor_from(c)
            keep = mag > threshold * floor
            gain = np.where(keep, 1.0, 10 ** (-reduction_db / 20))
            # smooth the mask over time to avoid "musical noise"
            kernel = np.ones(3) / 3
            gain = np.apply_along_axis(lambda g: np.convolve(g, kernel, "same"), 0, gain)
            r.Z = r.Z * gain
            return analysis.istft(r)
        return self._apply(gate, f"spectral gating {reduction_db:g} dB")

    def normalize(self, peak: float = 0.9) -> "AudioProcessor":
        m = np.abs(self.samples).max()
        y = self.samples * (peak / m) if m else self.samples.copy()
        return AudioProcessor(y, self.fs, self.history + [f"normalize to {peak:g}"])

    # ---- analysis / output ---------------------------------------------
    def spectrogram(self, nperseg: int = 1024, overlap: float = 0.75):
        return analysis.spectrogram(self.mono, self.fs, nperseg, overlap)

    def peaks(self, n: int = 5) -> List[analysis.Peak]:
        return analysis.peak_frequencies(self.mono, self.fs, n_peaks=n,
                                         min_separation=5.0)

    def plot_spectrogram(self, save_path: Optional[str] = None, title: Optional[str] = None,
                         nperseg: int = 1024, max_freq: Optional[float] = None):
        from src.fft_analyzer.visualization.plots import plot_spectrogram
        return plot_spectrogram(self.mono, self.fs, nperseg=nperseg,
                                title=title or " → ".join(self.history[-3:]),
                                max_freq=max_freq, save_path=save_path)

    def to_wav(self, path: str, sample_width: int = 2) -> str:
        write_wav(path, self.samples, self.fs, sample_width)
        return path
