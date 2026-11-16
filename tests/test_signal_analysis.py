"""Test suite for spectral analysis: periodogram (Parseval), Welch PSD,
STFT / inverse STFT, spectrograms, and peak frequencies."""

import numpy as np
import pytest

from src.fft_analyzer.signal import analysis as SA
from src.fft_analyzer.signal import generators as G

FS = 8000


@pytest.mark.parametrize("window", ["rectangular", "hann", "blackman"])
@pytest.mark.parametrize("n", [1000, 1024, 1001])
def test_periodogram_parseval(window, n):
    x = G.white_noise(n, 1.0, seed=1)
    f, p = SA.periodogram(x, FS, window)
    assert len(f) == n // 2 + 1 and f[-1] <= FS / 2
    power = p.sum() * FS / n
    if window == "rectangular":
        assert power == pytest.approx(np.var(x), rel=1e-9)
    else:
        assert power == pytest.approx(np.var(x), rel=0.1)


def test_periodogram_tone_location_and_errors():
    _, x = G.sine(1000, 1.0, FS)
    f, p = SA.periodogram(x, FS)
    assert f[np.argmax(p)] == pytest.approx(1000)
    with pytest.raises(ValueError):
        SA.periodogram([], FS)


def test_welch_reduces_variance():
    x = G.white_noise(1 << 14, seed=2)
    _, p1 = SA.periodogram(x, 1.0, "hann")
    _, p2 = SA.welch_psd(x, 1.0, 256)
    cv = lambda p: p[5:-5].std() / p[5:-5].mean()  # noqa: E731
    assert cv(p2) < cv(p1) / 4
    assert p2.mean() == pytest.approx(2.0, rel=0.1)  # one-sided density of unit white noise
    with pytest.raises(ValueError):
        SA.welch_psd(x, 1.0, 256, overlap=1.0)


@pytest.mark.parametrize("nperseg,overlap", [(256, 0.5), (512, 0.75), (255, 0.75)])
def test_stft_round_trip(nperseg, overlap):
    _, x = G.chirp(100, 3000, 0.5, FS)
    r = SA.stft(x, FS, nperseg, overlap)
    assert r.Z.shape[1] == nperseg // 2 + 1
    assert np.allclose(SA.istft(r), x, atol=1e-10)


def test_spectrogram_tracks_chirp():
    t, x = G.chirp(500, 3000, 2.0, FS)
    times, freqs, power = SA.spectrogram(x, FS, 512)
    ridge = freqs[np.argmax(power, axis=1)]
    expected = G.instantaneous_frequency(500, 3000, 2.0, times)
    inside = (times > 0.1) & (times < 1.9)
    assert np.abs(ridge[inside] - expected[inside]).max() < 40


class TestPeaks:

    def test_multitone(self):
        _, x = G.multitone([440, 1250, 3000], 1.0, FS, [1.0, 0.5, 0.25])
        peaks = SA.peak_frequencies(x, FS, 3)
        assert [round(p.frequency) for p in peaks] == [440, 1250, 3000]
        assert [p.magnitude for p in peaks] == pytest.approx([1.0, 0.5, 0.25], rel=0.1)

    def test_sub_bin_accuracy(self):
        _, x = G.sine(440.37, 1.0, FS)
        assert SA.dominant_frequency(x, FS) == pytest.approx(440.37, abs=0.05)

    def test_threshold_and_separation(self):
        _, x = G.multitone([1000, 1010, 2000], 1.0, FS, [1.0, 0.8, 0.0001])
        assert len(SA.peak_frequencies(x, FS, 5, threshold_db=-40)) == 2
        sep = SA.peak_frequencies(x, FS, 5, min_separation=50)
        assert len(sep) == 1

    def test_short_and_silent(self):
        assert SA.peak_frequencies([1.0, 2.0], FS) == []
        assert SA.dominant_frequency(np.zeros(16), FS) == 0.0    # no peaks at all


def test_band_power_and_centroid():
    _, x = G.multitone([500, 2500], 1.0, FS, [1.0, 1.0])
    low, high = SA.band_power(x, FS, 400, 600), SA.band_power(x, FS, 2400, 2600)
    assert low == pytest.approx(high, rel=0.01) and low == pytest.approx(0.5, rel=0.05)
    assert SA.spectral_centroid(x, FS) == pytest.approx(1500, rel=0.01)
