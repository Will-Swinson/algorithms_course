"""Test suite for signal generators, window functions, FFT-domain
filters, hum removal, and windowed-sinc FIR design."""

import numpy as np
import pytest

from src.fft_analyzer.signal import filters as SF
from src.fft_analyzer.signal import generators as G

FS = 8000


def tones(*freqs, duration=1.0):
    return G.multitone(list(freqs), duration, FS)[1]


class TestGenerators:

    def test_sine_and_time_axis(self):
        t, x = G.sine(5, 1.0, 100, amplitude=2.0)
        assert len(t) == 100 and t[1] == 0.01
        assert np.isclose(x.max(), 2.0, atol=0.01)
        with pytest.raises(ValueError):
            G.time_axis(1.0, 0)

    def test_multitone_validation(self):
        with pytest.raises(ValueError):
            G.multitone([1, 2], 1.0, 100, amplitudes=[1])

    @pytest.mark.parametrize("method", ["linear", "exponential"])
    def test_chirp_frequency(self, method):
        from src.fft_analyzer.signal.analysis import dominant_frequency
        t, x = G.chirp(200, 2000, 2.0, FS, method)
        # the frequency in the middle segment matches the formula
        mid = slice(len(t) // 2 - 400, len(t) // 2 + 400)
        expected = G.instantaneous_frequency(200, 2000, 2.0, 1.0, method)
        assert abs(dominant_frequency(x[mid], FS) - expected) < 15
        with pytest.raises(ValueError):
            G.chirp(0, 10, 1.0, FS, "exponential")
        with pytest.raises(ValueError):
            G.chirp(1, 10, 1.0, FS, "quadratic")

    def test_square_and_sawtooth(self):
        _, sq = G.square(10, 1.0, 1000)
        _, saw = G.sawtooth(10, 1.0, 1000)
        assert set(np.unique(sq)) == {-1.0, 1.0}
        assert -1 <= saw.min() and saw.max() < 1

    def test_noise_spectra(self):
        assert abs(G.spectral_slope(G.white_noise(1 << 15, seed=1))) < 0.1
        assert G.spectral_slope(G.pink_noise(1 << 15, seed=1)) == pytest.approx(-1, abs=0.1)
        assert G.pink_noise(1000, std=2.0, seed=2).std() == pytest.approx(2.0)
        assert len(G.pink_noise(0)) == 0

    def test_add_noise_snr(self):
        x = tones(440)
        assert G.snr_db(x, G.add_noise(x, 10, seed=3)) == pytest.approx(10, abs=0.3)
        assert G.snr_db(x, x) == float("inf")
        assert len(G.add_noise(x, 5, seed=1, kind="pink")) == len(x)


class TestWindows:

    @pytest.mark.parametrize("name,ref", [("hann", np.hanning), ("hamming", np.hamming),
                                          ("blackman", np.blackman)])
    def test_match_numpy(self, name, ref):
        for n in (1, 2, 9, 64):
            assert np.allclose(SF.window(name, n), ref(n))

    def test_rectangular_and_edge_cases(self):
        assert np.all(SF.window("rectangular", 5) == 1)
        assert len(SF.window("hann", 0)) == 0
        with pytest.raises(ValueError):
            SF.window("kaiser", 8)
        assert np.isclose(SF.window("hann", 8, symmetric=False)[0], 0)

    @pytest.mark.parametrize("name,sidelobe", [("rectangular", -13.3), ("hann", -31.5),
                                               ("hamming", -42.7), ("blackman", -58.1)])
    def test_sidelobe_levels(self, name, sidelobe):
        assert SF.window_properties(name)["sidelobe_db"] == pytest.approx(sidelobe, abs=1.0)

    def test_main_lobe_widens(self):
        widths = [SF.window_properties(w)["mainlobe_halfwidth_bins"] for w in SF.WINDOWS]
        assert widths == sorted(widths)


class TestFFTFilters:

    def test_lowpass_highpass_bandpass_bandstop(self):
        x = tones(200, 1000, 3000)
        cases = [(SF.lowpass(x, FS, 500), {200}),
                 (SF.highpass(x, FS, 2000), {3000}),
                 (SF.bandpass(x, FS, 800, 1500), {1000}),
                 (SF.bandstop(x, FS, 800, 1500), {200, 3000})]
        for y, kept in cases:
            for f in (200, 1000, 3000):
                change = SF.measure_attenuation(y, x, FS, (f - 5, f + 5))
                if f in kept:
                    assert change > -0.5
                else:
                    assert change < -40

    def test_mask_symmetry_keeps_signal_real(self):
        m = SF.frequency_mask(64, FS, "lowpass", 1000)
        assert np.allclose(m[1:], m[1:][::-1])

    def test_transition_is_smooth(self):
        m = SF.frequency_mask(1024, FS, "lowpass", 1000, transition=400)
        vals = np.unique(np.round(m, 6))
        assert len(vals) > 10                          # not just 0 and 1
        assert m[0] == 1.0

    def test_wider_transition_less_ringing(self):
        n = 8192
        step = np.r_[np.zeros(n // 2), np.ones(n // 2)]
        sharp = SF.ringing(SF.lowpass(step, FS, 1000))
        soft = SF.ringing(SF.lowpass(step, FS, 1000, transition=1600))
        assert sharp > 0.08 and soft < 0.5 * sharp

    @pytest.mark.parametrize("kind,cutoff", [("lowpass", 0), ("lowpass", 5000),
                                             ("bandpass", (300, 200)), ("notch", 10)])
    def test_invalid(self, kind, cutoff):
        with pytest.raises(ValueError):
            SF.frequency_mask(64, FS, kind, cutoff)

    def test_batched_and_empty(self):
        X = np.stack([tones(200, 3000), tones(300, 3000)])
        Y = SF.lowpass(X, FS, 1000)
        assert Y.shape == X.shape
        assert len(SF.lowpass(np.zeros(0), FS, 100)) == 0


class TestHum:

    @pytest.mark.parametrize("duration", [1.0, 1.013])
    def test_subtract_beats_fft_notch(self, duration):
        t, music = G.multitone([440, 660], duration, FS, [0.5, 0.3])
        hum = 0.3 * np.sin(2 * np.pi * 60.2 * t + 0.4) + 0.1 * np.sin(2 * np.pi * 180.6 * t)
        rms = lambda v: np.sqrt(np.mean(v ** 2))  # noqa: E731
        sub = SF.notch_harmonics(music + hum, FS, 60, 5, method="subtract")
        fft = SF.notch_harmonics(music + hum, FS, 60, 5, method="fft")
        assert 20 * np.log10(rms(sub - music) / rms(hum)) < -35
        assert rms(sub - music) < rms(fft - music)
        with pytest.raises(ValueError):
            SF.notch_harmonics(music, FS, 60, method="magic")

    def test_refine_frequency(self):
        _, x = G.sine(60.37, 2.0, FS)
        assert SF.refine_frequency(x, FS, 60) == pytest.approx(60.37, abs=0.01)

    def test_subtract_sinusoids_empty(self):
        x = tones(100)
        assert np.allclose(SF.subtract_sinusoids(x, FS, []), x)


class TestFIR:

    def test_lowpass_taps(self):
        h = SF.fir_lowpass(101, 1000, FS, "hamming")
        assert len(h) == 101 and np.isclose(h.sum(), 1.0)
        assert np.allclose(h, h[::-1])                 # linear phase
        with pytest.raises(ValueError):
            SF.fir_lowpass(0, 1000, FS)

    def test_window_sets_stopband(self):
        atten = {w: SF.stopband_attenuation(SF.fir_lowpass(101, 1000, FS, w), 1000, 300, FS)
                 for w in SF.WINDOWS}
        assert atten["rectangular"] > -35              # sinc truncation: ~-31 dB
        assert max(atten["hann"], atten["hamming"]) < -40
        assert atten["blackman"] < -70
        assert atten["blackman"] < min(atten["hann"], atten["hamming"])

    def test_highpass(self):
        h = SF.fir_highpass(101, 1000, FS)
        f, db = SF.frequency_response(h, FS)
        assert db[0] < -40 and db[-1] > -1
        with pytest.raises(ValueError):
            SF.fir_highpass(100, 1000, FS)

    def test_apply_fir(self):
        x = tones(200, 3000)
        y = SF.apply_fir(x, SF.fir_lowpass(201, 1000, FS, "blackman"))
        assert len(y) == len(x)
        assert SF.measure_attenuation(y, x, FS, (2990, 3010)) < -60
