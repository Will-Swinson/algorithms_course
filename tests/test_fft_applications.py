"""Test suite for the audio processor (WAV I/O, filters, hum removal,
spectral gating), the image processor, and the time-series analyzer."""

import numpy as np
import pytest

from src.fft_analyzer.applications import (
    AudioProcessor, ImageProcessor, TimeSeriesAnalyzer, autocorrelation,
    detect_periodicities, load_image, read_wav, seasonal_strength,
    smooth, synthetic_series, write_wav,
)
from src.fft_analyzer.applications.data_analyzer import autocorrelation_direct, detrend
from src.fft_analyzer.image.filters import add_periodic_noise, sample_image
from src.fft_analyzer.signal import generators as G
from src.fft_analyzer.signal.filters import measure_attenuation

FS = 8000


class TestWav:

    @pytest.mark.parametrize("width,tol", [(1, 1 / 64), (2, 1 / 16000), (4, 1e-8)])
    def test_round_trip(self, tmp_path, width, tol):
        _, x = G.sine(440, 0.1, FS, 0.8)
        path = str(tmp_path / "a.wav")
        write_wav(path, x, FS, width)
        y, fs = read_wav(path)
        assert fs == FS and y.shape == x.shape and np.abs(y - x).max() < tol

    def test_stereo_and_clipping(self, tmp_path):
        x = np.stack([np.linspace(-2, 2, 100), np.zeros(100)], axis=1)
        path = str(tmp_path / "s.wav")
        write_wav(path, x, FS)
        y, _ = read_wav(path)
        assert y.shape == (100, 2) and y.max() <= 1.0 and y.min() >= -1.0

    def test_24_bit(self, tmp_path):
        import wave
        path = str(tmp_path / "b.wav")
        values = np.array([0, 1 << 22, -(1 << 22), (1 << 23) - 1], dtype=np.int64)
        raw = b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in values)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(3)
            w.setframerate(FS)
            w.writeframes(raw)
        y, _ = read_wav(path)
        assert y.tolist() == pytest.approx([0, 0.5, -0.5, 1 - 2 ** -23])

    def test_bad_width(self, tmp_path):
        with pytest.raises(ValueError):
            write_wav(str(tmp_path / "x.wav"), np.zeros(4), FS, 3)


class TestAudioProcessor:

    def make(self, duration=1.0):
        t, music = G.multitone([440, 660], duration, FS, [0.5, 0.3])
        return t, music

    def test_filters_chain_and_history(self, tmp_path):
        t, music = self.make()
        noisy = music + 0.2 * np.sin(2 * np.pi * 3000 * t)
        a = AudioProcessor.from_signal(noisy, FS)
        b = a.lowpass(1500).highpass(100)
        assert measure_attenuation(b.samples, noisy, FS, (2990, 3010)) < -40
        assert measure_attenuation(b.samples, noisy, FS, (435, 445)) > -0.5
        assert b.history[-2:] == ["low-pass 1500 Hz", "high-pass 100 Hz"]
        assert a.samples is noisy or np.allclose(a.samples, noisy)   # immutable
        c = a.bandpass(400, 500)
        assert [round(p.frequency) for p in c.peaks(1)] == [440]
        d = a.bandstop(2900, 3100)
        assert measure_attenuation(d.samples, noisy, FS, (2990, 3010)) < -30
        path = d.normalize(0.5).to_wav(str(tmp_path / "out.wav"))
        y, _ = read_wav(path)
        assert np.abs(y).max() == pytest.approx(0.5, abs=1e-3)

    def test_from_wav_and_stereo(self, tmp_path):
        _, music = self.make(0.2)
        path = str(tmp_path / "st.wav")
        write_wav(path, np.stack([music, 0.5 * music], axis=1), FS)
        a = AudioProcessor.from_wav(path)
        assert a.samples.shape[1] == 2 and a.duration == pytest.approx(0.2)
        b = a.lowpass(1000)
        assert b.samples.shape == a.samples.shape
        assert a.mono.shape == (len(music),)

    def test_hum_removal(self):
        t, music = self.make()
        hum = 0.3 * np.sin(2 * np.pi * 60 * t + 0.2)
        a = AudioProcessor.from_signal(music + hum, FS)
        clean = a.remove_hum(60, 3)
        resid = np.sqrt(np.mean((clean.samples - music) ** 2)) / np.sqrt(np.mean(hum ** 2))
        assert resid < 0.02

    def test_spectral_gating_improves_snr(self):
        t, music = self.make(2.0)
        noise = G.white_noise(len(t), 0.1, seed=3)
        a = AudioProcessor.from_signal(music + noise, FS)
        before = G.snr_db(music, a.samples)
        assert G.snr_db(music, a.denoise().samples) > before + 5
        clip = G.white_noise(FS, 0.1, seed=9)
        assert G.snr_db(music, a.denoise(noise_clip=clip).samples) > before + 5

    def test_spectrogram_and_plot(self, tmp_path):
        _, x = G.chirp(200, 2000, 1.0, FS)
        a = AudioProcessor.from_signal(x, FS)
        times, freqs, power = a.spectrogram(512)
        assert power.shape == (len(times), len(freqs))
        path = str(tmp_path / "spec.png")
        a.plot_spectrogram(path, max_freq=3000)
        import os
        assert os.path.getsize(path) > 5000


class TestImageProcessor:

    def test_pipeline(self, tmp_path):
        ip = ImageProcessor.from_array(sample_image(64))
        out = ip.blur(1.5).sharpen().denoise(0.3)
        assert out.shape == (64, 64) and len(out.history) == 4
        assert 0 <= out.image.min() and out.image.max() <= 1
        for method in ("sobel", "laplacian"):
            e = ip.edges(method)
            assert e.image.max() == pytest.approx(1.0)
        with pytest.raises(ValueError):
            ip.edges("canny")

    def test_scaling_rules(self):
        assert ImageProcessor.from_array(np.full((4, 4), 255, np.uint8)).image.max() == 1.0
        noisy = sample_image(32) + 0.2                  # floats slightly above 1
        assert ImageProcessor.from_array(noisy).image.max() == pytest.approx(1.2)  # not /255
        rgb = np.zeros((4, 4, 3))
        rgb[..., 0] = 0.9
        assert ImageProcessor.from_array(rgb).image.max() == pytest.approx(0.3)

    def test_periodic_noise_and_compare(self):
        clean = ImageProcessor.from_array(sample_image(128))
        noisy = ImageProcessor.from_array(add_periodic_noise(sample_image(128), (0, 0.2), 0.2))
        fixed = noisy.remove_periodic_noise()
        assert clean.compare(fixed)["psnr"] > clean.compare(noisy)["psnr"] + 5
        assert "notched" in fixed.history[-1]

    def test_compression_and_files(self, tmp_path):
        ip = ImageProcessor.from_array(sample_image(64))
        decoded, c = ip.compress(50)
        assert c.bits_per_pixel < 2 and decoded.compare(ip)["psnr"] > 25
        sweep = ip.quality_sweep((20, 80))
        assert sweep[0]["psnr"] < sweep[1]["psnr"]
        size = ip.save_compressed(str(tmp_path / "c.npz"), 50)
        assert 0 < size < 64 * 64
        back = ImageProcessor.load_compressed(str(tmp_path / "c.npz"))
        assert np.allclose(back.image, decoded.image)
        png = ip.save(str(tmp_path / "img.png"))
        assert np.abs(load_image(png) - ip.image).max() <= 1 / 255
        assert ip.spectrum().shape == (64, 64)
        ip.plot_frequency_domain(str(tmp_path / "f.png"))


class TestTimeSeries:

    def test_autocorrelation_fft_matches_direct(self):
        x = np.random.default_rng(1).normal(size=700)
        assert np.allclose(autocorrelation(x), autocorrelation_direct(x))
        assert np.allclose(autocorrelation(x, 20, normalize=False),
                           autocorrelation_direct(x, 20, normalize=False))
        assert autocorrelation(x)[0] == 1.0
        assert len(autocorrelation([])) == 0

    def test_acf_of_sinusoid(self):
        t = np.arange(400)
        acf = autocorrelation(np.sin(2 * np.pi * t / 20))
        assert acf[20] > 0.9 and acf[10] < -0.9

    def test_detect_periodicities(self):
        series = synthetic_series(1460, seed=4)
        periods = [p.period for p in detect_periodicities(series, n=3)]
        assert any(abs(p - 7) < 0.2 for p in periods)
        assert any(abs(p - 365.25) < 15 for p in periods)

    def test_dt_scaling(self):
        t = np.arange(512)
        x = np.sin(2 * np.pi * t / 32)                  # 32 samples per cycle
        assert detect_periodicities(x, dt=0.5, n=1)[0].period == pytest.approx(16, rel=0.02)

    def test_seasonal_strength(self):
        t = np.arange(700)
        strong = np.sin(2 * np.pi * t / 7) + 0.1 * np.random.default_rng(2).normal(size=700)
        assert seasonal_strength(strong, 7) > 0.95
        assert seasonal_strength(strong, 5) < 0.2
        with pytest.raises(ValueError):
            seasonal_strength(strong, 1)

    @pytest.mark.parametrize("kind", ["moving_average", "gaussian", "hann"])
    def test_smooth(self, kind):
        rng = np.random.default_rng(3)
        x = np.sin(np.linspace(0, 6, 500)) + rng.normal(0, 0.3, 500)
        y = smooth(x, 31, kind)
        assert len(y) == 500
        assert np.std(np.diff(y)) < np.std(np.diff(x)) / 5
        assert np.allclose(smooth(x, 1), x)
        with pytest.raises(ValueError):
            smooth(x, 5, "median")

    def test_detrend(self):
        t = np.arange(200.0)
        y = 3 + 0.5 * t + 0.001 * t ** 2
        assert np.abs(detrend(y, 2)).max() < 1e-8
        assert np.abs(detrend(y, 1)).max() > 0.1

    def test_price_series_and_analyzer(self):
        p = synthetic_series(400, seed=1, price=True)
        assert p.min() > 0
        an = TimeSeriesAnalyzer(synthetic_series(1460, seed=5), name="sales")
        summary = an.summary()
        assert summary["length"] == 1460 and 7.0 in [round(x) for x in summary["periods"]]
        assert an.seasonality([7, 1000])[7] > 0
        assert len(an.acf(30)) == 31 and len(an.smoothed(9)) == 1460
        assert len(an.trend()) == 1460
