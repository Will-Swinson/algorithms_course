"""Test suite for the FFTAnalyzer plots and animations (headless)."""

import os

import numpy as np

from src.fft_analyzer.image.filters import sample_image
from src.fft_analyzer.signal import generators as G
from src.fft_analyzer.signal.filters import fir_lowpass, frequency_response
from src.fft_analyzer.visualization import (
    animate_filtering, animate_fft_stages, fft_stage_frames, filtering_frames,
    plot_compression_grid, plot_condition_numbers, plot_fft2_heatmap,
    plot_filter_response, plot_signal_and_spectrum, plot_spectrogram,
    plot_spectrum, plot_windows,
)

FS = 8000


def saved(path):
    return os.path.exists(path) and os.path.getsize(path) > 3000


def test_signal_plots(tmp_path):
    _, x = G.multitone([440, 1200], 0.5, FS)
    p = str(tmp_path / "s.png")
    plot_signal_and_spectrum(x, FS, "two tones", max_time=0.01, save_path=p)
    assert saved(p)
    fig = plot_spectrum({"a": x, "b": 0.5 * x}, FS)
    assert fig.axes[0].get_legend() is not None
    p2 = str(tmp_path / "sg.png")
    plot_spectrogram(G.chirp(100, 3000, 1.0, FS)[1], FS, save_path=p2)
    assert saved(p2)


def test_image_and_filter_plots(tmp_path):
    p = str(tmp_path / "f2.png")
    plot_fft2_heatmap(sample_image(64), save_path=p)
    assert saved(p)
    p = str(tmp_path / "fr.png")
    plot_filter_response({"hamming": frequency_response(fir_lowpass(51, 1000, FS), FS)},
                         save_path=p)
    assert saved(p)
    p = str(tmp_path / "w.png")
    plot_windows(save_path=p)
    assert saved(p)


def test_condition_and_compression_plots(tmp_path):
    p = str(tmp_path / "c.png")
    plot_condition_numbers([2, 3, 4], {"A": [10, 100, 1e4]}, {"A": [1e-15, 1e-14, 1e-12]},
                           save_path=p)
    assert saved(p)
    from src.fft_analyzer.image import compression as C
    img = sample_image(64)
    results, recon = [], []
    for q in (10, 50):
        r, c = C.round_trip(img, q)
        recon.append(r)
        results.append(C.quality_report(img, q))
    p = str(tmp_path / "g.png")
    plot_compression_grid(img, results, recon, save_path=p)
    assert saved(p)


def test_fft_stage_frames():
    x = np.arange(8.0)
    frames = fft_stage_frames(x)
    assert len(frames) == 4
    assert np.allclose(frames[-1]["values"], np.fft.fft(x))
    assert len(frames[1]["butterflies"]) == 4 and frames[1]["span"] == 2
    assert frames[3]["butterflies"][0] == (0, 4, 0)


def test_filtering_frames():
    t, x = G.multitone([100, 2000], 0.1, FS)
    frames = filtering_frames(x, FS, [3000, 500])
    assert frames[0]["energy_kept"] > 0.99
    assert 0.4 < frames[1]["energy_kept"] < 0.6


def test_gifs(tmp_path):
    p1, p2 = str(tmp_path / "fft.gif"), str(tmp_path / "filt.gif")
    animate_fft_stages(n=8, save_path=p1)
    _, x = G.multitone([100, 2000], 0.05, FS)
    animate_filtering(x, FS, [3000, 1000, 200], save_path=p2)
    for p in (p1, p2):
        assert saved(p)
        with open(p, "rb") as fh:
            assert fh.read(3) == b"GIF"
