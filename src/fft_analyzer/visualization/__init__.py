"""Plots (spectra, spectrograms, 2-D FFT heatmaps, condition numbers,
compression grids) and animations (FFT stages, filtering)."""

from src.fft_analyzer.visualization.plots import (
    plot_compression_grid, plot_condition_numbers, plot_fft2_heatmap,
    plot_filter_response, plot_signal_and_spectrum, plot_spectrogram,
    plot_spectrum, plot_windows,
)
from src.fft_analyzer.visualization.animations import (
    animate_filtering, animate_fft_stages, fft_stage_frames,
    filtering_frames,
)
