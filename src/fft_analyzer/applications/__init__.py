"""End-to-end workflows: audio (WAV) processing, image processing and
compression, and time-series analytics."""

from src.fft_analyzer.applications.audio_processor import (
    AudioProcessor, read_wav, write_wav,
)
from src.fft_analyzer.applications.image_processor import (
    ImageProcessor, load_image, save_image,
)
from src.fft_analyzer.applications.data_analyzer import (
    TimeSeriesAnalyzer, autocorrelation, detect_periodicities,
    seasonal_strength, smooth, synthetic_series,
)
