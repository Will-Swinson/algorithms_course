"""
Image processing workflow: load → FFT filters → DCT compression →
frequency-domain views → save.

    ImageProcessor.from_file(path) / from_array(a)   grayscale in [0, 1]
        .blur(sigma) .sharpen() .edges() .denoise(cutoff)
        .remove_periodic_noise()              — each returns a new processor
        .compress(quality) → (processor, CompressedImage)
        .spectrum()                           — centred log |FFT2|
        .quality_sweep(qualities)             — PSNR / SSIM / size table
        .save(path)                           — PNG (or any Pillow format)
        .save_compressed(path)                — our DCT format (.npz)
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.fft_analyzer.image import compression, filters


def load_image(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=float) / 255.0


def save_image(array, path: str) -> str:
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = np.clip(np.asarray(array, dtype=float), 0, 1)
    Image.fromarray(np.round(data * 255).astype(np.uint8)).save(path)
    return path


@dataclass
class ImageProcessor:
    image: np.ndarray
    history: List[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "ImageProcessor":
        return cls(load_image(path), [f"loaded {path}"])

    @classmethod
    def from_array(cls, array) -> "ImageProcessor":
        """Integer arrays (or floats above 2) are taken as 0–255; float
        arrays otherwise as [0, 1] (values slightly outside are kept,
        e.g. after adding noise)."""
        raw = np.asarray(array)
        a = raw.astype(float)
        if a.ndim == 3:
            a = a[..., :3].mean(axis=2)
        if a.size and (np.issubdtype(raw.dtype, np.integer) or a.max() > 2.0):
            a = a / 255.0
        return cls(a, ["from array"])

    def _new(self, image: np.ndarray, label: str) -> "ImageProcessor":
        return ImageProcessor(np.clip(image, 0.0, 1.0), self.history + [label])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.image.shape

    # ---- filters --------------------------------------------------------
    def blur(self, sigma: float = 2.0) -> "ImageProcessor":
        return self._new(filters.gaussian_blur(self.image, sigma), f"blur σ={sigma:g}")

    def sharpen(self, sigma: float = 1.5, amount: float = 1.0) -> "ImageProcessor":
        return self._new(filters.sharpen(self.image, sigma, amount),
                         f"sharpen ×{amount:g}")

    def edges(self, method: str = "sobel", smooth_sigma: float = 1.0) -> "ImageProcessor":
        if method == "sobel":
            e = filters.sobel_edges(self.image, smooth_sigma)[0]
        elif method == "laplacian":
            e = np.abs(filters.laplacian_edges(self.image, smooth_sigma))
        else:
            raise ValueError("method must be 'sobel' or 'laplacian'")
        top = e.max()
        return self._new(e / top if top else e, f"{method} edges")

    def denoise(self, cutoff: float = 0.15) -> "ImageProcessor":
        return self._new(filters.lowpass_denoise(self.image, cutoff),
                         f"low-pass denoise {cutoff:g} c/px")

    def remove_periodic_noise(self, strength: float = 8.0) -> "ImageProcessor":
        cleaned, notched = filters.remove_periodic_noise(self.image, strength=strength)
        return self._new(cleaned, f"notched {notched} spectral spikes")

    # ---- compression ----------------------------------------------------
    def compress(self, quality: int = 75) -> Tuple["ImageProcessor", compression.CompressedImage]:
        c = compression.compress(self.image, quality)
        return self._new(compression.decompress(c), f"DCT quality {quality}"), c

    def quality_sweep(self, qualities: Iterable[int] = (10, 25, 50, 75, 90)) -> List[Dict[str, float]]:
        return [compression.quality_report(self.image, q) for q in qualities]

    # ---- views / output --------------------------------------------------
    def spectrum(self) -> np.ndarray:
        return filters.log_magnitude_spectrum(self.image)

    def compare(self, other: "ImageProcessor") -> Dict[str, float]:
        return {"psnr": compression.psnr(self.image, other.image),
                "ssim": compression.ssim(self.image, other.image)}

    def save(self, path: str) -> str:
        return save_image(self.image, path)

    def save_compressed(self, path: str, quality: int = 75) -> int:
        """Write the quantized DCT coefficients; returns bytes written."""
        return compression.compress(self.image, quality).save(path)

    @staticmethod
    def load_compressed(path: str) -> "ImageProcessor":
        c = compression.CompressedImage.load(path)
        return ImageProcessor(compression.decompress(c),
                              [f"decoded {path} (quality {c.quality})"])

    def plot_frequency_domain(self, save_path: Optional[str] = None, title: Optional[str] = None):
        from src.fft_analyzer.visualization.plots import plot_fft2_heatmap
        return plot_fft2_heatmap(self.image, title=title or self.history[-1],
                                 save_path=save_path)
