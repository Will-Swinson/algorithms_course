"""
Image filtering in the frequency domain.

2-D convolution theorem: image ∗ kernel = IFFT2(FFT2(image)·FFT2(kernel)).
fft_convolve2d zero-pads both to (H + kh − 1) × (W + kw − 1) rounded up
to powers of two, so the product is a linear (not circular) convolution,
then crops the "same"-size centre. Edges are handled by padding the
image first ("reflect", "edge", or "zero"), the usual fix for dark
borders after blurring. Cost O(HW log HW) regardless of kernel size —
direct convolution is O(HW·kh·kw), so FFT wins for big kernels (the
benchmark finds the crossover).

    gaussian_blur      multiply by the Gaussian's own transform,
                       G(u, v) = exp(−2π²σ²(u² + v²)) — no kernel needed
    sharpen            unsharp mask: image + amount·(image − blur)
    sobel_edges        gradient magnitude from the two Sobel kernels
    laplacian_edges    response to the 3×3 discrete Laplacian
    lowpass_denoise    keep frequencies inside a radius (smooth roll-off)
    remove_periodic_noise  find isolated spectral spikes far from DC (a
                       stripe/moiré pattern is a pair of delta functions in
                       frequency) and notch them out
    sample_image       synthetic test images (shapes, gradient, text-like
                       bars, checkerboard) in [0, 1]
"""
from typing import Optional, Tuple

import numpy as np

from src.fft_analyzer.core.fft import fft2, fftfreq, ifft2, next_power_of_two

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_Y = SOBEL_X.T
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
PADDING = ("reflect", "edge", "zero")


def _image(image) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {img.shape}")
    return img


def _pad(img: np.ndarray, ph: int, pw: int, padding: str) -> np.ndarray:
    if padding not in PADDING:
        raise ValueError(f"padding must be one of {PADDING}")
    if ph == 0 and pw == 0:
        return img
    mode = {"reflect": "reflect", "edge": "edge", "zero": "constant"}[padding]
    if mode == "reflect" and (ph >= img.shape[0] or pw >= img.shape[1]):
        mode = "symmetric" if ph <= img.shape[0] and pw <= img.shape[1] else "edge"
    return np.pad(img, ((ph, ph), (pw, pw)), mode=mode)


def fft_convolve2d(image, kernel, padding: str = "reflect") -> np.ndarray:
    """'same'-size 2-D convolution via zero-padded FFTs."""
    img, k = _image(image), _image(kernel)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    src = _pad(img, ph, pw, padding)
    H, W = src.shape[0] + kh - 1, src.shape[1] + kw - 1
    sh, sw = next_power_of_two(H), next_power_of_two(W)
    a = np.zeros((sh, sw))
    a[:src.shape[0], :src.shape[1]] = src
    b = np.zeros((sh, sw))
    b[:kh, :kw] = k
    full = ifft2(fft2(a) * fft2(b)).real
    # output pixel y sits at full-convolution index y + kh − 1 of the
    # padded image (= 2·ph for odd kernels; even kernels centre at
    # (kh − 1)//2, matching numpy/scipy "same")
    top, left = kh - 1, kw - 1
    return full[top:top + img.shape[0], left:left + img.shape[1]]


def direct_convolve2d(image, kernel, padding: str = "reflect") -> np.ndarray:
    """O(HW·kh·kw) reference (shift-and-add over the kernel taps)."""
    img, k = _image(image), _image(kernel)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    src = _pad(img, ph, pw, padding)
    out = np.zeros_like(img)
    h, w = img.shape
    for i in range(kh):
        for j in range(kw):
            # convolution flips the kernel
            out += k[kh - 1 - i, kw - 1 - j] * src[i:i + h, j:j + w]
    return out


def gaussian_kernel(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    radius = int(np.ceil(3 * sigma)) if radius is None else radius
    ax = np.arange(-radius, radius + 1)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def gaussian_blur(image, sigma: float, padding: str = "reflect") -> np.ndarray:
    """Blur by multiplying the spectrum by the Gaussian transfer function
    exp(−2π²σ²(u² + v²)) (u, v in cycles per pixel)."""
    img = _image(image)
    if sigma <= 0:
        return img.copy()
    pad = int(np.ceil(3 * sigma))
    src = _pad(img, pad, pad, padding)
    h, w = src.shape
    sh, sw = next_power_of_two(h), next_power_of_two(w)
    a = np.zeros((sh, sw))
    a[:h, :w] = src
    if padding == "reflect" or padding == "edge":
        # fill the power-of-two margin by continuing the edge values, so
        # the circular transform does not see a jump
        a[:h, w:] = src[:, -1:]
        a[h:, :] = a[h - 1:h, :]
    u = fftfreq(sh)[:, None]
    v = fftfreq(sw)[None, :]
    G = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (u ** 2 + v ** 2))
    out = ifft2(fft2(a) * G).real
    return out[pad:pad + img.shape[0], pad:pad + img.shape[1]]


def sharpen(image, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask: boost what blurring removes."""
    img = _image(image)
    return img + amount * (img - gaussian_blur(img, sigma))


def sobel_edges(image, smooth_sigma: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(magnitude, gx, gy) of the Sobel gradient (optionally pre-blurred
    to suppress noise)."""
    img = _image(image)
    if smooth_sigma > 0:
        img = gaussian_blur(img, smooth_sigma)
    # Sobel kernels are written as correlation templates; flip for convolution
    gx = fft_convolve2d(img, SOBEL_X[::-1, ::-1])
    gy = fft_convolve2d(img, SOBEL_Y[::-1, ::-1])
    return np.hypot(gx, gy), gx, gy


def laplacian_edges(image, smooth_sigma: float = 1.0) -> np.ndarray:
    """Laplacian of Gaussian response (zero crossings mark edges)."""
    img = _image(image)
    if smooth_sigma > 0:
        img = gaussian_blur(img, smooth_sigma)
    return fft_convolve2d(img, LAPLACIAN)


def radial_frequency(shape: Tuple[int, int]) -> np.ndarray:
    """|frequency| of every 2-D FFT bin, in cycles per pixel."""
    u = fftfreq(shape[0])[:, None]
    v = fftfreq(shape[1])[None, :]
    return np.sqrt(u ** 2 + v ** 2)


def lowpass_denoise(image, cutoff: float = 0.15, rolloff: float = 0.05) -> np.ndarray:
    """Keep radial frequencies below `cutoff` cycles/pixel (raised-cosine
    roll-off of width `rolloff`); reflect-padded to avoid edge leakage."""
    img = _image(image)
    h, w = img.shape
    src = np.pad(img, ((0, h), (0, w)), mode="symmetric")   # periodic-friendly
    r = radial_frequency(src.shape)
    if rolloff > 0:
        z = np.clip((r - cutoff) / rolloff, 0, 1)
        mask = 0.5 + 0.5 * np.cos(np.pi * z)
    else:
        mask = (r <= cutoff).astype(float)
    return ifft2(fft2(src) * mask).real[:h, :w]


def remove_periodic_noise(image, min_radius: float = 0.05, strength: float = 8.0,
                          notch_radius: int = 2) -> Tuple[np.ndarray, int]:
    """
    Notch out spectral spikes: bins farther than min_radius from DC whose
    magnitude exceeds `strength` × the median of their neighbourhood.
    Returns (cleaned image, number of notched bins).
    """
    img = _image(image)
    F = fft2(img - img.mean())
    mag = np.abs(F)
    r = radial_frequency(img.shape)
    # local background: median-ish via a blurred log-magnitude
    log_mag = np.log1p(mag)
    background = np.expm1(gaussian_blur(log_mag, 3.0, padding="edge"))
    spikes = (mag > strength * (background + 1e-12)) & (r > min_radius)
    mask = np.ones(img.shape)
    ys, xs = np.nonzero(spikes)
    for y, x in zip(ys, xs):
        for dy in range(-notch_radius, notch_radius + 1):
            for dx in range(-notch_radius, notch_radius + 1):
                mask[(y + dy) % img.shape[0], (x + dx) % img.shape[1]] = 0.0
    cleaned = ifft2(F * mask).real + img.mean()
    return cleaned, int(spikes.sum())


def log_magnitude_spectrum(image) -> np.ndarray:
    """log(1 + |FFT2|), centred — the usual way to look at a spectrum."""
    F = fft2(_image(image))
    return np.fft.fftshift(np.log1p(np.abs(F)))


# ----------------------------------------------------------------------
# test images
# ----------------------------------------------------------------------
def sample_image(size: int = 128, kind: str = "shapes", seed: int = 0) -> np.ndarray:
    """Grayscale test image in [0, 1]."""
    y, x = np.mgrid[0:size, 0:size] / max(size - 1, 1)
    if kind == "shapes":
        img = 0.15 + 0.25 * x                          # soft gradient background
        img = np.where((x - 0.3) ** 2 + (y - 0.35) ** 2 < 0.18 ** 2, 0.9, img)
        img = np.where((abs(x - 0.72) < 0.15) & (abs(y - 0.3) < 0.1), 0.65, img)
        ring = (x - 0.5) ** 2 + (y - 0.75) ** 2
        img = np.where((ring < 0.17 ** 2) & (ring > 0.1 ** 2), 0.8, img)
        stripes = (abs(y - 0.55) < 0.04) & (np.sin(2 * np.pi * 12 * x) > 0)
        return np.where(stripes, 1.0, img)
    if kind == "checkerboard":
        return ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(float)
    if kind == "gradient":
        return 0.5 * (x + y)
    if kind == "texture":
        rng = np.random.default_rng(seed)
        base = rng.random((size, size))
        return np.clip(gaussian_blur(base, 2.0) * 2 - 0.5, 0, 1)
    raise ValueError("kind must be shapes, checkerboard, gradient or texture")


def add_periodic_noise(image, frequency: Tuple[float, float] = (0.0, 0.2),
                       amplitude: float = 0.2) -> np.ndarray:
    """Superimpose a sinusoidal stripe pattern (cycles per pixel)."""
    img = _image(image)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return img + amplitude * np.sin(2 * np.pi * (frequency[0] * yy + frequency[1] * xx))


def psnr(reference, test, peak: float = 1.0) -> float:
    mse = float(np.mean((np.asarray(reference, float) - np.asarray(test, float)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)
