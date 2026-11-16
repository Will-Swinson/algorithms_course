"""
JPEG-style image compression with the 2-D DCT.

Pipeline (baseline JPEG, luminance only):

  1. Split the image (pixel values 0–255, shifted to −128…127) into
     8×8 blocks, padding the edges by replication.
  2. 2-D DCT-II of every block: F = C·B·Cᵀ with the orthonormal DCT
     matrix C. Smooth image content lands in the few low-frequency
     coefficients near F[0, 0].
  3. Quantize: q = round(F / Q·s), with the standard JPEG luminance
     table Q (coarser steps for high frequencies, which the eye notices
     least) and the IJG quality scaling s(quality). This is the only
     lossy step — most high-frequency coefficients become 0.
  4. Entropy-code the integers. We report two sizes: an entropy
     estimate of the zig-zag run-length symbols (what Huffman coding
     would approach), and the real size of the quantized coefficients
     written with numpy.savez_compressed (zlib).
  5. Decode: dequantize (q·Q·s), inverse DCT (Cᵀ·F·C), add 128, clip.

dct2 / idct2 also exist in an FFT-based form (Makhoul's algorithm:
reorder, one complex FFT, multiply by a twiddle), which ties the DCT to
the FFT and handles any block size in O(n log n).

Quality metrics: MSE, PSNR (dB) and SSIM (structural similarity, with
Gaussian-weighted local statistics computed by our FFT blur).
"""
import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.fft_analyzer.core.fft import fft, ifft

BLOCK = 8

# JPEG Annex K luminance quantization table
JPEG_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=float)


# ----------------------------------------------------------------------
# DCT
# ----------------------------------------------------------------------
def dct_matrix(n: int = BLOCK) -> np.ndarray:
    """Orthonormal DCT-II matrix: C[k, j] = α_k cos(π(2j + 1)k / 2n)."""
    k = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    C = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * j + 1) * k / (2 * n))
    C[0, :] = np.sqrt(1.0 / n)
    return C


def dct_fft(x, axis: int = -1) -> np.ndarray:
    """Orthonormal DCT-II along an axis via one FFT (Makhoul 1980):
    v = [x_0, x_2, …, x_3, x_1] (evens, then odds reversed),
    X_k = α_k · Re(e^{−iπk/2n} · FFT(v)_k)."""
    x = np.moveaxis(np.asarray(x, dtype=float), axis, -1)
    n = x.shape[-1]
    v = np.concatenate([x[..., 0::2], x[..., 1::2][..., ::-1]], axis=-1)
    V = fft(v)
    k = np.arange(n)
    X = (np.exp(-1j * np.pi * k / (2 * n)) * V).real
    X *= np.sqrt(2.0 / n)
    X[..., 0] /= np.sqrt(2.0)
    return np.moveaxis(X, -1, axis)


def idct_fft(X, axis: int = -1) -> np.ndarray:
    """Inverse of dct_fft (orthonormal DCT-III) via one inverse FFT."""
    X = np.moveaxis(np.asarray(X, dtype=float), axis, -1).copy()
    n = X.shape[-1]
    X[..., 0] *= np.sqrt(2.0)
    X /= np.sqrt(2.0 / n)
    k = np.arange(n)
    # V_k = e^{iπk/2n}(X_k − i X_{n−k}), X_n := 0
    Xr = np.concatenate([np.zeros(X.shape[:-1] + (1,)), X[..., :0:-1]], axis=-1)
    V = np.exp(1j * np.pi * k / (2 * n)) * (X - 1j * Xr)
    v = ifft(V).real
    x = np.empty_like(v)
    half = (n + 1) // 2
    x[..., 0::2] = v[..., :half]
    x[..., 1::2] = v[..., half:][..., ::-1]
    return np.moveaxis(x, -1, axis)


def dct2(block) -> np.ndarray:
    """2-D DCT-II (FFT-based, any size)."""
    return dct_fft(dct_fft(block, axis=-1), axis=-2)


def idct2(coeffs) -> np.ndarray:
    return idct_fft(idct_fft(coeffs, axis=-2), axis=-1)


def blockwise_dct(blocks: np.ndarray) -> np.ndarray:
    """DCT of a stack of n×n blocks with the DCT matrix: C·B·Cᵀ."""
    C = dct_matrix(blocks.shape[-1])
    return C @ blocks @ C.T


def blockwise_idct(coeffs: np.ndarray) -> np.ndarray:
    C = dct_matrix(coeffs.shape[-1])
    return C.T @ coeffs @ C


# ----------------------------------------------------------------------
# blocks, quantization, zig-zag
# ----------------------------------------------------------------------
def to_blocks(image: np.ndarray, n: int = BLOCK) -> Tuple[np.ndarray, Tuple[int, int]]:
    """(blocks of shape (by, bx, n, n), original shape); edges padded by
    replication."""
    h, w = image.shape
    H, W = -(-h // n) * n, -(-w // n) * n
    padded = np.pad(image, ((0, H - h), (0, W - w)), mode="edge")
    blocks = padded.reshape(H // n, n, W // n, n).swapaxes(1, 2)
    return blocks, (h, w)


def from_blocks(blocks: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    by, bx, n, _ = blocks.shape
    return blocks.swapaxes(1, 2).reshape(by * n, bx * n)[:shape[0], :shape[1]]


def quality_scale(quality: int) -> float:
    """IJG scaling: quality 50 → table as published; 100 → all ones."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in 1..100")
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return s / 100.0


def quantization_table(quality: int) -> np.ndarray:
    return np.clip(np.floor(JPEG_LUMA * quality_scale(quality) + 0.5), 1, 255)


def zigzag_order(n: int = BLOCK) -> np.ndarray:
    """Flat indices of an n×n block in JPEG zig-zag order (low → high
    frequency)."""
    order = sorted(((i, j) for i in range(n) for j in range(n)),
                   key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else p[1]))
    return np.array([i * n + j for i, j in order])


def run_length_symbols(block: np.ndarray, zz: np.ndarray):
    """JPEG-style AC symbols of one quantized block: (zero run, value)
    pairs in zig-zag order, ending with an end-of-block marker."""
    flat = block.ravel()[zz]
    symbols, run = [], 0
    for v in flat[1:]:
        if v == 0:
            run += 1
            continue
        while run > 15:
            symbols.append((15, 0))                 # ZRL
            run -= 16
        symbols.append((run, int(v)))
        run = 0
    symbols.append((0, 0))                          # EOB
    return symbols


def _bit_length(v: int) -> int:
    return int(abs(v)).bit_length()


def entropy_bits(quantized: np.ndarray) -> float:
    """
    Estimated coded size in bits: DC differences and AC (run, size)
    symbols are entropy-coded at their empirical entropy (≈ an optimal
    Huffman code), plus the raw magnitude bits JPEG appends.
    """
    blocks = quantized.reshape(-1, BLOCK, BLOCK)
    zz = zigzag_order()
    dc = blocks[:, 0, 0].astype(int)
    dc_diff = np.diff(np.concatenate([[0], dc]))
    dc_sizes = [_bit_length(d) for d in dc_diff]
    ac_symbols, extra = [], sum(dc_sizes)
    for b in blocks:
        for run, v in run_length_symbols(b, zz):
            ac_symbols.append((run, _bit_length(v)))
            extra += _bit_length(v)

    def entropy(symbols) -> float:
        if not symbols:
            return 0.0
        _, counts = np.unique(np.array(symbols).reshape(len(symbols), -1),
                              axis=0, return_counts=True)
        p = counts / counts.sum()
        return float(-(counts * np.log2(p)).sum())

    return entropy(dc_sizes) + entropy(ac_symbols) + extra


# ----------------------------------------------------------------------
# compress / decompress
# ----------------------------------------------------------------------
@dataclass
class CompressedImage:
    coefficients: np.ndarray        # int16, (by, bx, 8, 8) quantized DCT
    quality: int
    shape: Tuple[int, int]

    @property
    def estimated_bits(self) -> float:
        return entropy_bits(self.coefficients)

    @property
    def bits_per_pixel(self) -> float:
        return self.estimated_bits / (self.shape[0] * self.shape[1])

    @property
    def compression_ratio(self) -> float:
        """Against 8 bits per pixel, using the entropy estimate."""
        return 8.0 / self.bits_per_pixel if self.bits_per_pixel else math.inf

    @property
    def zero_fraction(self) -> float:
        return float(np.mean(self.coefficients == 0))

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, q=self.coefficients,
                            meta=np.array([self.quality, *self.shape]))
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompressedImage":
        with np.load(io.BytesIO(data)) as f:
            quality, h, w = (int(v) for v in f["meta"])
            return cls(f["q"], quality, (h, w))

    def save(self, path: str) -> int:
        data = self.to_bytes()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return len(data)

    @classmethod
    def load(cls, path: str) -> "CompressedImage":
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())


def _to_255(image) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("expected a 2-D grayscale image")
    return img * 255.0 if img.size and img.max() <= 1.0 else img


def compress(image, quality: int = 75) -> CompressedImage:
    """Image in [0, 1] (or 0–255) → quantized DCT coefficients."""
    img = _to_255(image)
    blocks, shape = to_blocks(img - 128.0)
    F = blockwise_dct(blocks)
    q = np.round(F / quantization_table(quality)).astype(np.int16)
    return CompressedImage(q, quality, shape)


def decompress(c: CompressedImage, scale_01: bool = True) -> np.ndarray:
    F = c.coefficients.astype(float) * quantization_table(c.quality)
    pixels = np.clip(from_blocks(blockwise_idct(F), c.shape) + 128.0, 0, 255)
    return pixels / 255.0 if scale_01 else pixels


def round_trip(image, quality: int = 75) -> Tuple[np.ndarray, CompressedImage]:
    c = compress(image, quality)
    return decompress(c), c


# ----------------------------------------------------------------------
# quality metrics
# ----------------------------------------------------------------------
def mse(a, b) -> float:
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def psnr(reference, test, peak: float = 1.0) -> float:
    e = mse(reference, test)
    return math.inf if e == 0 else 10 * math.log10(peak ** 2 / e)


def ssim(reference, test, peak: float = 1.0, sigma: float = 1.5) -> float:
    """Mean structural similarity (Wang et al. 2004) with Gaussian
    windows; local means and (co)variances via the FFT Gaussian blur."""
    from src.fft_analyzer.image.filters import gaussian_blur
    x, y = np.asarray(reference, float), np.asarray(test, float)
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    mx, my = gaussian_blur(x, sigma), gaussian_blur(y, sigma)
    vx = gaussian_blur(x * x, sigma) - mx * mx
    vy = gaussian_blur(y * y, sigma) - my * my
    cxy = gaussian_blur(x * y, sigma) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())


def quality_report(image, quality: int) -> Dict[str, float]:
    recon, c = round_trip(image, quality)
    ref = np.asarray(image, float)
    ref = ref / 255.0 if ref.max() > 1 else ref
    raw_bytes = ref.size                                  # 8-bit grayscale
    return {"quality": quality, "psnr": psnr(ref, recon), "ssim": ssim(ref, recon),
            "bits_per_pixel": c.bits_per_pixel,
            "ratio_entropy": c.compression_ratio,
            "ratio_file": raw_bytes / len(c.to_bytes()),
            "zero_fraction": c.zero_fraction}
