"""Test suite for FFT image filters and JPEG-style DCT compression."""

import numpy as np
import pytest

from src.fft_analyzer.image import compression as C
from src.fft_analyzer.image import filters as IF

RNG = np.random.default_rng(5)
IMG = IF.sample_image(64, "shapes")


class TestConvolution2D:

    @pytest.mark.parametrize("ks", [(3, 3), (5, 5), (4, 6), (1, 2), (9, 1)])
    @pytest.mark.parametrize("padding", IF.PADDING)
    def test_fft_matches_direct(self, ks, padding):
        k = RNG.normal(size=ks)
        assert np.allclose(IF.fft_convolve2d(IMG, k, padding),
                           IF.direct_convolve2d(IMG, k, padding))

    def test_zero_padding_matches_definition(self):
        k = RNG.normal(size=(4, 6))
        h, w = IMG.shape
        full = np.zeros((h + 3, w + 5))
        for i in range(4):
            for j in range(6):
                full[i:i + h, j:j + w] += k[i, j] * IMG
        assert np.allclose(IF.fft_convolve2d(IMG, k, "zero"), full[1:1 + h, 2:2 + w])

    def test_identity_kernel_and_errors(self):
        delta = np.zeros((3, 3))
        delta[1, 1] = 1
        assert np.allclose(IF.fft_convolve2d(IMG, delta), IMG)
        with pytest.raises(ValueError):
            IF.fft_convolve2d(IMG, delta, padding="wrap")
        with pytest.raises(ValueError):
            IF.fft_convolve2d(np.ones(5), delta)


class TestFilters:

    def test_gaussian_blur_matches_kernel(self):
        a = IF.gaussian_blur(IMG, 1.5)
        b = IF.direct_convolve2d(IMG, IF.gaussian_kernel(1.5, 8))
        assert np.abs(a - b).max() < 1e-3
        assert np.allclose(IF.gaussian_blur(IMG, 0), IMG)

    def test_blur_preserves_mean_and_smooths(self):
        noise = RNG.random((64, 64))
        b = IF.gaussian_blur(noise, 2.0)
        assert b.mean() == pytest.approx(noise.mean(), abs=0.01)
        assert b.std() < noise.std() / 3

    def test_sharpen_increases_edge_contrast(self):
        s = IF.sharpen(IMG, 1.5, 1.0)
        assert np.abs(np.diff(s, axis=1)).max() > np.abs(np.diff(IMG, axis=1)).max()

    def test_sobel_on_step(self):
        step = np.zeros((32, 32))
        step[:, 16:] = 1.0
        mag, gx, gy = IF.sobel_edges(step)
        cols = mag.mean(axis=0)
        assert np.argmax(cols) in (15, 16)
        assert np.abs(gy[4:-4, 4:-4]).max() < 1e-9
        assert gx[16, 16] > 0                          # dark → bright to the right

    def test_laplacian_flat_is_zero(self):
        assert np.abs(IF.laplacian_edges(np.full((32, 32), 0.5), 1.0)).max() < 1e-9

    def test_lowpass_denoise_improves_psnr(self):
        # needs noise strong relative to the image's own fine detail: on a
        # small, sharp image a low-pass removes real edges too
        img = IF.sample_image(128, "shapes")
        noisy = img + np.random.default_rng(5).normal(0, 0.15, img.shape)
        assert IF.psnr(img, IF.lowpass_denoise(noisy, 0.15)) > IF.psnr(img, noisy) + 2
        assert IF.psnr(img, IF.lowpass_denoise(noisy, 0.15, rolloff=0)) > IF.psnr(img, noisy)

    def test_periodic_noise_removed(self):
        img = IF.sample_image(128, "shapes")
        noisy = IF.add_periodic_noise(img, (0.0, 0.2), 0.2)
        cleaned, notched = IF.remove_periodic_noise(noisy)
        assert notched > 0
        assert IF.psnr(img, cleaned) > IF.psnr(img, noisy) + 8
        _, none = IF.remove_periodic_noise(img)
        assert none == 0                               # no false positives

    def test_spectrum_and_samples(self):
        spec = IF.log_magnitude_spectrum(IMG)
        assert spec.shape == IMG.shape and np.unravel_index(spec.argmax(), spec.shape) == (32, 32)
        for kind in ("shapes", "checkerboard", "gradient", "texture"):
            s = IF.sample_image(32, kind)
            assert s.shape == (32, 32) and 0 <= s.min() and s.max() <= 1
        with pytest.raises(ValueError):
            IF.sample_image(8, "noise")
        assert IF.psnr(IMG, IMG) == float("inf")


class TestDCT:

    @pytest.mark.parametrize("n", [1, 2, 7, 8, 16])
    def test_fft_dct_matches_matrix(self, n):
        x = RNG.normal(size=(3, n))
        Cm = C.dct_matrix(n)
        assert np.allclose(C.dct_fft(x), x @ Cm.T)
        assert np.allclose(C.idct_fft(C.dct_fft(x)), x)

    def test_matrix_orthonormal_and_2d(self):
        Cm = C.dct_matrix()
        assert np.allclose(Cm @ Cm.T, np.eye(8))
        B = RNG.normal(size=(8, 8))
        assert np.allclose(C.dct2(B), Cm @ B @ Cm.T)
        assert np.allclose(C.idct2(C.dct2(B)), B)
        blocks = RNG.normal(size=(2, 3, 8, 8))
        assert np.allclose(C.blockwise_idct(C.blockwise_dct(blocks)), blocks)

    def test_constant_block_energy_in_dc(self):
        F = C.dct2(np.full((8, 8), 3.0))
        assert F[0, 0] == pytest.approx(24.0) and np.abs(F.ravel()[1:]).max() < 1e-12


class TestCompression:

    def test_blocks_round_trip(self):
        img = RNG.random((13, 21))
        blocks, shape = C.to_blocks(img)
        assert blocks.shape == (2, 3, 8, 8)
        assert np.allclose(C.from_blocks(blocks, shape), img)

    def test_quality_scale_and_table(self):
        assert C.quality_scale(50) == 1.0
        assert np.all(C.quantization_table(100) == 1)
        assert np.all(C.quantization_table(10) >= C.quantization_table(90))
        with pytest.raises(ValueError):
            C.quality_scale(0)

    def test_zigzag(self):
        zz = C.zigzag_order()
        assert zz[:10].tolist() == [0, 1, 8, 16, 9, 2, 3, 10, 17, 24]
        assert sorted(zz.tolist()) == list(range(64))

    def test_run_length(self):
        block = np.zeros((8, 8), dtype=int)
        block[0, 0], block[0, 1], block[7, 7] = 5, 3, 1
        symbols = C.run_length_symbols(block, C.zigzag_order())
        assert symbols[0] == (0, 3) and symbols[-1] == (0, 0)
        assert (15, 0) in symbols                      # long zero run → ZRL

    @pytest.mark.parametrize("quality", [5, 25, 50, 75, 95])
    def test_round_trip_error_bounded(self, quality):
        recon, c = C.round_trip(IMG, quality)
        assert recon.shape == IMG.shape and 0 <= recon.min() and recon.max() <= 1
        assert C.psnr(IMG, recon) > 20

    def test_quality_trade_off(self):
        reports = [C.quality_report(IMG, q) for q in (10, 50, 90)]
        psnr = [r["psnr"] for r in reports]
        bpp = [r["bits_per_pixel"] for r in reports]
        assert psnr == sorted(psnr) and bpp == sorted(bpp)
        assert reports[0]["zero_fraction"] > reports[2]["zero_fraction"]
        assert reports[0]["ratio_entropy"] > 8 and reports[0]["ratio_file"] > 2

    def test_quality_100_nearly_lossless(self):
        recon, _ = C.round_trip(IMG, 100)
        assert C.psnr(IMG, recon) > 50

    def test_serialization(self, tmp_path):
        c = C.compress(IMG, 40)
        path = str(tmp_path / "img.npz")
        size = c.save(path)
        back = C.CompressedImage.load(path)
        assert np.array_equal(back.coefficients, c.coefficients)
        assert back.quality == 40 and back.shape == IMG.shape
        assert size == len(c.to_bytes()) < IMG.size

    def test_accepts_0_255(self):
        a = C.decompress(C.compress(IMG * 255, 75))
        b = C.decompress(C.compress(IMG, 75))
        assert np.allclose(a, b)
        assert C.decompress(C.compress(IMG, 75), scale_01=False).max() > 1

    def test_metrics(self):
        assert C.mse(IMG, IMG) == 0 and C.psnr(IMG, IMG) == float("inf")
        assert C.ssim(IMG, IMG) == pytest.approx(1.0)
        noisy = IMG + RNG.normal(0, 0.1, IMG.shape)
        assert C.ssim(IMG, noisy) < C.ssim(IMG, IF.gaussian_blur(noisy, 1.0)) < 1
        with pytest.raises(ValueError):
            C.compress(np.ones(10))
