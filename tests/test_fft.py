"""Test suite for the FFT core: DFT, recursive and iterative
Cooley-Tukey, Bluestein, inverse, 2-D, and FFT convolution."""

import importlib

import numpy as np
import pytest

F = importlib.import_module("src.fft_analyzer.core.fft")

RNG = np.random.default_rng(0)


def rand(n, complex_=True):
    x = RNG.normal(size=n)
    return x + 1j * RNG.normal(size=n) if complex_ else x


class TestTransforms:

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 256, 1024])
    def test_power_of_two_variants(self, n):
        x = rand(n)
        ref = np.fft.fft(x)
        assert np.allclose(F.fft_iterative(x), ref)
        for cutoff in (1, 4, 32):
            assert np.allclose(F.fft_recursive(x, cutoff), ref)
        if n <= 256:
            assert np.allclose(F.fft_recursive_python(x), ref)
            assert np.allclose(F.dft(x), ref)

    @pytest.mark.parametrize("n", [0, 1, 3, 5, 6, 7, 12, 100, 127, 1000, 1023])
    def test_any_length(self, n):
        x = rand(n)
        assert np.allclose(F.fft(x), np.fft.fft(x) if n else x)

    def test_bluestein_prime(self):
        x = rand(1009)
        assert np.allclose(F.fft_bluestein(x), np.fft.fft(x))

    @pytest.mark.parametrize("n", [1, 7, 64, 1000])
    def test_inverse_round_trip(self, n):
        x = rand(n)
        assert np.allclose(F.ifft(F.fft(x)), x)
        assert np.allclose(F.ifft(np.fft.fft(x)), np.fft.ifft(np.fft.fft(x)))

    def test_rejects_non_power_of_two(self):
        with pytest.raises(ValueError):
            F.fft_iterative(rand(12))
        with pytest.raises(ValueError):
            F.fft_recursive(rand(6))
        with pytest.raises(ValueError):
            F.fft_recursive_python([1, 2, 3])

    def test_batched_last_axis(self):
        X = RNG.normal(size=(3, 4, 32))
        assert np.allclose(F.fft(X), np.fft.fft(X))
        Y = RNG.normal(size=(5, 12))                  # Bluestein, batched
        assert np.allclose(F.fft(Y), np.fft.fft(Y))

    def test_2d(self):
        img = RNG.normal(size=(16, 24))
        assert np.allclose(F.fft2(img), np.fft.fft2(img))
        assert np.allclose(F.ifft2(F.fft2(img)).real, img)

    def test_large_2_20(self):
        x = rand(1 << 20, complex_=False)
        X = F.fft_iterative(x)
        ref = np.fft.fft(x)
        assert np.abs(X - ref).max() / np.abs(ref).max() < 1e-12


class TestProperties:

    def test_impulse_and_constant(self):
        e = np.zeros(8)
        e[0] = 1
        assert np.allclose(F.fft(e), np.ones(8))
        assert np.allclose(F.fft(np.ones(8)), 8 * e)

    def test_linearity_and_parseval(self):
        a, b = rand(64), rand(64)
        assert np.allclose(F.fft(2 * a + 3 * b), 2 * F.fft(a) + 3 * F.fft(b))
        assert np.isclose(np.sum(np.abs(a) ** 2), np.sum(np.abs(F.fft(a)) ** 2) / 64)

    def test_real_input_hermitian(self):
        X = F.fft(rand(32, complex_=False))
        assert np.allclose(X[1:], np.conj(X[1:][::-1]))

    def test_shift_theorem(self):
        x = rand(32)
        k = np.arange(32)
        assert np.allclose(F.fft(np.roll(x, 3)), F.fft(x) * np.exp(-2j * np.pi * k * 3 / 32))

    def test_record_stages(self):
        X, stages = F.fft_iterative(rand(16), record=True)
        assert len(stages) == 5 and np.allclose(stages[-1], X)

    def test_bit_reversal(self):
        assert F.bit_reverse_indices(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]
        assert F.bit_reverse_indices(1).tolist() == [0]

    def test_helpers(self):
        assert F.is_power_of_two(64) and not F.is_power_of_two(0) and not F.is_power_of_two(12)
        assert [F.next_power_of_two(n) for n in (0, 1, 5, 8, 9)] == [1, 1, 8, 8, 16]
        for n in (7, 8):
            assert np.allclose(F.fftfreq(n, 0.1), np.fft.fftfreq(n, 0.1))
            assert np.allclose(F.rfftfreq(n, 0.1), np.fft.rfftfreq(n, 0.1))
            assert np.allclose(F.fftshift(np.arange(n)), np.fft.fftshift(np.arange(n)))
        m = np.arange(12).reshape(3, 4)
        assert np.allclose(F.fftshift(m), np.fft.fftshift(m))
        assert F.fft_flops(1024) == 5 * 1024 * 10


class TestConvolution:

    @pytest.mark.parametrize("n,m", [(37, 11), (11, 37), (5, 5), (1, 9), (64, 64), (100, 1)])
    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_linear_matches_numpy(self, n, m, mode):
        a, b = rand(n, False), rand(m, False)
        assert np.allclose(F.linear_convolution(a, b, mode), np.convolve(a, b, mode))
        assert np.allclose(F.direct_convolution(a, b, mode), np.convolve(a, b, mode))

    def test_complex_inputs(self):
        a, b = rand(10), rand(4)
        assert np.allclose(F.linear_convolution(a, b), np.convolve(a, b))

    def test_circular(self):
        a, b = rand(16, False), rand(16, False)
        ref = [sum(a[j] * b[(k - j) % 16] for j in range(16)) for k in range(16)]
        assert np.allclose(F.circular_convolution(a, b), ref)
        with pytest.raises(ValueError):
            F.circular_convolution(a, b[:5])

    def test_polynomial_multiplication(self):
        # (1 + 2x)(3 + x + x²) = 3 + 7x + 3x² + 2x³
        assert np.allclose(F.linear_convolution([1, 2], [3, 1, 1]), [3, 7, 3, 2])

    def test_cross_correlation(self):
        a, b = rand(30, False), rand(7, False)
        assert np.allclose(F.cross_correlation(a, b), np.correlate(a, b, "full"))

    def test_empty_and_bad_mode(self):
        assert len(F.linear_convolution([], [1, 2])) == 0
        assert len(F.direct_convolution([1], [])) == 0
        with pytest.raises(ValueError):
            F.linear_convolution([1, 2], [1], mode="circular")
