"""Numerical core: FFT and convolution, dense matrix algorithms, and
floating-point stability utilities."""

from src.fft_analyzer.core.fft import (
    circular_convolution, dft, fft, fft2, fft_bluestein, fft_iterative,
    fft_recursive, fftfreq, fftshift, ifft, ifft2, linear_convolution,
    rfftfreq,
)
from src.fft_analyzer.core.matrix import (
    determinant, inverse, lu_decompose, lu_solve, matmul_classical,
    power_method, qr, qr_gram_schmidt, qr_householder,
    qr_modified_gram_schmidt, solve, strassen,
)
from src.fft_analyzer.core.numerical import (
    check_conditioning, condition_number, estimate_condition_1, hilbert,
    relative_error, vector_norm,
)
