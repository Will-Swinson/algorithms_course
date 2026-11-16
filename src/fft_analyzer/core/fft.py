"""
Fast Fourier Transform: Cooley–Tukey radix-2, its inverse, and
convolution.

The DFT of x_0 … x_{n−1} is

    X_k = Σ_j x_j · ω^{jk},    ω = e^{−2πi/n},

an O(n²) matrix–vector product. Cooley–Tukey splits it by index
parity: with E = DFT(x_even) and O = DFT(x_odd), both of size n/2,

    X_k         = E_k + ω^k · O_k
    X_{k + n/2} = E_k − ω^k · O_k          (k < n/2)

— one "butterfly" per k. Recursing gives T(n) = 2T(n/2) + O(n) =
O(n log n).

Implementations (all accept any leading dimensions and transform the
last axis):

    dft             the O(n²) definition, as a matrix product (reference)
    fft_recursive   the split above, recursively. Each level's
                    butterflies are numpy-vectorized; below `cutoff`
                    points it switches to the direct DFT (cutoff=1 is the
                    textbook recursion all the way down)
    fft_iterative   bottom-up: permute the input into bit-reversed order,
                    then log₂n stages of butterflies over spans 2, 4, …, n,
                    each stage a single vectorized operation. No recursion,
                    O(n) extra memory — this is the version that handles
                    2^20 points in a fraction of a second
    fft             dispatcher: iterative for powers of two, Bluestein's
                    chirp-z algorithm otherwise (rewrites a length-n DFT as
                    a convolution, computed with power-of-two FFTs), so any
                    length is O(n log n)
    ifft            conj(fft(conj(X))) / n

Convolution theorem: circular convolution becomes pointwise
multiplication, a ⊛ b = IFFT(FFT(a) · FFT(b)). Zero-padding both
inputs to ≥ len(a) + len(b) − 1 makes the circular wrap-around land in
the padding, giving linear convolution in O((n + m) log(n + m))
instead of O(nm).
"""

import numpy as np


def _as_complex(x) -> np.ndarray:
    return np.asarray(x, dtype=np.complex128)


def is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


# ----------------------------------------------------------------------
# reference DFT
# ----------------------------------------------------------------------
def dft_matrix(n: int, inverse: bool = False) -> np.ndarray:
    k = np.arange(n)
    sign = 1 if inverse else -1
    return np.exp(sign * 2j * np.pi * np.outer(k, k) / n)


def dft(x) -> np.ndarray:
    """Direct O(n²) DFT along the last axis."""
    x = _as_complex(x)
    n = x.shape[-1]
    if n == 0:
        return x.copy()
    return x @ dft_matrix(n).T


# ----------------------------------------------------------------------
# recursive Cooley–Tukey
# ----------------------------------------------------------------------
def fft_recursive(x, cutoff: int = 16) -> np.ndarray:
    """
    Radix-2 decimation-in-time FFT by recursion. Length must be a power
    of two. Subproblems of ≤ cutoff points use the direct DFT.
    """
    x = _as_complex(x)
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"fft_recursive needs a power-of-two length, got {n}")
    return _fft_rec(x, max(1, cutoff))


def _fft_rec(x: np.ndarray, cutoff: int) -> np.ndarray:
    n = x.shape[-1]
    if n <= cutoff:
        return x.copy() if n == 1 else dft(x)
    even = _fft_rec(x[..., 0::2], cutoff)
    odd = _fft_rec(x[..., 1::2], cutoff)
    twiddled = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + twiddled, even - twiddled], axis=-1)


def fft_recursive_python(x) -> list:
    """The textbook recursion on Python lists of complex numbers, no
    numpy (for small n and for reading)."""
    x = [complex(v) for v in x]
    n = len(x)
    if not is_power_of_two(n):
        raise ValueError(f"needs a power-of-two length, got {n}")

    def rec(a):
        m = len(a)
        if m == 1:
            return a
        even, odd = rec(a[0::2]), rec(a[1::2])
        out = [0j] * m
        for k in range(m // 2):
            t = np.exp(-2j * np.pi * k / m) * odd[k]
            out[k] = even[k] + t
            out[k + m // 2] = even[k] - t
        return out

    return rec(x)


# ----------------------------------------------------------------------
# iterative Cooley–Tukey
# ----------------------------------------------------------------------
_REV_CACHE = {}
_TWIDDLE_CACHE = {}


def bit_reverse_indices(n: int) -> np.ndarray:
    """Permutation taking i to its log₂n-bit reversal (cached)."""
    rev = _REV_CACHE.get(n)
    if rev is None:
        bits = n.bit_length() - 1
        rev = np.zeros(n, dtype=np.int64)
        idx = np.arange(n)
        for b in range(bits):
            rev |= ((idx >> b) & 1) << (bits - 1 - b)
        if n <= 1 << 22:
            _REV_CACHE[n] = rev
    return rev


def _twiddles(n: int) -> np.ndarray:
    """ω^k for k < n/2 (cached); stage with span s uses every (n/s)-th."""
    w = _TWIDDLE_CACHE.get(n)
    if w is None:
        w = np.exp(-2j * np.pi * np.arange(n // 2) / n)
        if n <= 1 << 22:
            _TWIDDLE_CACHE[n] = w
    return w


def fft_iterative(x, record: bool = False):
    """
    Bit-reversal + log₂n vectorized butterfly stages. Power-of-two
    length. With record=True returns (X, stages): the array after the
    permutation and after every stage (for the butterfly animation).
    """
    x = _as_complex(x)
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"fft_iterative needs a power-of-two length, got {n}")
    a = x[..., bit_reverse_indices(n)]
    stages = [a.copy()] if record else None
    lead = a.shape[:-1]
    w_all = _twiddles(n)
    span = 2
    while span <= n:
        half = span // 2
        w = w_all[:: n // span]                      # ω_span^k, k < half
        blocks = a.reshape(*lead, n // span, span)
        top = blocks[..., :half]
        bottom = blocks[..., half:] * w
        a = np.concatenate([top + bottom, top - bottom], axis=-1).reshape(*lead, n)
        if record:
            stages.append(a.copy())
        span *= 2
    return (a, stages) if record else a


# ----------------------------------------------------------------------
# any length: Bluestein
# ----------------------------------------------------------------------
def fft_bluestein(x) -> np.ndarray:
    """
    DFT of any length n via jk = (j² + k² − (k − j)²)/2:

        X_k = w_k · Σ_j (x_j w_j) · conj(w_{k−j}),   w_j = e^{−iπ j²/n}

    a convolution of length n, done with power-of-two FFTs of size
    ≥ 2n − 1. O(n log n) for every n.
    """
    x = _as_complex(x)
    n = x.shape[-1]
    if n <= 1:
        return x.copy()
    k = np.arange(n)
    chirp = np.exp(-1j * np.pi * (k * k % (2 * n)) / n)
    m = next_power_of_two(2 * n - 1)
    a = np.zeros(x.shape[:-1] + (m,), dtype=np.complex128)
    a[..., :n] = x * chirp
    b = np.zeros(m, dtype=np.complex128)
    b[:n] = np.conj(chirp)
    b[m - n + 1:] = np.conj(chirp[1:][::-1])
    conv = ifft(fft_iterative(a) * fft_iterative(b))
    return conv[..., :n] * chirp


def fft(x) -> np.ndarray:
    """FFT of any length along the last axis."""
    x = _as_complex(x)
    n = x.shape[-1]
    if n <= 1:
        return x.copy()
    return fft_iterative(x) if is_power_of_two(n) else fft_bluestein(x)


def ifft(X) -> np.ndarray:
    """Inverse FFT: conj(FFT(conj(X))) / n."""
    X = _as_complex(X)
    n = X.shape[-1]
    if n == 0:
        return X.copy()
    return np.conj(fft(np.conj(X))) / n


def fft2(image) -> np.ndarray:
    """2-D FFT: 1-D FFTs along rows, then along columns."""
    a = fft(_as_complex(image))
    return np.swapaxes(fft(np.swapaxes(a, -1, -2)), -1, -2)


def ifft2(X) -> np.ndarray:
    a = ifft(_as_complex(X))
    return np.swapaxes(ifft(np.swapaxes(a, -1, -2)), -1, -2)


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Frequencies of the FFT bins (cycles per unit of d)."""
    k = np.arange(n)
    k[(n + 1) // 2:] -= n
    return k / (n * d)


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    return np.arange(n // 2 + 1) / (n * d)


def fftshift(X, axes=None) -> np.ndarray:
    """Move the zero frequency to the centre."""
    X = np.asarray(X)
    axes = range(X.ndim) if axes is None else ([axes] if np.isscalar(axes) else axes)
    for ax in axes:
        X = np.roll(X, X.shape[ax] // 2, axis=ax)
    return X


# ----------------------------------------------------------------------
# convolution
# ----------------------------------------------------------------------
def circular_convolution(a, b) -> np.ndarray:
    """(a ⊛ b)_k = Σ_j a_j b_{(k−j) mod n}; inputs of equal length n."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("circular convolution needs equal lengths")
    out = ifft(fft(a) * fft(b))
    return out.real if np.isrealobj(a) and np.isrealobj(b) else out


def linear_convolution(a, b, mode: str = "full") -> np.ndarray:
    """
    Linear convolution via zero-padded FFTs (size: next power of two
    ≥ len(a) + len(b) − 1). mode: "full" (length n + m − 1), "same"
    (length max(n, m), centred) or "valid" (only full overlaps) — the
    same conventions as numpy.convolve.
    """
    a, b = np.asarray(a), np.asarray(b)
    n, m = a.shape[-1], b.shape[-1]
    if n == 0 or m == 0:
        return np.zeros(0)
    size = n + m - 1
    p = next_power_of_two(size)
    fa = fft(np.concatenate([a, np.zeros(a.shape[:-1] + (p - n,))], axis=-1))
    fb = fft(np.concatenate([b, np.zeros(b.shape[:-1] + (p - m,))], axis=-1))
    full = ifft(fa * fb)[..., :size]
    if np.isrealobj(a) and np.isrealobj(b):
        full = full.real
    return _crop(full, n, m, mode)


def direct_convolution(a, b, mode: str = "full") -> np.ndarray:
    """O(nm) reference: shift-and-add over the shorter input."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.zeros(0)
    if m > n:
        a, b, n, m = b, a, m, n
        swapped = True
    else:
        swapped = False
    full = np.zeros(n + m - 1)
    for j in range(m):
        full[j:j + n] += b[j] * a
    return _crop(full, *((m, n) if swapped else (n, m)), mode)


def _crop(full: np.ndarray, n: int, m: int, mode: str) -> np.ndarray:
    if mode == "full":
        return full
    if mode == "same":
        start = (min(n, m) - 1) // 2
        return full[..., start:start + max(n, m)]
    if mode == "valid":
        lo, hi = min(n, m) - 1, max(n, m)
        return full[..., lo:hi]
    raise ValueError(f"mode must be 'full', 'same' or 'valid', got {mode!r}")


def cross_correlation(a, b) -> np.ndarray:
    """Linear cross-correlation r_k = Σ_j a_{j+k} b_j for lags
    −(len(b)−1) … len(a)−1, via FFT."""
    b = np.asarray(b)
    return linear_convolution(a, np.conj(b[::-1]))


def fft_flops(n: int) -> float:
    """Classic operation count 5 n log₂ n (for GFLOP/s estimates)."""
    return 5 * n * np.log2(max(n, 2))
