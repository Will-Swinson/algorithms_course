# Week 12 Technical Report: FFTAnalyzer — Numerical Algorithms

**Author:** Will Swinson
**Course:** Advanced Algorithms
**Assignment:** A modular FFT and numerical computing toolkit

## Executive Summary

`src/fft_analyzer/` is a numerical toolkit with the assignment's
modules: `core`, `signal`, `image`, `applications` and
`visualization`. All of it runs on numpy arrays, with no scipy. The
benchmark compared every result against numpy or an exact reference,
and all 368 configurations agreed.

- **FFT.** The iterative Cooley–Tukey FFT transforms 2²⁰ points in
  0.11 s, with relative error 1.1·10⁻¹⁵. Already at 4,096 points it is
  2,000× faster than the direct DFT; it is 5× slower than numpy's C
  implementation. Bluestein's algorithm extends O(n log n) to any
  length.
- **Stability.** Every textbook effect appeared in measurement:
  - Solve error tracked κ·ε across Hilbert, Vandermonde and random
    matrices.
  - Classical Gram–Schmidt lost all orthogonality at κ = 10⁹, while
    Householder stayed at 5·10⁻¹⁵.
  - Six cancellation-prone formulas were off by up to 669% until
    rewritten.
- **Applications.**
  - DCT compression reached 34.6 dB PSNR at 0.41 bits per pixel, an
    11:1 file-size ratio.
  - Least-squares hum subtraction removed 46.7 dB of mains hum; FFT
    notch filters managed 16.5 dB.
  - Spectral gating added 6–8 dB of SNR.
  - The time-series analyzer found the weekly and yearly cycles in
    10 of 10 synthetic series.

## Numerical Algorithm Theory

**Matrix multiplication.** The classical algorithm uses n³ scalar
multiplications. Strassen's algorithm forms 7 half-size products
instead of 8, giving O(n^2.807). At n = 256 with a 32-point cutoff it
did 11.2 M multiplications against 16.8 M, and ran 12% faster in pure
Python (0.30 s vs 0.34 s). Its extra additions cost accuracy: in
float32 its error was 9× larger at n = 512 (1.2·10⁻⁵ vs 1.3·10⁻⁶).
Both are dwarfed by BLAS at 0.08 ms, about 4,000× faster than either. Here
the hardware-tuned library matters far more than the asymptotic
exponent.

**LU with partial pivoting** factors PA = LU and swaps the largest
pivot into place at each step, so every multiplier is at most 1 in
magnitude. On 60×60 random matrices:

| | Element growth | Solve error |
|---|---|---|
| With pivoting | 4 | 2·10⁻¹⁴ |
| Without pivoting | 1,803 | 6·10⁻¹² (300× worse) |

The 2×2 matrix [[10⁻²⁰, 1], [1, 1]] shows it most starkly. Without
pivoting it returns x = [0, 1] instead of [1, 1].

**QR** factors A into an orthonormal Q and an upper-triangular R:
- Classical Gram–Schmidt (CGS) loses orthogonality like κ²ε.
- Modified Gram–Schmidt (MGS) loses it like κε.
- Householder reflections are backward stable.

Storing only the reflectors (never the m×m Q, which for the
32,000-row hum fit would need 8 GB) makes that fit take 10 ms.

**Power method.** Each multiplication by A shrinks the error by
|λ₂/λ₁|. Measured iteration counts (34, 99, 203, 403 and 1,892 for
ratios 0.5–0.99) matched the prediction log(tol)/log|λ₂/λ₁|. Testing
showed that stopping when the eigenvalue estimate stops changing is
not safe. With eigenvalues 1 and 0.999999 it declared convergence
after one step, so the stopping rule now requires a small residual
‖Ax − λx‖.

## FFT and Spectral Analysis

Cooley–Tukey splits a length-n DFT into transforms of its even and
odd samples and combines them with n/2 butterflies:
X_k = E_k ± ω^k O_k, giving O(n log n). I implemented four variants
(`fft_performance.png`):

- **Direct DFT** as a matrix product: 0.34 s at n = 4,096. Its error
  grew linearly with n, reaching 7.6·10⁻¹³.
- **Textbook recursion on Python lists:** 18 ms at 8,192.
- **Recursion with numpy-vectorized butterflies:** 0.87 s at 2²⁰.
- **Iterative:** a bit-reversal permutation, then log₂n stages, each
  one vectorized operation. 0.11 s at 2²⁰.

The iterative version's round-trip error ifft(fft(x)) stayed below
6.5·10⁻¹⁶ up to 2²⁰ points, under the ε·log₂n bound.

**Bluestein's algorithm** rewrites a DFT of any length as a
convolution computed with power-of-two FFTs. It handled 2¹⁹ + 1
points in 0.76 s, and a 1,000,003-point transform in 0.83 s, with
errors near 10⁻¹⁵.

**Convolution.** By the convolution theorem, convolving two signals
equals multiplying their FFTs. Zero-padding both to at least
n + m − 1 points avoids circular wrap-around. On a 2¹⁶-point signal
the FFT method cost a flat 24–30 ms. It matched numpy's C direct
convolution at about 2,000 taps and was 3.6× faster at 8,193 taps. In 2-D on a 256×256 image, the crossover against direct
shift-add came at about a 61×61 kernel. Below that, numpy's
vectorized direct method won.

**Spectral analysis** (`signal_analysis.png`):

- **Windows.** Window sidelobes measured −13.3 dB (rectangular),
  −31.5 dB (Hann), −42.4 dB (Hamming) and −58.1 dB (Blackman), in
  exchange for wider main lobes.
- **Peak frequencies.** Parabolic interpolation located them to 1/100
  of a bin with Hann and 1/240 of a bin with Blackman. The rectangular
  window kept a 0.24 Hz bias at every SNR.
- **PSD variance.** The periodogram's relative standard deviation on
  white noise was 1.00, and more data does not reduce it. Averaging 127
  Welch segments cut it to 0.089, exactly 1/√127.
- **Filters.**
  - FFT low-pass filters suppressed stopband tones by 72 dB.
  - A brick-wall mask overshoots by 8.4% on a step (Gibbs ringing); a
    1,200 Hz raised-cosine transition reduced that to 5.3%.
  - For 101-tap FIR filters, the window set the stopband: −31 dB for
    rectangular, −75 dB for Blackman.

## Image and Audio Results

**Compression** (`image_processing.png`, `image_examples.png`) follows
the JPEG pipeline:
1. 8×8 block DCT.
2. Quantization with the JPEG luminance table, scaled by quality.
3. A zig-zag run-length entropy estimate.

The DCT itself is also implemented through a single FFT and checked
against the DCT matrix. Results on the synthetic "shapes" image:

| Quality | PSNR | SSIM | Bits per pixel | File ratio |
|---|---|---|---|---|
| 5 | 26.3 dB | 0.887 | 0.12 | 25:1 |
| 50 | 34.6 dB | 0.968 | 0.41 | 11:1 |
| 95 | 51.9 dB | 0.999 | 0.97 | 6:1 |

The file ratio comes from writing the quantized coefficients with zlib.
A busier texture image needed about twice the bits at the same
quality setting (0.78 vs 0.41 at quality 50).

**Image filters:**
- Gaussian blur raised PSNR from 20.0 to 25.6 dB at noise σ = 0.1. A
  radial FFT low-pass reached 24.7 dB, but at σ = 0.05 it slightly
  hurt (25.7 vs 26.0 dB) because it also removes real edges.
- Periodic stripe noise is a pair of spikes in the 2-D spectrum.
  Notching them lifted PSNR from 17.0 to 32.9 dB.

**Audio.** The audio processor reads and writes WAV with Python's
standard `wave` module.
- **Hum removal with FFT notches** left 16.5 dB of hum. The clip is
  not a whole number of hum periods, so the tone's abrupt start and end
  leak through at the edges.
- **Least-squares subtraction** estimates each harmonic's frequency
  from the spectrum, fits its amplitude and phase, and subtracts it.
  This reached 46.7 dB.
- **Spectral gating** attenuates time–frequency cells near a noise
  floor estimated from the STFT. It improved SNR by 5.8 dB at 0 dB
  input and 7.8 dB at 20 dB.

  My first version estimated the floor per frequency over time. A
  steady tone then counted as its own noise floor and was erased,
  dropping SNR from 16 to 1 dB. Capping the floor at a multiple of the
  global median fixed it.
- **End to end,** the demo cleaned a 3-second chord from 4.1 dB to
  26.2 dB SNR.

**Time series.** Autocorrelation via the FFT of the power spectrum
took 3.6 ms against 31 ms directly at n = 16,000.

## Stability Considerations

**Conditioning** (`numerical_stability.png`). The Hilbert matrix's
condition number κ₂ grows about e^{3.5n} and reaches 1/ε at n = 12.
Solve errors stayed just below κ·ε throughout. For example, at n = 8
(κ = 1.5·10¹⁰) the error was 1.2·10⁻⁷. Past 1/ε, results are
meaningless: at n = 12 the error was 22%, and at n = 14 it was 890%.

My first κ₂ implementation formed AᵀA, which squares the condition
number. It reported 6·10⁸ for Hilbert(12) instead of 1.7·10¹⁶, so it
silently under-reported exactly the matrices it exists to flag.
Inverse iteration with LU solves on A and Aᵀ now matches numpy up to
10¹⁶. `check_conditioning` turns κ into a verdict and the number of
digits expected to be lost.

**QR orthogonality at κ = 10⁹** (40×40):

| Method | ‖QᵀQ − I‖ |
|---|---|
| Classical Gram–Schmidt | 6.4 (orthogonality gone) |
| Modified Gram–Schmidt | 8·10⁻⁸ |
| Householder | 5·10⁻¹⁵ (unchanged at κ = 10¹⁵) |

All three reconstruct A to 10⁻¹⁶. Only orthogonality distinguishes
them.

**Cancellation.**
- (1 − cos x)/x² at x = 10⁻⁸ returned 0, a 100% error;
  2sin²(x/2)/x² is exact.
- The textbook quadratic formula lost 25% on x² + 10⁸x + 1.
- The E[x²] − E[x]² variance formula was off by 669% for data near
  10⁹; Welford's method was exact.
- The Taylor series for e⁻²⁰ was 152% wrong; 1/e²⁰ was exact.
- Summing 10⁶ terms naively gave a relative error of 9.8·10⁻¹⁵; Kahan
  summation was exact.

A 400×400 determinant (~10⁴⁰⁰) overflows a double, so the toolkit
also provides a log-determinant.

## Benchmarking Plots

`fft_performance.png` (FFT runtime, accuracy, convolution crossover),
`matrix_performance.png`, `numerical_stability.png`,
`signal_analysis.png`, `image_processing.png`, `image_examples.png`
and `time_series_analysis.png` hold every result above.
`fft_stages.gif` animates the butterfly network stage by stage, and
`filter_animation.gif` sweeps a low-pass cutoff downward.

## Final Reflections

The FFT's O(n log n) cost is what makes signal and image processing
feasible at all: at 4,096 points it is already 2,000× faster than the
DFT. Implementation decided the constant factors. Vectorizing each
butterfly stage gained 8× over recursion, and numpy's C still won by 5×
over that.

The stability work was the more instructive half, because the
failures were silent:
- A condition number computed through AᵀA.
- A power method that "converged" one step in.
- A noise floor that erased the signal.
- A Householder Q too large to store.

Each returned plausible numbers until checked against an independent
reference.
