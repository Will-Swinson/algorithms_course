"""
Binary image segmentation by s–t minimum cut (Boykov–Jolly graph cuts).

Label every pixel foreground (F) or background (B) to minimize

    E(L) = Σ_p D_p(L_p)  +  Σ_{p~q, L_p ≠ L_q} w_pq

    data term    D_p(F) = λ·(I_p − μ_F)²,  D_p(B) = λ·(I_p − μ_B)²
                 (negative log-likelihood of Gaussian intensity models)
    smoothness   w_pq = exp(−(I_p − I_q)² / 2σ²) for 4-neighbours:
                 cutting between similar pixels is expensive, cutting
                 along a strong edge is cheap

As a network: s stands for foreground, t for background, one node per
pixel.

    s → p   capacity D_p(B)    cut when p ends on the sink side (B)
    p → t   capacity D_p(F)    cut when p stays on the source side (F)
    p ↔ q   capacity w_pq      cut when p and q are separated

Every s–t cut (S, T) is a labeling (S = foreground) whose cut capacity
is exactly E(L), so the minimum cut — found by any max-flow solver,
then read off as the residual-reachable set — is the globally optimal
segmentation. Two refinements:

  * Reparametrization: subtracting min(D_p(F), D_p(B)) from both
    terminal capacities shifts every labeling's energy by the same
    constant, so each pixel needs only one terminal arc.
  * Hard constraints: user seed pixels get an "infinite" (larger than
    every finite cut) arc to their terminal and none to the other.

Capacities are scaled and rounded to integers so all solvers are exact.

Unknown μ_F, μ_B are estimated by 1-D 2-means on the intensities, with
the brighter cluster as foreground (override with fg_mean / bg_mean or
seeds).
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from src.flow.core import get_solver
from src.flow.core.network import FlowNetwork, FlowResult
from src.flow.utils.generators import grid_edges

Pixel = Tuple[int, int]


@dataclass
class SegmentationResult:
    mask: np.ndarray                # True = foreground
    energy: float                   # E(mask) in the unscaled units
    flow: FlowResult
    fg_mean: float
    bg_mean: float
    num_nodes: int
    num_edges: int


def as_grayscale(image) -> np.ndarray:
    """2-D float array in [0, 1] from uint8/float, gray or RGB(A)."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2:
        raise ValueError("image must be 2-D (grayscale) or 3-D (color)")
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def two_means(image: np.ndarray, iterations: int = 50) -> Tuple[float, float]:
    """1-D 2-means on pixel intensities: (bright mean, dark mean)."""
    values = image.ravel()
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return hi, lo
    for _ in range(iterations):
        mid = (lo + hi) / 2
        bright, dark = values[values >= mid], values[values < mid]
        new_hi = float(bright.mean()) if bright.size else hi
        new_lo = float(dark.mean()) if dark.size else lo
        if abs(new_hi - hi) < 1e-9 and abs(new_lo - lo) < 1e-9:
            break
        hi, lo = new_hi, new_lo
    return hi, lo


def threshold_segment(image, threshold: Optional[float] = None) -> np.ndarray:
    """Baseline: foreground = brighter than the midpoint of the 2-means."""
    image = as_grayscale(image)
    if threshold is None:
        fg, bg = two_means(image)
        threshold = (fg + bg) / 2
    return image >= threshold


def build_segmentation_network(image, fg_mean: float, bg_mean: float,
                               smoothness: float = 4.0, data_weight: float = 8.0,
                               sigma: Optional[float] = None,
                               fg_seeds: Iterable[Pixel] = (),
                               bg_seeds: Iterable[Pixel] = (),
                               scale: int = 1000):
    """
    The graph-cut network for `image`. Pixel (r, c) is node r·w + c;
    s = h·w, t = h·w + 1. Returns (net, s, t, offset) where offset is
    the constant removed by reparametrization, so that
    E(L) = (cut capacity of L) / scale + offset.
    """
    image = as_grayscale(image)
    h, w = image.shape
    flat = image.ravel()
    cells = h * w
    s, t = cells, cells + 1
    net = FlowNetwork(cells + 2)

    d_fg = data_weight * (flat - fg_mean) ** 2
    d_bg = data_weight * (flat - bg_mean) ** 2
    low = np.minimum(d_fg, d_bg)
    offset = float(low.sum())
    to_source = np.rint((d_bg - low) * scale).astype(np.int64)
    to_sink = np.rint((d_fg - low) * scale).astype(np.int64)

    pairs = grid_edges(h, w)
    if sigma is None:
        diffs = np.array([flat[p] - flat[q] for p, q in pairs]) if pairs else np.zeros(1)
        sigma = max(float(np.sqrt(np.mean(diffs ** 2))), 1e-3)
    weights = [int(round(smoothness * scale *
                         np.exp(-(flat[p] - flat[q]) ** 2 / (2 * sigma ** 2))))
               for p, q in pairs]

    fg_set = {r * w + c for r, c in fg_seeds}
    bg_set = {r * w + c for r, c in bg_seeds}
    if fg_set & bg_set:
        raise ValueError("a pixel cannot be both a foreground and a "
                         "background seed")
    for p in fg_set | bg_set:
        if not 0 <= p < cells:
            raise ValueError("seed pixel outside the image")
    hard = int(to_source.sum() + to_sink.sum() + 2 * sum(weights) + 1)

    add = net.add_edge
    for p in range(cells):
        if p in fg_set:
            add(s, p, hard)
        elif p in bg_set:
            add(p, t, hard)
        elif to_source[p] > 0:
            add(s, p, int(to_source[p]))
        elif to_sink[p] > 0:
            add(p, t, int(to_sink[p]))
    for (p, q), wt in zip(pairs, weights):
        if wt > 0:
            add(p, q, wt)
            add(q, p, wt)
    return net, s, t, offset


def segment_image(image, fg_mean: Optional[float] = None,
                  bg_mean: Optional[float] = None, smoothness: float = 4.0,
                  data_weight: float = 8.0, sigma: Optional[float] = None,
                  fg_seeds: Iterable[Pixel] = (), bg_seeds: Iterable[Pixel] = (),
                  solver: str = "pr", scale: int = 1000) -> SegmentationResult:
    """
    Optimal foreground/background labeling of a grayscale image by
    minimum s–t cut. Larger `smoothness` (relative to `data_weight`)
    gives smoother, blobbier regions; 0 reduces to per-pixel
    classification.
    """
    image = as_grayscale(image)
    fg_seeds, bg_seeds = list(fg_seeds), list(bg_seeds)
    h, w = image.shape
    for r, c in fg_seeds + bg_seeds:
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError(f"seed pixel ({r}, {c}) outside the {h}x{w} image")
    if fg_mean is None or bg_mean is None:
        bright, dark = two_means(image)
        if fg_seeds:
            fg_est = float(np.mean([image[r, c] for r, c in fg_seeds]))
            bright, dark = (bright, dark) if abs(fg_est - bright) <= abs(fg_est - dark) \
                else (dark, bright)
        fg_mean = bright if fg_mean is None else fg_mean
        bg_mean = dark if bg_mean is None else bg_mean
    net, s, t, offset = build_segmentation_network(
        image, fg_mean, bg_mean, smoothness, data_weight, sigma,
        fg_seeds, bg_seeds, scale)
    result = get_solver(solver)(net, s, t)
    mask = np.zeros(h * w, dtype=bool)
    for v in result.min_cut:
        if v < h * w:
            mask[v] = True
    return SegmentationResult(mask.reshape(h, w),
                              result.value / scale + offset, result,
                              float(fg_mean), float(bg_mean), net.n,
                              net.num_edges)


def labeling_energy(net: FlowNetwork, s: int, mask: np.ndarray,
                    scale: int = 1000, offset: float = 0.0) -> float:
    """Energy of an arbitrary labeling on a network from
    build_segmentation_network (cut capacity of {s} ∪ foreground)."""
    side = [s] + np.flatnonzero(mask.ravel()).tolist()
    return net.cut_capacity(side) / scale + offset


def segmentation_scores(mask: np.ndarray, truth: np.ndarray) -> dict:
    """Pixel accuracy and foreground intersection-over-union."""
    mask, truth = np.asarray(mask, bool), np.asarray(truth, bool)
    union = np.logical_or(mask, truth).sum()
    return {"accuracy": float((mask == truth).mean()),
            "iou": float(np.logical_and(mask, truth).sum() / union)
            if union else 1.0}
