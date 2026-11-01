"""
Seeded generators for flow networks and application inputs.

Networks come back as (FlowNetwork, s, t) with s = 0 and t = n − 1.

    sparse_network     ~avg_degree·n random arcs (m = O(n)); s and t
                       attach to ~√n random nodes each, so the max flow
                       is spread over many paths rather than limited by
                       one or two terminal edges
    dense_network      each ordered pair is an arc with probability
                       `density` (m = Θ(n²))
    grid_network       h × w 4-neighbour grid with s and t attached to
                       every cell — the shape of a graph-cut
                       segmentation problem
    bipartite_graph    random left/right edge list for matching
    synthetic_image    grayscale shapes + Gaussian noise, with the
                       ground-truth mask

Capacities are drawn by sample_capacities() from one of
CAPACITY_DISTRIBUTIONS (integers ≥ 1):
    "uniform"      U{low, …, high}
    "unit"         all 1 (unit-capacity networks, e.g. matching)
    "exponential"  geometric-like, mean ≈ (low + high) / 4: many small,
                   a few large
    "powerlaw"     Pareto(α = 1.5) scaled from low, capped at high:
                   heavy-tailed, the case where capacity scaling helps
"""
from typing import List, Optional, Tuple

import numpy as np

from src.flow.core.network import FlowNetwork

CAPACITY_DISTRIBUTIONS = ("uniform", "unit", "exponential", "powerlaw")


def sample_capacities(k: int, dist: str = "uniform", low: int = 1,
                      high: int = 100,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """k integer capacities in [low, high] from the named distribution."""
    if dist not in CAPACITY_DISTRIBUTIONS:
        raise ValueError(f"unknown capacity distribution {dist!r}; choose "
                         f"from {CAPACITY_DISTRIBUTIONS}")
    if low < 1 or high < low:
        raise ValueError("need 1 <= low <= high")
    rng = rng if rng is not None else np.random.default_rng()
    if dist == "uniform":
        return rng.integers(low, high + 1, size=k)
    if dist == "unit":
        return np.ones(k, dtype=np.int64)
    if dist == "exponential":
        values = low + rng.exponential((high - low) / 4 + 1e-9, size=k)
    else:
        values = low * (1 + rng.pareto(1.5, size=k))
    return np.clip(np.round(values), low, high).astype(np.int64)


def _add_arcs(net: FlowNetwork, us, vs, caps, costs=None) -> None:
    add = net.add_edge
    if costs is None:
        for u, v, c in zip(us.tolist(), vs.tolist(), caps.tolist()):
            add(u, v, c)
    else:
        for u, v, c, w in zip(us.tolist(), vs.tolist(), caps.tolist(),
                              costs.tolist()):
            add(u, v, c, w)


def _costs(rng, k: int, cost_range: Optional[Tuple[int, int]]):
    if cost_range is None:
        return None
    return rng.integers(cost_range[0], cost_range[1] + 1, size=k)


def _attach_terminals(net: FlowNetwork, rng, s: int, t: int, inner: int,
                      k: int, dist: str, cap_range, cost_range) -> None:
    """s → k random inner nodes, k random inner nodes → t."""
    k = max(1, min(k, inner))
    for side in ("out", "in"):
        nodes = rng.choice(inner, size=k, replace=False) + 1
        caps = sample_capacities(k, dist, *cap_range, rng=rng)
        costs = _costs(rng, k, cost_range)
        ends = np.full(k, s if side == "out" else t)
        if side == "out":
            _add_arcs(net, ends, nodes, caps, costs)
        else:
            _add_arcs(net, nodes, ends, caps, costs)


def sparse_network(n: int, avg_degree: float = 4.0, capacity: str = "uniform",
                   cap_range: Tuple[int, int] = (1, 100),
                   cost_range: Optional[Tuple[int, int]] = None,
                   terminal_degree: Optional[int] = None,
                   seed: Optional[int] = None) -> Tuple[FlowNetwork, int, int]:
    """
    Random sparse network: n nodes, ≈ avg_degree·(n − 2) arcs among the
    inner nodes (no self-loops or duplicates) plus terminal arcs.
    """
    if n < 3:
        raise ValueError("sparse_network needs n >= 3")
    rng = np.random.default_rng(seed)
    inner = n - 2
    m = int(avg_degree * inner)
    us = rng.integers(0, inner, size=int(m * 1.1) + 8)
    vs = rng.integers(0, inner, size=us.size)
    keys = us * inner + vs
    keys = keys[us != vs]
    _, first = np.unique(keys, return_index=True)
    keys = keys[np.sort(first)][:m]
    us, vs = keys // inner + 1, keys % inner + 1
    net = FlowNetwork(n)
    _add_arcs(net, us, vs, sample_capacities(us.size, capacity, *cap_range,
                                             rng=rng),
              _costs(rng, us.size, cost_range))
    k = terminal_degree if terminal_degree is not None else int(np.sqrt(n)) + 1
    _attach_terminals(net, rng, 0, n - 1, inner, k, capacity, cap_range,
                      cost_range)
    return net, 0, n - 1


def dense_network(n: int, density: float = 0.5, capacity: str = "uniform",
                  cap_range: Tuple[int, int] = (1, 100),
                  cost_range: Optional[Tuple[int, int]] = None,
                  seed: Optional[int] = None) -> Tuple[FlowNetwork, int, int]:
    """
    Random dense network: every ordered pair of distinct nodes
    (other than into s or out of t) is an arc with probability density.
    """
    if n < 2:
        raise ValueError("dense_network needs n >= 2")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    mask[:, 0] = False                   # nothing enters s
    mask[n - 1, :] = False               # nothing leaves t
    us, vs = np.nonzero(mask)
    net = FlowNetwork(n)
    _add_arcs(net, us, vs, sample_capacities(us.size, capacity, *cap_range,
                                             rng=rng),
              _costs(rng, us.size, cost_range))
    return net, 0, n - 1


def grid_edges(h: int, w: int) -> List[Tuple[int, int]]:
    """Undirected 4-neighbour pairs (p, q), p < q, of an h × w grid
    with cells numbered row-major from 0."""
    pairs = []
    for r in range(h):
        for c in range(w):
            p = r * w + c
            if c + 1 < w:
                pairs.append((p, p + 1))
            if r + 1 < h:
                pairs.append((p, p + w))
    return pairs


def grid_network(h: int, w: int, capacity: str = "uniform",
                 cap_range: Tuple[int, int] = (1, 100),
                 terminal_fraction: float = 0.5,
                 seed: Optional[int] = None) -> Tuple[FlowNetwork, int, int]:
    """
    Segmentation-shaped network: cell (r, c) is node 1 + r·w + c,
    neighbours are joined in both directions, and each cell gets one
    terminal arc — from s with probability terminal_fraction, else to
    t — as after the usual reparametrization of a graph-cut energy.
    """
    if h < 1 or w < 1:
        raise ValueError("grid needs h, w >= 1")
    rng = np.random.default_rng(seed)
    cells = h * w
    n = cells + 2
    s, t = 0, n - 1
    net = FlowNetwork(n)
    pairs = np.array(grid_edges(h, w), dtype=np.int64).reshape(-1, 2) + 1
    caps = sample_capacities(2 * len(pairs), capacity, *cap_range, rng=rng)
    _add_arcs(net, pairs[:, 0], pairs[:, 1], caps[:len(pairs)])
    _add_arcs(net, pairs[:, 1], pairs[:, 0], caps[len(pairs):])
    from_source = rng.random(cells) < terminal_fraction
    tcaps = sample_capacities(cells, capacity, *cap_range, rng=rng)
    cell_ids = np.arange(1, cells + 1)
    _add_arcs(net, np.full(from_source.sum(), s), cell_ids[from_source],
              tcaps[from_source])
    _add_arcs(net, cell_ids[~from_source], np.full((~from_source).sum(), t),
              tcaps[~from_source])
    return net, s, t


def bipartite_graph(n_left: int, n_right: int, avg_degree: float = 3.0,
                    seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Random bipartite edges (l, r), l in range(n_left), r in
    range(n_right); each left vertex gets ~Poisson(avg_degree) distinct
    neighbours (at least one).
    """
    if n_left < 0 or n_right < 0 or (n_left and not n_right):
        raise ValueError("need n_left >= 0 and n_right >= 1")
    rng = np.random.default_rng(seed)
    edges = []
    for left in range(n_left):
        k = int(min(n_right, max(1, rng.poisson(avg_degree))))
        for right in rng.choice(n_right, size=k, replace=False).tolist():
            edges.append((left, right))
    return edges


def synthetic_image(h: int = 64, w: int = 64, shape: str = "circle",
                    noise: float = 0.2, foreground: float = 0.75,
                    background: float = 0.25,
                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grayscale test image in [0, 1] and its boolean foreground mask.

    shape: "circle" (one disc), "shapes" (disc, rectangle and a ring),
    or "rect". Gaussian noise with std `noise` is added, then clipped.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:h, 0:w]
    y, x = rows / max(h - 1, 1), cols / max(w - 1, 1)
    if shape == "circle":
        mask = (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.3 ** 2
    elif shape == "rect":
        mask = (abs(x - 0.5) < 0.3) & (abs(y - 0.5) < 0.2)
    elif shape == "shapes":
        disc = (x - 0.28) ** 2 + (y - 0.3) ** 2 < 0.18 ** 2
        rect = (abs(x - 0.72) < 0.16) & (abs(y - 0.3) < 0.12)
        r2 = (x - 0.5) ** 2 + (y - 0.75) ** 2
        ring = (r2 < 0.18 ** 2) & (r2 > 0.09 ** 2)
        mask = disc | rect | ring
    else:
        raise ValueError(f"unknown shape {shape!r}")
    image = np.where(mask, foreground, background).astype(float)
    image += rng.normal(0, noise, size=image.shape)
    return np.clip(image, 0, 1), mask


def random_jobs(num_jobs: int, horizon: int, max_length: int = 4,
                seed: Optional[int] = None) -> dict:
    """
    {job: (processing_time, release, deadline)} with deadline ≤ horizon
    and each window at least as long as the processing time.
    """
    rng = np.random.default_rng(seed)
    jobs = {}
    for j in range(num_jobs):
        p = int(rng.integers(1, max_length + 1))
        release = int(rng.integers(0, max(1, horizon - p + 1)))
        slack = int(rng.integers(0, max(1, horizon - release - p + 1)))
        jobs[f"J{j}"] = (p, release, release + p + slack)
    return jobs
