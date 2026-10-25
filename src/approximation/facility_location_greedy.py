"""
Uncapacitated Facility Location (UFL).

Facilities i = 0..m−1 cost f_i to open; client j pays c_ij to be served
by facility i. Open some facilities and connect every client to an open
one, minimizing

    Σ_{i open} f_i  +  Σ_j min_{i open} c_ij.

UFL is NP-hard (Set Cover is the special case with c_ij ∈ {0, ∞}).
Opening a facility trades a fixed cost against cheaper connections, the
same trade-off as warehouses, cell towers or cache servers.

Algorithms:

  greedy_facility_location — the star greedy. A "star" is a facility i
    plus a set of still-unconnected clients. Its cost-effectiveness is

        (f_i' + Σ_{j in star} c_ij) / |star|,   f_i' = 0 once i is open,

    and for each facility the best star is a prefix of its unconnected
    clients sorted by distance. Repeatedly open the most cost-effective
    star and connect its clients, then finally reconnect every client
    to its nearest open facility (which can only lower the cost).
    Guarantee: H_n ≈ ln n in general — the same argument as greedy Set
    Cover. On metric instances (distances obey the triangle inequality)
    this rule is the 1.861-approximation of Mahdian et al. (2001).

  local_search_facility_location — start from a solution and apply
    any improving move: open one facility, close one, or swap one open
    for one closed. On metric instances a local optimum under these
    moves costs ≤ 3·OPT (Arya et al. 2004).

  exact_facility_location — try every nonempty set of open facilities;
    O(2^m · m · n), for small m only.

Unless P = NP, metric UFL cannot be approximated within 1.463
(Guha–Khuller 1999), so the 1.861 greedy is not far off the limit.
"""
import math
import random
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Sequence

import numpy as np

MAX_EXACT_FACILITIES = 18


@dataclass
class FacilityInstance:
    opening_costs: List[float]               # f_i
    connection: List[List[float]]            # c[i][j]: facility i, client j
    facility_points: Optional[list] = None   # for plotting metric instances
    client_points: Optional[list] = None

    def __post_init__(self) -> None:
        m = len(self.opening_costs)
        if m == 0:
            raise ValueError("need at least one facility")
        if len(self.connection) != m:
            raise ValueError("need one connection row per facility")
        n = len(self.connection[0])
        if any(len(row) != n for row in self.connection):
            raise ValueError("connection rows must have equal length")
        if any(f < 0 for f in self.opening_costs):
            raise ValueError("opening costs must be non-negative")

    @property
    def num_facilities(self) -> int:
        return len(self.opening_costs)

    @property
    def num_clients(self) -> int:
        return len(self.connection[0])


@dataclass
class FacilitySolution:
    open: FrozenSet[int]
    assignment: List[int]                    # client j -> facility
    cost: float
    opening_cost: float
    connection_cost: float
    steps: int = 0                           # greedy stars / local moves
    extra: dict = field(default_factory=dict)


def evaluate(inst: FacilityInstance, open_set) -> FacilitySolution:
    """Cost of opening `open_set` and connecting each client to its
    nearest open facility. O(|open| · n)."""
    open_set = frozenset(open_set)
    if not open_set:
        raise ValueError("at least one facility must be open")
    c = inst.connection
    assignment = [min(open_set, key=lambda i: c[i][j])
                  for j in range(inst.num_clients)]
    opening = sum(inst.opening_costs[i] for i in open_set)
    connect = sum(c[i][j] for j, i in enumerate(assignment))
    return FacilitySolution(open_set, assignment, opening + connect,
                            opening, connect)


def greedy_facility_location(inst: FacilityInstance) -> FacilitySolution:
    """
    Star greedy (see module docstring).

    Each facility's clients are presorted by distance once; a round
    scans every facility's sorted list, skipping connected clients, and
    computes the best prefix ratio in one pass.

    Complexity: O(m·n log n) to sort, O(m·n) per round, at most n
    rounds: O(m·n²).
    """
    m, n = inst.num_facilities, inst.num_clients
    c = inst.connection
    order = [sorted(range(n), key=lambda j: c[i][j]) for i in range(m)]
    connected = [False] * n
    remaining = n
    opened = set()
    rounds = 0
    while remaining:
        best_ratio, best_i, best_k = math.inf, -1, 0
        for i in range(m):
            total = 0.0 if i in opened else inst.opening_costs[i]
            k = 0
            for j in order[i]:
                if connected[j]:
                    continue
                total += c[i][j]
                k += 1
                ratio = total / k
                if ratio < best_ratio:
                    best_ratio, best_i, best_k = ratio, i, k
        opened.add(best_i)
        k = 0
        for j in order[best_i]:
            if k == best_k:
                break
            if not connected[j]:
                connected[j] = True
                remaining -= 1
                k += 1
        rounds += 1
    solution = evaluate(inst, opened)
    solution.steps = rounds
    return solution


def local_search_facility_location(inst: FacilityInstance,
                                   start: Optional[Sequence[int]] = None,
                                   max_moves: int = 10_000,
                                   min_gain: float = 1e-9
                                   ) -> FacilitySolution:
    """
    Add / drop / swap local search, taking the best improving move each
    round until none improves by more than `min_gain` (relative).

    Distances are kept in a numpy array so each candidate solution
    costs one vectorized min over its open rows.

    Args:
        start: Initial open facilities (default: the single facility
            that is cheapest on its own).

    Complexity: O(m² · m·n) per round for the swap neighborhood.
    """
    m = inst.num_facilities
    c = np.asarray(inst.connection, dtype=float)
    f = np.asarray(inst.opening_costs, dtype=float)

    def cost(open_list) -> float:
        idx = list(open_list)
        return float(f[idx].sum() + c[idx].min(axis=0).sum())

    if start is None:
        start = [min(range(m), key=lambda i: cost([i]))]
    current = set(start)
    current_cost = cost(current)
    moves = 0
    while moves < max_moves:
        best_cost, best_set = current_cost, None
        closed = [i for i in range(m) if i not in current]
        candidates = [current | {i} for i in closed]
        if len(current) > 1:
            candidates += [current - {i} for i in current]
        candidates += [(current - {i}) | {k} for i in current for k in closed]
        for cand in candidates:
            value = cost(cand)
            if value < best_cost:
                best_cost, best_set = value, cand
        if best_set is None or best_cost > current_cost * (1 - min_gain):
            break
        current, current_cost = best_set, best_cost
        moves += 1
    solution = evaluate(inst, current)
    solution.steps = moves
    return solution


def exact_facility_location(inst: FacilityInstance) -> FacilitySolution:
    """
    Optimal solution by trying every nonempty subset of facilities.

    Subsets are processed in Gray-code order so each one differs from
    the previous by one facility; a client's best distance still needs
    a min over the open rows, done with numpy.

    Complexity: O(2^m · m · n).
    """
    m = inst.num_facilities
    if m > MAX_EXACT_FACILITIES:
        raise ValueError(f"{m} facilities exceeds the exact limit of "
                         f"{MAX_EXACT_FACILITIES}")
    c = np.asarray(inst.connection, dtype=float)
    f = np.asarray(inst.opening_costs, dtype=float)
    best_cost, best_mask = math.inf, 0
    mask = 0
    for k in range(1, 1 << m):
        mask ^= k & -k                      # Gray code: flip one bit
        rows = [i for i in range(m) if mask >> i & 1]
        value = f[rows].sum() + c[rows].min(axis=0).sum()
        if value < best_cost:
            best_cost, best_mask = value, mask
    return evaluate(inst, [i for i in range(m) if best_mask >> i & 1])


def random_metric_instance(num_facilities: int, num_clients: int,
                           opening_range: tuple = (20.0, 60.0),
                           size: float = 100.0,
                           seed: Optional[int] = None) -> FacilityInstance:
    """
    Facilities and clients at random points in a size × size square;
    c_ij is Euclidean distance (so the instance is metric), opening
    costs uniform in opening_range.
    """
    rng = random.Random(seed)
    fac = [(rng.uniform(0, size), rng.uniform(0, size))
           for _ in range(num_facilities)]
    cli = [(rng.uniform(0, size), rng.uniform(0, size))
           for _ in range(num_clients)]
    conn = [[math.dist(p, q) for q in cli] for p in fac]
    costs = [round(rng.uniform(*opening_range), 2)
             for _ in range(num_facilities)]
    return FacilityInstance(costs, conn, fac, cli)
