"""
Set Cover by LP relaxation and rounding.

Given a universe U = {0, ..., n−1} and sets S_1..S_m with costs c_j,
choose the cheapest collection of sets whose union is U.

Integer program:

    minimize    Σ_j c_j x_j
    subject to  Σ_{j : e ∈ S_j} x_j ≥ 1      for every element e
                x_j ∈ {0, 1}

LP relaxation: replace x_j ∈ {0, 1} with x_j ≥ 0. (x_j ≤ 1 is never
binding with positive costs.) Every integer solution is also an LP
solution, so LP* ≤ OPT: the LP value is a lower bound we can compute
in polynomial time, even when OPT is out of reach.

Two ways to turn the fractional x* into a real cover:

  Deterministic (threshold) rounding — take every set with x_j ≥ 1/f,
  where f is the largest number of sets containing any one element.
    Feasible: an element's ≤ f sets have x-values summing to ≥ 1, so
    one of them is ≥ 1/f.
    Cost: each chosen set has 1 ≤ f·x_j, so cost ≤ f·LP* ≤ f·OPT.
    (f = 2 is Vertex Cover, where this is the LP 2-approximation.)

  Randomized rounding — in each of t rounds, take set j independently
  with probability x_j. An element is missed in one round with
  probability Π(1 − x_j) ≤ e^(−Σx_j) ≤ 1/e, so after t = ⌈ln n⌉ + 1
  rounds it is missed with probability ≤ 1/(e·n). Expected cost ≤
  t·LP* = O(log n)·OPT. Any element still uncovered is patched with
  the cheapest set containing it (rare, and it keeps the output always
  feasible).

The classic greedy algorithm (most new elements per unit cost) is an
H_n ≈ ln n approximation and serves as the baseline. Unless P = NP,
no polynomial algorithm beats (1 − ε)·ln n (Dinur–Steurer 2014), so
the O(log n) of rounding and greedy is essentially the best possible.
"""
import heapq
import math
import random
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Sequence

from src.utils.lp_solver import solve_lp
from src.utils.timer import CallStats


@dataclass
class SetCoverInstance:
    num_elements: int
    sets: List[FrozenSet[int]]
    costs: List[float]

    def __post_init__(self) -> None:
        if len(self.sets) != len(self.costs):
            raise ValueError("need one cost per set")
        if any(c <= 0 for c in self.costs):
            raise ValueError("costs must be positive")
        self.sets = [frozenset(s) for s in self.sets]
        covered = frozenset().union(*self.sets) if self.sets else frozenset()
        if covered != frozenset(range(self.num_elements)):
            raise ValueError("the sets do not cover the universe")

    @property
    def max_frequency(self) -> int:
        """f: the most sets any single element belongs to."""
        counts = [0] * self.num_elements
        for s in self.sets:
            for e in s:
                counts[e] += 1
        return max(counts, default=0)


@dataclass
class Cover:
    chosen: List[int]                      # indices of chosen sets
    cost: float
    repaired: int = 0                      # sets added to patch gaps
    extra: dict = field(default_factory=dict)


def is_cover(inst: SetCoverInstance, chosen: Sequence[int]) -> bool:
    covered = set()
    for j in chosen:
        covered |= inst.sets[j]
    return len(covered) == inst.num_elements


def cover_cost(inst: SetCoverInstance, chosen: Sequence[int]) -> float:
    return sum(inst.costs[j] for j in set(chosen))


def _cover(inst: SetCoverInstance, chosen, repaired: int = 0) -> Cover:
    chosen = sorted(set(chosen))
    return Cover(chosen, cover_cost(inst, chosen), repaired)


# ---------------------------------------------------------------------------
# LP relaxation and rounding
# ---------------------------------------------------------------------------

def lp_relaxation(inst: SetCoverInstance) -> tuple:
    """
    Solve the LP relaxation.

    Returns:
        (x, LP*) — fractional values per set and the LP optimum.
    """
    a = [[1.0 if e in s else 0.0 for s in inst.sets]
         for e in range(inst.num_elements)]
    result = solve_lp(inst.costs, A_ge=a, b_ge=[1.0] * inst.num_elements)
    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.status}")
    return [float(v) for v in result.x], result.objective


def deterministic_rounding(inst: SetCoverInstance,
                           x: Optional[Sequence[float]] = None) -> Cover:
    """Take every set with x_j ≥ 1/f. Cost ≤ f·LP*."""
    if x is None:
        x, _ = lp_relaxation(inst)
    threshold = 1.0 / inst.max_frequency - 1e-9
    return _cover(inst, [j for j, v in enumerate(x) if v >= threshold])


def default_rounds(num_elements: int) -> int:
    return math.ceil(math.log(max(num_elements, 2))) + 1


def randomized_rounding(inst: SetCoverInstance,
                        x: Optional[Sequence[float]] = None,
                        rounds: Optional[int] = None,
                        rng: Optional[random.Random] = None) -> Cover:
    """
    Take each set with probability x_j, repeated `rounds` times
    (default ⌈ln n⌉ + 1); patch any uncovered element with its cheapest
    set. Always returns a valid cover; Cover.repaired counts patches.
    """
    if x is None:
        x, _ = lp_relaxation(inst)
    rng = rng or random
    rounds = rounds if rounds is not None else default_rounds(
        inst.num_elements)
    chosen = set()
    for _ in range(rounds):
        for j, v in enumerate(x):
            if v > 0 and rng.random() < v:
                chosen.add(j)
    covered = set()
    for j in chosen:
        covered |= inst.sets[j]
    repaired = 0
    for e in range(inst.num_elements):
        if e not in covered:
            j = min((j for j, s in enumerate(inst.sets) if e in s),
                    key=lambda j: inst.costs[j])
            chosen.add(j)
            covered |= inst.sets[j]
            repaired += 1
    return _cover(inst, chosen, repaired)


# ---------------------------------------------------------------------------
# Baselines: greedy and exact
# ---------------------------------------------------------------------------

def greedy_set_cover(inst: SetCoverInstance) -> Cover:
    """
    Repeatedly take the set with the lowest cost per newly covered
    element. H_n-approximation (Chvátal 1979).

    Lazy heap: a set's ratio only gets worse as elements are covered,
    so a popped entry is re-checked and re-pushed if it is stale.

    Complexity: O(Σ|S_j| · log m).
    """
    uncovered = set(range(inst.num_elements))
    heap = [(c / len(s), j) for j, (s, c) in
            enumerate(zip(inst.sets, inst.costs)) if s]
    heapq.heapify(heap)
    chosen = []
    while uncovered:
        ratio, j = heapq.heappop(heap)
        new = len(inst.sets[j] & uncovered)
        if new == 0:
            continue
        current = inst.costs[j] / new
        if current > ratio + 1e-12:
            heapq.heappush(heap, (current, j))       # stale: try later
            continue
        chosen.append(j)
        uncovered -= inst.sets[j]
    return _cover(inst, chosen)


def exact_set_cover(inst: SetCoverInstance,
                    stats: Optional[CallStats] = None) -> Cover:
    """
    Optimal cover by branch and bound over bitmasks.

    Branch on the uncovered element with the fewest candidate sets: one
    of them must be chosen. Prune when cost + lower bound ≥ best, where
    the bound charges every uncovered element at least the cheapest
    cost per element any set can offer for the remaining elements.
    Greedy supplies the first upper bound.

    Complexity: exponential in the worst case; fine for ~30–60 sets.
    """
    stats = stats if stats is not None else CallStats()
    masks = [sum(1 << e for e in s) for s in inst.sets]
    costs = inst.costs
    full = (1 << inst.num_elements) - 1
    containing = [[j for j, mk in enumerate(masks) if mk >> e & 1]
                  for e in range(inst.num_elements)]
    for opts in containing:
        opts.sort(key=lambda j: costs[j])
    greedy = greedy_set_cover(inst)
    best = [greedy.cost, list(greedy.chosen)]

    def bound(uncovered: int) -> float:
        left = bin(uncovered).count("1")
        rate = min(costs[j] / bin(masks[j] & uncovered).count("1")
                   for j in range(len(masks)) if masks[j] & uncovered)
        return left * rate

    def search(covered: int, cost: float, chosen: List[int]) -> None:
        stats.enter()
        try:
            if covered == full:
                if cost < best[0] - 1e-12:
                    best[0], best[1] = cost, list(chosen)
                return
            uncovered = full & ~covered
            if cost + bound(uncovered) >= best[0] - 1e-12:
                return
            e = min((e for e in range(inst.num_elements)
                     if uncovered >> e & 1),
                    key=lambda e: len(containing[e]))
            for j in containing[e]:
                chosen.append(j)
                search(covered | masks[j], cost + costs[j], chosen)
                chosen.pop()
        finally:
            stats.exit()

    search(0, 0.0, [])
    return _cover(inst, best[1])


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

def random_set_cover(num_elements: int, num_sets: int,
                     density: float = 0.1,
                     cost_range: tuple = (1.0, 10.0),
                     seed: Optional[int] = None) -> SetCoverInstance:
    """
    Each element joins each set independently with probability
    `density`; any element left in no set joins one at random. Costs are
    uniform in cost_range.
    """
    if num_elements < 1 or num_sets < 1:
        raise ValueError("need at least one element and one set")
    rng = random.Random(seed)
    members = [set() for _ in range(num_sets)]
    for e in range(num_elements):
        hit = [j for j in range(num_sets) if rng.random() < density]
        if not hit:
            hit = [rng.randrange(num_sets)]
        for j in hit:
            members[j].add(e)
    # an empty set is useless but legal; give it one element so every
    # set is a real option
    for s in members:
        if not s:
            s.add(rng.randrange(num_elements))
    costs = [round(rng.uniform(*cost_range), 2) for _ in range(num_sets)]
    return SetCoverInstance(num_elements, [frozenset(s) for s in members],
                            costs)


def greedy_bad_instance(k: int) -> SetCoverInstance:
    """
    Unit costs where greedy pays k while OPT = 2.

    Universe of 2(2^k − 1) elements in two rows. OPT takes the two rows.
    Greedy is lured by column blocks of sizes 2, 4, ..., 2^k, each just
    larger than half of what remains in a row — so it picks all k
    blocks: ratio k/2 = Θ(log n).
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    half = 2 ** k - 1                   # elements per row
    top = frozenset(range(half))
    bottom = frozenset(range(half, 2 * half))
    sets = [top, bottom]
    start = 0
    for i in range(k):
        width = 2 ** i
        cols = range(start, start + width)
        sets.append(frozenset(cols) | frozenset(c + half for c in cols))
        start += width
    return SetCoverInstance(2 * half, sets, [1.0] * len(sets))
