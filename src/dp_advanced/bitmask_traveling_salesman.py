"""
Traveling Salesman Problem — state compression with bitmasks.

Visit every city exactly once and return to the start (city 0) at
minimum total distance.

Brute force tries every order of the other n-1 cities: (n-1)! tours.
That is wasteful because tours share structure. After visiting the
set S of cities and standing at city i, the cheapest way to finish
does NOT depend on the order S was visited in — only on S and i. That
is the DP state, and S is stored as a bitmask (src/utils/bitmask_utils):

    dp[mask][i] = minimum cost of a path that starts at city 0,
                  visits exactly the cities in `mask`, and ends at i

    dp[{0}][0] = 0
    dp[mask | bit(j)][j] = min(dp[mask][i] + dist[i][j])   for j not in mask
    answer = min over i of dp[ALL][i] + dist[i][0]

This is Held–Karp. There are 2ⁿ masks × n endpoints = O(n·2ⁿ) states,
each extended in O(n) ways: O(n²·2ⁿ) time. Still exponential — TSP is
NP-hard — but n²·2ⁿ grows far slower than n!: at n = 12 the bound is
about 590,000 (56,331 transitions actually evaluated, since a state
only extends to unvisited cities) versus 39,916,800 brute-force tours.

Bit operations do all the set work: `mask >> j & 1` tests membership,
`mask | (1 << j)` adds a city, and `unvisited & -unvisited` peels off
one unvisited city at a time without scanning visited ones.
"""
import itertools
from typing import List, Optional, Sequence, Tuple

from src.utils.bitmask_utils import full_mask
from src.utils.timer import CallStats

INF = float("inf")
MAX_BITMASK_CITIES = 20      # 2^20 × 20 states ≈ 21M cells
MAX_BRUTE_FORCE_CITIES = 13  # 12! ≈ 479M tours


def _validate(dist: Sequence[Sequence[float]], limit: int) -> int:
    n = len(dist)
    if n == 0:
        raise ValueError("need at least one city")
    for row in dist:
        if len(row) != n:
            raise ValueError("distance matrix must be square")
    if n > limit:
        raise ValueError(f"{n} cities exceeds the limit of {limit} "
                         "for this method")
    return n


def tour_cost(dist: Sequence[Sequence[float]], tour: Sequence[int]) -> float:
    """Total length of a tour given as [0, c1, ..., c_{n-1}, 0]."""
    return sum(dist[a][b] for a, b in zip(tour, tour[1:]))


def tsp_bitmask(dist: Sequence[Sequence[float]],
                stats: Optional[CallStats] = None
                ) -> Tuple[float, Optional[List[int]]]:
    """
    Optimal tour by Held–Karp bitmask DP.

    Args:
        dist: n×n distance matrix; dist[i][j] is the cost of i -> j
            (need not be symmetric; INF marks a missing road).
        stats: Optional counter; receives one count per transition
            dp[mask][i] -> dp[mask | bit(j)][j] evaluated.

    Returns:
        (minimum cost, tour) where tour = [0, ..., 0] lists the cities
        in visiting order. A single city gives (0, [0]). If no tour
        exists (missing edges), returns (INF, None).

    Raises:
        ValueError: If dist is empty, not square, or has more than
            MAX_BITMASK_CITIES cities.

    Complexity: O(n²·2ⁿ) time, O(n·2ⁿ) space.
    """
    n = _validate(dist, MAX_BITMASK_CITIES)
    if n == 1:
        return 0, [0]

    all_cities = full_mask(n)
    size = 1 << n
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    dp[1][0] = 0                      # at city 0, having visited {0}

    transitions = 0
    for mask in range(1, size, 2):    # odd masks: city 0 always visited
        row = dp[mask]
        unvisited_all = all_cities ^ mask
        if not unvisited_all:
            continue
        for last in range(n):
            cost = row[last]
            if cost == INF:
                continue              # unreachable state (or last not in mask)
            dist_last = dist[last]
            unvisited = unvisited_all
            while unvisited:
                low = unvisited & -unvisited       # lowest unvisited city
                nxt = low.bit_length() - 1
                unvisited ^= low
                candidate = cost + dist_last[nxt]
                new_row = dp[mask | low]
                if candidate < new_row[nxt]:
                    new_row[nxt] = candidate
                    parent[mask | low][nxt] = last
                transitions += 1

    best, best_last = INF, -1
    final = dp[all_cities]
    for last in range(1, n):
        total = final[last] + dist[last][0]
        if total < best:
            best, best_last = total, last
    if stats is not None:
        stats.count(transitions)
    if best == INF:
        return INF, None

    # Walk parents backward from the full set, removing one city a step
    tour = [0]
    mask, city = all_cities, best_last
    while city != 0:
        tour.append(city)
        previous = parent[mask][city]
        mask ^= 1 << city
        city = previous
    tour.append(0)
    tour.reverse()
    return best, tour


def tsp_brute_force(dist: Sequence[Sequence[float]],
                    stats: Optional[CallStats] = None
                    ) -> Tuple[float, Optional[List[int]]]:
    """
    Optimal tour by trying all (n-1)! orders of cities 1..n-1.

    Args / Returns: as tsp_bitmask (stats counts tours evaluated).

    Raises:
        ValueError: If dist is empty, not square, or has more than
            MAX_BRUTE_FORCE_CITIES cities.

    Complexity: O(n · (n-1)!) time, O(n) space.
    """
    n = _validate(dist, MAX_BRUTE_FORCE_CITIES)
    if n == 1:
        return 0, [0]

    best, best_order = INF, None
    dist_0 = dist[0]
    tours = 0
    for order in itertools.permutations(range(1, n)):
        cost = dist_0[order[0]]
        previous = order[0]
        for city in order[1:]:
            cost += dist[previous][city]
            previous = city
        cost += dist[previous][0]
        if cost < best:
            best, best_order = cost, order
        tours += 1

    if stats is not None:
        stats.count(tours)
    if best == INF:
        return INF, None
    return best, [0, *best_order, 0]
