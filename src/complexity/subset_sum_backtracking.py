"""
Subset Sum: given non-negative integers and a target T, is there a
subset that adds up to exactly T?

Subset Sum is NP-complete (src/complexity/reduction_demo.py reduces
3-SAT to it). Three ways to attack it:

    subset_sum_backtracking   exact; explores the include/exclude
                              decision tree with pruning        O(2^n)
    subset_sum_dp             exact; bitset of reachable sums   O(n·T)
    greedy_subset_sum         heuristic; largest-first          O(n log n)

The DP looks polynomial but is not: T is written in log T bits, so
O(n·T) is exponential in the input's length (pseudo-polynomial). With
values around 10^12 its table would need 10^12 bits, while
backtracking's cost depends only on n. With values below a few
thousand, the DP is the fastest exact method by far.

Greedy guarantee: sort descending and take every item that still
fits. If it skips some item x ≤ T, then (current sum) + x > T, and the
current sum already contains an item at least as large as x, so the
sum is more than T/2. Otherwise it took every item ≤ T, which is
optimal. Either way greedy reaches at least half of the best
achievable sum: a 1/2-approximation for the optimization version.
"""
import random
from typing import List, Optional, Sequence, Tuple

from src.utils.timer import CallStats


def _validate(nums: Sequence[int], target: int) -> None:
    if target < 0:
        raise ValueError(f"target must be non-negative, got {target}")
    for x in nums:
        if x < 0:
            raise ValueError(f"values must be non-negative, got {x}")


def is_subset_sum_solution(nums: Sequence[int], target: int,
                           indices: Sequence[int]) -> bool:
    """Certificate check: distinct valid indices whose values sum to target."""
    return (len(set(indices)) == len(indices)
            and all(0 <= i < len(nums) for i in indices)
            and sum(nums[i] for i in indices) == target)


def subset_sum_backtracking(nums: Sequence[int], target: int,
                            stats: Optional[CallStats] = None
                            ) -> Optional[List[int]]:
    """
    Exact Subset Sum by depth-first search.

    Each level of the tree decides one item: include it, or not. Items
    are visited largest first, and a branch is abandoned when

        remaining < 0                         (overshot: include skipped
                                               when the item doesn't fit)
        sum of all items still undecided < remaining   (can't reach T)

    Both bounds prune enormous subtrees on "yes" instances, but on a
    "no" instance the search must rule out every subset, and the
    bounds cut only a constant fraction of the 2^(n+1) − 1 nodes.

    Args:
        nums: Non-negative integers.
        target: Non-negative integer T.
        stats: Optional counter; stats.calls = nodes visited,
            stats.max_depth = deepest level reached.

    Returns:
        Sorted indices into `nums` of a subset summing to target, or
        None if no subset does. target = 0 gives [].

    Complexity: O(2^n) time, O(n) space.
    """
    _validate(nums, target)
    stats = stats if stats is not None else CallStats()
    order = sorted(range(len(nums)), key=lambda i: -nums[i])
    values = [nums[i] for i in order]
    n = len(values)
    suffix = [0] * (n + 1)                 # suffix[i] = sum(values[i:])
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]
    chosen: List[int] = []

    def search(i: int, remaining: int) -> bool:
        stats.enter()
        try:
            if remaining == 0:
                return True
            if i == n or suffix[i] < remaining:
                return False
            if values[i] <= remaining:
                chosen.append(i)
                if search(i + 1, remaining - values[i]):
                    return True
                chosen.pop()
            return search(i + 1, remaining)
        finally:
            stats.exit()

    if not search(0, target):
        return None
    return sorted(order[i] for i in chosen)


def subset_sum_dp(nums: Sequence[int], target: int) -> int:
    """
    Largest subset sum that does not exceed target (exact).

    Bit s of `reachable` is set when some subset sums to s. Adding item
    x shifts every reachable sum up by x: reachable |= reachable << x.
    Sums above target are masked off. The answer to the decision
    problem is "yes" exactly when the result equals target.

    Python's big integers do the shift-and-OR a machine word at a time,
    so this costs O(n·T / 64) word operations — fast for small values,
    hopeless for T ≈ 10^12.

    Complexity: O(n·T) bit operations, O(T) bits of space.
    """
    _validate(nums, target)
    mask = (1 << (target + 1)) - 1
    reachable = 1                           # the empty subset sums to 0
    for x in nums:
        reachable = (reachable | (reachable << x)) & mask
    return reachable.bit_length() - 1


def greedy_subset_sum(nums: Sequence[int], target: int) -> List[int]:
    """
    Heuristic: largest item first, take each one that still fits.

    Never exceeds target and always reaches at least half the best
    achievable sum (see module docstring), but may miss an exact
    solution that exists.

    Returns:
        Sorted indices of the chosen items.

    Complexity: O(n log n) for the sort, then O(n).
    """
    _validate(nums, target)
    chosen = []
    remaining = target
    for i in sorted(range(len(nums)), key=lambda i: -nums[i]):
        if nums[i] <= remaining:
            chosen.append(i)
            remaining -= nums[i]
            if remaining == 0:
                break
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------

def random_subset_sum(n: int, max_value: int = 1_000,
                      solvable: bool = True,
                      seed: Optional[int] = None) -> Tuple[List[int], int]:
    """
    n random values in [1, max_value] and a target.

    solvable=True plants a solution: the target is the sum of a random
    half of the values. solvable=False aims at half the total, which
    with large max_value (≈ 10^12) is almost surely unreachable:
    2^n subsets cannot cover ~n·10^12 possible sums. That is the worst
    case for backtracking, which must then explore the whole tree.

    Returns:
        (values, target)
    """
    if n < 0 or max_value < 1:
        raise ValueError("need n >= 0 and max_value >= 1")
    rng = random.Random(seed)
    nums = [rng.randint(1, max_value) for _ in range(n)]
    if solvable:
        target = sum(rng.sample(nums, n // 2))
    else:
        target = sum(nums) // 2
    return nums, target
