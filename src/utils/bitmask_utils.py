"""
Bit-manipulation helpers for state-compressed DP.

A bitmask packs a SET of small integers {0, 1, ..., n-1} into one int:
bit i is 1 exactly when i is in the set. For n <= ~25 this turns "which
cities have I visited?" into a single machine-word-sized key, so a DP
table can be indexed by sets the same way it is indexed by numbers:

    {0, 2, 3}  ->  0b1101  ->  13

Every set operation becomes one or two bit operations:

    membership    mask >> i & 1          add      mask | (1 << i)
    remove        mask & ~(1 << i)       size     popcount(mask)
    lowest bit    mask & -mask           all n    (1 << n) - 1
"""
from typing import Iterable, Iterator, List


def bit(i: int) -> int:
    """The mask containing only element i."""
    return 1 << i


def has_bit(mask: int, i: int) -> bool:
    """True if element i is in the set."""
    return (mask >> i) & 1 == 1


def set_bit(mask: int, i: int) -> int:
    """The set with element i added."""
    return mask | (1 << i)


def clear_bit(mask: int, i: int) -> int:
    """The set with element i removed."""
    return mask & ~(1 << i)


def full_mask(n: int) -> int:
    """The set {0, 1, ..., n-1}."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return (1 << n) - 1


def popcount(mask: int) -> int:
    """Number of elements in the set."""
    return bin(mask).count("1")


def iter_bits(mask: int) -> Iterator[int]:
    """
    Yield each element of the set in increasing order.

    `mask & -mask` isolates the lowest set bit (two's complement), so
    the loop runs once per ELEMENT rather than once per bit position.
    """
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def mask_from(elements: Iterable[int]) -> int:
    """Build a mask from an iterable of element indices."""
    mask = 0
    for i in elements:
        if i < 0:
            raise ValueError(f"element indices must be >= 0, got {i}")
        mask |= 1 << i
    return mask


def to_list(mask: int) -> List[int]:
    """The set's elements as a sorted list."""
    return list(iter_bits(mask))


def format_mask(mask: int, n: int) -> str:
    """
    Fixed-width binary string with element 0 on the RIGHT, matching
    how the integer is written: format_mask(0b1101, 5) -> '01101'.
    """
    return format(mask, f"0{n}b")
