# File: examples/week2_demo.py
"""
Week 2 demo: the divide-and-conquer sorts in action.

Run from the project root:

    python examples/week2_demo.py

Shows (1) all five algorithms sorting the same small list, (2) a head-
to-head timing at n = 2,000 where the O(n log n) algorithms pull away,
and (3) merge sort's stability versus quicksort's instability.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sorting import (
    bubble_sort, selection_sort, insertion_sort, merge_sort, quick_sort,
)

ALL_SORTS = [bubble_sort, selection_sort, insertion_sort,
             merge_sort, quick_sort]


def demo_correctness():
    print("=" * 60)
    print("1. All five algorithms, same input")
    print("=" * 60)
    data = [38, 27, 43, 3, 9, 82, 10]
    print(f"Input: {data}\n")
    for sort_func in ALL_SORTS:
        print(f"{sort_func.__name__:15s} -> {sort_func(data)}")
    print(f"{'(input after)':15s} -> {data}   <- never modified")


def demo_timing():
    print()
    print("=" * 60)
    print("2. Head-to-head at n = 2,000 (random data)")
    print("=" * 60)
    rng = random.Random(42)
    data = [rng.randint(0, 2000) for _ in range(2000)]
    times = {}
    for sort_func in ALL_SORTS:
        start = time.perf_counter()
        sort_func(data)
        times[sort_func.__name__] = time.perf_counter() - start
    fastest = min(times.values())
    for name, elapsed in sorted(times.items(), key=lambda kv: kv[1]):
        print(f"{name:15s} {elapsed * 1000:8.2f} ms   "
              f"({elapsed / fastest:5.1f}x slowest-vs-fastest baseline"
              f"{')' if elapsed > fastest else ', fastest)'}")


def demo_stability():
    print()
    print("=" * 60)
    print("3. Stability: merge sort keeps ties in order, quicksort may not")
    print("=" * 60)
    # Records sorted by grade only; letters record the original order.
    # NOTE: the list must be longer than quick_sort's insertion-sort
    # threshold (10) — below it, quicksort delegates the whole array to
    # insertion sort, which IS stable, and the demo would prove nothing.
    rng = random.Random(7)
    records = [(letter, rng.choice([70, 80, 90]))
               for letter in "abcdefghijklmnop"]

    class ByGrade:
        def __init__(self, record):
            self.record = record

        def __lt__(self, other):
            return self.record[1] < other.record[1]

        def __gt__(self, other):
            return self.record[1] > other.record[1]

    def show(name, sort_func):
        result = [item.record for item in sort_func(wrapped)]
        stable = result == sorted(records, key=lambda r: r[1])
        print(f"{name:12s} {result}")
        print(f"{'':12s} ties in original order? {'YES (stable)' if stable else 'NO (not stable)'}")

    wrapped = [ByGrade(r) for r in records]
    print(f"Input:       {records}")
    show("merge_sort:", merge_sort)
    show("quick_sort:", quick_sort)


if __name__ == "__main__":
    demo_correctness()
    demo_timing()
    demo_stability()
