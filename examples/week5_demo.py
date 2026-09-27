"""
Week 5 demo: dynamic programming on real decisions.

Run from the project root:

    python examples/week5_demo.py

Shows (1) the Fibonacci call explosion and how memoization collapses
it, (2) knapsack as a budget-constrained project picker with the
chosen items traced back out of the table, (3) LCS as DNA sequence
comparison, with the shared subsequence reconstructed, and (4) the
filled LCS table itself, so the recurrence is visible.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dp import (
    fib_naive, fib_memo, fib_tabulation,
    knapsack_recursive, knapsack_with_items,
    lcs, build_lcs_table,
)
from src.utils.timer import CallStats


def demo_fibonacci():
    print("=" * 64)
    print("1. Fibonacci: same answer, wildly different work")
    print("=" * 64)
    n = 30
    for name, func in [("naive", fib_naive), ("memoization", fib_memo),
                       ("tabulation", fib_tabulation)]:
        stats = CallStats()
        start = time.perf_counter()
        value = func(n, stats=stats)
        elapsed = time.perf_counter() - start
        print(f"  {name:12s} F({n}) = {value:,}  "
              f"{stats.calls:>10,} calls/cells  {elapsed * 1000:9.3f} ms")
    print("\n  Naive recursion recomputes F(28) twice, F(27) three times,\n"
          "  F(26) five times... the repeat counts are themselves\n"
          "  Fibonacci numbers. Memoization solves each F(k) exactly once.")


def demo_knapsack():
    print()
    print("=" * 64)
    print("2. Knapsack: pick projects under a 10-week budget")
    print("=" * 64)
    projects = [
        ("Search redesign", 5, 50),
        ("Mobile app", 5, 48),
        ("Billing migration", 4, 30),
        ("Analytics dashboard", 3, 20),
        ("API rate limiting", 1, 12),
        ("Onboarding flow", 6, 45),
    ]
    names = [p[0] for p in projects]
    weeks = [p[1] for p in projects]
    impact = [p[2] for p in projects]
    budget = 10

    best, chosen = knapsack_with_items(weeks, impact, budget)
    assert best == knapsack_recursive(weeks, impact, budget)
    for i in chosen:
        print(f"  ✓ {names[i]:22s} {weeks[i]:2d} wk  impact {impact[i]}")
    print(f"  {'':24s}{sum(weeks[i] for i in chosen):2d} wk  "
          f"impact {best}")

    # Greedy by impact-per-week grabs the 12/week task first, which
    # leaves no room for the second 5-week project
    order = sorted(range(len(projects)),
                   key=lambda i: impact[i] / weeks[i], reverse=True)
    used, greedy = 0, 0
    for i in order:
        if used + weeks[i] <= budget:
            used += weeks[i]
            greedy += impact[i]
    print(f"\n  Greedy by impact/week would reach only {greedy} — "
          f"DP proves {best} is the best possible.")


def demo_lcs():
    print()
    print("=" * 64)
    print("3. LCS: how similar are two DNA fragments?")
    print("=" * 64)
    human = "ATGCTAGCTAGGCTTACGATCG"
    mouse = "ATGCAAGCTTGGCTACGTTCG"
    length, shared = lcs(human, mouse)
    print(f"  fragment A: {human}")
    print(f"  fragment B: {mouse}")
    print(f"  LCS ({length} bases): {shared}")
    print(f"  similarity: {length / max(len(human), len(mouse)):.0%} "
          "of the longer fragment is conserved in order")


def demo_lcs_table():
    print()
    print("=" * 64)
    print("4. The LCS table for 'ABCBDAB' vs 'BDCABA'")
    print("=" * 64)
    x, y = "ABCBDAB", "BDCABA"
    table = build_lcs_table(x, y)
    print("       " + "  ".join(f"{c}" for c in " " + y))
    for i, row in enumerate(table):
        label = x[i - 1] if i else " "
        print(f"    {label}  " + "  ".join(str(v) for v in row))
    print(f"\n  Each cell = LCS of the prefixes ending there; bottom-right "
          f"({table[-1][-1]})\n  is the answer, and walking back from it "
          f"recovers '{lcs(x, y)[1]}'.")


if __name__ == "__main__":
    demo_fibonacci()
    demo_knapsack()
    demo_lcs()
    demo_lcs_table()
