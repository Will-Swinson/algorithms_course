# File: src/dp/__init__.py
"""Dynamic programming: Fibonacci, 0/1 knapsack, and LCS, each as
naive recursion, top-down memoization, and bottom-up tabulation."""

from src.dp.fibonacci import (
    fib_naive,
    fib_memo,
    fib_tabulation,
    naive_call_count,
)
from src.dp.knapsack import (
    knapsack_recursive,
    knapsack_memo,
    knapsack_tabulation,
    build_knapsack_table,
    trace_solution,
    knapsack_with_items,
)
from src.dp.lcs import (
    lcs_recursive,
    lcs_memo,
    lcs_tabulation,
    build_lcs_table,
    reconstruct_lcs,
    lcs,
)

__all__ = [
    "fib_naive",
    "fib_memo",
    "fib_tabulation",
    "naive_call_count",
    "knapsack_recursive",
    "knapsack_memo",
    "knapsack_tabulation",
    "build_knapsack_table",
    "trace_solution",
    "knapsack_with_items",
    "lcs_recursive",
    "lcs_memo",
    "lcs_tabulation",
    "build_lcs_table",
    "reconstruct_lcs",
    "lcs",
]
