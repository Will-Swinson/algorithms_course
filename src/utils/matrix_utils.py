"""
Matrix helpers for the Week 6 algorithms.

- Matrix chains: random dimension arrays, plus real matrix
  multiplication that COUNTS scalar multiplications — so a matrix-chain
  answer can be checked against arithmetic actually performed.
- Graphs as matrices: convert a Week 4 Graph into the V×V weight
  matrix Floyd–Warshall works on (INF = no edge, 0 on the diagonal).
- Cities: random points and their Euclidean distance matrix, the
  standard TSP test input.
- format_matrix() pretty-prints any of the above for demos.
"""
import math
import random
from typing import Any, List, Optional, Sequence, Tuple

from src.graphs.graph import Graph

INF = math.inf
Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# Matrix chains
# ---------------------------------------------------------------------------

def random_chain_dimensions(n: int, low: int = 5, high: int = 100,
                            seed: Optional[int] = None) -> List[int]:
    """
    Dimension array p for a chain of n matrices: matrix A_i is
    p[i-1] × p[i], so the array has n + 1 entries.
    """
    if n < 1:
        raise ValueError(f"need at least one matrix, got n={n}")
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(n + 1)]


def random_matrix(rows: int, cols: int,
                  seed: Optional[int] = None) -> Matrix:
    """rows × cols matrix of small random ints (exact arithmetic)."""
    rng = random.Random(seed)
    return [[rng.randint(-5, 5) for _ in range(cols)] for _ in range(rows)]


def matrix_multiply(a: Matrix, b: Matrix) -> Tuple[Matrix, int]:
    """
    Naive product a · b.

    Returns:
        (product, scalar multiplications performed). Multiplying a
        p×q matrix by a q×r matrix always costs exactly p·q·r — the
        quantity matrix-chain ordering minimizes.

    Raises:
        ValueError: If the inner dimensions disagree.
    """
    p, q = len(a), len(a[0])
    if len(b) != q:
        raise ValueError(f"cannot multiply {p}x{q} by {len(b)}x{len(b[0])}")
    r = len(b[0])
    product = [[0] * r for _ in range(p)]
    for i in range(p):
        a_row = a[i]
        out = product[i]
        for k in range(q):
            a_ik = a_row[k]
            b_row = b[k]
            for j in range(r):
                out[j] += a_ik * b_row[j]
    return product, p * q * r


def multiply_chain(matrices: Sequence[Matrix],
                   split: List[List[int]]) -> Tuple[Matrix, int]:
    """
    Multiply A_1 · ... · A_n in the order a split table prescribes.

    Args:
        matrices: The n matrices (0-indexed list; A_i is matrices[i-1]).
        split: 1-indexed table where split[i][j] = k means the product
            A_i..A_j is computed as (A_i..A_k)(A_{k+1}..A_j).

    Returns:
        (product, total scalar multiplications performed).
    """
    def solve(i: int, j: int) -> Tuple[Matrix, int]:
        if i == j:
            return matrices[i - 1], 0
        k = split[i][j]
        left, left_cost = solve(i, k)
        right, right_cost = solve(k + 1, j)
        product, cost = matrix_multiply(left, right)
        return product, left_cost + right_cost + cost

    return solve(1, len(matrices))


def left_to_right_cost(p: Sequence[int]) -> int:
    """Scalar multiplications for the naive order ((A1 A2) A3) ..."""
    return sum(p[0] * p[i] * p[i + 1] for i in range(1, len(p) - 1))


# ---------------------------------------------------------------------------
# Graphs as weight matrices
# ---------------------------------------------------------------------------

def graph_to_matrix(graph: Graph) -> Tuple[List[Any], Matrix]:
    """
    Convert a Graph into a V×V weight matrix.

    Returns:
        (nodes, matrix) where matrix[i][j] is the weight of edge
        nodes[i] -> nodes[j], INF if there is no such edge, and 0 on the
        diagonal (staying put is free) unless a negative self-loop says
        otherwise.
    """
    nodes = graph.nodes()
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    matrix = [[INF] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for u in nodes:
        for v, weight in graph.get_neighbors(u):
            i, j = index[u], index[v]
            if i != j or weight < 0:
                matrix[i][j] = weight
    return nodes, matrix


# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------

def random_points(n: int, size: float = 100.0,
                  seed: Optional[int] = None) -> List[Tuple[float, float]]:
    """n random points in a size × size square."""
    rng = random.Random(seed)
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def euclidean_distance_matrix(points: Sequence[Tuple[float, float]]
                              ) -> Matrix:
    """Symmetric matrix of straight-line distances between points."""
    return [[math.dist(a, b) for b in points] for a in points]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_matrix(matrix: Matrix, labels: Optional[Sequence[Any]] = None,
                  width: int = 7, precision: int = 0) -> str:
    """Render a matrix as aligned text, with ∞ for INF and · for None."""
    n_cols = len(matrix[0]) if matrix else 0
    labels = list(labels) if labels is not None else list(range(len(matrix)))

    def cell(v: Any) -> str:
        if v is None:
            return "·".rjust(width)
        if v == INF:
            return "∞".rjust(width)
        return f"{v:{width}.{precision}f}"

    col_labels = labels if len(labels) == n_cols else list(range(n_cols))
    header = " " * width + "".join(str(c).rjust(width) for c in col_labels)
    rows = [str(labels[i]).rjust(width) + "".join(cell(v) for v in row)
            for i, row in enumerate(matrix)]
    return "\n".join([header] + rows)
