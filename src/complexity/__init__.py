"""Computational complexity: polynomial-time reductions between
NP-complete problems, exact exponential solvers (SAT, Subset Sum,
Vertex Cover), and polynomial-time heuristics and approximation
algorithms (WalkSAT, greedy Subset Sum, 2-approximate Vertex Cover,
nearest-neighbor / MST / 2-opt TSP)."""

from src.complexity.satisfiability_solver import (
    PHASE_TRANSITION_RATIO,
    LocalSearchResult,
    backtracking_sat,
    brute_force_sat,
    count_satisfied,
    format_cnf,
    is_satisfied,
    planted_ksat,
    random_assignment,
    random_ksat,
    two_sat,
    walksat,
)
from src.complexity.subset_sum_backtracking import (
    greedy_subset_sum,
    is_subset_sum_solution,
    random_subset_sum,
    subset_sum_backtracking,
    subset_sum_dp,
)
from src.complexity.vertex_cover_approx import (
    exact_vertex_cover,
    greedy_vertex_cover,
    is_vertex_cover,
    vertex_cover_2approx,
)
from src.complexity.traveling_salesman_approx import (
    mst_tour,
    mst_weight,
    nearest_neighbor_tour,
    optimal_tour,
    repeated_nearest_neighbor,
    two_opt,
)
from src.complexity.reduction_demo import (
    CATALOG,
    Clique,
    IndependentSet,
    IndependentSetToClique,
    Reduction,
    ReductionChain,
    SubsetSum,
    ThreeSAT,
    ThreeSatToSubsetSum,
    ThreeSatToVertexCover,
    VertexCover,
    VertexCoverToIndependentSet,
    check_reduction,
    classify,
)
