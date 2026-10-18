"""
Week 8 benchmark: exact exponential algorithms vs polynomial-time
heuristics and approximations for NP-complete problems.

Experiments (results land in benchmarks/results/):

1. Subset Sum: backtracking on "no" instances with 12-digit values
   (the worst case) and planted "yes" instances, n = 4 .. 28; the
   pseudo-polynomial bitset DP; greedy's runtime up to n = 300,000 and
   its approximation ratio against the DP optimum.
2. SAT: brute force (2^n), plain backtracking, and DPLL on random
   3-SAT at the phase transition (m = 4.26n); WalkSAT and a random
   assignment on the same formulas; 2-SAT in linear time for contrast.
3. Vertex Cover: branch-and-bound exact solver vs the matching
   2-approximation and max-degree greedy on random graphs (average
   degree 4); the tight K_{n,n} family where the ratio is exactly 2.
4. TSP: brute force and Held–Karp (Week 6) vs nearest neighbor,
   repeated nearest neighbor, MST double-tree, and 2-opt on random
   Euclidean cities. Ratios vs the optimum for n ≤ 13 and vs the MST
   lower bound up to n = 2,000.
5. Reductions: 3-SAT → Vertex Cover and 3-SAT → Subset Sum built for
   formulas of up to 3,000 clauses (time and output size grow
   polynomially), and every reduction validated against the exact
   solvers on hundreds of random formulas.

Outputs:
    runtime_growth.png           exponential vs polynomial runtime
    approximation_quality.png    approximation ratio vs input size
    tradeoff_curves.png          runtime ↔ accuracy for each problem
    reduction_3sat_vertex_cover.png   the gadget graph of a small formula
    comparison_table.csv         one row per configuration

Usage:
    python benchmarks/week8_complexity_benchmark.py          # ~40 s
    python benchmarks/week8_complexity_benchmark.py --quick  # ~10 s
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.complexity import (
    PHASE_TRANSITION_RATIO, IndependentSetToClique, ReductionChain, ThreeSAT,
    ThreeSatToSubsetSum, ThreeSatToVertexCover, VertexCoverToIndependentSet,
    backtracking_sat, brute_force_sat, check_reduction, exact_vertex_cover, format_cnf,
    greedy_subset_sum, greedy_vertex_cover, is_satisfied, is_vertex_cover,
    mst_tour, mst_weight, nearest_neighbor_tour, optimal_tour,
    random_assignment, random_ksat, random_subset_sum,
    repeated_nearest_neighbor, subset_sum_backtracking, subset_sum_dp,
    two_opt, two_sat, vertex_cover_2approx, walksat,
)
from src.dp_advanced.bitmask_traveling_salesman import tsp_brute_force
from src.utils.graph_utils import complete_bipartite_edges, random_graph_edges
from src.utils.matrix_utils import euclidean_distance_matrix, random_points
from src.utils.probability_analysis import summarize
from src.utils.timer import CallStats, time_call
from src.utils.visualization import draw_highlighted_graph, plot_mixed_panels

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_NAME = "comparison_table.csv"
SEED = 8

FIELDS = ["experiment", "problem", "method", "approach", "input_size",
          "parameter", "instances", "mean_time_s", "std_time_s",
          "mean_work", "work_unit", "quality_metric", "mean_quality",
          "worst_quality", "bound", "success_rate"]

SS_HARD_SIZES = list(range(4, 29, 2))
SS_INSTANCES = 3
SS_HARD_MAX = 10 ** 12           # values ~10^12: DP table impossible
SS_SMALL_MAX = 1_000             # values ≤ 1,000: DP is fast
SS_DP_SIZES = [10, 30, 100, 300, 1_000, 3_000]
SS_GREEDY_SIZES = [100, 1_000, 10_000, 100_000, 300_000]
SS_RATIO_SIZES = [4, 6, 8, 10, 15, 20, 50, 100, 300, 1_000]
SS_RATIO_INSTANCES = 30

SAT_BF_SIZES = [4, 6, 8, 10, 12, 14, 16]
SAT_PLAIN_SIZES = [5, 10, 15, 20, 25, 30]
SAT_DPLL_SIZES = [10, 20, 30, 40, 50, 60, 70]
SAT_INSTANCES = 10
SAT_SMALL_INSTANCES = 4          # brute force / plain backtracking
WALKSAT_FLIPS = 20_000
WALKSAT_BUDGETS = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000]
WALKSAT_BUDGET_N = 50
WALKSAT_BUDGET_SEEDS = 5
TWO_SAT_SIZES = [100, 1_000, 10_000, 100_000]

VC_EXACT_SIZES = [20, 60, 100, 140, 180, 220, 260]
VC_DEGREE = 4
VC_INSTANCES = 3
VC_FAST_SIZES = [100, 1_000, 10_000, 100_000]
VC_BIPARTITE = [2, 5, 10, 20, 50]

TSP_EXACT_SIZES = list(range(5, 14))
TSP_BRUTE_MAX = 10
TSP_INSTANCES = 5
TSP_LARGE_SIZES = [20, 50, 100, 200, 500, 1_000, 2_000]
TSP_LARGE_INSTANCES = 3
TSP_RNN_MAX = 200                # O(n³)
TSP_2OPT_MAX = 500
TSP_PASS_N = 200
TSP_PASSES = [1, 2, 3, 5, 8, 13, None]

REDUCTION_CLAUSES = [10, 30, 100, 300, 1_000, 3_000]
REDUCTION_VALIDATION = 300

GADGET_FORMULA = ThreeSAT([(1, -2, 3), (-1, 2, 3), (-1, -2, -3)], 3)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def make_row(experiment: str, problem: str, method: str, approach: str,
             size: int, parameter: str = "", **fields) -> Dict:
    row = {f: None for f in FIELDS}
    row.update(experiment=experiment, problem=problem, method=method,
               approach=approach, input_size=size, parameter=parameter,
               **fields)
    _print_row(row)
    return row


def _print_row(row: Dict) -> None:
    parts = [f"  {row['experiment']:14s} {row['method']:24s} "
             f"n={row['input_size']:<7} {row['parameter'] or '':12s}"]
    if row["mean_time_s"] is not None:
        parts.append(f"{row['mean_time_s'] * 1e3:11.3f} ms")
    if row["mean_work"] is not None:
        parts.append(f"{row['mean_work']:>14,.0f} {row['work_unit'] or ''}")
    if row["mean_quality"] is not None:
        parts.append(f"{row['quality_metric']}={row['mean_quality']:.4f}")
    if row["success_rate"] is not None:
        parts.append(f"success={row['success_rate']:.2f}")
    print(" ".join(parts), flush=True)


def timed(func: Callable[[], object]) -> Tuple[float, object]:
    """(seconds, result) of one call — for slow exponential runs."""
    start = time.perf_counter()
    result = func()
    return time.perf_counter() - start, result


def stats_fields(times: Sequence[float],
                 work: Optional[Sequence[float]] = None,
                 quality: Optional[Sequence[float]] = None,
                 higher_is_better: bool = False) -> Dict:
    """Mean/std of time, mean work, mean and worst quality."""
    t = summarize(times)
    out = {"instances": len(times), "mean_time_s": t.mean,
           "std_time_s": t.std}
    if work:
        out["mean_work"] = summarize(work).mean
    if quality:
        out["mean_quality"] = summarize(quality).mean
        out["worst_quality"] = min(quality) if higher_is_better \
            else max(quality)
    return out


# ---------------------------------------------------------------------------
# 1. Subset Sum
# ---------------------------------------------------------------------------

def bench_subset_sum(hard_sizes: Sequence[int] = SS_HARD_SIZES,
                     instances: int = SS_INSTANCES,
                     dp_sizes: Sequence[int] = SS_DP_SIZES,
                     greedy_sizes: Sequence[int] = SS_GREEDY_SIZES,
                     ratio_sizes: Sequence[int] = SS_RATIO_SIZES,
                     ratio_instances: int = SS_RATIO_INSTANCES) -> List[Dict]:
    rows = []
    for n in hard_sizes:
        for solvable, label in ((False, "no-instance"), (True, "planted")):
            times, nodes = [], []
            for i in range(instances):
                nums, target = random_subset_sum(
                    n, SS_HARD_MAX, solvable, seed=SEED + 100 * n + i)
                stats = CallStats()
                seconds, found = timed(
                    lambda: subset_sum_backtracking(nums, target, stats))
                if solvable:
                    assert found is not None, "planted solution missed"
                times.append(seconds)
                nodes.append(stats.calls)
            rows.append(make_row(
                "ss_exact", "subset_sum", "backtracking", "exact", n, label,
                work_unit="nodes", bound=2 ** (n + 1) - 1,
                **stats_fields(times, nodes)))
        nums, target = random_subset_sum(n, SS_HARD_MAX, False, seed=SEED + n)
        rows.append(make_row(
            "ss_exact", "subset_sum", "greedy", "heuristic", n,
            "no-instance",
            **stats_fields([time_call(lambda: greedy_subset_sum(nums,
                                                                target))])))

    for n in dp_sizes:
        nums, target = random_subset_sum(n, SS_SMALL_MAX, False, seed=SEED + n)
        rows.append(make_row(
            "ss_dp", "subset_sum", "bitset_dp", "exact", n,
            f"T={target}", work_unit="n·T bit-ops", mean_work=n * target,
            **stats_fields([time_call(lambda: subset_sum_dp(nums, target))])))

    for n in greedy_sizes:
        nums, target = random_subset_sum(n, SS_SMALL_MAX, False, seed=SEED + n)
        rows.append(make_row(
            "ss_greedy", "subset_sum", "greedy", "heuristic", n,
            **stats_fields([time_call(lambda: greedy_subset_sum(nums,
                                                                target))])))

    for n in ratio_sizes:
        ratios, exact_hits, times = [], 0, []
        for i in range(ratio_instances):
            rng = random.Random(SEED * 7 + 1_000 * n + i)
            nums = [rng.randint(1, SS_SMALL_MAX) for _ in range(n)]
            target = rng.randint(1, sum(nums))
            best = subset_sum_dp(nums, target)
            seconds, chosen = timed(lambda: greedy_subset_sum(nums, target))
            got = sum(nums[j] for j in chosen)
            ratios.append(got / best if best else 1.0)
            exact_hits += got == target
            times.append(seconds)
        rows.append(make_row(
            "ss_ratio", "subset_sum", "greedy", "heuristic", n,
            quality_metric="sum/optimum", bound=0.5,
            success_rate=exact_hits / ratio_instances,
            **stats_fields(times, quality=ratios, higher_is_better=True)))
    return rows


# ---------------------------------------------------------------------------
# 2. SAT
# ---------------------------------------------------------------------------

def phase_formula(n: int, i: int) -> List[Tuple[int, ...]]:
    return random_ksat(n, round(PHASE_TRANSITION_RATIO * n),
                       seed=SEED * 1_000 + 31 * n + i)


def bench_sat(bf_sizes: Sequence[int] = SAT_BF_SIZES,
              plain_sizes: Sequence[int] = SAT_PLAIN_SIZES,
              dpll_sizes: Sequence[int] = SAT_DPLL_SIZES,
              instances: int = SAT_INSTANCES,
              small_instances: int = SAT_SMALL_INSTANCES,
              two_sat_sizes: Sequence[int] = TWO_SAT_SIZES,
              budgets: Sequence[int] = WALKSAT_BUDGETS,
              budget_n: int = WALKSAT_BUDGET_N,
              budget_seeds: int = WALKSAT_BUDGET_SEEDS) -> List[Dict]:
    rows = []
    for n in bf_sizes:
        times, work = [], []
        for i in range(small_instances):
            cnf = phase_formula(n, i)
            stats = CallStats()
            seconds, _ = timed(lambda: brute_force_sat(cnf, n, stats))
            times.append(seconds)
            work.append(stats.calls)
        rows.append(make_row("sat_exact", "3-sat", "brute_force", "exact", n,
                             work_unit="assignments", bound=2 ** n,
                             **stats_fields(times, work)))

    for n in plain_sizes:
        times, work = [], []
        for i in range(small_instances):
            cnf = phase_formula(n, i)
            stats = CallStats()
            seconds, _ = timed(lambda: backtracking_sat(
                cnf, n, unit_propagation=False, stats=stats))
            times.append(seconds)
            work.append(stats.calls)
        rows.append(make_row("sat_exact", "3-sat", "backtracking", "exact",
                             n, work_unit="nodes", bound=2 ** (n + 1) - 1,
                             **stats_fields(times, work)))

    for n in dpll_sizes:
        times, work, sat_formulas = [], [], []
        ws_times, ws_frac, rand_frac, ws_solved = [], [], [], 0
        ws_all_times, rand_times = [], []
        for i in range(instances):
            cnf = phase_formula(n, i)
            stats = CallStats()
            seconds, found = timed(lambda: backtracking_sat(cnf, n,
                                                            stats=stats))
            times.append(seconds)
            work.append(stats.calls)
            rng = random.Random(SEED + 17 * n + i)
            seconds, ws = timed(lambda: walksat(cnf, n, WALKSAT_FLIPS,
                                                rng=rng))
            ws_all_times.append(seconds)
            ws_frac.append(ws.fraction)
            seconds, ra = timed(lambda: random_assignment(cnf, n, rng))
            rand_times.append(seconds)
            rand_frac.append(ra.fraction)
            if found is not None:
                sat_formulas.append(cnf)
                ws_times.append(ws_all_times[-1])
                ws_solved += ws.solved
            elif ws.solved:
                raise AssertionError("WalkSAT satisfied an UNSAT formula")
        sat_share = len(sat_formulas) / instances
        rows.append(make_row("sat_exact", "3-sat", "dpll", "exact", n,
                             work_unit="nodes", bound=2 ** (n + 1) - 1,
                             success_rate=sat_share,
                             **stats_fields(times, work)))
        if ws_times:
            rows.append(make_row(
                "sat_exact", "3-sat", "walksat", "heuristic", n,
                f"flips≤{WALKSAT_FLIPS}",
                success_rate=ws_solved / len(sat_formulas),
                **stats_fields(ws_times)))
        rows.append(make_row(
            "sat_quality", "3-sat", "walksat", "heuristic", n,
            quality_metric="clauses satisfied", bound=1.0,
            **stats_fields(ws_all_times, quality=ws_frac,
                           higher_is_better=True)))
        rows.append(make_row(
            "sat_quality", "3-sat", "random_assignment", "heuristic", n,
            quality_metric="clauses satisfied", bound=7 / 8,
            **stats_fields(rand_times, quality=rand_frac,
                           higher_is_better=True)))

    for n in two_sat_sizes:
        cnf = random_ksat(n, n, k=2, seed=SEED + n)
        result = two_sat(cnf, n)
        assert result is None or is_satisfied(cnf, result)
        rows.append(make_row("sat_2sat", "2-sat", "scc", "polynomial", n,
                             "m=n", success_rate=float(result is not None),
                             **stats_fields([time_call(
                                 lambda: two_sat(cnf, n))])))

    # WalkSAT flip budget: runtime ↔ success trade-off
    formulas = []
    i = 0
    while len(formulas) < 10:
        cnf = phase_formula(budget_n, 500 + i)
        if backtracking_sat(cnf, budget_n) is not None:
            formulas.append(cnf)
        i += 1
    for budget in budgets:
        times, solved = [], 0
        for f_i, cnf in enumerate(formulas):
            for s in range(budget_seeds):
                rng = random.Random(SEED + 1_000 * f_i + s)
                seconds, ws = timed(lambda: walksat(cnf, budget_n, budget,
                                                    rng=rng))
                times.append(seconds)
                solved += ws.solved
        rows.append(make_row(
            "sat_budget", "3-sat", "walksat", "heuristic", budget_n,
            f"flips≤{budget}", success_rate=solved / len(times),
            **stats_fields(times)))
    dpll_times = [timed(lambda: backtracking_sat(cnf, budget_n))[0]
                  for cnf in formulas]
    rows.append(make_row("sat_budget", "3-sat", "dpll", "exact", budget_n,
                         success_rate=1.0, **stats_fields(dpll_times)))
    return rows


# ---------------------------------------------------------------------------
# 3. Vertex Cover
# ---------------------------------------------------------------------------

VC_METHODS = [("2approx_matching", vertex_cover_2approx, "approximation"),
              ("greedy_max_degree", greedy_vertex_cover, "heuristic")]


def bench_vertex_cover(exact_sizes: Sequence[int] = VC_EXACT_SIZES,
                       instances: int = VC_INSTANCES,
                       fast_sizes: Sequence[int] = VC_FAST_SIZES,
                       bipartite: Sequence[int] = VC_BIPARTITE) -> List[Dict]:
    rows = []
    for n in exact_sizes:
        graphs = [random_graph_edges(n, VC_DEGREE, seed=SEED + 50 * n + i)
                  for i in range(instances)]
        times, nodes, optimum = [], [], []
        for edges in graphs:
            stats = CallStats()
            seconds, cover = timed(lambda: exact_vertex_cover(edges, stats))
            assert is_vertex_cover(edges, cover)
            times.append(seconds)
            nodes.append(stats.calls)
            optimum.append(len(cover))
        rows.append(make_row("vc_exact", "vertex_cover", "branch_and_bound",
                             "exact", n, f"deg≈{VC_DEGREE}",
                             work_unit="nodes", quality_metric="size/optimum",
                             **stats_fields(times, nodes, [1.0] * instances)))
        for name, func, approach in VC_METHODS:
            times, ratios = [], []
            for edges, opt in zip(graphs, optimum):
                times.append(time_call(lambda: func(edges)))
                ratios.append(len(func(edges)) / opt)
            rows.append(make_row(
                "vc_exact", "vertex_cover", name, approach, n,
                f"deg≈{VC_DEGREE}", quality_metric="size/optimum",
                bound=2.0 if approach == "approximation" else None,
                **stats_fields(times, quality=ratios)))

    for n in fast_sizes:
        edges = random_graph_edges(n, VC_DEGREE, seed=SEED + n)
        for name, func, approach in VC_METHODS:
            rows.append(make_row(
                "vc_fast", "vertex_cover", name, approach, n,
                f"E={len(edges)}", work_unit="edges", mean_work=len(edges),
                **stats_fields([time_call(lambda: func(edges))])))

    for half in bipartite:
        edges = complete_bipartite_edges(half, half)
        ratio = len(vertex_cover_2approx(edges)) / len(
            exact_vertex_cover(edges))
        rows.append(make_row("vc_tight", "vertex_cover", "2approx_matching",
                             "approximation", 2 * half, f"K_{half},{half}",
                             quality_metric="size/optimum", bound=2.0,
                             mean_quality=ratio, worst_quality=ratio,
                             instances=1))
    return rows


# ---------------------------------------------------------------------------
# 4. TSP
# ---------------------------------------------------------------------------

def nn_two_opt(dist, max_passes=None):
    return two_opt(dist, nearest_neighbor_tour(dist)[1], max_passes)


TSP_HEURISTICS = [
    ("nearest_neighbor", nearest_neighbor_tour, "heuristic", None),
    ("repeated_nn", repeated_nearest_neighbor, "heuristic", TSP_RNN_MAX),
    ("mst_double_tree", mst_tour, "approximation", None),
    ("nn_2opt", nn_two_opt, "heuristic", TSP_2OPT_MAX),
]


def cities(n: int, i: int):
    return euclidean_distance_matrix(random_points(n, seed=SEED * 31 + 97 * n
                                                   + i))


def bench_tsp(exact_sizes: Sequence[int] = TSP_EXACT_SIZES,
              instances: int = TSP_INSTANCES,
              brute_max: int = TSP_BRUTE_MAX,
              large_sizes: Sequence[int] = TSP_LARGE_SIZES,
              large_instances: int = TSP_LARGE_INSTANCES,
              pass_n: int = TSP_PASS_N,
              passes: Sequence[Optional[int]] = TSP_PASSES) -> List[Dict]:
    rows = []
    for n in exact_sizes:
        dists = [cities(n, i) for i in range(instances)]
        times, optimum = [], []
        for d in dists:
            seconds, (cost, _) = timed(lambda: optimal_tour(d))
            times.append(seconds)
            optimum.append(cost)
        rows.append(make_row("tsp_exact", "tsp", "held_karp", "exact", n,
                             quality_metric="length/optimum",
                             **stats_fields(times, quality=[1.0] * instances)))
        if n <= brute_max:
            seconds, (cost, _) = timed(lambda: tsp_brute_force(dists[0]))
            assert math.isclose(cost, optimum[0])
            rows.append(make_row("tsp_exact", "tsp", "brute_force", "exact",
                                 n, work_unit="tours",
                                 mean_work=math.factorial(n - 1),
                                 quality_metric="length/optimum",
                                 **stats_fields([seconds],
                                                quality=[1.0])))
        for name, func, approach, _ in TSP_HEURISTICS:
            times, ratios = [], []
            for d, opt in zip(dists, optimum):
                times.append(time_call(lambda: func(d)))
                ratios.append(func(d)[0] / opt)
            rows.append(make_row(
                "tsp_exact", "tsp", name, approach, n,
                quality_metric="length/optimum",
                bound=2.0 if approach == "approximation" else None,
                **stats_fields(times, quality=ratios)))

    for n in large_sizes:
        dists = [cities(n, i) for i in range(large_instances)]
        bounds = [mst_weight(d) for d in dists]
        for name, func, approach, limit in TSP_HEURISTICS:
            if limit is not None and n > limit:
                continue
            times, ratios = [], []
            for d, lb in zip(dists, bounds):
                seconds, (cost, _) = timed(lambda: func(d))
                times.append(seconds)
                ratios.append(cost / lb)
            rows.append(make_row(
                "tsp_large", "tsp", name, approach, n,
                quality_metric="length/MST",
                bound=2.0 if approach == "approximation" else None,
                **stats_fields(times, quality=ratios)))

    dists = [cities(pass_n, i) for i in range(large_instances)]
    bounds = [mst_weight(d) for d in dists]
    for cap in passes:
        times, ratios = [], []
        for d, lb in zip(dists, bounds):
            seconds, (cost, _) = timed(lambda: nn_two_opt(d, cap))
            times.append(seconds)
            ratios.append(cost / lb)
        rows.append(make_row(
            "tsp_passes", "tsp", "nn_2opt", "heuristic", pass_n,
            f"passes≤{cap}" if cap else "passes=∞",
            quality_metric="length/MST",
            **stats_fields(times, quality=ratios)))
    return rows


# ---------------------------------------------------------------------------
# 5. Reductions
# ---------------------------------------------------------------------------

def bench_reductions(clause_counts: Sequence[int] = REDUCTION_CLAUSES,
                     validation: int = REDUCTION_VALIDATION) -> List[Dict]:
    rows = []
    for m in clause_counts:
        n = max(3, round(m / PHASE_TRANSITION_RATIO))
        phi = ThreeSAT(random_ksat(n, m, seed=SEED + m), n)
        vc = ThreeSatToVertexCover()
        target = vc.reduce(phi)
        rows.append(make_row(
            "reduction_size", "3-sat→vertex_cover", "reduce", "reduction", m,
            f"n={n}", work_unit="vertices+edges",
            mean_work=len(target.vertices) + len(target.edges),
            **stats_fields([time_call(lambda: vc.reduce(phi))])))
        ss = ThreeSatToSubsetSum()
        target = ss.reduce(phi)
        rows.append(make_row(
            "reduction_size", "3-sat→subset_sum", "reduce", "reduction", m,
            f"n={n}", work_unit="numbers×digits",
            mean_work=len(target.numbers) * (n + m),
            **stats_fields([time_call(lambda: ss.reduce(phi))])))

    reductions = [
        ThreeSatToVertexCover(), ThreeSatToSubsetSum(),
        ReductionChain([ThreeSatToVertexCover(),
                        VertexCoverToIndependentSet(),
                        IndependentSetToClique()]),
    ]
    rng = random.Random(SEED)
    formulas = []
    for _ in range(validation):
        n = rng.randint(2, 6)
        m = rng.randint(1, 12)
        clauses = [tuple(v if rng.random() < 0.5 else -v
                         for v in rng.sample(range(1, n + 1),
                                             rng.randint(1, min(3, n))))
                   for _ in range(m)]
        formulas.append(ThreeSAT(clauses, n))
    for r in reductions:
        start = time.perf_counter()
        checks = [check_reduction(r, phi) for phi in formulas]
        elapsed = time.perf_counter() - start
        yes = sum(c.source_yes for c in checks)
        rows.append(make_row(
            "reduction_check", r.name, "check_reduction", "reduction",
            validation, f"{yes} sat / {validation - yes} unsat",
            instances=validation, mean_time_s=elapsed / validation,
            success_rate=sum(c.ok for c in checks) / validation))
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _select(rows: List[Dict], experiment: str, method: str,
            field: str = "mean_time_s", parameter: Optional[str] = None,
            problem: Optional[str] = None) -> Tuple[List, List]:
    pts = [(r["input_size"], r[field]) for r in rows
           if r["experiment"] == experiment and r["method"] == method
           and (parameter is None or r["parameter"] == parameter)
           and (problem is None or r["problem"] == problem)]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def _ref(xs: Sequence[float], ys: Sequence[float], shape: Callable,
         anchor: int = -1) -> Tuple[List, List]:
    """Reference curve c·shape(x) through the point at index anchor."""
    if not xs:
        return [], []
    c = ys[anchor] / shape(xs[anchor])
    return list(xs), [c * shape(x) for x in xs]


def plot_runtime_growth(rows: List[Dict], results_dir: str) -> None:
    def ms(series):
        return series[0], [y * 1e3 for y in series[1]]

    ss_bt = ms(_select(rows, "ss_exact", "backtracking",
                       parameter="no-instance"))
    sat_bf = ms(_select(rows, "sat_exact", "brute_force"))
    sat_dpll = ms(_select(rows, "sat_exact", "dpll"))
    vc_exact = ms(_select(rows, "vc_exact", "branch_and_bound"))
    hk = ms(_select(rows, "tsp_exact", "held_karp"))
    brute = ms(_select(rows, "tsp_exact", "brute_force"))
    panels = [
        {"title": "Subset Sum: backtracking is O(2ⁿ)",
         "xlabel": "n (items)", "ylabel": "time (ms)",
         "series": {"backtracking, no solution (worst case)": ss_bt,
                    "backtracking, planted solution": ms(_select(
                        rows, "ss_exact", "backtracking",
                        parameter="planted")),
                    "greedy O(n log n)": ms(_select(rows, "ss_exact",
                                                    "greedy"))},
         "refs": {"∝ 1.8ⁿ": _ref(*ss_bt, lambda n: 1.8 ** n)},
         "logy": True},
        {"title": "3-SAT at the phase transition (m = 4.26n)",
         "xlabel": "n (variables)", "ylabel": "time (ms)",
         "series": {"brute force (2ⁿ assignments)": sat_bf,
                    "backtracking, no unit propagation": ms(_select(
                        rows, "sat_exact", "backtracking")),
                    "DPLL": sat_dpll,
                    "WalkSAT (satisfiable formulas)": ms(_select(
                        rows, "sat_exact", "walksat"))},
         "refs": {"∝ 2ⁿ": _ref(*sat_bf, lambda n: 2 ** n)},
         "logy": True},
        {"title": "Vertex Cover: exact vs approximation",
         "xlabel": "n (vertices, average degree 4)", "ylabel": "time (ms)",
         "series": {"branch and bound (exact)": vc_exact,
                    "2-approx (matching)": ms(_select(
                        rows, "vc_exact", "2approx_matching")),
                    "greedy max-degree": ms(_select(
                        rows, "vc_exact", "greedy_max_degree"))},
         "logy": True},
        {"title": "TSP: exact vs heuristics (Euclidean cities)",
         "xlabel": "n (cities)", "ylabel": "time (ms)",
         "series": {"brute force O(n!)": brute,
                    "Held–Karp O(n²2ⁿ)": hk,
                    "nearest neighbor O(n²)": ms(_select(
                        rows, "tsp_exact", "nearest_neighbor")),
                    "MST double-tree O(n²)": ms(_select(
                        rows, "tsp_exact", "mst_double_tree")),
                    "NN + 2-opt": ms(_select(rows, "tsp_exact", "nn_2opt"))},
         "logy": True},
        {"title": "Polynomial algorithms at scale",
         "xlabel": "n", "ylabel": "time (ms)",
         "series": {"Subset Sum greedy": ms(_select(rows, "ss_greedy",
                                                    "greedy")),
                    "2-SAT (SCC)": ms(_select(rows, "sat_2sat", "scc")),
                    "VC 2-approx": ms(_select(rows, "vc_fast",
                                              "2approx_matching")),
                    "VC greedy": ms(_select(rows, "vc_fast",
                                            "greedy_max_degree")),
                    "TSP nearest neighbor": ms(_select(
                        rows, "tsp_large", "nearest_neighbor")),
                    "TSP MST": ms(_select(rows, "tsp_large",
                                          "mst_double_tree"))},
         "refs": {"∝ n": _ref(*ms(_select(rows, "sat_2sat", "scc")),
                              lambda n: n),
                  "∝ n²": _ref(*ms(_select(rows, "tsp_large",
                                           "nearest_neighbor")),
                               lambda n: n * n)},
         "logx": True, "logy": True},
        {"title": "Search-tree size vs input size",
         "xlabel": "n", "ylabel": "nodes / assignments explored",
         "series": {"Subset Sum backtracking": _select(
                        rows, "ss_exact", "backtracking", "mean_work",
                        parameter="no-instance"),
                    "SAT brute force": _select(rows, "sat_exact",
                                               "brute_force", "mean_work"),
                    "SAT backtracking": _select(rows, "sat_exact",
                                                "backtracking", "mean_work"),
                    "SAT DPLL": _select(rows, "sat_exact", "dpll",
                                        "mean_work"),
                    "VC branch and bound": _select(
                        rows, "vc_exact", "branch_and_bound", "mean_work")},
         "refs": {"2ⁿ": (list(range(4, 31)),
                         [2 ** n for n in range(4, 31)])},
         "logy": True},
        {"title": "Reductions run in polynomial time",
         "xlabel": "m (clauses)", "ylabel": "time to build (ms)",
         "series": {"3-SAT → Vertex Cover": ms(_select(
                        rows, "reduction_size", "reduce",
                        problem="3-sat→vertex_cover")),
                    "3-SAT → Subset Sum": ms(_select(
                        rows, "reduction_size", "reduce",
                        problem="3-sat→subset_sum"))},
         "refs": {"∝ m": _ref(*ms(_select(rows, "reduction_size", "reduce",
                                          problem="3-sat→vertex_cover")),
                              lambda m: m),
                  "∝ m²": _ref(*ms(_select(rows, "reduction_size", "reduce",
                                           problem="3-sat→subset_sum")),
                               lambda m: m * m)},
         "logx": True, "logy": True},
        {"title": "Subset Sum DP: polynomial in n·T, not in input length",
         "xlabel": "n (values ≤ 1,000, T ≈ 250n)", "ylabel": "time (ms)",
         "series": {"bitset DP O(n·T)": ms(_select(rows, "ss_dp",
                                                   "bitset_dp"))},
         "refs": {"∝ n²": _ref(*ms(_select(rows, "ss_dp", "bitset_dp")),
                               lambda n: n * n)},
         "logx": True, "logy": True},
    ]
    plot_mixed_panels(panels, os.path.join(results_dir, "runtime_growth.png"),
                      "Runtime growth: exponential exact search vs "
                      "polynomial algorithms")


def plot_approximation_quality(rows: List[Dict], results_dir: str) -> None:
    q = "mean_quality"
    worst = "worst_quality"
    panels = [
        {"title": "Vertex Cover: size / optimum",
         "xlabel": "n (vertices)", "ylabel": "approximation ratio",
         "series": {"2-approx (matching), mean": _select(
                        rows, "vc_exact", "2approx_matching", q),
                    "2-approx, worst": _select(
                        rows, "vc_exact", "2approx_matching", worst),
                    "greedy max-degree, mean": _select(
                        rows, "vc_exact", "greedy_max_degree", q),
                    "2-approx on K(n/2, n/2)": _select(
                        rows, "vc_tight", "2approx_matching", q)},
         "refs": {"proven bound 2": ([10, 260], [2, 2]),
                  "optimal": ([10, 260], [1, 1])}},
        {"title": "TSP (n ≤ 13): length / optimal (Held–Karp)",
         "xlabel": "n (cities)", "ylabel": "approximation ratio",
         "series": {"nearest neighbor": _select(rows, "tsp_exact",
                                                "nearest_neighbor", q),
                    "repeated NN": _select(rows, "tsp_exact",
                                           "repeated_nn", q),
                    "MST double-tree": _select(rows, "tsp_exact",
                                               "mst_double_tree", q),
                    "NN + 2-opt": _select(rows, "tsp_exact", "nn_2opt", q),
                    "NN worst instance": _select(rows, "tsp_exact",
                                                 "nearest_neighbor", worst)},
         "refs": {"MST bound 2": ([5, 13], [2, 2]),
                  "optimal": ([5, 13], [1, 1])}},
        {"title": "TSP (large n): length / MST lower bound",
         "xlabel": "n (cities)", "ylabel": "length / MST  (≥ length / OPT)",
         "series": {"nearest neighbor": _select(rows, "tsp_large",
                                                "nearest_neighbor", q),
                    "repeated NN": _select(rows, "tsp_large",
                                           "repeated_nn", q),
                    "MST double-tree": _select(rows, "tsp_large",
                                               "mst_double_tree", q),
                    "NN + 2-opt": _select(rows, "tsp_large", "nn_2opt", q)},
         "refs": {"MST bound 2": ([20, 2_000], [2, 2])},
         "logx": True},
        {"title": "Subset Sum greedy: sum / best achievable",
         "xlabel": "n (values ≤ 1,000, random target)",
         "ylabel": "ratio (higher is better)",
         "series": {"greedy, mean": _select(rows, "ss_ratio", "greedy", q),
                    "greedy, worst": _select(rows, "ss_ratio", "greedy",
                                             worst),
                    "greedy hits target exactly (share)": _select(
                        rows, "ss_ratio", "greedy", "success_rate")},
         "refs": {"proven bound 1/2": ([4, 1_000], [0.5, 0.5])},
         "logx": True},
        {"title": "3-SAT (m = 4.26n): fraction of clauses satisfied",
         "xlabel": "n (variables)", "ylabel": "fraction satisfied",
         "series": {"WalkSAT (20k flips), mean": _select(
                        rows, "sat_quality", "walksat", q),
                    "random assignment, mean": _select(
                        rows, "sat_quality", "random_assignment", q),
                    "random assignment, worst": _select(
                        rows, "sat_quality", "random_assignment", worst)},
         "refs": {"expected 7/8": ([10, 70], [7 / 8, 7 / 8])}},
        {"title": "WalkSAT success on satisfiable formulas",
         "xlabel": "n (variables)", "ylabel": "share solved",
         "series": {"WalkSAT (20k flips)": _select(rows, "sat_exact",
                                                   "walksat",
                                                   "success_rate"),
                    "share of formulas satisfiable (DPLL)": _select(
                        rows, "sat_exact", "dpll", "success_rate")}},
    ]
    plot_mixed_panels(panels,
                      os.path.join(results_dir, "approximation_quality.png"),
                      "Approximation quality vs input size")


def plot_tradeoffs(rows: List[Dict], results_dir: str) -> None:
    def points(experiment, size, field="mean_quality", methods=None):
        out = {}
        for r in rows:
            if (r["experiment"] == experiment and r["input_size"] == size
                    and r[field] is not None
                    and (methods is None or r["method"] in methods)):
                out[r["method"]] = ([r["mean_time_s"] * 1e3], [r[field]])
        return out

    tsp_n = max(r["input_size"] for r in rows
                if r["experiment"] == "tsp_exact")
    vc_n = max(r["input_size"] for r in rows if r["experiment"] == "vc_exact")
    budget = [r for r in rows if r["experiment"] == "sat_budget"
              and r["method"] == "walksat"]
    passes = [r for r in rows if r["experiment"] == "tsp_passes"]
    dpll = next((r for r in rows if r["experiment"] == "sat_budget"
                 and r["method"] == "dpll"), None)
    tsp_large = max((r["input_size"] for r in rows
                     if r["experiment"] == "tsp_large"
                     and r["method"] == "repeated_nn"), default=None)
    panels = [
        {"kind": "scatter",
         "title": f"TSP, n = {tsp_n}: time vs tour quality",
         "xlabel": "time (ms)", "ylabel": "length / optimal",
         "points": points("tsp_exact", tsp_n), "logx": True,
         "measured_ticks": False},
        {"kind": "scatter",
         "title": f"TSP, n = {tsp_large}: time vs tour quality",
         "xlabel": "time (ms)", "ylabel": "length / MST lower bound",
         "points": points("tsp_large", tsp_large), "logx": True,
         "measured_ticks": False},
        {"kind": "scatter",
         "title": f"Vertex Cover, n = {vc_n}: time vs cover size",
         "xlabel": "time (ms)", "ylabel": "size / optimum",
         "points": points("vc_exact", vc_n), "logx": True,
         "measured_ticks": False},
        {"title": f"WalkSAT flip budget (n = {WALKSAT_BUDGET_N}, "
                  "satisfiable)",
         "xlabel": "mean time per run (ms)", "ylabel": "share solved",
         "series": {"WalkSAT, budget 10 → 30,000 flips": (
             [r["mean_time_s"] * 1e3 for r in budget],
             [r["success_rate"] for r in budget])},
         "refs": ({"DPLL (exact) time": (
             [dpll["mean_time_s"] * 1e3] * 2, [0, 1])} if dpll else {}),
         "logx": True, "measured_ticks": False},
        {"title": f"2-opt passes (n = {TSP_PASS_N})",
         "xlabel": "time (ms)", "ylabel": "length / MST lower bound",
         "series": {"NN + 2-opt, 1 pass → converged": (
             [r["mean_time_s"] * 1e3 for r in passes],
             [r["mean_quality"] for r in passes])}},
    ]
    plot_mixed_panels(panels, os.path.join(results_dir,
                                           "tradeoff_curves.png"),
                      "Trade-off curves: runtime ↔ accuracy")


def plot_gadget(results_dir: str) -> None:
    """The 3-SAT → Vertex Cover graph for GADGET_FORMULA and the cover
    that a satisfying assignment maps to."""
    r = ThreeSatToVertexCover()
    vc = r.reduce(GADGET_FORMULA)
    assignment = backtracking_sat(GADGET_FORMULA.clauses,
                                  GADGET_FORMULA.num_vars)
    cover = r.forward(assignment)
    pos = {}
    n = GADGET_FORMULA.num_vars
    m = len(GADGET_FORMULA.clauses)
    span = max(3 * n, 3 * m)
    for i in range(1, n + 1):
        x = (i - 0.5) * span / n
        pos[f"x{i}"] = (x - 0.6, 2.0)
        pos[f"¬x{i}"] = (x + 0.6, 2.0)
    for j, slots in enumerate(r.clause_slots):
        cx = (j + 0.5) * span / m
        offsets = [(-0.7, -1.6), (0.0, -0.4), (0.7, -1.6)]
        for slot, (dx, dy) in zip(slots, offsets):
            pos[slot] = (cx + dx, dy)
    truth = ", ".join(f"x{v}={'T' if assignment[v] else 'F'}"
                      for v in range(1, n + 1))
    draw_highlighted_graph(
        vc.edges, pos, sorted(cover),
        os.path.join(results_dir, "reduction_3sat_vertex_cover.png"),
        "3-SAT → Vertex Cover: " + format_cnf(GADGET_FORMULA.clauses),
        f"Satisfying assignment {truth} ↦ cover of size {len(cover)} = "
        f"k = n + 2m = {vc.k} (orange). Top: variable gadgets; bottom: "
        "clause triangles; each slot links to its literal.")


def make_plots(rows: List[Dict], results_dir: str) -> None:
    plot_runtime_growth(rows, results_dir)
    plot_approximation_quality(rows, results_dir)
    plot_tradeoffs(rows, results_dir)
    plot_gadget(results_dir)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r[k]) for k in FIELDS})


def print_summary(rows: List[Dict]) -> None:
    """Condensed console tables for the headline comparisons."""
    print("\n--- Exact solvers: largest size measured ---")
    for exp in ("ss_exact", "sat_exact", "vc_exact", "tsp_exact"):
        exact = [r for r in rows if r["experiment"] == exp
                 and r["approach"] == "exact"]
        for method in dict.fromkeys(r["method"] for r in exact):
            r = max((r for r in exact if r["method"] == method),
                    key=lambda r: r["input_size"])
            print(f"  {r['problem']:14s} {method:18s} n={r['input_size']:<5} "
                  f"{r['mean_time_s'] * 1e3:10.2f} ms")
    print("\n--- Approximation ratios (mean / worst) ---")
    for exp in ("vc_exact", "tsp_exact", "tsp_large", "ss_ratio"):
        for r in rows:
            if (r["experiment"] == exp and r["approach"] != "exact"
                    and r["input_size"] == max(
                        x["input_size"] for x in rows
                        if x["experiment"] == exp
                        and x["method"] == r["method"])):
                print(f"  {exp:10s} {r['method']:18s} n={r['input_size']:<5} "
                      f"{r['quality_metric']:16s} {r['mean_quality']:.3f} / "
                      f"{r['worst_quality']:.3f}")
    print("\n--- Reduction validation ---")
    for r in rows:
        if r["experiment"] == "reduction_check":
            print(f"  {r['problem']:48s} {r['parameter']:22s} "
                  f"consistent: {r['success_rate']:.0%}")


def run(results_dir: str = RESULTS_DIR, quick: bool = False) -> List[Dict]:
    os.makedirs(results_dir, exist_ok=True)
    rows: List[Dict] = []
    print("Subset Sum")
    rows += bench_subset_sum(
        hard_sizes=[n for n in SS_HARD_SIZES if not quick or n <= 22],
        greedy_sizes=[n for n in SS_GREEDY_SIZES
                      if not quick or n <= 100_000],
        ratio_instances=10 if quick else SS_RATIO_INSTANCES)
    print("SAT")
    rows += bench_sat(
        bf_sizes=[n for n in SAT_BF_SIZES if not quick or n <= 14],
        plain_sizes=[n for n in SAT_PLAIN_SIZES if not quick or n <= 25],
        dpll_sizes=[n for n in SAT_DPLL_SIZES if not quick or n <= 50],
        instances=5 if quick else SAT_INSTANCES,
        budgets=[b for b in WALKSAT_BUDGETS if not quick or b <= 10_000],
        budget_seeds=2 if quick else WALKSAT_BUDGET_SEEDS)
    print("Vertex Cover")
    rows += bench_vertex_cover(
        exact_sizes=[n for n in VC_EXACT_SIZES if not quick or n <= 180])
    print("TSP")
    rows += bench_tsp(
        exact_sizes=[n for n in TSP_EXACT_SIZES if not quick or n <= 12],
        instances=3 if quick else TSP_INSTANCES,
        large_sizes=[n for n in TSP_LARGE_SIZES if not quick or n <= 1_000])
    print("Reductions")
    rows += bench_reductions(
        clause_counts=[m for m in REDUCTION_CLAUSES if not quick or m <= 1_000],
        validation=100 if quick else REDUCTION_VALIDATION)

    csv_path = os.path.join(results_dir, CSV_NAME)
    write_csv(rows, csv_path)
    print(f"\nSaved {csv_path} ({len(rows)} rows)")
    print_summary(rows)
    make_plots(rows, results_dir)
    print(f"\nSaved plots to {results_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true",
                        help="smaller sizes and fewer instances")
    args = parser.parse_args()
    start = time.perf_counter()
    run(quick=args.quick)
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
