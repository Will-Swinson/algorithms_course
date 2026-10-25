"""
Week 9 benchmark: approximation algorithms against exact optima and
their proven guarantees.

Every experiment runs on seeded synthetic instances. Where the exact
optimum is computable (small n), ratios are measured against it; at
larger n they are measured against a polynomial-time bound (LP*, MST,
total weight W), which can only overstate the true ratio.

1. Vertex Cover: greedy edge-picking (matching) and LP rounding vs the
   exact branch-and-bound optimum (Week 8); the per-run certificate
   |C|/|M|; the tight K(n, n) family; runtime to n = 100,000.
2. Set Cover: LP relaxation (own simplex), deterministic threshold
   rounding (≤ f·OPT), randomized rounding (O(log n)·OPT, 30 trials per
   instance for mean and variance), and greedy (H_n) vs exact branch
   and bound; a rounds sweep; the family where greedy is k/2 × OPT.
3. Facility Location: star greedy, add/drop/swap local search, and
   greedy + local search vs exhaustive search over facility subsets.
4. Metric TSP: MST doubling (2) and Christofides (3/2), with optimal
   and greedy matching, vs Held–Karp (Week 6) and vs the MST bound.
5. Max-Cut: random cut (1,000 trials per graph), derandomized cut, and
   local search vs exhaustive search; empirical mean and variance vs
   W/2 and Σw²/4; runtime to n = 10,000.

Outputs (benchmarks/results/):
    approx_ratios.png              ratio vs n with guarantee lines
    cost_vs_optimal.png            per-instance cost vs optimum
    randomized_maxcut_results.png  distributions, variance, local search
    set_cover_rounding.png         LP gap, rounding variance, rounds sweep
    approx_runtime.png             runtime and scalability
    tsp_tours.png                  doubling vs Christofides vs optimal
    comparison_table.csv           one row per configuration

Usage:
    python benchmarks/week9_approx_benchmark.py          # ~1 min
    python benchmarks/week9_approx_benchmark.py --quick  # ~10 s
"""
import argparse
import csv
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.approximation import (
    christofides_tour, derandomized_cut, deterministic_rounding,
    exact_facility_location, exact_maxcut, exact_set_cover,
    expected_random_cut, greedy_bad_instance, greedy_facility_location,
    greedy_set_cover, local_search_cut, local_search_facility_location,
    lp_relaxation, lp_rounding_vertex_cover, matching_vertex_cover,
    mst_doubling_tour, optimal_tour, random_cut, random_metric_instance,
    random_set_cover, randomized_rounding, total_weight,
)
from src.approximation.set_cover_lpround import default_rounds
from src.complexity.vertex_cover_approx import exact_vertex_cover
from src.utils.graph_utils import (
    complete_bipartite_edges, random_graph_edges, random_weighted_edges,
)
from src.utils.matrix_utils import euclidean_distance_matrix, random_points
from src.utils.probability_analysis import harmonic, summarize
from src.utils.timer import time_call
from src.utils.visualization import draw_tours, plot_mixed_panels

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_NAME = "comparison_table.csv"
SEED = 9

FIELDS = ["experiment", "problem", "method", "approach", "input_size",
          "parameter", "instances", "mean_time_s", "std_time_s",
          "mean_cost", "mean_ratio", "worst_ratio", "std_ratio",
          "ratio_basis", "guarantee"]

VC_SIZES = [10, 20, 40, 60, 80, 100, 140, 180]
VC_INSTANCES = 5
VC_DEGREE = 4
VC_FAST_SIZES = [100, 1_000, 10_000, 100_000]
VC_LP_SIZES = [25, 50, 100, 200]
VC_BIPARTITE = [2, 5, 10, 25, 50]

SC_SIZES = [20, 30, 40, 60, 80, 100]
SC_SETS_PER_ELEMENT = 0.6
SC_DENSITY = 0.1
SC_INSTANCES = 5
SC_TRIALS = 30
SC_LARGE_SIZES = [200, 400]
SC_ROUNDS = [1, 2, 3, 4, 5, 6, 8]
SC_ROUNDS_N = 80
SC_BAD_K = [2, 3, 4, 5, 6, 7, 8]

FL_SIZES = [(5, 20), (8, 30), (10, 40), (12, 50), (14, 60), (16, 80)]
FL_INSTANCES = 5
FL_FAST_SIZES = [(10, 40), (25, 100), (50, 200), (100, 400)]

TSP_EXACT_SIZES = list(range(5, 14))
TSP_INSTANCES = 5
TSP_LARGE_SIZES = [20, 50, 100, 200, 500, 1_000]
TSP_LARGE_INSTANCES = 3
TSP_CHRISTOFIDES_MAX = 500
TSP_TOUR_N = 13

MC_SIZES = [8, 10, 12, 14, 16, 18, 20]
MC_INSTANCES = 5
MC_DEGREE = 4
MC_TRIALS = 1_000
MC_LS_TRIALS = 100
MC_LARGE_SIZES = [100, 1_000, 10_000]
MC_HIST_N = 20


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
    parts = [f"  {row['experiment']:13s} {row['method']:22s} "
             f"n={row['input_size']:<7} {row['parameter'] or '':14s}"]
    if row["mean_time_s"] is not None:
        parts.append(f"{row['mean_time_s'] * 1e3:10.3f} ms")
    if row["mean_ratio"] is not None:
        parts.append(f"ratio/{row['ratio_basis']}={row['mean_ratio']:.4f}"
                     f" (worst {row['worst_ratio']:.4f})")
    print(" ".join(parts), flush=True)


def timed(func: Callable[[], object]) -> Tuple[float, object]:
    start = time.perf_counter()
    result = func()
    return time.perf_counter() - start, result


def ratio_fields(times: Sequence[float], costs: Sequence[float],
                 ratios: Sequence[float], basis: str,
                 maximize: bool = False) -> Dict:
    """Mean/std time, mean cost, mean/worst/std ratio."""
    t = summarize(times)
    r = summarize(ratios)
    return {"instances": len(ratios), "mean_time_s": t.mean,
            "std_time_s": t.std, "mean_cost": summarize(costs).mean,
            "mean_ratio": r.mean, "std_ratio": r.std,
            "worst_ratio": min(ratios) if maximize else max(ratios),
            "ratio_basis": basis}


def _pairs(samples: Dict, key: str) -> List:
    return samples.setdefault("pairs", {}).setdefault(key, [])


# ---------------------------------------------------------------------------
# 1. Vertex Cover
# ---------------------------------------------------------------------------

def bench_vertex_cover(sizes: Sequence[int] = VC_SIZES,
                       instances: int = VC_INSTANCES,
                       fast_sizes: Sequence[int] = VC_FAST_SIZES,
                       lp_sizes: Sequence[int] = VC_LP_SIZES,
                       bipartite: Sequence[int] = VC_BIPARTITE,
                       samples: Optional[Dict] = None) -> List[Dict]:
    samples = samples if samples is not None else {}
    rows = []
    for n in sizes:
        graphs = [random_graph_edges(n, VC_DEGREE, seed=SEED + 31 * n + i)
                  for i in range(instances)]
        optimum, times = [], []
        for edges in graphs:
            seconds, cover = timed(lambda: exact_vertex_cover(edges))
            optimum.append(len(cover))
            times.append(seconds)
        rows.append(make_row("vc", "vertex_cover", "exact_branch_bound",
                             "exact", n, **ratio_fields(
                                 times, optimum, [1.0] * instances, "OPT")))
        for name, func, approach in (
                ("greedy_matching", matching_vertex_cover, "greedy"),
                ("lp_rounding", lp_rounding_vertex_cover, "lp_rounding")):
            times, costs, ratios, certified = [], [], [], []
            for edges, opt in zip(graphs, optimum):
                seconds, res = timed(lambda: func(edges))
                times.append(seconds)
                costs.append(res.cost)
                ratios.append(res.cost / opt)
                certified.append(res.certified_ratio)
                _pairs(samples, f"vc:{name}").append((opt, res.cost))
            rows.append(make_row("vc", "vertex_cover", name, approach, n,
                                 guarantee=2.0, **ratio_fields(
                                     times, costs, ratios, "OPT")))
            rows.append(make_row(
                "vc_certified", "vertex_cover", name, approach, n,
                f"vs {'|M|' if 'match' in name else 'LP*'}", guarantee=2.0,
                **ratio_fields(times, costs, certified, "own lower bound")))

    for half in bipartite:
        edges = complete_bipartite_edges(half, half)
        res = matching_vertex_cover(edges)
        ratio = res.cost / len(exact_vertex_cover(edges))
        rows.append(make_row("vc_tight", "vertex_cover", "greedy_matching",
                             "greedy", 2 * half, f"K({half},{half})",
                             guarantee=2.0, instances=1, mean_ratio=ratio,
                             worst_ratio=ratio, mean_cost=res.cost,
                             ratio_basis="OPT"))

    for n in fast_sizes:
        edges = random_graph_edges(n, VC_DEGREE, seed=SEED + n)
        rows.append(make_row(
            "vc_runtime", "vertex_cover", "greedy_matching", "greedy", n,
            f"E={len(edges)}", instances=1, mean_time_s=time_call(
                lambda: matching_vertex_cover(edges))))
    for n in lp_sizes:
        edges = random_graph_edges(n, VC_DEGREE, seed=SEED + n)
        seconds, _ = timed(lambda: lp_rounding_vertex_cover(edges))
        rows.append(make_row("vc_runtime", "vertex_cover", "lp_rounding",
                             "lp_rounding", n, f"E={len(edges)}",
                             instances=1, mean_time_s=seconds))
    return rows


# ---------------------------------------------------------------------------
# 2. Set Cover
# ---------------------------------------------------------------------------

def sc_instance(n: int, i: int):
    return random_set_cover(n, max(2, round(SC_SETS_PER_ELEMENT * n)),
                            SC_DENSITY, seed=SEED * 100 + 7 * n + i)


def bench_set_cover(sizes: Sequence[int] = SC_SIZES,
                    instances: int = SC_INSTANCES,
                    trials: int = SC_TRIALS,
                    large_sizes: Sequence[int] = SC_LARGE_SIZES,
                    rounds_sweep: Sequence[int] = SC_ROUNDS,
                    rounds_n: int = SC_ROUNDS_N,
                    bad_k: Sequence[int] = SC_BAD_K,
                    samples: Optional[Dict] = None) -> List[Dict]:
    samples = samples if samples is not None else {}
    rows = []
    for n in list(sizes) + list(large_sizes):
        exact = n in sizes
        basis = "OPT" if exact else "LP"
        exp = "sc" if exact else "sc_large"
        insts = [sc_instance(n, i) for i in range(instances)]
        lps, lp_times, reference = [], [], []
        for inst in insts:
            seconds, (x, lp) = timed(lambda: lp_relaxation(inst))
            lps.append((x, lp))
            lp_times.append(seconds)
        if exact:
            times = []
            for inst in insts:
                seconds, cover = timed(lambda: exact_set_cover(inst))
                times.append(seconds)
                reference.append(cover.cost)
            rows.append(make_row("sc", "set_cover", "exact_branch_bound",
                                 "exact", n, **ratio_fields(
                                     times, reference, [1.0] * instances,
                                     basis)))
            rows.append(make_row(
                "sc", "set_cover", "lp_relaxation", "lower_bound", n,
                "LP*/OPT", **ratio_fields(
                    lp_times, [lp for _, lp in lps],
                    [lp / opt for (_, lp), opt in zip(lps, reference)],
                    basis)))
        else:
            reference = [lp for _, lp in lps]
        freq = max(inst.max_frequency for inst in insts)

        methods = [
            ("greedy", "greedy", harmonic(n),
             lambda inst, x, rng: greedy_set_cover(inst)),
            ("deterministic_rounding", "lp_rounding", float(freq),
             lambda inst, x, rng: deterministic_rounding(inst, x)),
        ]
        for name, approach, guarantee, func in methods:
            times, costs, ratios = [], [], []
            for inst, (x, _), ref in zip(insts, lps, reference):
                cover = func(inst, x, None)
                times.append(time_call(lambda: func(inst, x, None)))
                costs.append(cover.cost)
                ratios.append(cover.cost / ref)
                if exact:
                    _pairs(samples, f"sc:{name}").append((ref, cover.cost))
            rows.append(make_row(exp, "set_cover", name, approach, n,
                                 f"f={freq}", guarantee=guarantee,
                                 **ratio_fields(times, costs, ratios, basis)))

        # randomized rounding: `trials` independent roundings per instance
        times, costs, ratios, spreads, repairs = [], [], [], [], 0
        for i, (inst, (x, _), ref) in enumerate(zip(insts, lps, reference)):
            per = []
            for t in range(trials):
                rng = random.Random(SEED + 1_000 * i + t)
                seconds, cover = timed(lambda: randomized_rounding(
                    inst, x, rng=rng))
                times.append(seconds)
                costs.append(cover.cost)
                per.append(cover.cost / ref)
                repairs += cover.repaired
            ratios += per
            spreads.append(summarize(per).std)
            if exact:
                _pairs(samples, "sc:randomized_rounding").append(
                    (ref, summarize([r * ref for r in per]).mean))
            if n == rounds_n and summarize(per).std >= samples.get(
                    "sc_dist", {}).get("std", -1.0):
                samples["sc_dist"] = {"ratios": per, "n": n,
                                      "std": summarize(per).std}
        rounds = default_rounds(n)
        rows.append(make_row(
            exp, "set_cover", "randomized_rounding", "randomized", n,
            f"t={rounds}, repairs/run={repairs / len(costs):.3f}",
            guarantee=float(rounds), **ratio_fields(times, costs, ratios,
                                                    basis)))
        samples.setdefault("sc_spread", []).append(
            (n, summarize(spreads).mean))

    # rounds sweep: fewer rounds = cheaper but more repairs
    insts = [sc_instance(rounds_n, i) for i in range(instances)]
    lps = [lp_relaxation(inst) for inst in insts]
    opts = [exact_set_cover(inst).cost for inst in insts]
    for t_rounds in rounds_sweep:
        times, costs, ratios, repairs = [], [], [], 0
        for i, (inst, (x, _), opt) in enumerate(zip(insts, lps, opts)):
            for t in range(trials):
                rng = random.Random(SEED + 7_000 * i + t)
                seconds, cover = timed(lambda: randomized_rounding(
                    inst, x, rounds=t_rounds, rng=rng))
                times.append(seconds)
                costs.append(cover.cost)
                ratios.append(cover.cost / opt)
                repairs += cover.repaired
        rows.append(make_row(
            "sc_rounds", "set_cover", "randomized_rounding", "randomized",
            rounds_n, f"t={t_rounds}", guarantee=float(t_rounds),
            **ratio_fields(times, costs, ratios, "OPT")))
        samples.setdefault("sc_repairs", []).append(
            (t_rounds, repairs / len(costs)))

    for k in bad_k:
        inst = greedy_bad_instance(k)
        g = greedy_set_cover(inst)
        x, lp = lp_relaxation(inst)
        d = deterministic_rounding(inst, x)
        for name, cover, approach in (("greedy", g, "greedy"),
                                      ("deterministic_rounding", d,
                                       "lp_rounding")):
            rows.append(make_row(
                "sc_bad", "set_cover", name, approach, inst.num_elements,
                f"k={k}", instances=1, mean_cost=cover.cost,
                mean_ratio=cover.cost / 2, worst_ratio=cover.cost / 2,
                ratio_basis="OPT", guarantee=harmonic(inst.num_elements)))
    return rows


# ---------------------------------------------------------------------------
# 3. Facility Location
# ---------------------------------------------------------------------------

def fl_instance(m: int, n: int, i: int):
    return random_metric_instance(m, n, seed=SEED * 1_000 + 13 * m + n + i)


FL_METHODS = [
    ("greedy", "greedy", 1.861, greedy_facility_location),
    ("local_search", "local_search", 3.0, local_search_facility_location),
    ("greedy+local_search", "local_search", 1.861,
     lambda inst: local_search_facility_location(
         inst, start=greedy_facility_location(inst).open)),
]


def bench_facility(sizes: Sequence[Tuple[int, int]] = FL_SIZES,
                   instances: int = FL_INSTANCES,
                   fast_sizes: Sequence[Tuple[int, int]] = FL_FAST_SIZES,
                   samples: Optional[Dict] = None) -> List[Dict]:
    samples = samples if samples is not None else {}
    rows = []
    for m, n in sizes:
        insts = [fl_instance(m, n, i) for i in range(instances)]
        times, optimum = [], []
        for inst in insts:
            seconds, sol = timed(lambda: exact_facility_location(inst))
            times.append(seconds)
            optimum.append(sol.cost)
        rows.append(make_row("fl", "facility_location", "exact_enumeration",
                             "exact", n, f"m={m}", **ratio_fields(
                                 times, optimum, [1.0] * instances, "OPT")))
        for name, approach, guarantee, func in FL_METHODS:
            times, costs, ratios = [], [], []
            for inst, opt in zip(insts, optimum):
                seconds, sol = timed(lambda: func(inst))
                times.append(seconds)
                costs.append(sol.cost)
                ratios.append(sol.cost / opt)
                _pairs(samples, f"fl:{name}").append((opt, sol.cost))
            rows.append(make_row("fl", "facility_location", name, approach,
                                 n, f"m={m}", guarantee=guarantee,
                                 **ratio_fields(times, costs, ratios, "OPT")))
    for m, n in fast_sizes:
        inst = fl_instance(m, n, 0)
        g_time, g = timed(lambda: greedy_facility_location(inst))
        ls_time, ls = timed(lambda: local_search_facility_location(
            inst, start=g.open))
        rows.append(make_row("fl_runtime", "facility_location", "greedy",
                             "greedy", n, f"m={m}", instances=1,
                             mean_time_s=g_time, mean_cost=g.cost))
        rows.append(make_row("fl_runtime", "facility_location",
                             "greedy+local_search", "local_search", n,
                             f"m={m}", instances=1,
                             mean_time_s=g_time + ls_time,
                             mean_cost=ls.cost,
                             mean_ratio=g.cost / ls.cost,
                             worst_ratio=g.cost / ls.cost,
                             ratio_basis="greedy / greedy+LS"))
    return rows


# ---------------------------------------------------------------------------
# 4. Metric TSP
# ---------------------------------------------------------------------------

TSP_METHODS = [
    ("mst_doubling", 2.0, mst_doubling_tour),
    ("christofides", 1.5, christofides_tour),
    ("christofides_greedy_match", None,
     lambda d: christofides_tour(d, matching="greedy")),
]


def tsp_points(n: int, i: int):
    return random_points(n, seed=SEED * 77 + 101 * n + i)


def bench_tsp(exact_sizes: Sequence[int] = TSP_EXACT_SIZES,
              instances: int = TSP_INSTANCES,
              large_sizes: Sequence[int] = TSP_LARGE_SIZES,
              large_instances: int = TSP_LARGE_INSTANCES,
              christofides_max: int = TSP_CHRISTOFIDES_MAX,
              tour_n: int = TSP_TOUR_N,
              samples: Optional[Dict] = None) -> List[Dict]:
    samples = samples if samples is not None else {}
    rows = []
    for n in exact_sizes:
        dists = [euclidean_distance_matrix(tsp_points(n, i))
                 for i in range(instances)]
        times, optimum = [], []
        for d in dists:
            seconds, (cost, _) = timed(lambda: optimal_tour(d))
            times.append(seconds)
            optimum.append(cost)
        rows.append(make_row("tsp", "metric_tsp", "held_karp", "exact", n,
                             **ratio_fields(times, optimum,
                                            [1.0] * instances, "OPT")))
        for name, guarantee, func in TSP_METHODS:
            times, costs, ratios = [], [], []
            for d, opt in zip(dists, optimum):
                seconds, res = timed(lambda: func(d))
                times.append(seconds)
                costs.append(res.cost)
                ratios.append(res.cost / opt)
                _pairs(samples, f"tsp:{name}").append((opt, res.cost))
            rows.append(make_row("tsp", "metric_tsp", name, "greedy", n,
                                 guarantee=guarantee,
                                 **ratio_fields(times, costs, ratios, "OPT")))
    for n in large_sizes:
        dists = [euclidean_distance_matrix(tsp_points(n, i))
                 for i in range(large_instances)]
        for name, guarantee, func in TSP_METHODS:
            if name.startswith("christofides") and n > christofides_max:
                continue
            times, costs, ratios = [], [], []
            for d in dists:
                seconds, res = timed(lambda: func(d))
                times.append(seconds)
                costs.append(res.cost)
                ratios.append(res.certified_ratio)
            rows.append(make_row("tsp_large", "metric_tsp", name, "greedy",
                                 n, guarantee=guarantee,
                                 **ratio_fields(times, costs, ratios, "MST")))

    points = tsp_points(tour_n, 0)
    d = euclidean_distance_matrix(points)
    opt_cost, opt_tour = optimal_tour(d)
    samples["tours"] = {"points": points, "tours": {
        "Optimal (Held–Karp)": (opt_tour, opt_cost),
        "MST doubling (≤ 2×)": (lambda r: (r.tour, r.cost))(
            mst_doubling_tour(d)),
        "Christofides (≤ 1.5×)": (lambda r: (r.tour, r.cost))(
            christofides_tour(d)),
    }}
    return rows


# ---------------------------------------------------------------------------
# 5. Max-Cut
# ---------------------------------------------------------------------------

def mc_graph(n: int, i: int, weighted: bool):
    seed = SEED * 500 + 37 * n + i
    return (random_weighted_edges(n, MC_DEGREE, seed=seed) if weighted
            else random_graph_edges(n, MC_DEGREE, seed=seed))


def bench_maxcut(sizes: Sequence[int] = MC_SIZES,
                 instances: int = MC_INSTANCES,
                 trials: int = MC_TRIALS,
                 ls_trials: int = MC_LS_TRIALS,
                 large_sizes: Sequence[int] = MC_LARGE_SIZES,
                 hist_n: int = MC_HIST_N,
                 samples: Optional[Dict] = None) -> List[Dict]:
    samples = samples if samples is not None else {}
    rows = []
    theory = samples.setdefault("mc_theory", [])
    for weighted in (False, True):
        tag = "weighted" if weighted else "unweighted"
        for n in sizes:
            graphs = [mc_graph(n, i, weighted) for i in range(instances)]
            times, optimum = [], []
            for edges in graphs:
                seconds, best = timed(lambda: exact_maxcut(
                    edges, vertices=range(n)))
                times.append(seconds)
                optimum.append(best.value)
            rows.append(make_row("mc", "max_cut", "exact_enumeration",
                                 "exact", n, tag, **ratio_fields(
                                     times, optimum, [1.0] * instances,
                                     "OPT", maximize=True)))

            rand_times, rand_vals, rand_ratios = [], [], []
            ls_times, ls_vals, ls_ratios = [], [], []
            for i, (edges, opt) in enumerate(zip(graphs, optimum)):
                vals = []
                for t in range(trials):
                    rng = random.Random(SEED + 10_000 * i + t)
                    seconds, cut = timed(lambda: random_cut(
                        edges, range(n), rng))
                    rand_times.append(seconds)
                    vals.append(cut.value)
                rand_vals += vals
                rand_ratios += [v / opt for v in vals]
                mean, var = expected_random_cut(edges)
                s = summarize(vals)
                theory.append({"n": n, "weighted": weighted,
                               "mean_theory": mean, "mean_measured": s.mean,
                               "var_theory": var, "var_measured": s.variance,
                               "below_half": sum(v < opt / 2 for v in vals)
                               / len(vals)})
                _pairs(samples, "mc:random_cut").append((opt, s.mean))
                for t in range(ls_trials):
                    rng = random.Random(SEED + 20_000 * i + t)
                    seconds, cut = timed(lambda: local_search_cut(
                        edges, vertices=range(n), rng=rng))
                    ls_times.append(seconds)
                    ls_vals.append(cut.value)
                    ls_ratios.append(cut.value / opt)
                _pairs(samples, "mc:local_search").append(
                    (opt, summarize(ls_vals[-ls_trials:]).mean))
            rows.append(make_row("mc", "max_cut", "random_cut", "randomized",
                                 n, tag, guarantee=0.5, **ratio_fields(
                                     rand_times, rand_vals, rand_ratios,
                                     "OPT", maximize=True)))
            rows.append(make_row("mc", "max_cut", "random+local_search",
                                 "local_search", n, tag, guarantee=0.5,
                                 **ratio_fields(ls_times, ls_vals, ls_ratios,
                                                "OPT", maximize=True)))
            times, vals, ratios = [], [], []
            for edges, opt in zip(graphs, optimum):
                seconds, cut = timed(lambda: derandomized_cut(edges,
                                                              range(n)))
                times.append(seconds)
                vals.append(cut.value)
                ratios.append(cut.value / opt)
                _pairs(samples, "mc:derandomized").append((opt, cut.value))
            rows.append(make_row("mc", "max_cut", "derandomized",
                                 "greedy", n, tag, guarantee=0.5,
                                 **ratio_fields(times, vals, ratios, "OPT",
                                                maximize=True)))

    # one graph in detail: the full distribution of the random cut
    edges = mc_graph(hist_n, 0, False)
    opt = exact_maxcut(edges, range(hist_n)).value
    samples["mc_hist"] = {
        "n": hist_n, "opt": opt, "W": total_weight(edges),
        "random": [random_cut(edges, range(hist_n),
                              random.Random(SEED + t)).value
                   for t in range(5 * trials)],
        "local": [local_search_cut(edges, vertices=range(hist_n),
                                   rng=random.Random(SEED + t)).value
                  for t in range(5 * trials)],
        "derandomized": derandomized_cut(edges, range(hist_n)).value,
    }

    for n in large_sizes:
        edges = mc_graph(n, 0, False)
        w = total_weight(edges)
        for name, approach, func in (
                ("random_cut", "randomized",
                 lambda: random_cut(edges, range(n), random.Random(SEED))),
                ("derandomized", "greedy",
                 lambda: derandomized_cut(edges, range(n))),
                ("random+local_search", "local_search",
                 lambda: local_search_cut(edges, vertices=range(n),
                                          rng=random.Random(SEED)))):
            seconds, cut = timed(func)
            rows.append(make_row("mc_large", "max_cut", name, approach, n,
                                 f"E={len(edges)}", instances=1,
                                 mean_time_s=seconds, mean_cost=cut.value,
                                 mean_ratio=cut.value / w,
                                 worst_ratio=cut.value / w, ratio_basis="W",
                                 guarantee=0.5))
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _select(rows: List[Dict], experiment: str, method: str,
            field: str = "mean_ratio", parameter: Optional[str] = None
            ) -> Tuple[List, List]:
    pts = sorted((r["input_size"], r[field]) for r in rows
                 if r["experiment"] == experiment and r["method"] == method
                 and (parameter is None or r["parameter"] == parameter))
    return [p[0] for p in pts], [p[1] for p in pts]


def _band(rows, experiment, method, parameter=None):
    pts = sorted((r["input_size"], r["mean_ratio"], r["std_ratio"] or 0)
                 for r in rows if r["experiment"] == experiment
                 and r["method"] == method
                 and (parameter is None or r["parameter"] == parameter))
    return ([p[0] for p in pts], [p[1] - p[2] for p in pts],
            [p[1] + p[2] for p in pts])


def _flat(xs, y):
    return (list(xs), [y] * len(xs)) if xs else ([], [])


def _guarantee_curve(rows, experiment, method):
    pts = sorted((r["input_size"], r["guarantee"]) for r in rows
                 if r["experiment"] == experiment and r["method"] == method
                 and r["guarantee"] is not None)
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_approx_ratios(rows: List[Dict], results_dir: str) -> None:
    vc_x = _select(rows, "vc", "greedy_matching")[0]
    tsp_x = _select(rows, "tsp", "mst_doubling")[0]
    fl_x = _select(rows, "fl", "greedy")[0]
    mc_x = _select(rows, "mc", "random_cut", parameter="unweighted")[0]
    panels = [
        {"title": "Vertex Cover: size / OPT",
         "xlabel": "n (vertices, average degree 4)", "ylabel": "ratio",
         "series": {"greedy edge picking": _select(rows, "vc",
                                                   "greedy_matching"),
                    "LP rounding (x ≥ 1/2)": _select(rows, "vc",
                                                     "lp_rounding"),
                    "edge picking, worst": _select(rows, "vc",
                                                   "greedy_matching",
                                                   "worst_ratio"),
                    "edge picking on K(n/2, n/2)": _select(
                        rows, "vc_tight", "greedy_matching"),
                    "certified bound |C| / |M| (no OPT needed)": _select(
                        rows, "vc_certified", "greedy_matching")},
         "refs": {"guarantee 2": _flat(vc_x, 2.0),
                  "optimal": _flat(vc_x, 1.0)}},
        {"title": "Set Cover: cost / OPT",
         "xlabel": "n (elements; 0.6n sets)", "ylabel": "ratio",
         "series": {"greedy": _select(rows, "sc", "greedy"),
                    "deterministic rounding": _select(
                        rows, "sc", "deterministic_rounding"),
                    "randomized rounding (mean)": _select(
                        rows, "sc", "randomized_rounding"),
                    "LP* / OPT (integrality gap)": _select(
                        rows, "sc", "lp_relaxation")},
         "bands": {"randomized rounding (mean)": _band(
             rows, "sc", "randomized_rounding")},
         "refs": {"H_n (greedy guarantee)": _guarantee_curve(
             rows, "sc", "greedy")}},
        {"title": "Facility Location: cost / OPT",
         "xlabel": "n clients (5–16 facilities)", "ylabel": "ratio",
         "series": {"star greedy": _select(rows, "fl", "greedy"),
                    "local search": _select(rows, "fl", "local_search"),
                    "greedy + local search": _select(
                        rows, "fl", "greedy+local_search"),
                    "greedy, worst": _select(rows, "fl", "greedy",
                                             "worst_ratio")},
         "refs": {"metric greedy guarantee 1.861": _flat(fl_x, 1.861),
                  "optimal": _flat(fl_x, 1.0)}},
        {"title": "Metric TSP: length / OPT (Held–Karp)",
         "xlabel": "n (cities)", "ylabel": "ratio",
         "series": {"MST doubling": _select(rows, "tsp", "mst_doubling"),
                    "Christofides": _select(rows, "tsp", "christofides"),
                    "Christofides, greedy matching": _select(
                        rows, "tsp", "christofides_greedy_match"),
                    "MST doubling, worst": _select(rows, "tsp",
                                                   "mst_doubling",
                                                   "worst_ratio"),
                    "Christofides, worst": _select(rows, "tsp",
                                                   "christofides",
                                                   "worst_ratio")},
         "refs": {"guarantee 2": _flat(tsp_x, 2.0),
                  "guarantee 3/2": _flat(tsp_x, 1.5)}},
        {"title": "Max-Cut: value / OPT (higher is better)",
         "xlabel": "n (vertices, average degree 4)", "ylabel": "ratio",
         "series": {"random cut (mean of 1,000)": _select(
                        rows, "mc", "random_cut", parameter="unweighted"),
                    "random cut, worst draw": _select(
                        rows, "mc", "random_cut", "worst_ratio",
                        parameter="unweighted"),
                    "derandomized": _select(rows, "mc", "derandomized",
                                            parameter="unweighted"),
                    "random + local search": _select(
                        rows, "mc", "random+local_search",
                        parameter="unweighted")},
         "bands": {"random cut (mean of 1,000)": _band(
             rows, "mc", "random_cut", parameter="unweighted")},
         "refs": {"guarantee 1/2 (in expectation)": _flat(mc_x, 0.5)}},
        {"title": "Larger instances: ratio vs a computable bound",
         "xlabel": "n", "ylabel": "ratio vs bound (overstates true ratio)",
         "series": {"TSP doubling / MST": _select(rows, "tsp_large",
                                                  "mst_doubling"),
                    "TSP Christofides / MST": _select(rows, "tsp_large",
                                                      "christofides"),
                    "Set Cover greedy / LP*": _large_sc(rows, "greedy"),
                    "Set Cover rand. rounding / LP*": _large_sc(
                        rows, "randomized_rounding")},
         "logx": True},
    ]
    plot_mixed_panels(panels, os.path.join(results_dir, "approx_ratios.png"),
                      "Approximation ratios vs input size, with proven "
                      "guarantees")


def _large_sc(rows, method):
    pts = sorted((r["input_size"], r["mean_ratio"]) for r in rows
                 if r["experiment"] == "sc_large" and r["method"] == method)
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_cost_vs_optimal(samples: Dict, results_dir: str) -> None:
    pairs = samples.get("pairs", {})

    def pts(key):
        data = pairs.get(key, [])
        return [p[0] for p in data], [p[1] for p in data]

    def lines(keys, factors):
        xs = [p[0] for k in keys for p in pairs.get(k, [])]
        if not xs:
            return {}
        lo, hi = min(xs), max(xs)
        return {f"{f:g} × OPT": ([lo, hi], [f * lo, f * hi])
                for f in factors}

    panels = [
        {"kind": "scatter", "title": "Vertex Cover",
         "xlabel": "optimal cover size", "ylabel": "algorithm's cover size",
         "points": {"greedy edge picking": pts("vc:greedy_matching"),
                    "LP rounding": pts("vc:lp_rounding")},
         "refs": lines(["vc:greedy_matching"], [1, 2])},
        {"kind": "scatter", "title": "Set Cover",
         "xlabel": "optimal cost", "ylabel": "algorithm's cost",
         "points": {"greedy": pts("sc:greedy"),
                    "deterministic rounding": pts(
                        "sc:deterministic_rounding"),
                    "randomized rounding (mean)": pts(
                        "sc:randomized_rounding")},
         "refs": lines(["sc:greedy"], [1, 2])},
        {"kind": "scatter", "title": "Facility Location (metric)",
         "xlabel": "optimal cost", "ylabel": "algorithm's cost",
         "points": {"star greedy": pts("fl:greedy"),
                    "local search": pts("fl:local_search"),
                    "greedy + local search": pts("fl:greedy+local_search")},
         "refs": lines(["fl:greedy"], [1, 1.861])},
        {"kind": "scatter", "title": "Metric TSP",
         "xlabel": "optimal tour length", "ylabel": "algorithm's tour length",
         "points": {"MST doubling": pts("tsp:mst_doubling"),
                    "Christofides": pts("tsp:christofides")},
         "refs": lines(["tsp:mst_doubling"], [1, 1.5, 2])},
        {"kind": "scatter", "title": "Max-Cut (maximization)",
         "xlabel": "optimal cut value", "ylabel": "algorithm's cut value",
         "points": {"random cut (mean)": pts("mc:random_cut"),
                    "derandomized": pts("mc:derandomized"),
                    "random + local search (mean)": pts(
                        "mc:local_search")},
         "refs": lines(["mc:random_cut"], [1, 0.5])},
    ]
    plot_mixed_panels(panels, os.path.join(results_dir,
                                           "cost_vs_optimal.png"),
                      "Algorithm cost vs optimal cost, one point per "
                      "instance")


def plot_maxcut(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    hist = samples.get("mc_hist")
    theory = samples.get("mc_theory", [])
    panels = []
    if hist:
        panels.append({
            "kind": "hist", "bins": 25,
            "title": f"One graph (n = {hist['n']}): 5,000 runs each",
            "xlabel": "cut value", "ylabel": "runs",
            "samples": {"random cut": hist["random"],
                        "random + local search": hist["local"]},
            "vlines": {"OPT": hist["opt"], "W/2 = E[random cut]":
                       hist["W"] / 2, "OPT/2": hist["opt"] / 2,
                       "derandomized": hist["derandomized"]},
            "colors": {"OPT": "black", "W/2 = E[random cut]": "tab:blue",
                       "OPT/2": "tab:red", "derandomized": "tab:green"}})
    if theory:
        lo = min(t["mean_theory"] for t in theory)
        hi = max(t["mean_theory"] for t in theory)
        vlo = min(t["var_theory"] for t in theory)
        vhi = max(t["var_theory"] for t in theory)
        panels += [
            {"kind": "scatter", "title": "Mean of 1,000 random cuts vs W/2",
             "xlabel": "theory: W/2", "ylabel": "measured mean",
             "points": {"unweighted": (
                 [t["mean_theory"] for t in theory if not t["weighted"]],
                 [t["mean_measured"] for t in theory if not t["weighted"]]),
                 "weighted": (
                 [t["mean_theory"] for t in theory if t["weighted"]],
                 [t["mean_measured"] for t in theory if t["weighted"]])},
             "refs": {"y = x": ([lo, hi], [lo, hi])}},
            {"kind": "scatter",
             "title": "Variance of 1,000 random cuts vs Σw²/4",
             "xlabel": "theory: Σ w² / 4", "ylabel": "measured variance",
             "points": {"unweighted": (
                 [t["var_theory"] for t in theory if not t["weighted"]],
                 [t["var_measured"] for t in theory if not t["weighted"]]),
                 "weighted": (
                 [t["var_theory"] for t in theory if t["weighted"]],
                 [t["var_measured"] for t in theory if t["weighted"]])},
             "refs": {"y = x": ([vlo, vhi], [vlo, vhi])},
             "logx": True, "logy": True, "measured_ticks": False},
        ]
    panels += [
        {"title": "Value / OPT by method (weighted graphs)",
         "xlabel": "n (vertices)", "ylabel": "ratio",
         "series": {"random cut (mean)": _select(rows, "mc", "random_cut",
                                                 parameter="weighted"),
                    "random cut, worst of 1,000": _select(
                        rows, "mc", "random_cut", "worst_ratio",
                        parameter="weighted"),
                    "derandomized": _select(rows, "mc", "derandomized",
                                            parameter="weighted"),
                    "random + local search": _select(
                        rows, "mc", "random+local_search",
                        parameter="weighted")},
         "refs": {"1/2": _flat(MC_SIZES, 0.5)}},
        {"title": "Share of random cuts below OPT/2",
         "xlabel": "n (vertices)", "ylabel": "share of 1,000 draws",
         "series": {
             "unweighted": _below_half(theory, False),
             "weighted": _below_half(theory, True)}},
        {"title": "Large graphs: value / W (W ≥ OPT)",
         "xlabel": "n (vertices, average degree 4)",
         "ylabel": "cut / total weight",
         "series": {m: _select(rows, "mc_large", m) for m in
                    ("random_cut", "derandomized", "random+local_search")},
         "refs": {"1/2": _flat(MC_LARGE_SIZES, 0.5)},
         "logx": True},
    ]
    plot_mixed_panels(panels, os.path.join(
        results_dir, "randomized_maxcut_results.png"),
        "Randomized Max-Cut: expectation, variance, and local search")


def _below_half(theory, weighted):
    by_n: Dict[int, List[float]] = {}
    for t in theory:
        if t["weighted"] == weighted:
            by_n.setdefault(t["n"], []).append(t["below_half"])
    xs = sorted(by_n)
    return xs, [sum(by_n[n]) / len(by_n[n]) for n in xs]


def plot_set_cover_rounding(rows: List[Dict], samples: Dict,
                            results_dir: str) -> None:
    rounds = [r for r in rows if r["experiment"] == "sc_rounds"]
    t_vals = [int(r["parameter"].split("=")[1]) for r in rounds]
    repairs = samples.get("sc_repairs", [])
    dist = samples.get("sc_dist")
    spread = samples.get("sc_spread", [])
    panels = [
        {"title": f"Randomized rounding: rounds t (n = {SC_ROUNDS_N})",
         "xlabel": "rounds t", "ylabel": "cost / OPT",
         "series": {"mean cost / OPT": (t_vals,
                                        [r["mean_ratio"] for r in rounds]),
                    "worst / OPT": (t_vals,
                                    [r["worst_ratio"] for r in rounds])},
         "bands": {"mean cost / OPT": (
             t_vals, [r["mean_ratio"] - r["std_ratio"] for r in rounds],
             [r["mean_ratio"] + r["std_ratio"] for r in rounds])}},
        {"title": "Repairs needed per run",
         "xlabel": "rounds t", "ylabel": "uncovered elements patched",
         "series": {"measured": ([p[0] for p in repairs],
                                 [p[1] for p in repairs])}},
        {"title": "Standard deviation of cost / OPT across 30 roundings",
         "xlabel": "n (elements)", "ylabel": "std of ratio (per instance)",
         "series": {"randomized rounding": ([p[0] for p in spread],
                                            [p[1] for p in spread])},
         "logx": True},
        {"title": "Greedy's bad family: cost / OPT = k/2",
         "xlabel": "n (elements) = 2(2ᵏ − 1)", "ylabel": "ratio",
         "series": {"greedy": _select(rows, "sc_bad", "greedy"),
                    "deterministic rounding": _select(
                        rows, "sc_bad", "deterministic_rounding")},
         "refs": {"H_n (greedy's bound)": _guarantee_curve(
             rows, "sc_bad", "greedy")},
         "logx": True},
    ]
    if dist:
        panels.insert(2, {
            "kind": "hist", "bins": 15,
            "title": f"{len(dist['ratios'])} randomized roundings of the "
                     f"most variable instance (n = {dist['n']})",
            "xlabel": "cost / OPT", "ylabel": "runs",
            "samples": {"randomized rounding": dist["ratios"]},
            "vlines": {"mean": summarize(dist["ratios"]).mean}})
    plot_mixed_panels(panels, os.path.join(results_dir,
                                           "set_cover_rounding.png"),
                      "Set Cover: LP relaxation and rounding")


def plot_runtime(rows: List[Dict], results_dir: str) -> None:
    def ms(series):
        return series[0], [y * 1e3 if y is not None else None
                           for y in series[1]]

    def t(exp, method, parameter=None):
        return ms(_select(rows, exp, method, "mean_time_s", parameter))

    def t_sc(method):
        a, b = t("sc", method), t("sc_large", method)
        pts = sorted(zip(a[0] + b[0], a[1] + b[1]))
        return [p[0] for p in pts], [p[1] for p in pts]

    panels = [
        {"title": "Vertex Cover", "xlabel": "n (vertices)",
         "ylabel": "time (ms)",
         "series": {"greedy edge picking": t("vc_runtime",
                                             "greedy_matching"),
                    "LP rounding (simplex)": t("vc_runtime", "lp_rounding"),
                    "exact branch and bound": t("vc", "exact_branch_bound")},
         "logx": True, "logy": True},
        {"title": "Set Cover", "xlabel": "n (elements)",
         "ylabel": "time (ms)",
         "series": {"greedy": t_sc("greedy"),
                    "LP relaxation": t("sc", "lp_relaxation"),
                    "randomized rounding (given LP)": t_sc(
                        "randomized_rounding"),
                    "exact branch and bound": t("sc", "exact_branch_bound")},
         "logx": True, "logy": True},
        {"title": "Facility Location", "xlabel": "n (clients)",
         "ylabel": "time (ms)",
         "series": {"greedy": t("fl_runtime", "greedy"),
                    "greedy + local search": t("fl_runtime",
                                               "greedy+local_search"),
                    "exact (2^m subsets)": t("fl", "exact_enumeration")},
         "logx": True, "logy": True},
        {"title": "Metric TSP", "xlabel": "n (cities)",
         "ylabel": "time (ms)",
         "series": {"MST doubling O(n²)": t("tsp_large", "mst_doubling"),
                    "Christofides O(n³)": t("tsp_large", "christofides"),
                    "Christofides, greedy matching": t(
                        "tsp_large", "christofides_greedy_match"),
                    "Held–Karp O(n²2ⁿ)": t("tsp", "held_karp")},
         "logx": True, "logy": True},
        {"title": "Max-Cut", "xlabel": "n (vertices)",
         "ylabel": "time (ms)",
         "series": {"random cut": t("mc_large", "random_cut"),
                    "derandomized": t("mc_large", "derandomized"),
                    "random + local search": t("mc_large",
                                               "random+local_search"),
                    "exact (2ⁿ⁻¹ cuts)": t("mc", "exact_enumeration",
                                           "unweighted")},
         "logx": True, "logy": True},
    ]
    for panel in panels:
        panel["measured_ticks"] = False
    plot_mixed_panels(panels, os.path.join(results_dir, "approx_runtime.png"),
                      "Runtime and scalability")


def make_plots(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    plot_approx_ratios(rows, results_dir)
    plot_cost_vs_optimal(samples, results_dir)
    plot_maxcut(rows, samples, results_dir)
    plot_set_cover_rounding(rows, samples, results_dir)
    plot_runtime(rows, results_dir)
    if "tours" in samples:
        draw_tours(samples["tours"]["points"], samples["tours"]["tours"],
                   os.path.join(results_dir, "tsp_tours.png"),
                   f"Metric TSP on {len(samples['tours']['points'])} "
                   "random cities")


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
    """Mean and worst ratio at the largest exactly solved size."""
    print("\n--- Ratios at the largest size with a known optimum "
          "(mean / worst / guarantee) ---")
    for exp in ("vc", "sc", "fl", "tsp", "mc"):
        group = [r for r in rows if r["experiment"] == exp
                 and r["approach"] not in ("exact",)
                 and r["ratio_basis"] == "OPT"]
        if not group:
            continue
        n = max(r["input_size"] for r in group)
        for r in group:
            if r["input_size"] == n:
                g = f"{r['guarantee']:.3g}" if r["guarantee"] else "—"
                print(f"  {r['problem']:18s} {r['method']:26s} n={n:<4} "
                      f"{r['parameter'] or '':12.12s} "
                      f"{r['mean_ratio']:.3f} / {r['worst_ratio']:.3f} / {g}")


def run(results_dir: str = RESULTS_DIR, quick: bool = False) -> List[Dict]:
    os.makedirs(results_dir, exist_ok=True)
    samples: Dict = {}
    rows: List[Dict] = []
    print("Vertex Cover")
    rows += bench_vertex_cover(
        sizes=[n for n in VC_SIZES if not quick or n <= 100],
        samples=samples)
    print("Set Cover")
    rows += bench_set_cover(
        sizes=[n for n in SC_SIZES if not quick or n <= 80],
        trials=10 if quick else SC_TRIALS,
        large_sizes=[n for n in SC_LARGE_SIZES if not quick or n <= 200],
        samples=samples)
    print("Facility Location")
    rows += bench_facility(
        sizes=[s for s in FL_SIZES if not quick or s[0] <= 12],
        fast_sizes=[s for s in FL_FAST_SIZES if not quick or s[0] <= 50],
        samples=samples)
    print("Metric TSP")
    rows += bench_tsp(
        exact_sizes=[n for n in TSP_EXACT_SIZES if not quick or n <= 11],
        instances=3 if quick else TSP_INSTANCES,
        large_sizes=[n for n in TSP_LARGE_SIZES if not quick or n <= 200],
        samples=samples)
    print("Max-Cut")
    rows += bench_maxcut(
        sizes=[n for n in MC_SIZES if not quick or n <= 16],
        trials=200 if quick else MC_TRIALS,
        ls_trials=30 if quick else MC_LS_TRIALS,
        large_sizes=[n for n in MC_LARGE_SIZES if not quick or n <= 1_000],
        samples=samples)

    csv_path = os.path.join(results_dir, CSV_NAME)
    write_csv(rows, csv_path)
    print(f"\nSaved {csv_path} ({len(rows)} rows)")
    print_summary(rows)
    make_plots(rows, samples, results_dir)
    print(f"\nSaved plots to {results_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true",
                        help="smaller sizes and fewer trials")
    args = parser.parse_args()
    start = time.perf_counter()
    run(quick=args.quick)
    print(f"Total wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
