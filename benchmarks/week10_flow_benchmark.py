"""
Week 10 benchmark: max-flow and min-cost flow algorithms on large
graphs, plus the flow applications.

Every flow returned by every solver is checked with the validators
(capacity, conservation, residual consistency, max-flow min-cut
certificate; min-cost optimality for MCF) and the row records whether
it passed. All instances are seeded.

1. Runtime scaling, Edmonds–Karp (EK) vs capacity scaling vs push–
   relabel (FIFO and highest-label) vs min-cost max-flow (MCF), with
   networkx's preflow_push as an outside reference:
     sparse   random networks, avg degree 4, n = 1k → 50k
     dense    arc probability 0.2, n = 100 → 1,200 (up to ~290k arcs)
     grid     segmentation-shaped grids, 32² → 224² (≈ 50k nodes)
   Slow methods stop at the largest size where they finish in seconds.
2. Memory: tracemalloc peak while each solver runs (network built
   beforehand), sparse n = 1k → 20k.
3. Augmentations: EK / scaling / MCF augmenting paths vs n; EK path
   length across one run (BFS distances never decrease); bottleneck
   distribution; EK vs scaling across capacity distributions.
4. Push–relabel convergence: flow reaching t and active-node count vs
   operations (FIFO vs highest-label); ablation of the gap and global-
   relabel heuristics (runs capped at a time limit).
5. Applications: matching vs networkx Hopcroft–Karp; segmentation
   accuracy vs noise and runtime vs image size; deadline-scheduling
   feasibility vs number of machines.

Outputs (benchmarks/results/):
    flow_runtime.png          runtime vs size: sparse, dense, grid
    flow_memory.png           peak memory and bytes per arc
    augmentations.png         augmenting-path counts, path lengths,
                              bottlenecks, capacity distributions
    pr_convergence.png        push–relabel profiles and heuristic ablation
    flow_applications.png     matching, segmentation, scheduling
    network_example.png       CLRS network: max flow and min cut
    matching_example.png      a maximum bipartite matching
    segmentation_example.png  graph cut vs threshold on a noisy image
    flow_storyboard.png       EK and push–relabel frames, side by side
    ek_animation.gif          Edmonds–Karp, one frame per augmentation
    pr_animation.gif          push–relabel, heights on the y axis
    flow_comparison_table.csv one row per configuration

Usage:
    python benchmarks/week10_flow_benchmark.py          # ~5 min
    python benchmarks/week10_flow_benchmark.py --quick  # ~30 s
"""
import argparse
import csv
import os
import signal
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.flow import preflow_push

from src.flow.applications import (
    bipartite_matching, schedule_jobs, segment_image, segmentation_scores,
    threshold_segment,
)
from src.flow.core import FlowNetwork, get_solver, push_relabel
from src.flow.utils.generators import (
    bipartite_graph, dense_network, grid_network, random_jobs,
    sparse_network, synthetic_image,
)
from src.flow.utils.validators import validate_flow
from src.flow.visualization import (
    animate_edmonds_karp, animate_push_relabel, plot_matching, plot_network,
    plot_segmentation,
)
from src.flow.visualization.animate_flow import (
    _draw_ek, _draw_pr, edmonds_karp_frames, push_relabel_frames, _subsample,
)
from src.flow.visualization.plot_network import layered_positions
from src.utils.timer import peak_memory
from src.utils.visualization import plot_mixed_panels

RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
CSV_NAME = "flow_comparison_table.csv"
SEED = 10

FIELDS = ["experiment", "family", "method", "n", "arcs", "parameter",
          "instances", "mean_time_s", "std_time_s", "flow_value",
          "augmentations", "pushes", "relabels", "peak_mem_mb", "valid",
          "note"]

METHODS = ["ek", "scaling", "pr-fifo", "pr-highest", "mcf", "networkx"]
LABELS = {"ek": "Edmonds–Karp", "scaling": "capacity scaling",
          "pr-fifo": "push–relabel FIFO", "pr-highest": "push–relabel HL",
          "mcf": "min-cost max-flow", "networkx": "networkx preflow_push"}
COLORS = {"ek": "#d62728", "scaling": "#ff7f0e", "pr-fifo": "#1f77b4",
          "pr-highest": "#17becf", "mcf": "#9467bd", "networkx": "#7f7f7f"}

SPARSE_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
DENSE_SIZES = [100, 200, 400, 800, 1_200]
DENSE_P = 0.2
GRID_SIDES = [32, 48, 64, 100, 141, 224]
# largest size each method runs at (per family); missing = no limit
LIMITS = {
    "sparse": {"ek": 20_000, "scaling": 20_000, "mcf": 5_000,
               "networkx": 20_000},
    "dense": {"mcf": 400, "networkx": 800},
    "grid": {"ek": 64 * 64, "scaling": 64 * 64, "mcf": 32 * 32,
             "networkx": 100 * 100},
}
INSTANCES = 3
COST_RANGE = (1, 20)
MEMORY_SIZES = [1_000, 5_000, 20_000]
ABLATION_CAP_S = 10.0


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class TimeLimit(Exception):
    pass


def _alarm(*_):
    raise TimeLimit


def run_capped(func: Callable[[], object], cap: Optional[float]):
    """(seconds, result) or (None, None) if func exceeded cap seconds."""
    if cap:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, cap)
    try:
        start = time.perf_counter()
        result = func()
        return time.perf_counter() - start, result
    except TimeLimit:
        return None, None
    finally:
        if cap:
            signal.setitimer(signal.ITIMER_REAL, 0)


def make_row(**kw) -> Dict:
    row = {f: "" for f in FIELDS}
    row.update(kw)
    return row


def to_networkx(net: FlowNetwork) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(net.n))
    for e in net.edges():
        if graph.has_edge(e.u, e.v):
            graph[e.u][e.v]["capacity"] += e.capacity
        else:
            graph.add_edge(e.u, e.v, capacity=e.capacity)
    return graph


def solve(method: str, net: FlowNetwork, s: int, t: int, **kw):
    if method == "networkx":
        graph = to_networkx(net)
        start = time.perf_counter()
        value = nx.maximum_flow_value(graph, s, t, flow_func=preflow_push)
        return time.perf_counter() - start, value, None
    solver = get_solver(method)
    start = time.perf_counter()
    result = solver(net, s, t, **kw)
    return time.perf_counter() - start, result.value, result


def check(net, s, t, result) -> bool:
    if result is None:
        return True
    return bool(validate_flow(net, s, t, value=result.value, max_flow=True,
                              min_cost=result.cost is not None,
                              potentials=result.stats.get("potentials")))


def build(family: str, size: int, seed: int):
    """(net, s, t) for one instance; `size` is n, or the side for grids."""
    if family == "sparse":
        return sparse_network(size, 4.0, seed=seed)
    if family == "dense":
        return dense_network(size, DENSE_P, seed=seed)
    return grid_network(size, size, seed=seed)


def with_costs(net: FlowNetwork, seed: int) -> FlowNetwork:
    """Copy of net with random per-unit costs in COST_RANGE (same arcs
    and capacities, so MCF solves the same max-flow instance)."""
    costed = net.copy()
    rng = np.random.default_rng(seed + 10_000)
    costed.cost = []
    for c in rng.integers(COST_RANGE[0], COST_RANGE[1] + 1,
                          size=net.num_edges).tolist():
        costed.cost += (c, -c)
    return costed


def _print_row(row: Dict) -> None:
    t = row["mean_time_s"]
    print(f"  {row['family']:7s} {row['method']:11s} n={row['n']:>6} "
          f"{'t=' + format(t, '.4f') + 's' if t != '' else row['note']:>14} "
          f"flow={row['flow_value']} valid={row['valid']}")


# ----------------------------------------------------------------------
# 1. runtime scaling
# ----------------------------------------------------------------------
def bench_scaling(family: str, sizes: Sequence[int], instances: int,
                  methods: Sequence[str] = METHODS,
                  limits: Optional[Dict[str, int]] = None,
                  samples: Optional[Dict] = None) -> List[Dict]:
    limits = LIMITS.get(family, {}) if limits is None else limits
    rows = []
    for size in sizes:
        nodes = size * size + 2 if family == "grid" else size
        per: Dict[str, Dict[str, list]] = {}
        for i in range(instances):
            net, s, t = build(family, size, SEED + i)
            cost_net = None
            values = set()
            for method in methods:
                if nodes > limits.get(method, float("inf")) + 2:
                    continue
                target = net
                if method == "mcf":
                    cost_net = cost_net or with_costs(net, SEED + i)
                    target = cost_net
                elapsed, value, result = solve(method, target, s, t)
                bucket = per.setdefault(method, {"t": [], "v": [], "ok": [],
                                                 "aug": [], "push": [],
                                                 "relabel": [], "arcs": []})
                bucket["t"].append(elapsed)
                bucket["v"].append(value)
                bucket["arcs"].append(target.num_edges)
                bucket["ok"].append(check(target, s, t, result))
                values.add(value)
                if result is not None:
                    st = result.stats
                    bucket["aug"].append(len(st.get("bottlenecks", [])))
                    bucket["push"].append(st.get("pushes", 0))
                    bucket["relabel"].append(st.get("relabels", 0))
                    if samples is not None and method in ("ek", "scaling",
                                                         "mcf") and i == 0:
                        samples.setdefault(("aug", family, method), {})[nodes] = st
                    if samples is not None and method.startswith("pr") and i == 0:
                        samples.setdefault(("pr", family, method), {})[nodes] = st
            if len(values) > 1:
                raise AssertionError(f"solvers disagree on {family} n={size} "
                                     f"instance {i}: {values}")
        for method, b in per.items():
            row = make_row(
                experiment="runtime", family=family, method=method, n=nodes,
                arcs=round(statistics.mean(b["arcs"])),
                parameter=f"p={DENSE_P}" if family == "dense" else
                ("avg_degree=4" if family == "sparse" else f"{size}x{size}"),
                instances=len(b["t"]), mean_time_s=statistics.mean(b["t"]),
                std_time_s=statistics.stdev(b["t"]) if len(b["t"]) > 1 else 0.0,
                flow_value=statistics.mean(b["v"]),
                augmentations=statistics.mean(b["aug"]) if b["aug"] else "",
                pushes=statistics.mean(b["push"]) if b["push"] else "",
                relabels=statistics.mean(b["relabel"]) if b["relabel"] else "",
                valid=all(b["ok"]))
            rows.append(row)
            _print_row(row)
    return rows


# ----------------------------------------------------------------------
# 2. memory
# ----------------------------------------------------------------------
def bench_memory(sizes: Sequence[int] = MEMORY_SIZES,
                 methods: Sequence[str] = ("ek", "scaling", "pr-fifo",
                                           "pr-highest", "mcf")) -> List[Dict]:
    rows = []
    for n in sizes:
        net, s, t = sparse_network(n, 4.0, cost_range=COST_RANGE, seed=SEED)
        base_peak, _ = peak_memory(
            lambda: sparse_network(n, 4.0, cost_range=COST_RANGE, seed=SEED))
        # (costs present so the same network also serves MCF)
        rows.append(make_row(experiment="memory", family="sparse",
                             method="network", n=n, arcs=net.num_edges,
                             peak_mem_mb=base_peak / 2 ** 20, valid=True,
                             note="building the FlowNetwork"))
        for method in methods:
            if n > LIMITS["sparse"].get(method, float("inf")):
                continue
            peak, result = peak_memory(lambda: get_solver(method)(net, s, t))
            rows.append(make_row(experiment="memory", family="sparse",
                                 method=method, n=n, arcs=net.num_edges,
                                 flow_value=result.value,
                                 peak_mem_mb=peak / 2 ** 20,
                                 valid=check(net, s, t, result),
                                 note="peak during solve (tracemalloc)"))
            print(f"  memory  {method:11s} n={n:>6} "
                  f"peak={peak / 2 ** 20:7.2f} MB")
    return rows


# ----------------------------------------------------------------------
# 3. augmentations across capacity distributions
# ----------------------------------------------------------------------
def bench_capacity_dists(n: int = 5_000, instances: int = INSTANCES,
                         dists: Sequence[str] = ("unit", "uniform",
                                                 "exponential", "powerlaw"),
                         high: int = 10_000) -> List[Dict]:
    rows = []
    for dist in dists:
        for method in ("ek", "scaling", "pr-fifo"):
            times, augs, ok = [], [], []
            for i in range(instances):
                net, s, t = sparse_network(n, 4.0, capacity=dist,
                                           cap_range=(1, high), seed=SEED + i)
                elapsed, _, result = solve(method, net, s, t)
                times.append(elapsed)
                augs.append(len(result.stats["bottlenecks"]) if method != "pr-fifo"
                            else result.stats["pushes"])
                ok.append(check(net, s, t, result))
            row = make_row(experiment="capacity_dist", family="sparse",
                           method=method, n=n, parameter=dist,
                           instances=instances,
                           mean_time_s=statistics.mean(times),
                           std_time_s=statistics.stdev(times) if instances > 1 else 0.0,
                           augmentations=statistics.mean(augs)
                           if method != "pr-fifo" else "",
                           pushes=statistics.mean(augs) if method == "pr-fifo" else "",
                           valid=all(ok), note=f"capacities 1..{high}")
            rows.append(row)
            print(f"  caps    {method:11s} {dist:11s} "
                  f"t={row['mean_time_s']:.3f}s work={statistics.mean(augs):.0f}")
    return rows


# ----------------------------------------------------------------------
# 4. push–relabel convergence and ablation
# ----------------------------------------------------------------------
def bench_convergence(cases: Sequence[Tuple[str, int]], samples: Dict,
                      profile_points: int = 400) -> List[Dict]:
    rows = []
    for family, size in cases:
        net, s, t = build(family, size, SEED)
        for variant in ("fifo", "highest"):
            probe = push_relabel(net, s, t, variant=variant)
            ops = probe.stats["pushes"] + probe.stats["relabels"]
            every = max(1, ops // profile_points)
            result = push_relabel(net, s, t, variant=variant,
                                  profile_every=every)
            nodes = net.n
            samples[("profile", family, variant)] = (nodes, result.value,
                                                     result.stats["profile"])
            rows.append(make_row(
                experiment="convergence", family=family,
                method=f"pr-{variant}", n=nodes, arcs=net.num_edges,
                flow_value=result.value, pushes=result.stats["pushes"],
                relabels=result.stats["relabels"],
                valid=check(net, s, t, result),
                note=f"{result.stats['global_relabels']} global relabels, "
                     f"{result.stats['gap_nodes']} gap-lifted nodes"))
    return rows


ABLATIONS = [("both heuristics", dict()),
             ("no gap", dict(gap=False)),
             ("no global relabel", dict(global_relabel=False)),
             ("neither", dict(gap=False, global_relabel=False))]


def bench_ablation(cases: Sequence[Tuple[str, int]], instances: int,
                   cap: float = ABLATION_CAP_S) -> List[Dict]:
    rows = []
    for family, size in cases:
        for label, kw in ABLATIONS:
            times, timeouts, ops = [], 0, []
            for i in range(instances):
                net, s, t = build(family, size, SEED + i)
                elapsed, result = run_capped(
                    lambda: push_relabel(net, s, t, **kw), cap)
                if elapsed is None:
                    timeouts += 1
                    times.append(cap)
                else:
                    times.append(elapsed)
                    ops.append(result.stats["pushes"] + result.stats["relabels"])
            nodes = size * size + 2 if family == "grid" else size
            rows.append(make_row(
                experiment="ablation", family=family, method="pr-fifo",
                n=nodes, parameter=label, instances=instances,
                mean_time_s=statistics.mean(times),
                std_time_s=statistics.stdev(times) if instances > 1 else 0.0,
                pushes=statistics.mean(ops) if ops else "", valid=True,
                note=f"{timeouts} of {instances} runs hit the {cap:g} s cap"
                if timeouts else ""))
            print(f"  ablate  {family:6s} {label:18s} "
                  f"t={statistics.mean(times):.3f}s timeouts={timeouts}")
    return rows


# ----------------------------------------------------------------------
# 5. applications
# ----------------------------------------------------------------------
MATCH_EK_LIMIT = 5_000


def bench_matching(sizes: Sequence[int], instances: int,
                   ek_limit: int = MATCH_EK_LIMIT) -> List[Dict]:
    rows = []
    for n in sizes:
        per = {m: [] for m in ("ek", "pr-fifo", "hopcroft-karp")}
        sizes_found = []
        for i in range(instances):
            edges = bipartite_graph(n, n, 3.0, seed=SEED + i)
            found = set()
            for method in ("ek", "pr-fifo"):
                if method == "ek" and n > ek_limit:
                    continue
                start = time.perf_counter()
                result = bipartite_matching(range(n), range(n), edges, method)
                per[method].append(time.perf_counter() - start)
                found.add(result.size)
            graph = nx.Graph()
            left = [("l", a) for a in range(n)]
            graph.add_nodes_from(left)
            graph.add_nodes_from(("r", b) for b in range(n))
            graph.add_edges_from((("l", a), ("r", b)) for a, b in edges)
            start = time.perf_counter()
            hk = nx.bipartite.hopcroft_karp_matching(graph, left)
            per["hopcroft-karp"].append(time.perf_counter() - start)
            found.add(len(hk) // 2)
            if len(found) != 1:
                raise AssertionError(f"matching sizes disagree: {found}")
            sizes_found.append(found.pop())
        for method, times in per.items():
            if not times:
                continue
            rows.append(make_row(
                experiment="matching", family="bipartite", method=method,
                n=2 * n, parameter="avg_degree=3", instances=instances,
                mean_time_s=statistics.mean(times),
                std_time_s=statistics.stdev(times) if instances > 1 else 0.0,
                flow_value=statistics.mean(sizes_found), valid=True,
                note="networkx reference" if method == "hopcroft-karp" else ""))
        print(f"  match   n={n:>6} per side, size={statistics.mean(sizes_found):.0f}, "
              f"pr={statistics.mean(per['pr-fifo']):.3f}s "
              f"ek={statistics.mean(per['ek']) if per['ek'] else float('nan'):.3f}s "
              f"hk={statistics.mean(per['hopcroft-karp']):.3f}s")
    return rows


def bench_segmentation(sides: Sequence[int], noises: Sequence[float],
                       instances: int, samples: Dict) -> List[Dict]:
    rows = []
    for side in sides:
        times, ious = [], []
        for i in range(instances):
            image, truth = synthetic_image(side, side, "shapes", noise=0.3,
                                           seed=SEED + i)
            start = time.perf_counter()
            result = segment_image(image)
            times.append(time.perf_counter() - start)
            ious.append(segmentation_scores(result.mask, truth)["iou"])
        rows.append(make_row(
            experiment="segmentation_size", family="image", method="graph-cut",
            n=side * side + 2, parameter="noise=0.3", instances=instances,
            mean_time_s=statistics.mean(times),
            std_time_s=statistics.stdev(times) if instances > 1 else 0.0,
            flow_value=statistics.mean(ious), valid=True,
            note="flow_value column holds mean IoU"))
        print(f"  segment {side}x{side}: t={statistics.mean(times):.3f}s "
              f"IoU={statistics.mean(ious):.3f}")
    for noise in noises:
        scores = {"graph-cut": [], "threshold": []}
        for i in range(instances):
            image, truth = synthetic_image(64, 64, "shapes", noise=noise,
                                           seed=SEED + i)
            scores["graph-cut"].append(
                segmentation_scores(segment_image(image).mask, truth)["iou"])
            scores["threshold"].append(
                segmentation_scores(threshold_segment(image), truth)["iou"])
        for method, vals in scores.items():
            rows.append(make_row(
                experiment="segmentation_noise", family="image", method=method,
                n=64 * 64 + 2, parameter=f"noise={noise}", instances=instances,
                flow_value=statistics.mean(vals), valid=True,
                note="flow_value column holds mean IoU"))
        print(f"  segment noise={noise}: cut IoU="
              f"{statistics.mean(scores['graph-cut']):.3f} threshold IoU="
              f"{statistics.mean(scores['threshold']):.3f}")
    return rows


def bench_scheduling(num_jobs: int, horizon: int, machine_counts: Sequence[int],
                     instances: int) -> List[Dict]:
    rows = []
    for m in machine_counts:
        feasible, fraction, times = 0, [], []
        for i in range(instances):
            jobs = random_jobs(num_jobs, horizon, seed=SEED + i)
            start = time.perf_counter()
            result = schedule_jobs(jobs, m)
            times.append(time.perf_counter() - start)
            feasible += result.feasible
            fraction.append(result.scheduled / result.required)
        rows.append(make_row(
            experiment="scheduling", family="deadline", method="pr-fifo",
            n=num_jobs, parameter=f"machines={m}", instances=instances,
            mean_time_s=statistics.mean(times),
            flow_value=feasible / instances,
            augmentations=statistics.mean(fraction), valid=True,
            note="flow_value = fraction feasible; augmentations = mean "
                 "fraction of work scheduled"))
        print(f"  sched   machines={m}: feasible {feasible}/{instances}, "
              f"work scheduled {statistics.mean(fraction):.3f}")
    return rows


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
def _series(rows, experiment, family, value="mean_time_s", key="n",
            methods=METHODS):
    out = {}
    for method in methods:
        pts = sorted((r[key], r[value]) for r in rows
                     if r["experiment"] == experiment and r["family"] == family
                     and r["method"] == method and r[value] != "")
        if pts:
            out[LABELS.get(method, method)] = ([p[0] for p in pts],
                                               [p[1] for p in pts])
    return out


def _colors():
    return {LABELS[m]: c for m, c in COLORS.items()}


def plot_runtime(rows: List[Dict], results_dir: str) -> None:
    panels = []
    for family, title in (("sparse", "Sparse random networks (avg degree 4)"),
                          ("dense", f"Dense random networks (p = {DENSE_P})"),
                          ("grid", "Segmentation-shaped grids")):
        series = _series(rows, "runtime", family)
        if series:
            panels.append({"title": title, "xlabel": "nodes",
                           "ylabel": "time (s)", "series": series,
                           "colors": _colors(), "logx": True, "logy": True})
    speed = {}
    for family in ("sparse", "dense", "grid"):
        ek = {r["n"]: r["mean_time_s"] for r in rows if r["experiment"] ==
              "runtime" and r["family"] == family and r["method"] == "ek"}
        pr = {r["n"]: r["mean_time_s"] for r in rows if r["experiment"] ==
              "runtime" and r["family"] == family and r["method"] == "pr-fifo"}
        xs = sorted(set(ek) & set(pr))
        if xs:
            speed[family] = (xs, [ek[x] / pr[x] for x in xs])
    if speed:
        panels.append({"title": "Speed-up of push–relabel FIFO over EK",
                       "xlabel": "nodes", "ylabel": "EK time / PR time",
                       "series": speed, "logx": True, "logy": True,
                       "refs": {"equal": ([min(min(v[0]) for v in speed.values()),
                                           max(max(v[0]) for v in speed.values())],
                                          [1, 1])}})
    plot_mixed_panels(panels, os.path.join(results_dir, "flow_runtime.png"),
                      "Max-flow runtime scaling", ncols=2)


def plot_memory(rows: List[Dict], results_dir: str) -> None:
    mem = [r for r in rows if r["experiment"] == "memory"]
    if not mem:
        return
    methods = ["network", "ek", "scaling", "pr-fifo", "pr-highest", "mcf"]
    labels = dict(LABELS, network="network itself")
    colors = dict(_colors(), **{"network itself": "#2ca02c"})
    series, per_arc = {}, {}
    for m in methods:
        pts = sorted((r["n"], r["peak_mem_mb"], r["arcs"]) for r in mem
                     if r["method"] == m)
        if pts:
            series[labels[m]] = ([p[0] for p in pts], [p[1] for p in pts])
            per_arc[labels[m]] = ([p[0] for p in pts],
                                  [p[1] * 2 ** 20 / p[2] for p in pts])
    plot_mixed_panels([
        {"title": "Peak memory allocated while solving", "xlabel": "nodes",
         "ylabel": "MB (tracemalloc peak)", "series": series,
         "colors": colors, "logx": True, "logy": True},
        {"title": "Memory per arc", "xlabel": "nodes", "ylabel": "bytes / arc",
         "series": per_arc, "colors": colors, "logx": True},
    ], os.path.join(results_dir, "flow_memory.png"),
        "Memory use (sparse networks, avg degree 4)", ncols=2)


def plot_augmentations(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for method in ("ek", "scaling", "mcf"):
        pts = sorted((r["n"], r["augmentations"]) for r in rows
                     if r["experiment"] == "runtime" and r["family"] == "sparse"
                     and r["method"] == method and r["augmentations"] != "")
        if pts:
            ax.plot(*zip(*pts), "o-", color=COLORS[method], label=LABELS[method])
    pts = sorted((r["n"], r["pushes"]) for r in rows if r["experiment"] ==
                 "runtime" and r["family"] == "sparse" and r["method"] ==
                 "pr-fifo" and r["pushes"] != "")
    if pts:
        ax.plot(*zip(*pts), "s--", color=COLORS["pr-fifo"],
                label="push–relabel pushes (for scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Augmenting paths vs n (sparse)")
    ax.set_xlabel("nodes")
    ax.set_ylabel("augmentations")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ek = samples.get(("aug", "sparse", "ek"), {})
    if ek:
        n = max(ek)
        lengths = ek[n]["path_lengths"]
        ax.plot(range(1, len(lengths) + 1), lengths, color=COLORS["ek"], lw=1.2)
        ax.set_title(f"EK path length per augmentation (n = {n:,})")
        ax.text(0.02, 0.95, "BFS distance d(s, t) never decreases",
                transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("augmentation")
    ax.set_ylabel("arcs on augmenting path")

    ax = axes[1, 0]
    for method in ("ek", "scaling"):
        st = samples.get(("aug", "sparse", method), {})
        if st:
            n = max(st)
            ax.hist(st[n]["bottlenecks"], bins=30, alpha=0.55,
                    color=COLORS[method],
                    label=f"{LABELS[method]} ({len(st[n]['bottlenecks'])} paths)")
    ax.set_yscale("log")
    ax.set_title("Bottleneck capacities (largest sparse instance run)")
    ax.set_xlabel("bottleneck")
    ax.set_ylabel("paths")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    caps = [r for r in rows if r["experiment"] == "capacity_dist"]
    dists = list(dict.fromkeys(r["parameter"] for r in caps))
    width = 0.38
    for k, method in enumerate(("ek", "scaling")):
        vals = [next((r["augmentations"] for r in caps if r["parameter"] == d
                      and r["method"] == method), 0) for d in dists]
        ax.bar(np.arange(len(dists)) + (k - 0.5) * width, vals, width,
               color=COLORS[method], label=LABELS[method])
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(dists)
    n_caps = caps[0]["n"] if caps else ""
    ax.set_title(f"Augmentations by capacity distribution (n = {n_caps:,})"
                 if caps else "Augmentations by capacity distribution")
    ax.set_ylabel("augmentations")
    ax.legend(fontsize=8)
    for a in axes.flat:
        a.grid(True, which="both", alpha=0.3)
    fig.suptitle("Augmenting paths: count, length and bottlenecks",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(results_dir, "augmentations.png"), dpi=150)
    plt.close(fig)


def plot_convergence(rows: List[Dict], samples: Dict, results_dir: str) -> None:
    """Columns 0-1: profile per family (flow at t, active nodes);
    column 2: heuristic ablation and operation counts vs n."""
    families = list(dict.fromkeys(k[1] for k in samples if k[0] == "profile"))[:2]
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9.5))
    for col, family in enumerate(families):
        nodes = 0
        for variant in ("fifo", "highest"):
            key = ("profile", family, variant)
            if key not in samples:
                continue
            nodes, value, prof = samples[key]
            ops = [p[0] for p in prof]
            color, label = COLORS[f"pr-{variant}"], LABELS[f"pr-{variant}"]
            axes[0, col].plot(ops, [p[1] / value if value else 1 for p in prof],
                              color=color, label=label)
            axes[1, col].plot(ops, [p[2] for p in prof], color=color,
                              label=label)
        axes[0, col].set_title(f"Flow reaching t: {family}, {nodes:,} nodes")
        axes[0, col].set_ylabel("fraction of max flow at t")
        axes[1, col].set_title(f"Active nodes: {family}")
        axes[1, col].set_ylabel("active nodes")
        axes[1, col].set_yscale("symlog")
        for row in (0, 1):
            axes[row, col].set_xlabel("operations (pushes + relabels)")

    ax = axes[0, 2]
    abl = [r for r in rows if r["experiment"] == "ablation"]
    cases = list(dict.fromkeys((r["family"], r["n"]) for r in abl))
    width = 0.8 / len(ABLATIONS)
    for k, (label, _) in enumerate(ABLATIONS):
        picked = [next(r for r in abl if (r["family"], r["n"]) == c
                       and r["parameter"] == label) for c in cases]
        bars = ax.bar(np.arange(len(cases)) + (k - (len(ABLATIONS) - 1) / 2)
                      * width, [r["mean_time_s"] for r in picked], width,
                      label=label)
        for b, r in zip(bars, picked):
            if r["note"]:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        r["note"].split(" runs")[0], ha="center",
                        va="bottom", fontsize=7)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"{f}\n{n:,} nodes" for f, n in cases])
    ax.set_yscale("log")
    ax.set_ylabel("mean time (s)")
    ax.set_title("Heuristic ablation, FIFO (labels: runs hitting the cap)")

    ax = axes[1, 2]
    for variant, marker in (("fifo", "o"), ("highest", "s")):
        for field, style in (("pushes", "-"), ("relabels", "--")):
            pts = sorted((r["n"], r[field]) for r in rows
                         if r["experiment"] == "runtime"
                         and r["family"] == "sparse"
                         and r["method"] == f"pr-{variant}" and r[field] != "")
            if pts:
                ax.plot(*zip(*pts), marker=marker, linestyle=style,
                        color=COLORS[f"pr-{variant}"],
                        label=f"{variant}: {field}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Push–relabel operations vs n (sparse)")
    ax.set_xlabel("nodes")
    ax.set_ylabel("operations")
    for a in axes.flat:
        a.grid(True, which="both", alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.suptitle("Push–relabel convergence and heuristics", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(results_dir, "pr_convergence.png"), dpi=150)
    plt.close(fig)


def plot_applications(rows: List[Dict], results_dir: str) -> None:
    panels = []
    match = _series(rows, "matching", "bipartite",
                    methods=["ek", "pr-fifo", "hopcroft-karp"])
    if match:
        panels.append({"title": "Bipartite matching (avg degree 3)",
                       "xlabel": "vertices (both sides)", "ylabel": "time (s)",
                       "series": match, "logx": True, "logy": True,
                       "colors": dict(_colors(), **{"hopcroft-karp": "#7f7f7f"})})
    noise = {}
    for method in ("graph-cut", "threshold"):
        pts = sorted((float(r["parameter"].split("=")[1]), r["flow_value"])
                     for r in rows if r["experiment"] == "segmentation_noise"
                     and r["method"] == method)
        if pts:
            noise[method] = ([p[0] for p in pts], [p[1] for p in pts])
    if noise:
        panels.append({"title": "Segmentation quality vs noise (64×64)",
                       "xlabel": "noise std", "ylabel": "foreground IoU",
                       "series": noise})
    seg = _series(rows, "segmentation_size", "image", methods=["graph-cut"])
    if seg:
        panels.append({"title": "Segmentation runtime vs image size",
                       "xlabel": "nodes (pixels + 2)", "ylabel": "time (s)",
                       "series": seg, "logx": True, "logy": True})
    sched = sorted((int(r["parameter"].split("=")[1]), r["flow_value"],
                    r["augmentations"]) for r in rows
                   if r["experiment"] == "scheduling")
    if sched:
        xs = [p[0] for p in sched]
        panels.append({"title": "Deadline scheduling feasibility",
                       "xlabel": "machines", "ylabel": "fraction",
                       "series": {"instances feasible": (xs, [p[1] for p in sched]),
                                  "work scheduled": (xs, [p[2] for p in sched])}})
    plot_mixed_panels(panels, os.path.join(results_dir, "flow_applications.png"),
                      "Flow applications", ncols=2)


CLRS_EDGES = [(0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (3, 2, 9),
              (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]


def plot_examples(results_dir: str) -> None:
    net = FlowNetwork.from_edges(6, CLRS_EDGES)
    push_relabel(net, 0, 5)
    plot_network(net, 0, 5, save_path=os.path.join(results_dir,
                                                   "network_example.png"),
                 title="CLRS network: max flow 23 = min-cut capacity 23")
    edges = bipartite_graph(10, 9, 2.0, seed=SEED)
    match = bipartite_matching(range(10), range(9), edges)
    plot_matching([f"L{i}" for i in range(10)], [f"R{j}" for j in range(9)],
                  [(f"L{a}", f"R{b}") for a, b in edges],
                  [(f"L{a}", f"R{b}") for a, b in match.pairs],
                  save_path=os.path.join(results_dir, "matching_example.png"))
    image, truth = synthetic_image(96, 96, "shapes", noise=0.35, seed=SEED)
    cut = segment_image(image)
    thr = threshold_segment(image)
    sc, st = segmentation_scores(cut.mask, truth), segmentation_scores(thr, truth)
    plot_segmentation(image, {f"threshold (IoU {st['iou']:.2f})": thr,
                              f"graph cut (IoU {sc['iou']:.2f})": cut.mask},
                      truth, title="Graph-cut segmentation, noise σ = 0.35",
                      save_path=os.path.join(results_dir,
                                             "segmentation_example.png"))
    # storyboard: EK frames on top, push–relabel frames below
    net = FlowNetwork.from_edges(6, CLRS_EDGES)
    pos = layered_positions(net, 0, 5)
    ek = _subsample(edmonds_karp_frames(net, 0, 5), 4)
    pr = _subsample(push_relabel_frames(net, 0, 5), 4)
    top = max(max(f["heights"]) for f in pr)
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))
    for ax, frame in zip(axes[0], ek):
        _draw_ek(ax, net, pos, frame, 0, 5)
        ax.title.set_fontsize(9)
    for ax, frame in zip(axes[1], pr):
        _draw_pr(ax, net, pos, frame, 0, 5, top)
        ax.title.set_fontsize(9)
    fig.text(0.005, 0.74, "Edmonds–Karp", rotation=90, fontsize=13,
             fontweight="bold", va="center")
    fig.text(0.005, 0.27, "Push–relabel", rotation=90, fontsize=13,
             fontweight="bold", va="center")
    fig.suptitle("Augmenting paths vs local pushes on the same network",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.015, 0, 1, 0.95))
    fig.savefig(os.path.join(results_dir, "flow_storyboard.png"), dpi=130)
    plt.close(fig)
    animate_edmonds_karp(FlowNetwork.from_edges(6, CLRS_EDGES), 0, 5,
                         save_path=os.path.join(results_dir, "ek_animation.gif"))
    animate_push_relabel(FlowNetwork.from_edges(6, CLRS_EDGES), 0, 5,
                         save_path=os.path.join(results_dir, "pr_animation.gif"))


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def print_summary(rows: List[Dict]) -> None:
    print("\nLargest size each method reached (runtime experiment):")
    for family in ("sparse", "dense", "grid"):
        fam = [r for r in rows if r["experiment"] == "runtime"
               and r["family"] == family]
        for method in METHODS:
            own = [r for r in fam if r["method"] == method]
            if own:
                top = max(own, key=lambda r: r["n"])
                print(f"  {family:6s} {LABELS[method]:24s} n={top['n']:>6,}  "
                      f"{top['mean_time_s']:.3f} s")
    invalid = [r for r in rows if r["valid"] is False]
    print(f"\nValidation: {len(invalid)} configuration(s) failed"
          if invalid else "\nValidation: every flow passed every check")


def run(results_dir: str = RESULTS_DIR, quick: bool = False) -> List[Dict]:
    os.makedirs(results_dir, exist_ok=True)
    samples: Dict = {}
    inst = 1 if quick else INSTANCES
    rows: List[Dict] = []
    print("1. Runtime scaling")
    rows += bench_scaling("sparse", [n for n in SPARSE_SIZES
                                     if not quick or n <= 5_000], inst,
                          samples=samples)
    rows += bench_scaling("dense", [n for n in DENSE_SIZES
                                    if not quick or n <= 400], inst,
                          samples=samples)
    rows += bench_scaling("grid", [g for g in GRID_SIDES
                                   if not quick or g <= 64], inst,
                          samples=samples)
    print("2. Memory")
    rows += bench_memory([n for n in MEMORY_SIZES if not quick or n <= 5_000])
    print("3. Capacity distributions")
    rows += bench_capacity_dists(n=2_000 if quick else 5_000, instances=inst)
    print("4. Push-relabel convergence and heuristics")
    rows += bench_convergence([("sparse", 5_000 if quick else 50_000),
                               ("grid", 64 if quick else 141)], samples)
    rows += bench_ablation([("sparse", 5_000), ("grid", 48 if quick else 64)],
                           inst, cap=2.0 if quick else ABLATION_CAP_S)
    print("5. Applications")
    rows += bench_matching([100, 1_000] if quick else [100, 1_000, 5_000, 20_000],
                           inst)
    rows += bench_segmentation([32, 64] if quick else [32, 64, 128, 200],
                               [0.1, 0.3, 0.5] if quick else
                               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], inst, samples)
    rows += bench_scheduling(30, 20, [1, 2, 3, 4, 5, 6, 8],
                             5 if quick else 20)

    csv_path = os.path.join(results_dir, CSV_NAME)
    write_csv(rows, csv_path)
    print(f"\nSaved {csv_path} ({len(rows)} rows)")
    print_summary(rows)
    plot_runtime(rows, results_dir)
    plot_memory(rows, results_dir)
    plot_augmentations(rows, samples, results_dir)
    plot_convergence(rows, samples, results_dir)
    plot_applications(rows, results_dir)
    plot_examples(results_dir)
    print(f"Saved plots and animations to {results_dir}")
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
