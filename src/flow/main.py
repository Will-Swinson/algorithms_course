"""
Command-line interface to the flow network solver.

    python -m src.flow.main [solver] [application] [input] [output]

Solver (default --pr):
    --ek                 Edmonds–Karp
    --pr                 push–relabel (--variant fifo|highest)
    --mcf                min-cost max-flow (successive shortest paths)
    --scaling            capacity-scaling augmenting paths

Application (default: plain max-flow on a graph):
    --match FILE         bipartite matching; FILE has "left right" lines
    --segment [IMAGE]    graph-cut segmentation of a grayscale image, or
                         of a synthetic test image if IMAGE is omitted
    --schedule FILE      JSON scheduling problem (see below)

Graph input (plain max-flow / min-cost flow):
    --input FILE         DIMACS .max/.min, or an edge list
                         "u v capacity [cost]" with --source/--sink
    --random N           random sparse network with N nodes instead
                         (--degree, --density for a dense one, --seed)

Output:
    results are printed; --plot PATH saves a picture (network, matching,
    segmentation or Gantt chart), --animate PATH a GIF of the algorithm
    (small graphs, --ek or --pr), --validate runs the flow validators.

Schedule JSON — "type" selects the model:
    {"type": "jobs", "machines": 2,
     "jobs": {"J1": [processing, release, deadline], ...}}
    {"type": "machines", "jobs": {"J1": hours, ...},
     "machines": {"M1": hours, ...}, "eligible": {"J1": ["M1"], ...},
     "costs": {"J1,M1": 3, ...}}                       (costs optional)
    {"type": "staffing", "projects": {"P": need, ...},
     "employees": {"ann": max_projects, ...},
     "qualified": {"ann": ["P", ...], ...},
     "preferences": {"P,ann": cost, ...}}              (optional)
    {"type": "tournament", "wins": {"A": 80, ...},
     "games": [["A", "B", games_left], ...]}
"""
import argparse
import os
import sys
import time
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from src.flow.applications import (
    assign_jobs_to_machines, bipartite_matching, schedule_jobs,
    segment_image, segmentation_scores, staff_projects, threshold_segment,
    tournament_elimination,
)
from src.flow.core import get_solver
from src.flow.utils import io
from src.flow.utils.generators import dense_network, sparse_network, synthetic_image
from src.flow.utils.validators import validate_flow

ANIMATE_LIMIT = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.flow.main",
        description="Flow network solver: max-flow, min-cost flow, matching, "
                    "segmentation and scheduling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Schedule JSON" + __doc__.split("Schedule JSON", 1)[1])
    solver = parser.add_mutually_exclusive_group()
    solver.add_argument("--ek", dest="solver", action="store_const", const="ek",
                        help="Edmonds-Karp")
    solver.add_argument("--pr", dest="solver", action="store_const", const="pr",
                        help="push-relabel (default)")
    solver.add_argument("--mcf", dest="solver", action="store_const", const="mcf",
                        help="min-cost max-flow")
    solver.add_argument("--scaling", dest="solver", action="store_const",
                        const="scaling", help="capacity scaling")
    parser.add_argument("--variant", choices=["fifo", "highest"], default="fifo",
                        help="push-relabel active-node rule")

    app = parser.add_mutually_exclusive_group()
    app.add_argument("--match", metavar="FILE", help="bipartite matching input")
    app.add_argument("--segment", metavar="IMAGE", nargs="?", const="",
                     help="segment IMAGE (synthetic test image if omitted)")
    app.add_argument("--schedule", metavar="FILE", help="scheduling JSON")

    graph = parser.add_argument_group("graph input")
    graph.add_argument("--input", "-i", metavar="FILE",
                       help="DIMACS or edge-list graph file")
    graph.add_argument("--source", "-s", type=int, help="source node (edge list)")
    graph.add_argument("--sink", "-t", type=int, help="sink node (edge list)")
    graph.add_argument("--random", type=int, metavar="N",
                       help="generate a random network with N nodes")
    graph.add_argument("--degree", type=float, default=4.0,
                       help="average out-degree of --random (sparse)")
    graph.add_argument("--density", type=float,
                       help="arc probability: makes --random dense")
    graph.add_argument("--max-cost", type=int, default=20,
                       help="random costs in [1, MAX] for --random with --mcf")
    graph.add_argument("--seed", type=int, default=0)

    seg = parser.add_argument_group("segmentation")
    seg.add_argument("--size", type=int, default=64,
                     help="synthetic image side length")
    seg.add_argument("--noise", type=float, default=0.3,
                     help="synthetic image noise level")
    seg.add_argument("--smoothness", type=float, default=4.0)

    out = parser.add_argument_group("output")
    out.add_argument("--plot", metavar="PATH", help="save a visualization")
    out.add_argument("--animate", metavar="PATH", help="save an algorithm GIF")
    out.add_argument("--validate", action="store_true",
                     help="check conservation, capacity, residual and optimality")
    out.add_argument("--show-flow", action="store_true",
                     help="print every arc carrying flow")
    return parser


def solver_kwargs(args) -> dict:
    return {"variant": args.variant} if args.solver == "pr" else {}


def solver_label(args) -> str:
    names = {"ek": "Edmonds-Karp", "mcf": "min-cost max-flow",
             "scaling": "capacity scaling",
             "pr": f"push-relabel ({args.variant})"}
    return names[args.solver]


# ----------------------------------------------------------------------
# plain max-flow / min-cost flow
# ----------------------------------------------------------------------
def run_graph(args) -> int:
    if args.input:
        net, s, t = io.read_graph(args.input, args.source, args.sink)
        origin = args.input
    elif args.random:
        costs = (1, args.max_cost) if args.solver == "mcf" else None
        if args.density:
            net, s, t = dense_network(args.random, args.density,
                                      cost_range=costs, seed=args.seed)
        else:
            net, s, t = sparse_network(args.random, args.degree,
                                       cost_range=costs, seed=args.seed)
        origin = f"random network (seed {args.seed})"
    else:
        print("error: give --input FILE or --random N (or an application)",
              file=sys.stderr)
        return 2
    print(f"{origin}: {net.n} nodes, {net.num_edges} arcs, s = {s}, t = {t}")
    start = time.perf_counter()
    result = get_solver(args.solver)(net, s, t, **solver_kwargs(args))
    elapsed = time.perf_counter() - start
    print(f"solver: {solver_label(args)}  ({elapsed * 1000:.1f} ms)")
    print(f"max flow value: {result.value:g}")
    if result.cost is not None:
        print(f"min cost: {result.cost:g}")
    side = result.min_cut
    print(f"min cut: |S| = {len(side)}, capacity {net.cut_capacity(side):g}, "
          f"{len(net.cut_edges(side))} arcs")
    for key, value in result.stats.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {key}: {value:g}")
    print(f"  iterations: {result.iterations}")
    if args.show_flow:
        for (u, v), f in sorted(result.flow.items()):
            print(f"  {u} -> {v}: {f:g}")
    if args.validate:
        report = validate_flow(net, s, t, value=result.value, max_flow=True,
                               min_cost=result.cost is not None,
                               potentials=result.stats.get("potentials"))
        print(f"validation: {report}")
    if args.plot:
        from src.flow.visualization import plot_network
        if net.n > 60:
            print("note: graph too large to draw legibly; skipping --plot")
        else:
            plot_network(net, s, t, save_path=args.plot)
            print(f"saved {args.plot}")
    if args.animate:
        animate(args, net, s, t)
    return 0 if not args.validate or report else 1


def animate(args, net, s, t) -> None:
    from src.flow.visualization import animate_edmonds_karp, animate_push_relabel
    if net.n > ANIMATE_LIMIT:
        print(f"note: --animate supports up to {ANIMATE_LIMIT} nodes; skipping")
        return
    if args.solver == "pr":
        animate_push_relabel(net, s, t, args.variant, save_path=args.animate)
    else:
        animate_edmonds_karp(net, s, t, save_path=args.animate)
    print(f"saved {args.animate}")


# ----------------------------------------------------------------------
# applications
# ----------------------------------------------------------------------
def run_matching(args) -> int:
    edges = io.read_pairs(args.match)
    left = list(dict.fromkeys(a for a, _ in edges))
    right = list(dict.fromkeys(b for _, b in edges))
    solver = "pr" if args.solver == "mcf" else args.solver
    start = time.perf_counter()
    result = bipartite_matching(left, right, edges, solver)
    elapsed = time.perf_counter() - start
    print(f"{len(left)} left, {len(right)} right, {len(edges)} edges")
    print(f"maximum matching: {result.size} pairs  ({elapsed * 1000:.1f} ms)")
    for a, b in sorted(result.pairs):
        print(f"  {a} - {b}")
    unmatched = [a for a in left if a not in result.as_dict()]
    if unmatched:
        print(f"unmatched left: {', '.join(unmatched)}")
    if args.plot:
        from src.flow.visualization import plot_matching
        plot_matching(left, right, edges, result.pairs, save_path=args.plot)
        print(f"saved {args.plot}")
    return 0


def run_segmentation(args) -> int:
    truth = None
    if args.segment:
        image = io.read_image(args.segment)
        origin = args.segment
    else:
        image, truth = synthetic_image(args.size, args.size, "shapes",
                                       noise=args.noise, seed=args.seed)
        origin = f"synthetic {args.size}x{args.size} image (noise {args.noise})"
    solver = "pr" if args.solver == "mcf" else args.solver
    start = time.perf_counter()
    result = segment_image(image, smoothness=args.smoothness, solver=solver)
    elapsed = time.perf_counter() - start
    print(f"{origin}: {result.num_nodes} nodes, {result.num_edges} arcs")
    print(f"solver: {solver}  ({elapsed * 1000:.1f} ms)")
    print(f"intensity models: foreground {result.fg_mean:.3f}, "
          f"background {result.bg_mean:.3f}")
    print(f"foreground pixels: {int(result.mask.sum())} / {result.mask.size}")
    print(f"min-cut energy: {result.energy:.3f}")
    baseline = threshold_segment(image)
    if truth is not None:
        cut, thr = segmentation_scores(result.mask, truth), \
            segmentation_scores(baseline, truth)
        print(f"graph cut: accuracy {cut['accuracy']:.3f}, IoU {cut['iou']:.3f}")
        print(f"threshold: accuracy {thr['accuracy']:.3f}, IoU {thr['iou']:.3f}")
    if args.plot:
        from src.flow.visualization import plot_segmentation
        plot_segmentation(image, {"threshold": baseline,
                                  "graph cut": result.mask}, truth,
                          save_path=args.plot)
        print(f"saved {args.plot}")
    return 0


def _pair_keys(mapping) -> dict:
    return {tuple(k.split(",")): v for k, v in (mapping or {}).items()}


def run_schedule(args) -> int:
    spec = io.read_json(args.schedule)
    kind = spec.get("type")
    solver = "pr" if args.solver == "mcf" else args.solver
    if kind == "jobs":
        jobs = {name: tuple(v) for name, v in spec["jobs"].items()}
        result = schedule_jobs(jobs, spec["machines"], solver)
        print(f"{len(jobs)} jobs on {spec['machines']} machines: "
              f"{'FEASIBLE' if result.feasible else 'INFEASIBLE'} "
              f"({result.scheduled}/{result.required} units scheduled)")
        for m, row in result.machines.items():
            print(f"  M{m}: " + " ".join(f"{k}:{job}" for k, job in row))
        for job, missing in result.late.items():
            print(f"  {job} misses its deadline by {missing} unit(s)")
        if args.plot:
            from src.flow.visualization import plot_schedule
            plot_schedule(result, save_path=args.plot)
            print(f"saved {args.plot}")
        return 0
    if kind in ("machines", "staffing"):
        if kind == "machines":
            result = assign_jobs_to_machines(spec["jobs"], spec["machines"],
                                             spec["eligible"],
                                             _pair_keys(spec.get("costs")) or None,
                                             solver)
        else:
            result = staff_projects(spec["projects"], spec["employees"],
                                    spec["qualified"],
                                    _pair_keys(spec.get("preferences")) or None,
                                    solver)
        print(f"{kind}: assigned {result.assigned:g} of {result.required:g} "
              f"({'FEASIBLE' if result.feasible else 'INFEASIBLE'})")
        if result.cost is not None:
            print(f"total cost: {result.cost:g}")
        for task, alloc in result.assignment.items():
            parts = ", ".join(f"{r} x{u:g}" if kind == "machines" else str(r)
                              for r, u in alloc.items())
            print(f"  {task}: {parts or '-'}")
        for task, short in result.unmet.items():
            print(f"  {task} is short by {short:g}")
        return 0
    if kind == "tournament":
        games = {(a, b): g for a, b, g in spec["games"]}
        for team, res in tournament_elimination(spec["wins"], games, solver).items():
            status = "ELIMINATED" if res.eliminated else "alive"
            print(f"  {team:12s} {status:10s} best {res.best_possible}: "
                  f"{res.reason}")
        return 0
    print(f"error: unknown schedule type {kind!r}", file=sys.stderr)
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.solver = args.solver or "pr"
    try:
        if args.match:
            return run_matching(args)
        if args.segment is not None:
            return run_segmentation(args)
        if args.schedule:
            return run_schedule(args)
        return run_graph(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
