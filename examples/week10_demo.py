"""
Week 10 demo: a modular flow network solver.

Run from the project root:

    python examples/week10_demo.py

Shows (1) the three solvers on the CLRS network, with the min cut that
certifies the answer, (2) how Edmonds–Karp and push–relabel scale on a
50,000-node network, (3) min-cost flow sending the cheap units first,
(4) bipartite matching and the assignment problem, (5) graph-cut
segmentation against plain thresholding, and (6) scheduling: deadlines
on machines, project staffing, and tournament elimination.
The command-line interface is python -m src.flow.main (see --help).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow.applications import (
    assignment_from_matrix, bipartite_matching, schedule_jobs,
    segment_image, segmentation_scores, staff_projects, threshold_segment,
    tournament_elimination,
)
from src.flow.applications.scheduling import roster
from src.flow.core import (
    FlowNetwork, edmonds_karp, min_cost_flow, min_cost_max_flow, push_relabel,
)
from src.flow.utils.generators import sparse_network, synthetic_image
from src.flow.utils.validators import validate_flow

CLRS = [(0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (3, 2, 9),
        (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]


def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_solvers():
    header("1. Three solvers, one answer, one certificate (CLRS Fig. 26.1)")
    for name, solve in (("Edmonds–Karp", edmonds_karp),
                        ("push–relabel", push_relabel),
                        ("min-cost max-flow", min_cost_max_flow)):
        net = FlowNetwork.from_edges(6, CLRS)
        r = solve(net, 0, 5)
        work = (f"{r.iterations} augmenting paths" if "push" not in r.algorithm
                else f"{r.stats['pushes']} pushes, {r.stats['relabels']} relabels")
        print(f"  {name:18s} max flow {r.value}   ({work})")
    net = FlowNetwork.from_edges(6, CLRS)
    r = edmonds_karp(net, 0, 5)
    cut = net.cut_edges(r.min_cut)
    print(f"  bottlenecks {r.stats['bottlenecks']}, path lengths "
          f"{r.stats['path_lengths']} (never decrease)")
    print(f"  min cut S = {sorted(r.min_cut)}: arcs "
          + ", ".join(f"{net.tail(e)}->{net.to[e]} ({net.base[e]})" for e in cut)
          + f" = {net.cut_capacity(r.min_cut)}")
    print(f"  validators: {validate_flow(net, 0, 5, value=23, max_flow=True)}")
    print("  flow matrix:")
    for row in r.flow_matrix().astype(int).tolist():
        print("   ", " ".join(f"{x:3d}" for x in row))


def demo_scaling():
    header("2. Scaling: 50,000-node sparse network")
    net, s, t = sparse_network(50_000, seed=1)
    print(f"  {net.n:,} nodes, {net.num_edges:,} arcs")
    for name, kw in (("push–relabel FIFO", dict(variant="fifo")),
                     ("push–relabel highest-label", dict(variant="highest"))):
        start = time.perf_counter()
        r = push_relabel(net, s, t, **kw)
        print(f"  {name:28s} {time.perf_counter() - start:6.2f} s   flow "
              f"{r.value:,}  ({r.stats['pushes']:,} pushes)")
    small, s2, t2 = sparse_network(10_000, seed=1)
    start = time.perf_counter()
    ek = edmonds_karp(small, s2, t2)
    ek_time = time.perf_counter() - start
    start = time.perf_counter()
    push_relabel(small, s2, t2)
    pr_time = time.perf_counter() - start
    print(f"  at 10,000 nodes: Edmonds–Karp {ek_time:.2f} s "
          f"({ek.iterations} BFS passes over the whole graph) vs push–relabel "
          f"{pr_time:.2f} s")


def demo_min_cost():
    header("3. Min-cost flow: cheap units first")
    # two warehouse routes: 3 units at 2/unit, 4 units at 5/unit
    routes = [(0, 1, 3, 1), (1, 3, 3, 1), (0, 2, 4, 2), (2, 3, 4, 3)]
    for amount in (2, 3, 5, 7, 9):
        r = min_cost_flow(FlowNetwork.from_edges(4, routes), 0, 3, amount)
        note = "" if r.stats["feasible"] else "  (infeasible: max is 7)"
        print(f"  send {amount}: cost {r.cost:3d}{note}")
    r = min_cost_max_flow(FlowNetwork.from_edges(4, routes), 0, 3)
    print(f"  per-unit path costs {r.stats['path_costs']}: marginal cost "
          "only rises — the cost curve is convex")


def demo_matching():
    header("4. Bipartite matching and assignment")
    edges = [("alice", "backend"), ("alice", "frontend"), ("bob", "backend"),
             ("carol", "frontend"), ("carol", "data"), ("dave", "data"),
             ("dave", "devops"), ("erin", "devops")]
    people = sorted({a for a, _ in edges})
    jobs = sorted({b for _, b in edges})
    r = bipartite_matching(people, jobs, edges)
    print(f"  {len(people)} applicants, {len(jobs)} jobs -> {r.size} hires: "
          + ", ".join(f"{a}->{b}" for a, b in sorted(r.pairs)))
    costs = [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]]
    a = assignment_from_matrix(costs)
    print(f"  assignment problem {costs}:")
    print(f"    rows->cols {sorted(a.pairs)}, total cost {a.cost} "
          "(min-cost flow solves the Hungarian algorithm's problem)")


def demo_segmentation():
    header("5. Image segmentation by minimum cut")
    for noise in (0.15, 0.3, 0.45):
        image, truth = synthetic_image(96, 96, "shapes", noise=noise, seed=3)
        start = time.perf_counter()
        r = segment_image(image)
        elapsed = time.perf_counter() - start
        cut = segmentation_scores(r.mask, truth)
        thr = segmentation_scores(threshold_segment(image), truth)
        print(f"  noise {noise:.2f}: graph cut IoU {cut['iou']:.3f} vs threshold "
              f"{thr['iou']:.3f}   ({r.num_nodes:,} nodes, {elapsed * 1000:.0f} ms)")


def demo_scheduling():
    header("6. Scheduling and resource allocation")
    jobs = {"render": (3, 0, 4), "build": (2, 0, 3), "test": (2, 2, 6),
            "deploy": (1, 5, 6), "backup": (3, 1, 6), "report": (1, 3, 5)}
    for machines in (1, 2):
        r = schedule_jobs(jobs, machines)
        print(f"  {machines} machine(s): "
              f"{'feasible' if r.feasible else 'infeasible'} "
              f"({r.scheduled}/{r.required} units)")
    for m, row in schedule_jobs(jobs, 2).machines.items():
        print(f"    M{m}: " + " ".join(f"{job}" for _, job in row))
    staff = staff_projects({"search": 2, "payments": 2, "mobile": 1},
                           {"ana": 1, "ben": 2, "chloe": 1, "dev": 1, "eli": 1},
                           {"ana": ["search", "payments"],
                            "ben": ["search", "mobile"], "chloe": ["payments"],
                            "dev": ["payments", "mobile"], "eli": ["search"]})
    print(f"  staffing: {roster(staff)}")
    wins = {"Atlanta": 83, "Philadelphia": 80, "New York": 78, "Montreal": 77}
    games = {("Atlanta", "Philadelphia"): 1, ("Atlanta", "New York"): 6,
             ("Atlanta", "Montreal"): 1, ("Philadelphia", "Montreal"): 2}
    for team, res in tournament_elimination(wins, games).items():
        print(f"  {team:12s} {'eliminated' if res.eliminated else 'alive':10s} "
              f"{res.reason}")


if __name__ == "__main__":
    demo_solvers()
    demo_scaling()
    demo_min_cost()
    demo_matching()
    demo_segmentation()
    demo_scheduling()
