"""
Scheduling and resource allocation as flow problems.

allocate — the general model behind the first two applications.
Tasks need units of work, resources supply units, and each task may
only use certain resources:

    s → task       capacity demand(task)
    task → res     capacity pair_capacity (∞ if None), cost(task, res)
    res → t        capacity supply(res)

A maximum flow assigns as much work as is possible at all ("maximize
assignment feasibility"); the demand is fully met iff the max flow
equals the total demand, and otherwise the min cut names the
bottleneck. With costs, min-cost max-flow returns the cheapest of the
maximum assignments.

    assign_jobs_to_machines   job hours → eligible machines with hour
                              budgets (work may be split)
    staff_projects            projects need k people; each employee is
                              qualified for some projects and can take
                              at most c of them; a person counts once
                              per project (pair capacity 1); optional
                              preference costs

schedule_jobs — preemptive scheduling with release times and
deadlines on m identical machines. Job j needs p_j unit time slots in
[r_j, d_j) and can run on at most one machine per slot:

    s → job j      capacity p_j
    job j → slot k capacity 1      for r_j ≤ k < d_j
    slot k → t     capacity m      (machines available in slot k)

A schedule exists iff max flow = Σ p_j (Horn 1974).

tournament_elimination — the "bracket constraint" problem (baseball
elimination, Schwartz 1966). Team x can still finish first iff the
remaining games among the other teams can be decided so that nobody
passes x's best possible total W = wins_x + remaining_x:

    s → game {i, j}   capacity games left between i and j
    game → i, game → j  capacity ∞ (the winner gets the point)
    team i → t        capacity W − wins_i

x is eliminated iff the max flow is less than the number of those
games; the teams on the source side of the min cut form the
certificate (together they must win more games than W allows).
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

from src.flow.core import get_solver, min_cost_max_flow
from src.flow.core.network import FlowNetwork, FlowResult


# ----------------------------------------------------------------------
# general allocation
# ----------------------------------------------------------------------
@dataclass
class AllocationResult:
    assignment: Dict[Hashable, Dict[Hashable, float]]   # task -> {resource: units}
    assigned: float
    required: float
    flow: FlowResult
    cost: Optional[float] = None
    unmet: Dict[Hashable, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.assigned >= self.required - 1e-9


def allocate(demands: Mapping[Hashable, float],
             capacities: Mapping[Hashable, float],
             eligible: Mapping[Hashable, Iterable[Hashable]],
             pair_capacity: Optional[float] = None,
             costs: Optional[Mapping[Tuple[Hashable, Hashable], float]] = None,
             solver: str = "pr") -> AllocationResult:
    """
    Assign task demands to resource capacities along eligible pairs,
    maximizing the total assigned (then minimizing cost, if costs are
    given — this switches to min-cost max-flow).
    """
    tasks, resources = list(demands), list(capacities)
    task_id = {x: 1 + i for i, x in enumerate(tasks)}
    res_id = {r: 1 + len(tasks) + i for i, r in enumerate(resources)}
    n = 2 + len(tasks) + len(resources)
    s, t = 0, n - 1
    net = FlowNetwork(n)
    total = float(sum(demands.values()))
    for x in tasks:
        if demands[x] < 0:
            raise ValueError(f"negative demand for {x!r}")
        net.add_edge(s, task_id[x], demands[x])
    for r in resources:
        if capacities[r] < 0:
            raise ValueError(f"negative capacity for {r!r}")
        net.add_edge(res_id[r], t, capacities[r])
    big = total + sum(capacities.values()) + 1
    pair_edges = {}
    for x in tasks:
        for r in dict.fromkeys(eligible.get(x, ())):
            if r not in res_id:
                raise ValueError(f"task {x!r} lists unknown resource {r!r}")
            cost = costs.get((x, r), 0) if costs else 0
            cap = big if pair_capacity is None else pair_capacity
            pair_edges[net.add_edge(task_id[x], res_id[r], cap, cost)] = (x, r)
    if costs is not None:
        result = min_cost_max_flow(net, s, t)
    else:
        result = get_solver(solver)(net, s, t)
    assignment: Dict[Hashable, Dict[Hashable, float]] = {x: {} for x in tasks}
    for e, (x, r) in pair_edges.items():
        f = net.flow_on(e)
        if f > 1e-9:
            assignment[x][r] = f
    unmet = {x: demands[x] - sum(assignment[x].values()) for x in tasks
             if demands[x] - sum(assignment[x].values()) > 1e-9}
    return AllocationResult(assignment, result.value, total, result,
                            result.cost, unmet)


def assign_jobs_to_machines(job_hours: Mapping[Hashable, float],
                            machine_hours: Mapping[Hashable, float],
                            eligible: Mapping[Hashable, Iterable[Hashable]],
                            costs: Optional[Mapping[Tuple, float]] = None,
                            solver: str = "pr") -> AllocationResult:
    """Split each job's hours over machines it may run on, within each
    machine's hour budget (cheapest split if hourly costs are given)."""
    return allocate(job_hours, machine_hours, eligible, None, costs, solver)


def staff_projects(project_needs: Mapping[Hashable, int],
                   employee_limits: Mapping[Hashable, int],
                   qualified: Mapping[Hashable, Iterable[Hashable]],
                   preferences: Optional[Mapping[Tuple, float]] = None,
                   solver: str = "pr") -> AllocationResult:
    """
    Staff projects: qualified maps employee -> projects they can work on.
    preferences[(project, employee)] is a cost (lower = preferred).
    The result's assignment maps project -> {employee: 1}.
    """
    eligible: Dict[Hashable, List[Hashable]] = defaultdict(list)
    for person, projects in qualified.items():
        for p in projects:
            eligible[p].append(person)
    return allocate(project_needs, employee_limits, eligible, 1,
                    preferences, solver)


def roster(result: AllocationResult) -> Dict[Hashable, List[Hashable]]:
    """project -> sorted list of assigned employees."""
    return {p: sorted(people, key=str) for p, people in result.assignment.items()}


# ----------------------------------------------------------------------
# deadline scheduling
# ----------------------------------------------------------------------
@dataclass
class ScheduleResult:
    feasible: bool
    slots: Dict[int, List[Hashable]]                 # time slot -> jobs running
    machines: Dict[int, List[Tuple[int, Hashable]]]  # machine -> [(slot, job)]
    scheduled: int
    required: int
    flow: FlowResult
    late: Dict[Hashable, int] = field(default_factory=dict)  # job -> units missing


def schedule_jobs(jobs: Mapping[Hashable, Tuple[int, int, int]],
                  machines: int, solver: str = "pr") -> ScheduleResult:
    """
    jobs: {name: (processing_time, release, deadline)} in integer time
    slots; the job may run in slots release, …, deadline − 1, one
    machine at a time, and may be interrupted (preemptive).
    """
    if machines < 1:
        raise ValueError("need at least one machine")
    for name, (p, r, d) in jobs.items():
        if p < 0 or r < 0 or d < r:
            raise ValueError(f"job {name!r} has an invalid (p, r, d)")
    names = list(jobs)
    horizon = max((d for _, _, d in jobs.values()), default=0)
    n = 2 + len(names) + horizon
    s, t = 0, n - 1
    net = FlowNetwork(n)
    slot_node = lambda k: 1 + len(names) + k          # noqa: E731
    job_slot_edges = []
    for j, name in enumerate(names):
        p, r, d = jobs[name]
        net.add_edge(s, 1 + j, p)
        for k in range(r, d):
            job_slot_edges.append((net.add_edge(1 + j, slot_node(k), 1), name, k))
    for k in range(horizon):
        net.add_edge(slot_node(k), t, machines)
    result = get_solver(solver)(net, s, t)

    slots: Dict[int, List[Hashable]] = {k: [] for k in range(horizon)}
    done: Dict[Hashable, int] = defaultdict(int)
    for e, name, k in job_slot_edges:
        if net.flow_on(e) > 0.5:
            slots[k].append(name)
            done[name] += 1
    # give each job the same machine as in its previous slot when possible
    timeline: Dict[int, List[Tuple[int, Hashable]]] = {m: [] for m in range(machines)}
    last_machine: Dict[Hashable, int] = {}
    for k in range(horizon):
        free = set(range(machines))
        placed = {}
        for name in slots[k]:
            m = last_machine.get(name)
            if m in free:
                placed[name] = m
                free.discard(m)
        for name in slots[k]:
            if name not in placed:
                placed[name] = min(free)
                free.discard(placed[name])
        for name, m in placed.items():
            timeline[m].append((k, name))
            last_machine[name] = m
    required = sum(p for p, _, _ in jobs.values())
    late = {name: jobs[name][0] - done[name] for name in names
            if done[name] < jobs[name][0]}
    return ScheduleResult(result.value == required, slots, timeline,
                          int(result.value), required, result, late)


# ----------------------------------------------------------------------
# tournament elimination
# ----------------------------------------------------------------------
@dataclass
class EliminationResult:
    team: Hashable
    eliminated: bool
    certificate: List[Hashable]      # teams that together exceed the bound
    best_possible: int
    reason: str


def _games_left(games: Mapping[Tuple[Hashable, Hashable], int], team) -> int:
    return sum(g for (a, b), g in games.items() if team in (a, b))


def team_eliminated(team: Hashable, wins: Mapping[Hashable, int],
                    games: Mapping[Tuple[Hashable, Hashable], int],
                    solver: str = "pr") -> EliminationResult:
    """Can `team` still finish with at least as many wins as everyone?"""
    best = wins[team] + _games_left(games, team)
    for other, w in wins.items():
        if w > best:
            return EliminationResult(team, True, [other], best,
                                     f"{other} already has {w} wins")
    others = [x for x in wins if x != team]
    remaining = [((a, b), g) for (a, b), g in games.items()
                 if g > 0 and team not in (a, b)]
    team_node = {x: 1 + len(remaining) + i for i, x in enumerate(others)}
    n = 2 + len(remaining) + len(others)
    s, t = 0, n - 1
    net = FlowNetwork(n)
    total = sum(g for _, g in remaining)
    for i, ((a, b), g) in enumerate(remaining):
        net.add_edge(s, 1 + i, g)
        net.add_edge(1 + i, team_node[a], total + 1)
        net.add_edge(1 + i, team_node[b], total + 1)
    for x in others:
        net.add_edge(team_node[x], t, best - wins[x])
    result = get_solver(solver)(net, s, t)
    if result.value >= total:
        return EliminationResult(team, False, [], best,
                                 "the remaining games can be split so "
                                 "nobody passes this team")
    certificate = sorted((x for x in others if team_node[x] in result.min_cut),
                         key=str)
    return EliminationResult(team, True, certificate, best,
                             f"teams {certificate} must win more games "
                             f"between them than a {best}-win ceiling allows")


def tournament_elimination(wins: Mapping[Hashable, int],
                           games: Mapping[Tuple[Hashable, Hashable], int],
                           solver: str = "pr") -> Dict[Hashable, EliminationResult]:
    """Elimination status of every team. games[(a, b)] = games left
    between a and b (each pair listed once)."""
    for (a, b) in games:
        if a not in wins or b not in wins:
            raise ValueError(f"game ({a!r}, {b!r}) involves an unknown team")
    return {team: team_eliminated(team, wins, games, solver) for team in wins}
