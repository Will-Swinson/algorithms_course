"""Test suite for flow-based scheduling and resource allocation."""

import itertools

import pytest

from src.flow.applications import (
    allocate, assign_jobs_to_machines, schedule_jobs, staff_projects,
    team_eliminated, tournament_elimination,
)
from src.flow.applications.scheduling import roster
from src.flow.utils.generators import random_jobs


class TestAllocate:

    def test_feasible(self):
        r = allocate({"a": 3, "b": 2}, {"x": 4, "y": 1},
                     {"a": ["x"], "b": ["x", "y"]})
        assert r.feasible and r.assigned == 5
        assert sum(r.assignment["a"].values()) == 3
        assert r.assignment["b"].get("y", 0) <= 1

    def test_infeasible_maximizes(self):
        r = allocate({"a": 5, "b": 5}, {"x": 6}, {"a": ["x"], "b": ["x"]})
        assert not r.feasible and r.assigned == 6
        assert sum(r.unmet.values()) == 4

    def test_pair_capacity(self):
        r = allocate({"a": 3}, {"x": 5, "y": 5}, {"a": ["x", "y"]},
                     pair_capacity=1)
        assert r.assigned == 2

    def test_costs_pick_cheapest(self):
        r = allocate({"a": 2}, {"x": 5, "y": 5}, {"a": ["x", "y"]},
                     costs={("a", "x"): 9, ("a", "y"): 1})
        assert r.assignment["a"] == {"y": 2} and r.cost == 2

    @pytest.mark.parametrize("bad", [
        dict(demands={"a": -1}, capacities={"x": 1}, eligible={}),
        dict(demands={"a": 1}, capacities={"x": -1}, eligible={}),
        dict(demands={"a": 1}, capacities={"x": 1}, eligible={"a": ["z"]}),
    ])
    def test_rejects(self, bad):
        with pytest.raises(ValueError):
            allocate(**bad)


class TestWrappers:

    def test_job_machines_split(self):
        r = assign_jobs_to_machines({"J1": 6}, {"m1": 4, "m2": 4},
                                    {"J1": ["m1", "m2"]})
        assert r.feasible and sum(r.assignment["J1"].values()) == 6

    def test_job_machines_cost(self):
        r = assign_jobs_to_machines({"J1": 6, "J2": 4}, {"m1": 7, "m2": 5},
                                    {"J1": ["m1", "m2"], "J2": ["m2"]},
                                    costs={("J1", "m1"): 3, ("J1", "m2"): 1,
                                           ("J2", "m2"): 1})
        # J2 must take 4 of m2's 5 hours; J1 gets 1 there and 5 on m1
        assert r.feasible and r.cost == 4 + 1 + 15

    def test_staffing(self):
        r = staff_projects({"A": 2, "B": 1}, {"ann": 1, "bob": 2, "cy": 1},
                           {"ann": ["A"], "bob": ["A", "B"], "cy": ["A"]})
        assert r.feasible
        team = roster(r)
        assert len(team["A"]) == 2 and len(team["B"]) == 1
        load = {}
        for people in team.values():
            for p in people:
                load[p] = load.get(p, 0) + 1
        assert load.get("ann", 0) <= 1 and load.get("cy", 0) <= 1

    def test_staffing_preferences(self):
        r = staff_projects({"A": 1}, {"ann": 1, "bob": 1},
                           {"ann": ["A"], "bob": ["A"]},
                           preferences={("A", "ann"): 5, ("A", "bob"): 1})
        assert roster(r)["A"] == ["bob"]

    def test_staffing_shortage(self):
        r = staff_projects({"A": 3}, {"ann": 1}, {"ann": ["A"]})
        assert not r.feasible and r.unmet == {"A": 2}


def brute_force_feasible(jobs, machines):
    """Exhaustive check for tiny instances: try every slot choice."""
    names = list(jobs)
    options = [list(itertools.combinations(range(r, d), p))
               for p, r, d in (jobs[n] for n in names)]
    for choice in itertools.product(*options):
        load = {}
        for slots in choice:
            for k in slots:
                load[k] = load.get(k, 0) + 1
        if all(v <= machines for v in load.values()):
            return True
    return False


class TestDeadlineScheduling:

    def test_feasible_schedule_is_valid(self):
        jobs = {"a": (2, 0, 3), "b": (2, 0, 2), "c": (1, 1, 3)}
        r = schedule_jobs(jobs, 2)
        assert r.feasible and r.scheduled == r.required == 5
        for name, (p, rel, dl) in jobs.items():
            used = [k for k, names in r.slots.items() if name in names]
            assert len(used) == p and all(rel <= k < dl for k in used)
        for k, names in r.slots.items():
            assert len(names) <= 2 and len(set(names)) == len(names)

    def test_machine_timeline(self):
        r = schedule_jobs({"a": (3, 0, 3), "b": (1, 0, 3)}, 2)
        assert all(job == "a" for _, job in r.machines[0]) or \
            all(job == "a" for _, job in r.machines[1])
        assert sum(len(row) for row in r.machines.values()) == 4
        for row in r.machines.values():         # one job per machine-slot
            slots = [k for k, _ in row]
            assert len(slots) == len(set(slots))

    def test_infeasible_reports_lateness(self):
        r = schedule_jobs({"a": (2, 0, 2), "b": (2, 0, 2)}, 1)
        assert not r.feasible and r.scheduled == 2
        assert sum(r.late.values()) == 2

    @pytest.mark.parametrize("seed", range(15))
    def test_matches_brute_force(self, seed):
        jobs = random_jobs(4, 5, max_length=3, seed=seed)
        for m in (1, 2):
            assert schedule_jobs(jobs, m).feasible == brute_force_feasible(jobs, m)

    def test_more_machines_never_hurt(self):
        jobs = random_jobs(25, 15, seed=3)
        scheduled = [schedule_jobs(jobs, m).scheduled for m in range(1, 7)]
        assert scheduled == sorted(scheduled)

    def test_rejects(self):
        with pytest.raises(ValueError):
            schedule_jobs({"a": (1, 0, 1)}, 0)
        with pytest.raises(ValueError):
            schedule_jobs({"a": (1, 3, 2)}, 1)


WINS = {"Atlanta": 83, "Philadelphia": 80, "New York": 78, "Montreal": 77}
GAMES = {("Atlanta", "Philadelphia"): 1, ("Atlanta", "New York"): 6,
         ("Atlanta", "Montreal"): 1, ("Philadelphia", "Montreal"): 2}


def brute_force_alive(team, wins, games):
    pairs = [(a, b) for (a, b), g in games.items() for _ in range(g)]
    best = wins[team] + sum(g for (a, b), g in games.items() if team in (a, b))
    for outcome in itertools.product((0, 1), repeat=len(pairs)):
        final = dict(wins)
        for (a, b), w in zip(pairs, outcome):
            final[a if w == 0 else b] += 1
        final[team] = best
        if all(v <= best for v in final.values()):
            return True
    return False


class TestTournament:

    def test_classic_example(self):
        status = tournament_elimination(WINS, GAMES)
        assert not status["Atlanta"].eliminated
        assert not status["New York"].eliminated
        assert status["Montreal"].eliminated
        assert status["Montreal"].certificate == ["Atlanta"]
        philly = status["Philadelphia"]
        assert philly.eliminated and philly.best_possible == 83
        assert philly.certificate == ["Atlanta", "New York"]

    def test_certificate_proves_elimination(self):
        r = team_eliminated("Philadelphia", WINS, GAMES)
        group = set(r.certificate)
        total = sum(WINS[x] for x in group) + sum(
            g for (a, b), g in GAMES.items() if a in group and b in group)
        assert total / len(group) > r.best_possible

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_brute_force(self, seed):
        import random
        rng = random.Random(seed)
        teams = "ABCD"
        wins = {x: rng.randint(0, 6) for x in teams}
        games = {(a, b): rng.randint(0, 2) for a, b in
                 itertools.combinations(teams, 2)}
        status = tournament_elimination(wins, games)
        for team in teams:
            assert status[team].eliminated == (not brute_force_alive(team, wins,
                                                                     games))

    def test_unknown_team(self):
        with pytest.raises(ValueError):
            tournament_elimination({"A": 1}, {("A", "Z"): 1})
