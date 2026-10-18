"""
Boolean satisfiability (SAT) — the first problem proved NP-complete
(Cook–Levin, 1971).

Representation (the DIMACS convention used by real SAT solvers):

    variable    an integer 1..n
    literal     +i means x_i, −i means ¬x_i
    clause      a tuple of literals, read as an OR:  (1, -2, 3) = x1 ∨ ¬x2 ∨ x3
    formula     a list of clauses, read as an AND (conjunctive normal form)
    assignment  a dict {variable: bool}

SAT is in NP because a claimed assignment (the certificate) is checked
in O(m·k) time by is_satisfied(). Nobody knows how to FIND one in
polynomial time; every exact solver here is exponential in the worst
case:

    brute_force_sat    tries all 2^n assignments                 Θ(2^n · m)
    backtracking_sat   DFS over partial assignments, abandoning a
                       branch as soon as a clause is falsified;
                       with unit_propagation=True this is DPLL   O(2^n · m) worst

Restricting every clause to two literals changes the picture
completely: two_sat() decides 2-SAT in O(n + m) through strongly
connected components. 2-SAT is in P; 3-SAT is NP-complete.

Heuristics trade the guarantee for speed:

    walksat            randomized local search (Selman, Kautz & Cohen 1994).
                       Finds satisfying assignments of large satisfiable
                       formulas quickly, but can never prove UNSAT.
    random_assignment  each 3-literal clause is satisfied with
                       probability 7/8, so a coin flip per variable
                       already satisfies 87.5% of clauses on average.
"""
import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils.timer import CallStats

Clause = Tuple[int, ...]
CNF = List[Clause]
Assignment = Dict[int, bool]

# Random 3-SAT with m/n near 4.26 sits at the satisfiability phase
# transition: about half the formulas are satisfiable, and both exact
# and heuristic solvers find them hardest there.
PHASE_TRANSITION_RATIO = 4.26


def validate_cnf(cnf: Sequence[Sequence[int]], num_vars: int) -> None:
    """Raise ValueError unless every literal names a variable 1..num_vars."""
    if num_vars < 0:
        raise ValueError(f"num_vars must be non-negative, got {num_vars}")
    for clause in cnf:
        for lit in clause:
            if lit == 0 or abs(lit) > num_vars:
                raise ValueError(f"literal {lit} outside 1..{num_vars}")


def literal_true(lit: int, assignment: Assignment) -> bool:
    """Value of one literal under a (complete) assignment."""
    return assignment[abs(lit)] == (lit > 0)


def count_satisfied(cnf: Sequence[Clause], assignment: Assignment) -> int:
    """Number of clauses with at least one true literal."""
    return sum(any(literal_true(lit, assignment) for lit in clause)
               for clause in cnf)


def is_satisfied(cnf: Sequence[Clause], assignment: Assignment) -> bool:
    """
    The NP certificate check: does `assignment` make every clause true?

    O(m·k) for m clauses of k literals — polynomial, which is what
    puts SAT in NP. An empty clause can never be satisfied.
    """
    return all(any(literal_true(lit, assignment) for lit in clause)
               for clause in cnf)


def format_cnf(cnf: Sequence[Clause]) -> str:
    """Human-readable formula: (x1 ∨ ¬x2 ∨ x3) ∧ (...)."""
    def lit(v: int) -> str:
        return f"x{v}" if v > 0 else f"¬x{-v}"
    return " ∧ ".join("(" + " ∨ ".join(lit(v) for v in c) + ")"
                      for c in cnf) or "⊤"


# ---------------------------------------------------------------------------
# Exact solvers
# ---------------------------------------------------------------------------

def brute_force_sat(cnf: Sequence[Clause], num_vars: int,
                    stats: Optional[CallStats] = None
                    ) -> Optional[Assignment]:
    """
    Try every one of the 2^n assignments in order.

    Args:
        cnf: The formula.
        num_vars: n; variables are 1..n.
        stats: Optional counter; receives one count per assignment
            tested.

    Returns:
        The first satisfying assignment, or None if unsatisfiable.

    Complexity: Θ(2^n · m·k) when unsatisfiable — every assignment is
    checked in full.
    """
    validate_cnf(cnf, num_vars)
    tested = 0
    try:
        for values in itertools.product((False, True), repeat=num_vars):
            tested += 1
            assignment = dict(zip(range(1, num_vars + 1), values))
            if is_satisfied(cnf, assignment):
                return assignment
        return None
    finally:
        if stats is not None:
            stats.count(tested)


def backtracking_sat(cnf: Sequence[Clause], num_vars: int,
                     unit_propagation: bool = True,
                     stats: Optional[CallStats] = None
                     ) -> Optional[Assignment]:
    """
    Depth-first search over partial assignments.

    At each node every clause is classified under the current partial
    assignment:

        satisfied    some literal is already true        -> ignore
        falsified    every literal assigned and false    -> backtrack now
        unit         exactly one literal left unassigned -> it is forced
        open         two or more literals unassigned

    A falsified clause prunes the whole subtree below the node, which is
    the entire advantage over brute force. With unit_propagation=True
    the forced literals are assigned before branching (this is DPLL,
    the core of every modern SAT solver); without it, the search
    branches on them like any other variable.

    Variables are branched on in the order 1..n, trying True then False.
    If every clause is satisfied before all variables are assigned, the
    rest are set False.

    Args:
        cnf, num_vars: The formula and its variable count.
        unit_propagation: Assign unit-clause literals without branching.
        stats: Optional counter; stats.calls = search nodes visited,
            stats.max_depth = deepest node.

    Returns:
        A satisfying assignment, or None if unsatisfiable.

    Complexity: O(2^n · m·k) worst case; usually far fewer nodes.
    """
    validate_cnf(cnf, num_vars)
    stats = stats if stats is not None else CallStats()
    values: List[Optional[bool]] = [None] * (num_vars + 1)

    def scan() -> Tuple[str, int]:
        """('conflict'|'unit'|'done'|'open', unit literal or 0)."""
        all_satisfied = True
        for clause in cnf:
            free, last_free = 0, 0
            for lit in clause:
                v = values[abs(lit)]
                if v is None:
                    free += 1
                    last_free = lit
                elif v == (lit > 0):
                    break                      # clause satisfied
            else:
                if free == 0:
                    return "conflict", 0
                all_satisfied = False
                if free == 1 and unit_propagation:
                    return "unit", last_free
        return ("done" if all_satisfied else "open"), 0

    def search() -> bool:
        stats.enter()
        trail: List[int] = []                  # variables forced here
        try:
            while True:
                status, lit = scan()
                if status != "unit":
                    break
                values[abs(lit)] = lit > 0
                trail.append(abs(lit))
            if status == "conflict":
                return _undo(trail)
            if status == "done":
                return True
            var = next(i for i in range(1, num_vars + 1) if values[i] is None)
            for value in (True, False):
                values[var] = value
                if search():
                    return True
            values[var] = None
            return _undo(trail)
        finally:
            stats.exit()

    def _undo(trail: List[int]) -> bool:
        for var in trail:
            values[var] = None
        return False

    if not search():
        return None
    return {i: bool(values[i]) for i in range(1, num_vars + 1)}


def two_sat(cnf: Sequence[Clause], num_vars: int) -> Optional[Assignment]:
    """
    Decide 2-SAT in linear time — the polynomial side of the P / NP line.

    Each clause (a ∨ b) is the pair of implications ¬a → b and ¬b → a.
    The formula is unsatisfiable exactly when some x_i and ¬x_i lie in
    the same strongly connected component of this implication graph
    (each forces the other). Otherwise, ordering the components
    topologically and setting each variable to the literal whose
    component comes later gives a satisfying assignment.

    Clauses of one literal (a) are treated as (a ∨ a).

    Args:
        cnf: Clauses of one or two literals.
        num_vars: n.

    Returns:
        A satisfying assignment, or None if unsatisfiable.

    Raises:
        ValueError: If a clause has more than two literals.

    Complexity: O(n + m) — Kosaraju's two-pass SCC algorithm, written
    iteratively so large formulas don't hit Python's recursion limit.
    """
    validate_cnf(cnf, num_vars)
    # node 2(v-1) is x_v, node 2(v-1)+1 is ¬x_v
    def node(lit: int) -> int:
        return 2 * (abs(lit) - 1) + (lit < 0)

    size = 2 * num_vars
    graph: List[List[int]] = [[] for _ in range(size)]
    reverse: List[List[int]] = [[] for _ in range(size)]
    for clause in cnf:
        if not 1 <= len(clause) <= 2:
            raise ValueError(f"2-SAT clauses need 1 or 2 literals: {clause}")
        a, b = clause[0], clause[-1]
        for u, v in ((node(-a), node(b)), (node(-b), node(a))):
            graph[u].append(v)
            reverse[v].append(u)

    # Pass 1: order nodes by DFS finishing time
    order: List[int] = []
    seen = [False] * size
    for start in range(size):
        if seen[start]:
            continue
        seen[start] = True
        stack = [(start, iter(graph[start]))]
        while stack:
            u, it = stack[-1]
            for v in it:
                if not seen[v]:
                    seen[v] = True
                    stack.append((v, iter(graph[v])))
                    break
            else:
                stack.pop()
                order.append(u)

    # Pass 2: DFS the reversed graph in decreasing finish time; the
    # components come out in topological order of the condensation
    comp = [-1] * size
    label = 0
    for start in reversed(order):
        if comp[start] != -1:
            continue
        comp[start] = label
        stack = [start]
        while stack:
            u = stack.pop()
            for v in reverse[u]:
                if comp[v] == -1:
                    comp[v] = label
                    stack.append(v)
        label += 1

    assignment: Assignment = {}
    for var in range(1, num_vars + 1):
        pos, neg = comp[node(var)], comp[node(-var)]
        if pos == neg:
            return None
        # later in topological order = nothing it implies can force it false
        assignment[var] = pos > neg
    return assignment


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

@dataclass
class LocalSearchResult:
    """Best assignment a heuristic found and how good it is."""
    assignment: Assignment
    satisfied: int          # clauses satisfied by `assignment`
    num_clauses: int
    flips: int = 0

    @property
    def solved(self) -> bool:
        return self.satisfied == self.num_clauses

    @property
    def fraction(self) -> float:
        return self.satisfied / self.num_clauses if self.num_clauses else 1.0


def random_assignment(cnf: Sequence[Clause], num_vars: int,
                      rng: Optional[random.Random] = None
                      ) -> LocalSearchResult:
    """
    Baseline: flip a fair coin for every variable. O(n + m·k).

    A clause of k distinct variables is false only if all k literals
    are, with probability 2^−k: for 3-SAT, 7/8 of clauses are
    satisfied in expectation.
    """
    validate_cnf(cnf, num_vars)
    rng = rng or random
    assignment = {i: rng.random() < 0.5 for i in range(1, num_vars + 1)}
    return LocalSearchResult(assignment, count_satisfied(cnf, assignment),
                             len(cnf))


def walksat(cnf: Sequence[Clause], num_vars: int, max_flips: int = 10_000,
            noise: float = 0.5, rng: Optional[random.Random] = None
            ) -> LocalSearchResult:
    """
    WalkSAT local search.

    Start from a random assignment. Repeatedly pick a random
    unsatisfied clause and flip one of its variables:

        with probability `noise`   a random variable from the clause
        otherwise                  the one whose flip breaks the fewest
                                   currently satisfied clauses (greedy)

    The random moves let the search escape local minima that pure
    greedy flipping gets stuck in. Stops at a satisfying assignment or
    after `max_flips`, returning the best assignment seen.

    Bookkeeping keeps each flip cheap: true_count[c] is the number of
    true literals in clause c, and only clauses containing the flipped
    variable are touched.

    Returns:
        LocalSearchResult; .solved is True if a satisfying assignment
        was found. A False result proves nothing — the formula may
        still be satisfiable (Monte Carlo, one-sided error).

    Complexity: O(n + m·k) setup, O(occurrences of a variable) per flip.
    """
    validate_cnf(cnf, num_vars)
    if not 0 <= noise <= 1:
        raise ValueError(f"noise must be in [0, 1], got {noise}")
    rng = rng or random
    m = len(cnf)
    values = [False] + [rng.random() < 0.5 for _ in range(num_vars)]

    occurs: Dict[int, List[int]] = {}
    for ci, clause in enumerate(cnf):
        for lit in clause:
            occurs.setdefault(lit, []).append(ci)

    true_count = [sum(values[abs(l)] == (l > 0) for l in c) for c in cnf]
    unsat: List[int] = [ci for ci in range(m) if true_count[ci] == 0]
    where = {ci: i for i, ci in enumerate(unsat)}   # position in unsat

    def remove_unsat(ci: int) -> None:
        i = where.pop(ci)
        last = unsat.pop()
        if last != ci:
            unsat[i] = last
            where[last] = i

    def breaks(var: int) -> int:
        true_lit = var if values[var] else -var
        return sum(true_count[ci] == 1 for ci in occurs.get(true_lit, ()))

    def flip(var: int) -> None:
        old_true = var if values[var] else -var
        values[var] = not values[var]
        for ci in occurs.get(old_true, ()):
            true_count[ci] -= 1
            if true_count[ci] == 0:
                where[ci] = len(unsat)
                unsat.append(ci)
        for ci in occurs.get(-old_true, ()):
            true_count[ci] += 1
            if true_count[ci] == 1:
                remove_unsat(ci)

    best_unsat, best_values = len(unsat), values[:]
    flips = 0
    while unsat and flips < max_flips:
        clause = cnf[unsat[rng.randrange(len(unsat))]]
        if not clause:
            break                                  # empty clause: hopeless
        if rng.random() < noise:
            var = abs(clause[rng.randrange(len(clause))])
        else:
            var = min((abs(l) for l in clause), key=breaks)
        flip(var)
        flips += 1
        if len(unsat) < best_unsat:
            best_unsat, best_values = len(unsat), values[:]

    assignment = {i: best_values[i] for i in range(1, num_vars + 1)}
    return LocalSearchResult(assignment, m - best_unsat, m, flips)


# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------

def random_ksat(num_vars: int, num_clauses: int, k: int = 3,
                seed: Optional[int] = None) -> CNF:
    """
    Uniform random k-SAT: each clause draws k distinct variables and
    negates each with probability 1/2.
    """
    if not 1 <= k <= num_vars:
        raise ValueError(f"need 1 <= k <= num_vars, got k={k}, n={num_vars}")
    rng = random.Random(seed)
    return [tuple(v if rng.random() < 0.5 else -v
                  for v in rng.sample(range(1, num_vars + 1), k))
            for _ in range(num_clauses)]


def planted_ksat(num_vars: int, num_clauses: int, k: int = 3,
                 seed: Optional[int] = None) -> Tuple[CNF, Assignment]:
    """
    Random k-SAT guaranteed satisfiable: draw a hidden assignment, then
    keep only random clauses it satisfies.

    Returns:
        (formula, the planted satisfying assignment)
    """
    if not 1 <= k <= num_vars:
        raise ValueError(f"need 1 <= k <= num_vars, got k={k}, n={num_vars}")
    rng = random.Random(seed)
    hidden = {i: rng.random() < 0.5 for i in range(1, num_vars + 1)}
    cnf: CNF = []
    while len(cnf) < num_clauses:
        clause = tuple(v if rng.random() < 0.5 else -v
                       for v in rng.sample(range(1, num_vars + 1), k))
        if any(literal_true(lit, hidden) for lit in clause):
            cnf.append(clause)
    return cnf, hidden
