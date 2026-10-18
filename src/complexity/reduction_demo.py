"""
Polynomial-time reductions, step by step.

A reduction A ≤p B is a polynomial-time function f that turns any
instance x of problem A into an instance f(x) of problem B with

    x is a "yes" instance of A   ⟺   f(x) is a "yes" instance of B.

So a fast algorithm for B would give a fast algorithm for A: B is at
least as hard as A. If A is NP-complete, and B is in NP (its solutions
can be checked in polynomial time), then B is NP-complete too. That is
how every NP-completeness proof after Cook–Levin works.

The classes, as used throughout this module:

    P             solvable in polynomial time (2-SAT, shortest paths, MST)
    NP            a "yes" answer has a certificate checkable in polynomial
                  time (an assignment, a cover, a subset)
    NP-hard       every problem in NP reduces to it (it need not be in NP:
                  TSP optimization, the halting problem)
    NP-complete   NP-hard AND in NP — the hardest problems in NP

Reductions implemented here, each recording a printed/logged
explanation of every step and mapping solutions back:

    3-SAT → Vertex Cover        variable edges + clause triangles
    Vertex Cover → Independent Set    complement of the chosen set
    Independent Set → Clique          complement of the graph
    3-SAT → Subset Sum          one decimal digit per variable and clause

and ReductionChain, which composes them (3-SAT → VC → IS → Clique),
showing that ≤p is transitive.

check_reduction() validates a reduction on a concrete instance with
the exact solvers from Part 2: both sides must give the same yes/no
answer, a target solution must map back to a valid source solution,
and a source solution must map forward to a valid target solution.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Hashable, List, Optional, Sequence,
                    Set)

from src.complexity.satisfiability_solver import (
    Assignment, Clause, backtracking_sat, format_cnf, is_satisfied,
    literal_true, validate_cnf,
)
from src.complexity.subset_sum_backtracking import (
    is_subset_sum_solution, subset_sum_backtracking,
)
from src.complexity.vertex_cover_approx import (
    exact_vertex_cover, is_vertex_cover,
)
from src.utils.graph_utils import Edge, adjacency_sets, complement_edges

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Problem instances
# ---------------------------------------------------------------------------

@dataclass
class ThreeSAT:
    """CNF formula over variables 1..num_vars, 1–3 literals per clause."""
    clauses: List[Clause]
    num_vars: int

    def __post_init__(self) -> None:
        validate_cnf(self.clauses, self.num_vars)
        for c in self.clauses:
            if not 1 <= len(c) <= 3:
                raise ValueError(f"3-SAT clauses need 1–3 literals: {c}")


@dataclass
class VertexCover:
    """Is there a set of ≤ k vertices touching every edge?"""
    vertices: List[Hashable]
    edges: List[Edge]
    k: int


@dataclass
class IndependentSet:
    """Is there a set of ≥ k vertices with no edge between any two?"""
    vertices: List[Hashable]
    edges: List[Edge]
    k: int


@dataclass
class Clique:
    """Is there a set of ≥ k vertices with an edge between every two?"""
    vertices: List[Hashable]
    edges: List[Edge]
    k: int


@dataclass
class SubsetSum:
    """Is there a subset of `numbers` summing exactly to `target`?"""
    numbers: List[int]
    target: int


# ---------------------------------------------------------------------------
# Certificate checks (polynomial time: this is what puts each problem in NP)
# ---------------------------------------------------------------------------

def verify_3sat(inst: ThreeSAT, assignment: Assignment) -> bool:
    return (all(v in assignment for v in range(1, inst.num_vars + 1))
            and is_satisfied(inst.clauses, assignment))


def verify_vertex_cover(inst: VertexCover, cover: Set[Hashable]) -> bool:
    return (set(cover) <= set(inst.vertices) and len(cover) <= inst.k
            and is_vertex_cover(inst.edges, cover))


def verify_independent_set(inst: IndependentSet, s: Set[Hashable]) -> bool:
    s = set(s)
    return (s <= set(inst.vertices) and len(s) >= inst.k
            and not any(u in s and v in s for u, v in inst.edges))


def verify_clique(inst: Clique, s: Set[Hashable]) -> bool:
    s = set(s)
    adj = adjacency_sets(inst.edges)
    return (s <= set(inst.vertices) and len(s) >= inst.k
            and all(v in adj.get(u, ()) for u in s for v in s if u != v))


def verify_subset_sum(inst: SubsetSum, indices: Sequence[int]) -> bool:
    return is_subset_sum_solution(inst.numbers, inst.target, indices)


# ---------------------------------------------------------------------------
# Exact solvers (exponential; from Part 2 of the assignment)
# ---------------------------------------------------------------------------

def solve_3sat(inst: ThreeSAT) -> Optional[Assignment]:
    return backtracking_sat(inst.clauses, inst.num_vars)


def solve_vertex_cover(inst: VertexCover) -> Optional[Set[Hashable]]:
    cover = exact_vertex_cover(inst.edges)
    return cover if len(cover) <= inst.k else None


def solve_independent_set(inst: IndependentSet) -> Optional[Set[Hashable]]:
    # the complement of a minimum cover is a maximum independent set
    s = set(inst.vertices) - exact_vertex_cover(inst.edges)
    return s if len(s) >= inst.k else None


def solve_clique(inst: Clique) -> Optional[Set[Hashable]]:
    # a clique in G is an independent set in G's complement
    return solve_independent_set(IndependentSet(
        inst.vertices, complement_edges(inst.vertices, inst.edges), inst.k))


def solve_subset_sum(inst: SubsetSum) -> Optional[List[int]]:
    return subset_sum_backtracking(inst.numbers, inst.target)


SOLVERS: Dict[type, Callable[[Any], Any]] = {
    ThreeSAT: solve_3sat, VertexCover: solve_vertex_cover,
    IndependentSet: solve_independent_set, Clique: solve_clique,
    SubsetSum: solve_subset_sum,
}
VERIFIERS: Dict[type, Callable[[Any, Any], bool]] = {
    ThreeSAT: verify_3sat, VertexCover: verify_vertex_cover,
    IndependentSet: verify_independent_set, Clique: verify_clique,
    SubsetSum: verify_subset_sum,
}


# ---------------------------------------------------------------------------
# Reduction framework
# ---------------------------------------------------------------------------

@dataclass
class ReductionStep:
    title: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.title}\n{self.detail}" if self.detail else self.title


class Reduction(ABC):
    """
    Base class. Subclasses implement:

        _reduce(source) -> target        the polynomial-time map f
        _recover(solution) -> solution   target solution -> source solution
        _forward(solution) -> solution   source solution -> target solution

    and call self._step(...) to record each stage. reduce() keeps the
    last source/target pair so recover() and forward() can use the
    mapping it built.
    """
    source_problem = ""
    target_problem = ""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.steps: List[ReductionStep] = []
        self.source: Any = None
        self.target: Any = None

    @property
    def name(self) -> str:
        return f"{self.source_problem} → {self.target_problem}"

    def reduce(self, source: Any) -> Any:
        self.steps = []
        self.source = source
        self.target = self._reduce(source)
        return self.target

    def recover(self, target_solution: Any) -> Any:
        """Map a solution of the target instance back to the source."""
        self._require_reduced()
        return self._recover(target_solution)

    def forward(self, source_solution: Any) -> Any:
        """Map a solution of the source instance to the target."""
        self._require_reduced()
        return self._forward(source_solution)

    def explain(self) -> str:
        lines = [f"Reduction: {self.name}"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  Step {i}. {step.title}")
            lines.extend("      " + l for l in step.detail.splitlines())
        return "\n".join(lines)

    def _step(self, title: str, detail: str = "") -> None:
        step = ReductionStep(title, detail)
        self.steps.append(step)
        logger.info("[%s] %s", self.name, str(step).replace("\n", " | "))
        if self.verbose:
            print(f"  Step {len(self.steps)}. {title}")
            for line in detail.splitlines():
                print("      " + line)

    def _require_reduced(self) -> None:
        if self.target is None:
            raise RuntimeError("call reduce() first")

    @abstractmethod
    def _reduce(self, source: Any) -> Any: ...

    @abstractmethod
    def _recover(self, solution: Any) -> Any: ...

    @abstractmethod
    def _forward(self, solution: Any) -> Any: ...


def _lit_name(lit: int) -> str:
    return f"x{lit}" if lit > 0 else f"¬x{-lit}"


def _preview(items: Sequence[Any], limit: int = 8) -> str:
    shown = ", ".join(str(x) for x in items[:limit])
    return shown + (f", … ({len(items)} total)" if len(items) > limit else "")


class ThreeSatToVertexCover(Reduction):
    """
    3-SAT ≤p Vertex Cover (Garey & Johnson).

    Variable gadget: an edge x_i — ¬x_i. Any cover takes at least one
    end; taking exactly one is choosing a truth value.

    Clause gadget: a clique on the clause's s literal slots (a triangle
    for 3 literals). Any cover takes at least s − 1 of them; the slot
    left out must have its edge to the matching literal vertex covered
    from the other side — i.e. that literal must be true.

    With k = n + Σ(s_j − 1) (= n + 2m for 3-literal clauses) there is no
    room for more than the minimum in any gadget, so a cover of size k
    exists exactly when the formula is satisfiable.

    Size: 2n + Σs_j vertices, n + Σ(s_j(s_j−1)/2 + s_j) edges — linear.
    """
    source_problem = "3-SAT"
    target_problem = "Vertex Cover"

    def _reduce(self, phi: ThreeSAT) -> VertexCover:
        n, clauses = phi.num_vars, phi.clauses
        self._step("Input formula",
                   f"{format_cnf(clauses)}\n"
                   f"n = {n} variables, m = {len(clauses)} clauses")

        vertices: List[str] = []
        edges: List[Edge] = []
        for i in range(1, n + 1):
            vertices += [f"x{i}", f"¬x{i}"]
            edges.append((f"x{i}", f"¬x{i}"))
        self._step("Variable gadgets: one edge x_i — ¬x_i per variable",
                   f"{n} edges: {_preview([f'x{i}—¬x{i}' for i in range(1, n + 1)])}\n"
                   "A cover must contain x_i or ¬x_i; the one it contains "
                   "is the literal set TRUE.")

        self.slot_literal: Dict[str, int] = {}
        self.clause_slots: List[List[str]] = []
        gadget_edges = []
        for j, clause in enumerate(clauses, 1):
            slots = [f"C{j}{'abc'[p]}" for p in range(len(clause))]
            self.clause_slots.append(slots)
            for slot, lit in zip(slots, clause):
                self.slot_literal[slot] = lit
            vertices += slots
            gadget_edges += [(slots[a], slots[b])
                             for a in range(len(slots))
                             for b in range(a + 1, len(slots))]
        edges += gadget_edges
        self._step("Clause gadgets: a triangle of literal slots per clause",
                   _preview([f"C{j}: " + ", ".join(
                       f"{s}={_lit_name(self.slot_literal[s])}"
                       for s in slots)
                       for j, slots in enumerate(self.clause_slots, 1)], 4)
                   + f"\n{len(gadget_edges)} triangle edges. A cover needs "
                   "all but one slot of each triangle.")

        links = [(slot, _lit_name(lit))
                 for slot, lit in self.slot_literal.items()]
        edges += links
        self._step("Connect each slot to the vertex of its literal",
                   f"{len(links)} edges: "
                   + _preview([f"{a}—{b}" for a, b in links])
                   + "\nThe uncovered slot's link must be covered by its "
                   "literal vertex, so that literal is TRUE.")

        k = n + sum(len(c) - 1 for c in clauses)
        self._step(f"Set k = n + Σ(|C_j| − 1) = {k}",
                   "Exactly one vertex per variable gadget and |C_j| − 1 "
                   "per clause gadget: no slack anywhere.")
        self._step("Result",
                   f"|V| = {len(vertices)}, |E| = {len(edges)}, k = {k} — "
                   "built in O(n + m) time.\n"
                   "Formula satisfiable ⟺ graph has a vertex cover of "
                   f"size ≤ {k}.")
        return VertexCover(vertices, edges, k)

    def _recover(self, cover: Set[Hashable]) -> Assignment:
        # Each variable gadget holds exactly one end of a size-k cover
        return {i: f"x{i}" in cover
                for i in range(1, self.source.num_vars + 1)}

    def _forward(self, assignment: Assignment) -> Set[str]:
        cover = {f"x{i}" if assignment[i] else f"¬x{i}"
                 for i in range(1, self.source.num_vars + 1)}
        for slots in self.clause_slots:
            # leave out one slot whose literal is true (if any)
            free = next((s for s in slots
                         if literal_true(self.slot_literal[s], assignment)),
                        slots[0])
            cover.update(s for s in slots if s != free)
        return cover


class VertexCoverToIndependentSet(Reduction):
    """
    Vertex Cover ≤p Independent Set.

    C is a vertex cover ⟺ V − C is an independent set: an edge with
    both ends outside C is exactly an edge C fails to cover. So a cover
    of size ≤ k exists ⟺ an independent set of size ≥ |V| − k exists.
    Same graph; O(1) work beyond copying it.
    """
    source_problem = "Vertex Cover"
    target_problem = "Independent Set"

    def _reduce(self, vc: VertexCover) -> IndependentSet:
        k = len(vc.vertices) - vc.k
        self._step("Keep the same graph",
                   f"|V| = {len(vc.vertices)}, |E| = {len(vc.edges)}")
        self._step(f"Set k' = |V| − k = {len(vc.vertices)} − {vc.k} = {k}",
                   "C covers every edge ⟺ no edge has both ends in V − C.")
        return IndependentSet(list(vc.vertices), list(vc.edges), k)

    def _recover(self, s: Set[Hashable]) -> Set[Hashable]:
        return set(self.source.vertices) - set(s)

    def _forward(self, cover: Set[Hashable]) -> Set[Hashable]:
        return set(self.source.vertices) - set(cover)


class IndependentSetToClique(Reduction):
    """
    Independent Set ≤p Clique.

    S has no edges in G ⟺ S has every edge in G's complement. Same k.
    Building the complement is O(V²).
    """
    source_problem = "Independent Set"
    target_problem = "Clique"

    def _reduce(self, ind: IndependentSet) -> Clique:
        comp = complement_edges(ind.vertices, ind.edges)
        total = len(ind.vertices) * (len(ind.vertices) - 1) // 2
        self._step("Take the complement graph",
                   f"{len(ind.edges)} edges become {len(comp)} "
                   f"(of {total} possible pairs).")
        self._step(f"Keep k = {ind.k}",
                   "No two vertices of S adjacent in G ⟺ every two "
                   "adjacent in the complement.")
        return Clique(list(ind.vertices), comp, ind.k)

    def _recover(self, s: Set[Hashable]) -> Set[Hashable]:
        return set(s)

    def _forward(self, s: Set[Hashable]) -> Set[Hashable]:
        return set(s)


class ThreeSatToSubsetSum(Reduction):
    """
    3-SAT ≤p Subset Sum (Sipser, Theorem 7.56, generalized to clauses
    of 1–3 literals).

    Numbers are written in decimal with one digit column per variable
    and one per clause:

        y_i   1 in column x_i, and 1 in column C_j for each occurrence
              of x_i in clause j        (choosing y_i = setting x_i TRUE)
        z_i   the same for ¬x_i         (choosing z_i = setting x_i FALSE)
        slack |C_j| − 1 numbers with a single 1 in column C_j
        target  1 in every variable column, |C_j| in clause column j

    No column can ever sum past 2·3 − 1 = 5, so there are no carries
    and each column must match independently. A variable column forces
    exactly one of y_i, z_i. A clause column receives t_j (the number
    of true literals) plus at most |C_j| − 1 from slack, so reaching
    |C_j| needs t_j ≥ 1: every clause satisfied.

    Size: 2n + Σ(|C_j| − 1) numbers of n + m digits each — polynomial.
    The numbers are exponentially LARGE (10^(n+m)), which is why the
    pseudo-polynomial O(n·T) DP gives no shortcut here.
    """
    source_problem = "3-SAT"
    target_problem = "Subset Sum"

    def _reduce(self, phi: ThreeSAT) -> SubsetSum:
        n, clauses = phi.num_vars, phi.clauses
        m = len(clauses)
        width = n + m
        self._step("Input formula",
                   f"{format_cnf(clauses)}\n"
                   f"Columns: x1..x{n} then C1..C{m} ({width} digits)")

        def number(digits: Dict[int, int]) -> int:
            # column c (0-based, left to right) has weight 10^(width-1-c)
            return sum(d * 10 ** (width - 1 - c) for c, d in digits.items())

        self.labels: List[str] = []
        self.numbers: List[int] = []
        for i in range(1, n + 1):
            for name, lit in ((f"y{i}", i), (f"z{i}", -i)):
                digits = {i - 1: 1}
                for j, clause in enumerate(clauses):
                    hits = clause.count(lit)
                    if hits:
                        digits[n + j] = hits
                self.labels.append(name)
                self.numbers.append(number(digits))
        self._step("Literal numbers y_i (x_i TRUE) and z_i (x_i FALSE)",
                   "\n".join(f"{l:>4} = {v:0{width}d}"
                             for l, v in zip(self.labels[:8],
                                             self.numbers[:8]))
                   + ("\n   …" if len(self.labels) > 8 else ""))

        slack_start = len(self.numbers)
        for j, clause in enumerate(clauses):
            for s in range(len(clause) - 1):
                self.labels.append(f"g{j + 1}.{s + 1}")
                self.numbers.append(number({n + j: 1}))
        self._step("Slack numbers: |C_j| − 1 per clause",
                   f"{len(self.numbers) - slack_start} numbers, each a "
                   "single 1 in its clause column; they top up a clause "
                   "with 1 or 2 true literals but cannot rescue one "
                   "with 0.")

        target = number({**{i: 1 for i in range(n)},
                         **{n + j: len(c) for j, c in enumerate(clauses)}})
        self._step(f"Target = {target:0{width}d}",
                   "1 per variable column (exactly one of y_i, z_i); "
                   "|C_j| per clause column (at least one true literal).\n"
                   f"{len(self.numbers)} numbers of {width} digits — "
                   "polynomial size, no carries possible.")
        return SubsetSum(list(self.numbers), target)

    def _recover(self, indices: Sequence[int]) -> Assignment:
        chosen = {self.labels[i] for i in indices}
        return {i: f"y{i}" in chosen
                for i in range(1, self.source.num_vars + 1)}

    def _forward(self, assignment: Assignment) -> List[int]:
        index = {label: i for i, label in enumerate(self.labels)}
        chosen = [index[f"y{i}" if assignment[i] else f"z{i}"]
                  for i in range(1, self.source.num_vars + 1)]
        for j, clause in enumerate(self.source.clauses, 1):
            true_lits = sum(literal_true(l, assignment) for l in clause)
            need = len(clause) - max(true_lits, 1)
            chosen += [index[f"g{j}.{s}"] for s in range(1, need + 1)]
        return sorted(chosen)


class ReductionChain(Reduction):
    """
    Compose reductions: if A ≤p B and B ≤p C then A ≤p C (polynomials
    compose into polynomials). Solutions map back through every link.
    """

    def __init__(self, reductions: Sequence[Reduction], verbose: bool = False):
        super().__init__(verbose)
        if not reductions:
            raise ValueError("need at least one reduction")
        self.reductions = list(reductions)
        self.source_problem = self.reductions[0].source_problem
        self.target_problem = self.reductions[-1].target_problem

    @property
    def name(self) -> str:
        return " → ".join([self.reductions[0].source_problem]
                          + [r.target_problem for r in self.reductions])

    def _reduce(self, source: Any) -> Any:
        instance = source
        for r in self.reductions:
            r.verbose = self.verbose
            if self.verbose:
                print(f"  -- {r.name} --")
            instance = r.reduce(instance)
            self.steps += [ReductionStep(f"[{r.name}] {s.title}", s.detail)
                           for s in r.steps]
        return instance

    def _recover(self, solution: Any) -> Any:
        for r in reversed(self.reductions):
            solution = r.recover(solution)
        return solution

    def _forward(self, solution: Any) -> Any:
        for r in self.reductions:
            solution = r.forward(solution)
        return solution


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ReductionCheck:
    """Outcome of check_reduction on one instance."""
    source_yes: bool
    target_yes: bool
    recovered_valid: Optional[bool] = None   # None when the answer is "no"
    forwarded_valid: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.source_yes == self.target_yes
                and self.recovered_valid is not False
                and self.forwarded_valid is not False)


def check_reduction(reduction: Reduction, source: Any) -> ReductionCheck:
    """
    Validate one reduction on one instance:

    1. Solve the source and the reduced target exactly; the yes/no
       answers must agree (the ⟺ in the definition).
    2. If yes: map the target's solution back and verify it solves the
       source, and map the source's solution forward and verify it
       solves the target.
    """
    target = reduction.reduce(source)
    src_sol = SOLVERS[type(source)](source)
    tgt_sol = SOLVERS[type(target)](target)
    check = ReductionCheck(src_sol is not None, tgt_sol is not None)
    if tgt_sol is not None:
        back = reduction.recover(tgt_sol)
        check.recovered_valid = VERIFIERS[type(source)](source, back)
    if src_sol is not None:
        fwd = reduction.forward(src_sol)
        check.forwarded_valid = VERIFIERS[type(target)](target, fwd)
    if not check.ok:
        check.notes.append(f"{reduction.name} failed on {source}")
    return check


# ---------------------------------------------------------------------------
# Complexity-class catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProblemInfo:
    name: str
    complexity_class: str
    best_known: str
    note: str


CATALOG: List[ProblemInfo] = [
    ProblemInfo("Sorting", "P", "O(n log n)", "Week 2"),
    ProblemInfo("Shortest paths", "P", "O((V + E) log V)",
                "Dijkstra, Week 4"),
    ProblemInfo("Minimum spanning tree", "P", "O(V²) dense",
                "Prim; used by mst_tour"),
    ProblemInfo("2-SAT", "P", "O(n + m)",
                "implication-graph SCCs (two_sat)"),
    ProblemInfo("3-SAT", "NP-complete", "O(1.3^n) best known",
                "Cook–Levin: the first NP-complete problem"),
    ProblemInfo("Vertex Cover", "NP-complete", "O(1.28^k + kn) (FPT)",
                "3-SAT ≤p VC; 2-approximation in O(E)"),
    ProblemInfo("Independent Set", "NP-complete", "O(1.2^n)",
                "VC ≤p IS via complement set"),
    ProblemInfo("Clique", "NP-complete", "O(1.2^n)",
                "IS ≤p Clique via complement graph"),
    ProblemInfo("Subset Sum", "NP-complete", "O(n·T) or O(2^(n/2))",
                "pseudo-polynomial DP; 3-SAT ≤p Subset Sum"),
    ProblemInfo("TSP (decision: tour ≤ L?)", "NP-complete", "O(n²·2^n)",
                "Held–Karp, Week 6"),
    ProblemInfo("TSP (find the shortest tour)", "NP-hard", "O(n²·2^n)",
                "an optimization problem, so not in NP itself"),
    ProblemInfo("Halting problem", "Undecidable (NP-hard)", "none",
                "no algorithm at all; NP-hard but not in NP"),
]


def classify(name: str) -> ProblemInfo:
    """Look up a problem in CATALOG by (case-insensitive) name prefix."""
    key = name.lower()
    for info in CATALOG:
        if info.name.lower().startswith(key):
            return info
    raise KeyError(f"unknown problem {name!r}")


def format_catalog() -> str:
    rows = [f"{'Problem':<30} {'Class':<22} {'Best known':<22} Note"]
    rows += [f"{p.name:<30} {p.complexity_class:<22} {p.best_known:<22} "
             f"{p.note}" for p in CATALOG]
    return "\n".join(rows)
