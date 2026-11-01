"""Flow network core: the residual-graph data structure and the
max-flow / min-cost-flow solvers that run on it.

get_solver(name) maps the CLI/application solver names to functions
with the common signature solver(net, s, t, **kwargs) -> FlowResult.
"""
from functools import partial
from typing import Callable, Dict

from src.flow.core.network import Edge, FlowNetwork, FlowResult, ResidualArc
from src.flow.core.edmonds_karp import capacity_scaling_max_flow, edmonds_karp
from src.flow.core.push_relabel import push_relabel
from src.flow.core.min_cost_flow import min_cost_flow, min_cost_max_flow

SOLVERS: Dict[str, Callable[..., FlowResult]] = {
    "ek": edmonds_karp,
    "scaling": capacity_scaling_max_flow,
    "pr": push_relabel,
    "pr-fifo": partial(push_relabel, variant="fifo"),
    "pr-highest": partial(push_relabel, variant="highest"),
    "mcf": min_cost_max_flow,
}


def get_solver(name: str) -> Callable[..., FlowResult]:
    try:
        return SOLVERS[name]
    except KeyError:
        raise ValueError(f"unknown solver {name!r}; choose from "
                         f"{', '.join(SOLVERS)}") from None


def max_flow(net: FlowNetwork, s: int, t: int, method: str = "pr",
             **kwargs) -> FlowResult:
    """Maximum s-t flow with the named solver (see SOLVERS)."""
    return get_solver(method)(net, s, t, **kwargs)
