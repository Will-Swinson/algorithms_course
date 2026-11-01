"""
File formats for the command-line interface.

Graphs
    DIMACS max-flow (.max)      c comment
                                p max <nodes> <arcs>
                                n <id> s    /   n <id> t
                                a <u> <v> <capacity>
    DIMACS min-cost (.min)      p min <nodes> <arcs>
                                n <id> <supply>     (> 0 source, < 0 sink)
                                a <u> <v> <low> <capacity> <cost>
                                (lower bounds must be 0)
    edge list (anything else)   u v capacity [cost]   one arc per line,
                                # comments; node ids are 0-based and
                                source/sink are given separately

DIMACS ids are 1-based and converted to 0-based. A .min file with
several supply / demand nodes gets a super-source and super-sink whose
arcs carry exactly those supplies.

Matching input: one "left right" pair per line (labels are strings).
Images: anything Pillow opens (PNG, JPEG, PGM, ...), as grayscale.
Schedules: JSON, described in main.py.
"""
import json
from typing import List, Optional, Tuple

import numpy as np

from src.flow.core.network import FlowNetwork


def _lines(path: str) -> List[List[str]]:
    with open(path) as fh:
        return [line.split() for line in fh
                if line.strip() and not line.lstrip().startswith(("c ", "#"))
                and line.strip() != "c"]


def read_dimacs(path: str) -> Tuple[FlowNetwork, int, int]:
    """(network, s, t) from a DIMACS .max or .min file."""
    rows = _lines(path)
    problem = next((r for r in rows if r[0] == "p"), None)
    if problem is None or problem[1] not in ("max", "min"):
        raise ValueError(f"{path}: missing 'p max' or 'p min' line")
    kind, n = problem[1], int(problem[2])
    net = FlowNetwork(n)
    s = t = None
    supplies = {}
    for row in rows:
        tag = row[0]
        if tag == "n":
            node = int(row[1]) - 1
            if kind == "max":
                if row[2] == "s":
                    s = node
                elif row[2] == "t":
                    t = node
                else:
                    raise ValueError(f"{path}: bad node line {' '.join(row)}")
            else:
                supplies[node] = float(row[2])
        elif tag == "a":
            u, v = int(row[1]) - 1, int(row[2]) - 1
            if kind == "max":
                net.add_edge(u, v, _number(row[3]))
            else:
                if _number(row[3]) != 0:
                    raise ValueError(f"{path}: arc lower bounds are not supported")
                net.add_edge(u, v, _number(row[4]), _number(row[5]))
    if kind == "min":
        sources = [v for v, b in supplies.items() if b > 0]
        sinks = [v for v, b in supplies.items() if b < 0]
        if len(sources) == 1 and len(sinks) == 1:
            s, t = sources[0], sinks[0]
        elif sources and sinks:
            s, t = net.add_node(), net.add_node()
            for v in sources:
                net.add_edge(s, v, _number(supplies[v]))
            for v in sinks:
                net.add_edge(v, t, _number(-supplies[v]))
    if s is None or t is None:
        raise ValueError(f"{path}: source and sink not specified")
    return net, s, t


def read_edge_list(path: str, source: int, sink: int) -> Tuple[FlowNetwork, int, int]:
    """(network, source, sink) from 'u v capacity [cost]' lines."""
    rows = _lines(path)
    arcs = []
    for row in rows:
        if len(row) not in (3, 4):
            raise ValueError(f"{path}: expected 'u v capacity [cost]', got "
                             f"{' '.join(row)}")
        arcs.append((int(row[0]), int(row[1]), _number(row[2]),
                     _number(row[3]) if len(row) == 4 else 0))
    n = 1 + max([source, sink] + [max(a[0], a[1]) for a in arcs])
    return FlowNetwork.from_edges(n, arcs), source, sink


def read_graph(path: str, source: Optional[int] = None,
               sink: Optional[int] = None) -> Tuple[FlowNetwork, int, int]:
    """DIMACS if the file has a 'p' line, else an edge list (which then
    needs source and sink)."""
    if any(r[0] == "p" for r in _lines(path)):
        net, s, t = read_dimacs(path)
        return net, s if source is None else source, t if sink is None else sink
    if source is None or sink is None:
        raise ValueError("edge-list input needs --source and --sink")
    return read_edge_list(path, source, sink)


def write_dimacs(net: FlowNetwork, s: int, t: int, path: str,
                 with_costs: bool = False) -> None:
    """Write a .max (or, with_costs, single-source .min) file."""
    with open(path, "w") as fh:
        if with_costs:
            flow = sum(net.base[e] for e in net.adj[s] if not e & 1)
            fh.write(f"p min {net.n} {net.num_edges}\n")
            fh.write(f"n {s + 1} {_fmt(flow)}\nn {t + 1} {_fmt(-flow)}\n")
            for e in net.edges():
                fh.write(f"a {e.u + 1} {e.v + 1} 0 {_fmt(e.capacity)} "
                         f"{_fmt(e.cost)}\n")
        else:
            fh.write(f"p max {net.n} {net.num_edges}\n")
            fh.write(f"n {s + 1} s\nn {t + 1} t\n")
            for e in net.edges():
                fh.write(f"a {e.u + 1} {e.v + 1} {_fmt(e.capacity)}\n")


def read_pairs(path: str) -> List[Tuple[str, str]]:
    """Bipartite edges from 'left right' lines."""
    pairs = []
    for row in _lines(path):
        if len(row) != 2:
            raise ValueError(f"{path}: expected 'left right', got {' '.join(row)}")
        pairs.append((row[0], row[1]))
    return pairs


def read_image(path: str) -> np.ndarray:
    """Grayscale float image in [0, 1]."""
    from PIL import Image
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=float) / 255.0


def save_image(array: np.ndarray, path: str) -> None:
    from PIL import Image
    Image.fromarray(np.clip(np.asarray(array) * 255, 0, 255)
                    .astype(np.uint8)).save(path)


def read_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _number(text):
    value = float(text)
    return int(value) if value == int(value) else value


def _fmt(x) -> str:
    return str(int(x)) if x == int(x) else repr(x)
