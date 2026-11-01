"""Test suite for the flow solver's file readers and command-line
interface (run in-process through main(argv))."""

import json
import os

import numpy as np
import pytest

from src.flow.core import push_relabel
from src.flow.main import main
from src.flow.utils import io
from src.flow.utils.generators import sparse_network

INPUTS = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                      "examples", "flow_inputs")


def inp(name):
    return os.path.join(INPUTS, name)


class TestReaders:

    def test_dimacs_max(self):
        net, s, t = io.read_dimacs(inp("clrs.max"))
        assert (net.n, net.num_edges, s, t) == (6, 9, 0, 5)
        assert push_relabel(net, s, t).value == 23

    def test_dimacs_min_super_terminals(self):
        net, s, t = io.read_dimacs(inp("shipping.min"))
        assert net.n == 8 and (s, t) == (6, 7)
        assert sorted(net.base[e] for e in net.adj[s] if not e & 1) == [5, 10]

    def test_dimacs_min_single_terminals(self, tmp_path):
        path = tmp_path / "one.min"
        path.write_text("p min 2 1\nn 1 4\nn 2 -4\na 1 2 0 5 3\n")
        net, s, t = io.read_dimacs(str(path))
        assert (s, t) == (0, 1) and net.cost[0] == 3

    def test_round_trip(self, tmp_path):
        net, s, t = sparse_network(50, cost_range=(1, 9), seed=1)
        for with_costs, name in ((False, "g.max"), (True, "g.min")):
            path = str(tmp_path / name)
            io.write_dimacs(net, s, t, path, with_costs=with_costs)
            back, s2, t2 = io.read_graph(path)
            assert (s2, t2) == (s, t)
            assert [(e.u, e.v, e.capacity) for e in back.edges()] == \
                [(e.u, e.v, e.capacity) for e in net.edges()][:back.num_edges]

    def test_edge_list(self):
        net, s, t = io.read_graph(inp("edges.txt"), 0, 3)
        assert net.num_edges == 5 and push_relabel(net, s, t).value == 5

    def test_edge_list_needs_terminals(self):
        with pytest.raises(ValueError):
            io.read_graph(inp("edges.txt"))

    @pytest.mark.parametrize("text", ["p max 2 1\na 1 2 3\n",
                                      "p foo 2 1\n",
                                      "p min 2 1\nn 1 1\nn 2 -1\na 1 2 1 5 3\n",
                                      "p max 2 1\nn 1 q\n"])
    def test_bad_dimacs(self, tmp_path, text):
        path = tmp_path / "bad.max"
        path.write_text(text)
        with pytest.raises(ValueError):
            io.read_dimacs(str(path))

    def test_bad_edge_list(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("0 1\n")
        with pytest.raises(ValueError):
            io.read_edge_list(str(path), 0, 1)

    def test_pairs(self, tmp_path):
        assert io.read_pairs(inp("matching.txt"))[0] == ("alice", "backend")
        path = tmp_path / "p.txt"
        path.write_text("a b c\n")
        with pytest.raises(ValueError):
            io.read_pairs(str(path))

    def test_image_round_trip(self, tmp_path):
        image = np.linspace(0, 1, 30).reshape(5, 6)
        path = str(tmp_path / "i.png")
        io.save_image(image, path)
        assert np.allclose(io.read_image(path), image, atol=1 / 255)


class TestCLI:

    def run(self, capsys, *argv):
        code = main(list(argv))
        return code, capsys.readouterr()

    @pytest.mark.parametrize("flag", ["--ek", "--pr", "--scaling", "--mcf"])
    def test_graph_solvers(self, capsys, flag):
        code, out = self.run(capsys, flag, "-i", inp("clrs.max"), "--validate")
        assert code == 0
        assert "max flow value: 23" in out.out and "valid" in out.out

    def test_mcf_cost(self, capsys):
        code, out = self.run(capsys, "--mcf", "-i", inp("shipping.min"),
                             "--show-flow")
        assert code == 0 and "min cost: 67" in out.out and "6 -> 0: 10" in out.out

    def test_highest_label_and_edge_list(self, capsys):
        code, out = self.run(capsys, "--pr", "--variant", "highest", "-i",
                             inp("edges.txt"), "-s", "0", "-t", "3")
        assert code == 0 and "push-relabel (highest)" in out.out
        assert "max flow value: 5" in out.out

    def test_random(self, capsys):
        code, out = self.run(capsys, "--random", "500", "--validate")
        assert code == 0 and "500 nodes" in out.out
        code, out = self.run(capsys, "--random", "60", "--density", "0.3",
                             "--mcf", "--validate")
        assert code == 0 and "min cost" in out.out

    def test_plot_and_animate(self, capsys, tmp_path):
        png, gif = tmp_path / "n.png", tmp_path / "a.gif"
        code, out = self.run(capsys, "--ek", "-i", inp("clrs.max"),
                             "--plot", str(png), "--animate", str(gif))
        assert code == 0 and png.exists() and gif.exists()

    def test_large_graph_skips_drawing(self, capsys, tmp_path):
        code, out = self.run(capsys, "--random", "200", "--plot",
                             str(tmp_path / "x.png"), "--animate",
                             str(tmp_path / "x.gif"))
        assert code == 0 and "skipping" in out.out
        assert not (tmp_path / "x.png").exists()

    def test_match(self, capsys, tmp_path):
        png = tmp_path / "m.png"
        code, out = self.run(capsys, "--match", inp("matching.txt"),
                             "--plot", str(png))
        assert code == 0 and "maximum matching: 4 pairs" in out.out
        assert png.exists()

    def test_segment_file_and_synthetic(self, capsys, tmp_path):
        code, out = self.run(capsys, "--segment", inp("shapes.png"), "--plot",
                             str(tmp_path / "s.png"))
        assert code == 0 and "foreground pixels" in out.out
        code, out = self.run(capsys, "--segment", "--size", "24", "--ek")
        assert code == 0 and "IoU" in out.out

    @pytest.mark.parametrize("name,expect", [
        ("jobs.json", "FEASIBLE"), ("staffing.json", "total cost: 5"),
        ("machines.json", "assigned 15 of 15"),
        ("tournament.json", "Philadelphia ELIMINATED")])
    def test_schedule(self, capsys, name, expect):
        code, out = self.run(capsys, "--schedule", inp(name))
        assert code == 0 and expect in " ".join(out.out.split())

    def test_schedule_gantt(self, capsys, tmp_path):
        png = tmp_path / "g.png"
        code, _ = self.run(capsys, "--schedule", inp("jobs.json"), "--plot",
                           str(png))
        assert code == 0 and png.exists()

    def test_errors(self, capsys, tmp_path):
        assert self.run(capsys)[0] == 2                          # no input
        assert self.run(capsys, "-i", "missing.max")[0] == 2
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"type": "lottery"}))
        assert self.run(capsys, "--schedule", str(bad))[0] == 2

    def test_mutually_exclusive_solvers(self):
        with pytest.raises(SystemExit):
            main(["--ek", "--pr", "--random", "10"])
