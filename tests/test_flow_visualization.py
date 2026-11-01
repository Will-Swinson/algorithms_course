"""Test suite for flow network drawings and algorithm animations
(rendered headless; checks files and frame contents, not pixels)."""

import os

import pytest

from src.flow.applications import bipartite_matching, schedule_jobs, segment_image
from src.flow.core import edmonds_karp, push_relabel
from src.flow.core.network import FlowNetwork
from src.flow.utils.generators import synthetic_image
from src.flow.visualization import (
    animate_edmonds_karp, animate_push_relabel, edmonds_karp_frames,
    layered_positions, plot_matching, plot_network, plot_schedule,
    plot_segmentation, push_relabel_frames, storyboard,
)

CLRS = [(0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (3, 2, 9),
        (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]


@pytest.fixture
def net():
    return FlowNetwork.from_edges(6, CLRS)


def nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 1_000


class TestLayout:

    def test_layers_follow_bfs(self, net):
        pos = layered_positions(net, 0, 5)
        assert pos[0][0] == 0
        assert pos[1][0] == pos[2][0] == 1
        assert pos[5][0] == max(x for x, _ in pos.values())

    def test_unreachable_nodes_placed(self):
        net = FlowNetwork.from_edges(4, [(0, 1, 1)])
        pos = layered_positions(net, 0, 3)
        assert set(pos) == {0, 1, 2, 3}


class TestStaticPlots:

    def test_plot_network(self, net, tmp_path):
        push_relabel(net, 0, 5)
        path = tmp_path / "net.png"
        fig = plot_network(net, 0, 5, save_path=str(path))
        assert nonempty(path)
        assert "23" in fig.axes[0].get_title()

    def test_plot_network_without_flow(self, net, tmp_path):
        path = tmp_path / "zero.png"
        plot_network(net, 0, 5, show_cut=False, save_path=str(path))
        assert nonempty(path)

    def test_plot_matching(self, tmp_path):
        edges = [("a", "x"), ("a", "y"), ("b", "x")]
        r = bipartite_matching("ab", "xy", edges)
        path = tmp_path / "m.png"
        plot_matching("ab", "xy", edges, r.pairs, save_path=str(path))
        assert nonempty(path)

    def test_plot_segmentation(self, tmp_path):
        image, truth = synthetic_image(24, 24, seed=1)
        path = tmp_path / "s.png"
        plot_segmentation(image, {"cut": segment_image(image).mask}, truth,
                          title="t", save_path=str(path))
        assert nonempty(path)

    def test_plot_schedule(self, tmp_path):
        r = schedule_jobs({"a": (2, 0, 3), "b": (1, 0, 2)}, 2)
        path = tmp_path / "g.png"
        plot_schedule(r, save_path=str(path))
        assert nonempty(path)


class TestFrames:

    def test_edmonds_karp_frames(self, net):
        frames = edmonds_karp_frames(net, 0, 5)
        assert len(frames) == edmonds_karp(net, 0, 5).iterations + 2
        assert frames[0]["path"] == [] and sum(frames[0]["flows"]) == 0
        assert frames[-1]["cut"]
        assert "23" in frames[-1]["title"]

    def test_push_relabel_frames(self, net):
        frames = push_relabel_frames(net, 0, 5)
        assert frames[0]["heights"][0] == 6
        assert any(f["node"] is not None for f in frames)    # a relabel
        assert any(f["edge"] for f in frames)                # a push
        assert "23" in frames[-1]["title"]


class TestAnimations:

    def test_gifs(self, net, tmp_path):
        ek, pr = tmp_path / "ek.gif", tmp_path / "pr.gif"
        animate_edmonds_karp(net, 0, 5, save_path=str(ek))
        animate_push_relabel(net, 0, 5, save_path=str(pr), max_frames=10)
        assert nonempty(ek) and nonempty(pr)
        with open(ek, "rb") as fh:
            assert fh.read(3) == b"GIF"

    def test_animation_object(self, net, tmp_path):
        anim = animate_edmonds_karp(net, 0, 5)
        anim.save(str(tmp_path / "x.gif"), writer="pillow")

    @pytest.mark.parametrize("algorithm", ["ek", "pr"])
    def test_storyboard(self, net, tmp_path, algorithm):
        path = tmp_path / f"{algorithm}.png"
        storyboard(net, 0, 5, algorithm, panels=4, ncols=2, save_path=str(path))
        assert nonempty(path)

    def test_storyboard_bad_algorithm(self, net):
        with pytest.raises(ValueError):
            storyboard(net, 0, 5, "mcf")
