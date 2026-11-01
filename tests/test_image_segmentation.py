"""Test suite for graph-cut image segmentation."""

import itertools

import numpy as np
import pytest

from src.flow.applications.image_seg import (
    as_grayscale, build_segmentation_network, labeling_energy,
    segment_image, segmentation_scores, threshold_segment, two_means,
)
from src.flow.utils.generators import synthetic_image


class TestHelpers:

    def test_as_grayscale(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0] = (255, 255, 255)
        gray = as_grayscale(rgb)
        assert gray.shape == (2, 2) and gray[0, 0] == 1.0 and gray[1, 1] == 0
        with pytest.raises(ValueError):
            as_grayscale(np.zeros(4))

    def test_two_means(self):
        image = np.array([[0.1, 0.12, 0.9], [0.88, 0.11, 0.92]])
        bright, dark = two_means(image)
        assert bright == pytest.approx(0.9, abs=0.02)
        assert dark == pytest.approx(0.11, abs=0.02)
        assert two_means(np.full((2, 2), 0.5)) == (0.5, 0.5)

    def test_scores(self):
        truth = np.array([[True, False], [True, False]])
        assert segmentation_scores(truth, truth) == {"accuracy": 1.0, "iou": 1.0}
        s = segmentation_scores(np.array([[True, True], [False, False]]), truth)
        assert s["accuracy"] == 0.5 and s["iou"] == pytest.approx(1 / 3)
        empty = np.zeros((2, 2), bool)
        assert segmentation_scores(empty, empty)["iou"] == 1.0


class TestOptimality:

    @pytest.mark.parametrize("seed", range(8))
    def test_min_cut_is_minimum_energy(self, seed):
        image, _ = synthetic_image(3, 3, noise=0.35, seed=seed)
        net, s, t, offset = build_segmentation_network(image, 0.75, 0.25)
        brute = min(labeling_energy(net, s, np.array(bits, bool).reshape(3, 3),
                                    offset=offset)
                    for bits in itertools.product((0, 1), repeat=9))
        result = segment_image(image, 0.75, 0.25)
        assert result.energy == pytest.approx(brute)
        assert labeling_energy(net, s, result.mask, offset=offset) == \
            pytest.approx(result.energy)

    @pytest.mark.parametrize("solver", ["ek", "pr", "pr-highest"])
    def test_solvers_agree(self, solver):
        image, _ = synthetic_image(16, 16, "shapes", noise=0.3, seed=1)
        ref = segment_image(image, solver="pr")
        assert segment_image(image, solver=solver).energy == \
            pytest.approx(ref.energy)

    def test_energy_beats_threshold_and_truth(self):
        image, truth = synthetic_image(32, 32, "shapes", noise=0.35, seed=2)
        result = segment_image(image)
        net, s, _, offset = build_segmentation_network(
            image, result.fg_mean, result.bg_mean)
        for other in (threshold_segment(image), truth):
            assert result.energy <= labeling_energy(net, s, other,
                                                    offset=offset) + 1e-9


class TestBehaviour:

    def test_clean_image_is_exact(self):
        image, truth = synthetic_image(32, 32, "circle", noise=0.0, seed=0)
        assert np.array_equal(segment_image(image).mask, truth)

    def test_beats_threshold_on_noise(self):
        image, truth = synthetic_image(64, 64, "shapes", noise=0.35, seed=3)
        cut = segmentation_scores(segment_image(image).mask, truth)["iou"]
        thr = segmentation_scores(threshold_segment(image), truth)["iou"]
        assert cut > thr + 0.15

    def test_zero_smoothness_is_pixelwise(self):
        image, _ = synthetic_image(20, 20, noise=0.3, seed=4)
        r = segment_image(image, 0.75, 0.25, smoothness=0.0)
        expected = (image - 0.75) ** 2 < (image - 0.25) ** 2
        ties = np.isclose((image - 0.75) ** 2, (image - 0.25) ** 2)
        assert np.array_equal(r.mask[~ties], expected[~ties])

    def test_more_smoothing_fewer_boundary_pixels(self):
        image, _ = synthetic_image(48, 48, "shapes", noise=0.4, seed=5)

        def boundary(mask):
            return int((mask[1:] != mask[:-1]).sum() +
                       (mask[:, 1:] != mask[:, :-1]).sum())
        assert boundary(segment_image(image, smoothness=8).mask) < \
            boundary(segment_image(image, smoothness=0.5).mask)

    def test_seeds_are_respected(self):
        image, _ = synthetic_image(20, 20, noise=0.2, seed=6)
        fg, bg = [(0, 0), (0, 1)], [(10, 10)]
        r = segment_image(image, fg_seeds=fg, bg_seeds=bg)
        assert r.mask[0, 0] and r.mask[0, 1] and not r.mask[10, 10]

    def test_seed_conflicts(self):
        image, _ = synthetic_image(5, 5, seed=0)
        with pytest.raises(ValueError):
            segment_image(image, fg_seeds=[(1, 1)], bg_seeds=[(1, 1)])
        with pytest.raises(ValueError):
            segment_image(image, fg_seeds=[(9, 9)])

    def test_dark_object_via_seed(self):
        image, truth = synthetic_image(32, 32, "circle", noise=0.1,
                                       foreground=0.2, background=0.8, seed=7)
        r = segment_image(image, fg_seeds=[(16, 16)])
        assert segmentation_scores(r.mask, truth)["iou"] > 0.9

    def test_result_metadata(self):
        image, _ = synthetic_image(10, 12, seed=8)
        r = segment_image(image)
        assert r.mask.shape == (10, 12) and r.num_nodes == 122
        assert r.fg_mean > r.bg_mean
