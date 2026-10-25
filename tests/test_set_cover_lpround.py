"""Test suite for Set Cover: LP relaxation, deterministic and randomized
rounding, greedy, exact branch and bound, and instance generators."""

import itertools
import math
import random

import pytest

from src.approximation.set_cover_lpround import (
    SetCoverInstance, cover_cost, default_rounds, deterministic_rounding,
    exact_set_cover, greedy_bad_instance, greedy_set_cover, is_cover,
    lp_relaxation, random_set_cover, randomized_rounding,
)
from src.utils.probability_analysis import harmonic

SMALL = SetCoverInstance(5, [{0, 1, 2}, {2, 3}, {3, 4}, {0, 4}, {1}],
                         [3, 1, 1, 2, 1])


def brute_force(inst):
    best = math.inf
    for r in range(1, len(inst.sets) + 1):
        for combo in itertools.combinations(range(len(inst.sets)), r):
            if is_cover(inst, combo):
                best = min(best, cover_cost(inst, combo))
    return best


def random_small(seed):
    rng = random.Random(seed)
    return random_set_cover(rng.randint(3, 12), rng.randint(2, 10),
                            rng.uniform(0.15, 0.5), seed=seed)


class TestInstance:

    def test_validation(self):
        with pytest.raises(ValueError):
            SetCoverInstance(3, [{0, 1}], [1])             # 2 uncovered
        with pytest.raises(ValueError):
            SetCoverInstance(1, [{0}], [0])                # zero cost
        with pytest.raises(ValueError):
            SetCoverInstance(1, [{0}], [1, 2])             # cost count

    def test_frequency(self):
        assert SMALL.max_frequency == 2

    def test_generator_covers_universe(self):
        for seed in range(20):
            inst = random_set_cover(30, 10, 0.05, seed=seed)
            assert is_cover(inst, range(len(inst.sets)))
            assert all(inst.sets)


class TestExactAndGreedy:

    def test_small_known(self):
        assert exact_set_cover(SMALL).cost == brute_force(SMALL) == 4

    @pytest.mark.parametrize("seed", range(40))
    def test_exact_matches_brute_force(self, seed):
        inst = random_small(seed)
        cover = exact_set_cover(inst)
        assert is_cover(inst, cover.chosen)
        assert cover.cost == pytest.approx(brute_force(inst))

    @pytest.mark.parametrize("seed", range(40))
    def test_greedy_within_harmonic_bound(self, seed):
        inst = random_small(seed)
        cover = greedy_set_cover(inst)
        assert is_cover(inst, cover.chosen)
        assert cover.cost <= harmonic(inst.num_elements) * brute_force(
            inst) + 1e-9

    @pytest.mark.parametrize("k", [2, 3, 5, 7])
    def test_greedy_bad_family(self, k):
        inst = greedy_bad_instance(k)
        assert inst.num_elements == 2 * (2 ** k - 1)
        assert greedy_set_cover(inst).cost == k
        assert exact_set_cover(inst).cost == 2

    def test_bad_family_needs_k_at_least_2(self):
        with pytest.raises(ValueError):
            greedy_bad_instance(1)


class TestLP:

    @pytest.mark.parametrize("seed", range(30))
    def test_lp_is_lower_bound_and_feasible(self, seed):
        inst = random_small(seed)
        x, lp = lp_relaxation(inst)
        assert lp <= brute_force(inst) + 1e-9
        for e in range(inst.num_elements):
            assert sum(x[j] for j, s in enumerate(inst.sets)
                       if e in s) >= 1 - 1e-7
        assert all(v >= -1e-12 for v in x)

    def test_integrality_gap_example(self):
        # three elements, three pair-sets: LP picks each at 1/2 = 1.5
        inst = SetCoverInstance(3, [{0, 1}, {1, 2}, {0, 2}], [1, 1, 1])
        x, lp = lp_relaxation(inst)
        assert lp == pytest.approx(1.5)
        assert exact_set_cover(inst).cost == 2


class TestRounding:

    @pytest.mark.parametrize("seed", range(40))
    def test_deterministic_bound(self, seed):
        inst = random_small(seed)
        x, lp = lp_relaxation(inst)
        cover = deterministic_rounding(inst, x)
        assert is_cover(inst, cover.chosen)
        assert cover.cost <= inst.max_frequency * lp + 1e-7

    def test_deterministic_solves_lp_itself(self):
        assert is_cover(SMALL, deterministic_rounding(SMALL).chosen)

    @pytest.mark.parametrize("seed", range(40))
    def test_randomized_always_feasible(self, seed):
        inst = random_small(seed)
        cover = randomized_rounding(inst, rng=random.Random(seed))
        assert is_cover(inst, cover.chosen)
        assert cover.cost == pytest.approx(cover_cost(inst, cover.chosen))

    def test_randomized_seeded(self):
        inst = random_set_cover(40, 25, 0.1, seed=3)
        x, _ = lp_relaxation(inst)
        a = randomized_rounding(inst, x, rng=random.Random(5))
        b = randomized_rounding(inst, x, rng=random.Random(5))
        assert a.chosen == b.chosen

    def test_expected_cost_within_rounds_times_lp(self):
        inst = random_set_cover(60, 36, 0.1, seed=2)
        x, lp = lp_relaxation(inst)
        t = default_rounds(inst.num_elements)
        costs = [randomized_rounding(inst, x, rng=random.Random(s)).cost
                 for s in range(200)]
        # E[cost] ≤ t·LP* plus rare repairs
        assert sum(costs) / len(costs) <= t * lp * 1.05

    def test_one_round_needs_repairs(self):
        inst = random_set_cover(80, 48, 0.1, seed=4)
        x, _ = lp_relaxation(inst)
        repairs = sum(randomized_rounding(inst, x, rounds=1,
                                          rng=random.Random(s)).repaired
                      for s in range(50))
        assert repairs > 0

    def test_default_rounds(self):
        assert default_rounds(1) == default_rounds(2) == 2
        assert default_rounds(100) == math.ceil(math.log(100)) + 1
