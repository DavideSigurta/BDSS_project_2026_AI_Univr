"""Robustness radii against brute force and witnesses."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.toy_forest import build_toy_forest  # noqa: E402
from bdss import bdd, compose, oracle, robustness, transitions  # noqa: E402
from bdss.discretize import Disc  # noqa: E402
from bdss.frontier import SELECTORS  # noqa: E402


def _own(forest, disc, code):
    return oracle.hard_label(
        forest, disc.code_cells(code), [disc.tau_array(t) for t in forest.trees]
    )


def _flipped_label(forest, disc, code, bits):
    sample = list(disc.code_bits(code))
    for level in bits:
        sample[level] ^= 1
    cells = disc.decode_bits(tuple(sample))
    return oracle.hard_label(forest, cells, [disc.tau_array(t) for t in forest.trees])


def test_radius_matches_bruteforce_and_witness():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    for code in disc.all_codes():
        own = _own(forest, disc, code)
        assert robustness.radius(build, code, own)[0] == 0
        for target in range(forest.n_classes):
            got, bits = robustness.radius(build, code, target)
            expected, _ = oracle.brute_radius(forest, disc, code, target)
            assert got == expected, (code, target, got, expected)
            assert _flipped_label(forest, disc, code, bits) == target, (code, target, bits)
        loss, bits = robustness.loss_radius(build, code, own)
        expected, _ = oracle.brute_loss(forest, disc, code, own)
        assert loss == expected, (code, own, loss, expected)
        assert _flipped_label(forest, disc, code, bits) != own
        rival, _bits, _target = robustness.min_rival_radius(build, code, own)
        assert rival == min(robustness.radius(build, code, t)[0]
                            for t in range(forest.n_classes) if t != own)
        # Binary with an odd number of trees: losing certification is the same event as a rival.
        assert loss == rival


def test_frontier_radii_equal_base():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    base = compose.build(forest, disc)
    for selector in SELECTORS:
        build = compose.build(forest, disc, budget=2, selector=selector)
        for code in disc.all_codes():
            own = _own(forest, disc, code)
            for target in range(forest.n_classes):
                assert robustness.radius(build, code, target)[0] == robustness.radius(
                    base, code, target
                )[0]
            assert robustness.loss_radius(build, code, own)[0] == robustness.loss_radius(
                base, code, own
            )[0]


def test_distance_to_false_is_infinite():
    mgr = bdd.new_manager()
    mgr.declare("x0")
    assert math.isinf(robustness.distance_bits(mgr, mgr.false, {0: 0})[0])
    assert robustness.distance_bits(mgr, mgr.true, {0: 0})[0] == 0


def test_fast_radius_matches_dp():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    names = transitions.input_names(disc)
    wins = {label: build.win(label) for label in range(forest.n_classes)}
    for code in disc.all_codes():
        own = _own(forest, disc, code)
        for target in range(forest.n_classes):
            fast, _ = robustness.radius_from_win(build.mgr, wins[target],
                                                 disc.code_bits(code), names)
            exact, _ = robustness.radius(build, code, target)
            assert fast == exact, (code, target, fast, exact)
        fast_loss, _ = robustness.loss_radius_from_win(
            build.mgr, wins[own], disc.code_bits(code), names
        )
        exact_loss, _ = robustness.loss_radius(build, code, own)
        assert fast_loss == exact_loss


def test_fast_radius_uses_dp_fallback_below_max_depth():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    names = transitions.input_names(disc)
    wins = {label: build.win(label) for label in range(forest.n_classes)}
    for code in disc.all_codes():
        for target in range(forest.n_classes):
            shallow, _ = robustness.radius_from_win(
                build.mgr, wins[target], disc.code_bits(code), names, max_depth=1
            )
            deep, _ = robustness.radius_from_win(
                build.mgr, wins[target], disc.code_bits(code), names, max_depth=0
            )
            assert shallow == deep
