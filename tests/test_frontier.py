"""Budgeted frontier: partition, exactness, selectors, cap and peaks."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.toy_forest import build_toy_forest  # noqa: E402
from bdss import bdd, compose, readouts  # noqa: E402
from bdss.discretize import Disc  # noqa: E402
from bdss.frontier import SELECTORS  # noqa: E402

SELECTORS = tuple(SELECTORS)


def _matches(build, code):
    sample = dict(enumerate(build.disc.code_bits(code)))
    return [leaf for leaf in build.leaves
            if all(sample[level] == value for level, value in leaf.rho.items())]


def test_base_peak_is_primitive_len():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    assert build.peak == 19
    assert build.peak > bdd.size(build.leaves[0].A)


def test_frontier_partition_and_exact_counts():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    base = compose.build(forest, disc)
    base_counts = readouts.label_counts(base)
    for selector in SELECTORS:
        for budget in (1, 2, 4, 8, 10, 12, 18, 19):
            build = compose.build(forest, disc, budget=budget, selector=selector)
            assert build.leaves
            assert all(leaf.t == forest.n_trees for leaf in build.leaves)
            assert build.peak >= max(bdd.size(leaf.A) for leaf in build.leaves)
            if not build.cap_suspended:
                assert max(bdd.size(leaf.A) for leaf in build.leaves) <= budget
            assert build.seconds > 0
            for code in disc.all_codes():
                assert len(_matches(build, code)) == 1, (selector, budget, code)
            assert readouts.label_counts(build) == base_counts
            for label in range(forest.n_classes):
                assert bdd.size(build.win(label)) == bdd.size(base.win(label))
                for code in disc.all_codes():
                    names = [f"x_{f}_{j}" for f, width in enumerate(disc.bits)
                             for j in range(width - 1, -1, -1)]
                    assignment = {name: bool(bit)
                                  for name, bit in zip(names, disc.code_bits(code))}
                    got = build.mgr.let(assignment, build.win(label)) == build.mgr.true
                    expected = base.mgr.let(assignment, base.win(label)) == base.mgr.true
                    assert got == expected, (selector, budget, label, code)


def test_budget_one_terminates_and_cap_is_reported():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    for selector in SELECTORS:
        build = compose.build(forest, disc, budget=1, selector=selector)
        assert build.status == "completed"
        assert build.cap_suspended  # init is over budget and has no input to split
        assert len(build.leaves) > 1


def test_random_selector_is_reproducible():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    a = compose.build(forest, disc, budget=2, selector="random")
    b = compose.build(forest, disc, budget=2, selector="random")
    assert a.peak == b.peak
    assert len(a.leaves) == len(b.leaves)
    assert [sorted(leaf.rho.items()) for leaf in a.leaves] == [
        sorted(leaf.rho.items()) for leaf in b.leaves
    ]


def test_rankings_on_frontier_equal_base():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    base = compose.build(forest, disc)
    for selector in SELECTORS:
        build = compose.build(forest, disc, budget=2, selector=selector)
        got = readouts.label_rankings(build)
        expected = readouts.label_rankings(base)
        for label in range(forest.n_classes):
            for key in ("ig", "shapley", "pvalue"):
                for level in range(disc.n_bits):
                    if key == "pvalue":
                        assert got[label][key][level]["s"] == expected[label][key][level]["s"]
                    else:
                        assert got[label][key][level] == expected[label][key][level]


def test_peak_is_max_of_size_calls(monkeypatch):
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    original = bdd.size
    seen = []

    def spy(root):
        value = original(root)
        seen.append(value)
        return value

    monkeypatch.setattr(bdd, "size", spy)
    base = compose.build(forest, disc)
    assert base.peak == max(seen)
    seen.clear()
    build = compose.build(forest, disc, budget=4, selector="ig")
    assert build.peak == max(seen)


def test_cap_not_suspended_once_init_fits():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc, budget=10, selector="ig")
    assert build.cap_suspended is False
