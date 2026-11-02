"""M4 robustness: radii on M3 leaves, witnesses, and brute force on a small real prefix."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.m4_bruteforce import check  # noqa: E402
from bdss import compose, config, robustness, study  # noqa: E402
from bdss.discretize import Disc  # noqa: E402
from bdss.forest import Forest, load_bundle  # noqa: E402


@pytest.mark.slow
def test_m4_banknote5_radii_and_witnesses():
    bundle = load_bundle(config.DATASETS_DIR / "banknote" / "banknote_rf.pkl")
    samples, targets, summary, distribution, fragile = study.m4_cell(
        bundle, 5, 100, "modal"
    )
    assert summary["samples"] == 275
    assert summary["certified"] == 275
    assert summary["own_zero"] == 275
    assert summary["target_mismatches"] == 0
    assert summary["witness_failures"] == 0
    assert summary["loss_le_rival"] == 275
    assert all(target["equal"] for target in targets)
    assert distribution
    assert fragile


@pytest.mark.slow
def test_m4_frontier_bfs_matches_dp_banknote5():
    bundle = load_bundle(config.DATASETS_DIR / "banknote" / "banknote_rf.pkl")
    base = Forest.from_arrays(bundle.forest.trees[:5], bundle.forest.n_features,
                              bundle.forest.labels)
    order = study.shallow_order(base)
    forest, disc = study.permute(base, order)
    frontier = compose.build(forest, disc, budget=100, selector="modal")
    names = study.transitions.input_names(disc)
    wins = {label: frontier.win(label) for label in range(forest.n_classes)}
    test = np.genfromtxt(bundle.path.parent / "banknote_test.csv", delimiter=",",
                         skip_header=1)
    for row in test:
        cells = disc.cells_of_row(row[:-1])
        bits = disc.encode_cells(cells)
        code = study._code_of(bits)
        for target in range(forest.n_classes):
            fast, _ = robustness.radius_from_win(frontier.mgr, wins[target], bits, names)
            exact, _ = robustness.radius(frontier, code, target)
            assert fast == exact, (code, target, fast, exact)


@pytest.mark.slow
def test_m4_bruteforce_real_prefix_12_bits():
    bundle = load_bundle(config.DATASETS_DIR / "banknote" / "banknote_rf.pkl")
    forest = Forest.from_arrays(
        bundle.forest.trees[:1], bundle.forest.n_features, bundle.forest.labels
    )
    disc = Disc.from_forest(forest)
    assert disc.n_bits == 12
    test = np.genfromtxt(bundle.path.parent / "banknote_test.csv", delimiter=",",
                         skip_header=1)
    codes = [
        sum(bit << (disc.n_bits - 1 - i)
            for i, bit in enumerate(disc.encode_cells(disc.cells_of_row(row[:-1]))))
        for row in test
    ]
    row = check(forest, disc, codes, "banknote/1", 1)
    assert row["radius_mismatch"] == 0
    assert row["loss_mismatch"] == 0
    assert row["witness_failures"] == 0
    assert row["targets"] == len(codes) * forest.n_classes
