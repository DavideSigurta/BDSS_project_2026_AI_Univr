"""Toy forest and discretization: the 4-bit example used by every later milestone."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.toy_forest import build_toy_forest, toy_cell_counts  # noqa: E402
from bdss.discretize import Disc  # noqa: E402
from bdss import oracle  # noqa: E402


def test_cells_and_bits():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    assert disc.n_features == 2
    assert disc.cells == toy_cell_counts() == (3, 4)
    assert disc.bits == (2, 2)
    assert disc.n_bits == 4
    np.testing.assert_array_equal(disc.thresholds[0], [2.0, 4.0])
    np.testing.assert_array_equal(disc.thresholds[1], [1.0, 3.0, 5.0])


def test_tau_arrays():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    t0 = disc.tau_array(forest.trees[0])
    assert t0[0] == 0          # f0 <= 2.0 -> cell 0
    assert t0[2] == 1          # f1 <= 3.0 -> cell 1
    t1 = disc.tau_array(forest.trees[1])
    assert t1[0] == 0          # f1 <= 1.0 -> cell 0
    assert t1[2] == 1          # f0 <= 4.0 -> cell 1
    t2 = disc.tau_array(forest.trees[2])
    assert t2[0] == 1          # f0 <= 4.0 -> cell 1
    assert t2[1] == 2          # f1 <= 5.0 -> cell 2


def test_cell_walk_matches_real_walk():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    taus = [disc.tau_array(tree) for tree in forest.trees]
    reps = oracle.cell_representatives(disc)
    for f0, v0 in enumerate(reps[0]):
        for f1, v1 in enumerate(reps[1]):
            row = [v0, v1]
            cells = (f0, f1)
            for tree, tau in zip(forest.trees, taus):
                assert tree.leaf_for_values(row) == tree.leaf_for_cells(cells, tau)


def test_hard_vote_all_cells():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    taus = [disc.tau_array(tree) for tree in forest.trees]
    for cells, counts, label in oracle.cell_table(forest, disc):
        got = forest.hard_label(cells, taus)
        assert got == label
        assert counts.sum() == forest.n_trees
        assert got is not None  # three trees: no ties


def test_encode_decode_and_clamping():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    assert disc.encode_cells((0, 0)) == (0, 0, 0, 0)
    assert disc.encode_cells((1, 2)) == (0, 1, 1, 0)
    assert disc.encode_cells((2, 3)) == (1, 0, 1, 1)
    for f0 in range(3):
        for f1 in range(4):
            assert disc.decode_bits(disc.encode_cells((f0, f1))) == (f0, f1)
    decoded = {disc.code_cells(code) for code in disc.all_codes()}
    assert len(decoded) == 12
    # The four surplus patterns of feature 0 (code 11) clamp to cell 2.
    assert disc.code_cells(0b1100) == (2, 0)
    assert disc.code_cells(0b1111) == (2, 3)
