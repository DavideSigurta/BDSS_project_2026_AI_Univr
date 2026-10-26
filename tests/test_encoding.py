"""Node tests and path formulas against the brute-force cell walk."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.toy_forest import build_toy_forest  # noqa: E402
from bdss import bdd, encoding  # noqa: E402
from bdss.discretize import Disc  # noqa: E402


def _input_names(disc):
    names = []
    for feature, width in enumerate(disc.bits):
        for j in range(width - 1, -1, -1):
            names.append(f"x_{feature}_{j}")
    return names


def _eval(mgr, u, code, disc):
    bits = disc.code_bits(code)
    names = _input_names(disc)
    assignment = {name: bool(bit) for name, bit in zip(names, bits)}
    return mgr.let(assignment, u) == mgr.true


def test_nt_matches_cell_test():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    mgr = bdd.new_manager()
    mgr.declare(*_input_names(disc))
    taus = [disc.tau_array(tree) for tree in forest.trees]
    for tree, tau in zip(forest.trees, taus):
        for node in range(tree.n_nodes):
            if tree.is_leaf(node):
                continue
            feature = int(tree.feature[node])
            formula = encoding.nt(mgr, disc, feature, int(tau[node]))
            for code in disc.all_codes():
                cells = disc.code_cells(code)
                expected = cells[feature] <= tau[node]
                assert _eval(mgr, formula, code, disc) == expected, (node, code)


def test_path_formulas_partition_the_codes():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    mgr = bdd.new_manager()
    mgr.declare(*_input_names(disc))
    taus = [disc.tau_array(tree) for tree in forest.trees]
    for tree, tau in zip(forest.trees, taus):
        paths = encoding.path_formulas(mgr, disc, tree)
        assert len(paths) == sum(tree.is_leaf(n) for n in range(tree.n_nodes))
        for code in disc.all_codes():
            cells = disc.code_cells(code)
            reached = tree.leaf_for_cells(cells, tau)
            true_leaves = [leaf for leaf, pi in paths if _eval(mgr, pi, code, disc)]
            assert true_leaves == [reached], (code, true_leaves, reached)
