"""Environment smoke tests: BDD backends and a real forest hard vote.

The `.pkl` bundles are local and outside the repository; this module is marked `slow` because
it needs `../datasets`.
"""

import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import pytest

from bdss import bdd
from bdss.config import DATASETS_DIR


def _leaf_label(tree, row):
    """Walk one sklearn tree and return the argmax class at the reached leaf."""
    node = 0
    while tree.children_left[node] != -1:
        feature = tree.feature[node]
        if row[feature] <= tree.threshold[node]:
            node = tree.children_left[node]
        else:
            node = tree.children_right[node]
    return int(np.argmax(tree.value[node][0]))


def test_bdd_backend_selected():
    assert bdd.BACKEND in ("cudd", "autoref")
    mgr = bdd.new_manager()
    mgr.declare("x", "y")
    u = mgr.add_expr("x & y")
    assert mgr.count(u, 2) == 1
    v = mgr.add_expr("x | y")
    assert mgr.count(v, 2) == 3


def test_autoref_fallback():
    """Blocking dd.cudd must make bdss.bdd select dd.autoref, not crash."""
    code = (
        "import sys; sys.modules['dd.cudd'] = None; "
        "from bdss import bdd; assert bdd.BACKEND == 'autoref'; "
        "m = bdd.new_manager(); m.declare('x'); u = m.add_expr('x'); "
        "assert m.count(u, 1) == 1"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.slow
def test_real_forest_hard_vote():
    name = "vehicle"
    bundle = joblib.load(DATASETS_DIR / name / f"{name}_rf.pkl")
    forest = bundle["model"]
    assert forest.n_features_in_ == bundle["n_features"]

    test = pd.read_csv(DATASETS_DIR / name / f"{name}_test.csv")
    X = test.drop(columns=[test.columns[-1]]).to_numpy(dtype=float)
    n_classes = len(forest.classes_)
    soft = forest.predict(X)

    chosen = None
    for row, soft_label in zip(X, soft):
        votes = np.array([_leaf_label(est.tree_, row) for est in forest.estimators_])
        counts = np.bincount(votes, minlength=n_classes)
        winner = int(np.argmax(counts))
        strict = 2 * counts[winner] > len(votes)
        if strict and winner == int(soft_label):
            chosen = (row, votes, winner)
            break

    assert chosen is not None, "no test sample with a strict hard vote matching rf.predict"
    row, votes, winner = chosen
    assert votes[0] == _leaf_label(forest.estimators_[0].tree_, row)
    assert all(
        _leaf_label(est.tree_, row) == int(est.predict(row.reshape(1, -1))[0])
        for est in forest.estimators_
    )
    assert winner in range(n_classes)
