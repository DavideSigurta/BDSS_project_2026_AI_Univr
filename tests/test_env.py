"""Environment smoke tests: BDD backends and real bundles.

The `.pkl` bundles are local and outside the repository; the bundle check is marked `slow`
because it needs `../datasets`.
"""

import subprocess
import sys

import pytest

from bdss import bdd
from bdss.config import DATASETS_DIR
from bdss.forest import load_bundle, validate_bundle


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
def test_real_bundle_label_map_and_votes():
    bundle = load_bundle(DATASETS_DIR / "vehicle" / "vehicle_rf.pkl")
    row = validate_bundle(bundle)
    assert row["soft_acc"] == pytest.approx(bundle.test_acc, abs=1e-12)
    assert 0.0 <= row["hard_acc"] <= 1.0
    assert len(bundle.forest.labels) == bundle.forest.n_classes
