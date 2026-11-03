"""Dataset bundles: label order, validation and the 54 provided forests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdss import config  # noqa: E402
from bdss.forest import load_bundle, validate_bundle  # noqa: E402


@pytest.mark.slow
def test_all_54_bundles_load_and_validate():
    paths = sorted(config.DATASETS_DIR.glob("*/*_rf.pkl"))
    assert len(paths) == 54
    for path in paths:
        bundle = load_bundle(path)
        row = validate_bundle(bundle)
        assert row["dataset"] == bundle.name
        assert row["soft_acc"] == pytest.approx(bundle.test_acc, abs=1e-12)
