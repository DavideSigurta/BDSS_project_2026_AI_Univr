"""M5 pilot: frozen trial list, campaign fields and projection."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdss import study  # noqa: E402


def test_m5_pilot_trials_are_the_frozen_six_models():
    trials = study.m5_pilot_trials()
    assert len(trials) == 45
    assert {model for model in study.M5_PILOT_MODELS} == {
        ("threeOf9", 1), ("banknote", 5), ("ann-thyroid", 10),
        ("vehicle", 5), ("vehicle", 10), ("factors", 1),
    }
    for name, trees, budget, selector in trials:
        assert budget in study.M5_PILOT_BUDGETS
        if selector == "ig":
            assert (name, trees) in study.M5_PILOT_SEMANTIC


def test_m5_pilot_artifacts_are_consistent():
    rows = list(csv.DictReader((ROOT / "results" / "m5_pilot.csv").open()))
    assert len(rows) == 45
    assert sum(row["status"] == "completed" for row in rows) == 27
    assert sum(row["status"] == "timeout" for row in rows) == 18
    for row in rows:
        assert int(row["n_bits"]) > 0
        if row["status"] == "completed":
            assert row["timeout"] == "False"
            assert float(row["build_seconds"]) >= 0.0
            assert float(row["compose_seconds"]) >= 0.0
            assert float(row["selector_seconds"]) >= 0.0
            assert float(row["readout_seconds"]) >= 0.0
            assert int(row["fallbacks"]) >= 0
            assert float(row["rss_mb"]) > 0.0
        else:
            assert row["timeout"] == "True"
            assert row["wall_seconds"] != ""

    projection = list(csv.DictReader(
        (ROOT / "results" / "m5_pilot_projection.csv").open()
    ))
    assert {row["scenario"] for row in projection} == {"pilot_mean", "cap120"}
    for row in projection:
        assert int(row["grid_runs"]) == 1350
        assert int(row["total_runs"]) == 1350 + int(row["prefix_runs"])
        assert float(row["cpu_hours"]) > 0.0
        for jobs in (2, 4, 8):
            assert float(row[f"wall_hours_{jobs}"]) > 0.0
