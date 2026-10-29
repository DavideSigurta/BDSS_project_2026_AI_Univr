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
