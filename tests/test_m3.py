"""M3 frontier: per-cell pilot grid, selector coverage and partition/sum checks."""

import sys
from pathlib import Path
from queue import Queue

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdss import config, study  # noqa: E402


def _trial(name, trees, budget, selector):
    queue = Queue()
    study._run_trial(name, trees, budget, selector, queue)
    result = queue.get()
    assert isinstance(result, dict), result
    return result


@pytest.fixture(scope="module")
def expected():
    return study._m3_expected()


def test_m3_trials_follow_the_frozen_grid():
    trials = study.m3_trials()
    assert len(trials) == 52
    assert ("banknote", 5, 1, "modal") in trials
    assert ("ann-thyroid", 5, 19_000, "shapley") in trials
    assert ("vehicle", 10, 10_000, "modal") not in trials  # the wall is a pilot row
    assert ("vehicle", 5, 500_000, "ig") not in trials
    for name, trees, budget, selector in trials:
        assert budget in config.M3_GRIDS[(name, trees)]
        assert selector in config.M3_CHEAP or budget in config.M3_SEMANTIC[(name, trees)]


@pytest.mark.slow
def test_m3_banknote5_b1_terminates(expected):
    result = _trial("banknote", 5, 1, "modal")
    assert result["status"] == "completed"
    assert result["cap_suspended"] is True
    assert result["cap_violations"] == result["leaves"]
    assert result["leaves_at_D"] is True
    assert result["sample_mismatch"] == 0
    assert result["counts"] == expected[("banknote", 5)]


@pytest.mark.slow
def test_m3_banknote5_all_selectors_agree(expected):
    for selector in (*config.M3_CHEAP, *config.M3_SEMANTIC_SELECTORS):
        result = _trial("banknote", 5, 100, selector)
        assert result["status"] == "completed"
        assert result["cap_violations"] == 0
        assert result["leaves_at_D"] is True
        assert result["sample_mismatch"] == 0
        assert result["counts"] == expected[("banknote", 5)]


def test_m3_artifacts_are_consistent():
    import csv
    from collections import Counter, defaultdict

    frontier = list(csv.DictReader((ROOT / "results" / "m3_frontier.csv").open()))
    assert len(frontier) == 54
    assert Counter(row["status"] for row in frontier) == {"completed": 51, "timeout": 3}
    for row in frontier:
        if row["status"] != "completed":
            assert row["dataset"] == "vehicle"
            assert row["selector"] in ("modal", "random")
            assert row["wall_seconds"]
            continue
        assert row["leaves_at_D"] == "True"
        assert row["sample_mismatch"] == "0"
        assert row["counts_match"] == "True"
        if int(row["cap_violations"]) > 0:
            assert row["cap_suspended"] == "True"

    sums = defaultdict(int)
    for row in csv.DictReader((ROOT / "results" / "m3_counts.csv").open()):
        sums[(row["dataset"], row["trees"], row["budget"], row["selector"])] += int(row["count"])
    for row in frontier:
        if row["status"] != "completed":
            continue
        key = (row["dataset"], row["trees"], row["budget"], row["selector"])
        assert sums[key] == int(row["counts_sum"])

    coverage = list(csv.DictReader((ROOT / "results" / "m3_coverage.csv").open()))
    assert len(coverage) == 30
    assert {row["selector"] for row in coverage} == {
        "modal", "random", "ig", "shapley", "pvalue"
    }
