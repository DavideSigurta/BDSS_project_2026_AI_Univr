"""Campaign planning: final grid, resume, cost estimate and row schema."""

import csv
import threading

import pytest

from bdss import campaign, config


def test_budget_soft_and_hard():
    assert config.SOFT_USD == 10.0
    assert config.MAX_USD == 30.0


def test_estimate_usd_worst_case():
    estimate = campaign.estimate_usd(1560, timeout=120.0)
    assert estimate["runs"] == 1560
    assert estimate["cpu_hours"] == pytest.approx(52.0, abs=1e-9)
    assert estimate["total_usd"] == pytest.approx(4.108, abs=1e-3)


def test_resume_keys_roundtrip(tmp_path):
    path = tmp_path / "campaign.csv"
    row = campaign._empty_row(("d", 5, None, "modal"), "completed")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=campaign.CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    assert campaign.resume_keys(path) == {("d", 5, "", "modal")}


def test_three_timeouts_do_not_grow_threads():
    before_threads = threading.active_count()
    for _ in range(3):
        row = campaign.run_one_limited(("threeOf9", 1, None, "modal"), timeout=0.001)
        assert row["status"] == "timeout"
    assert threading.active_count() <= before_threads + 1


def test_run_local_records_90_and_300(tmp_path):
    for timeout in (90.0, 300.0):
        out = tmp_path / f"run_{int(timeout)}.csv"
        rows = campaign.run_local([("threeOf9", 1, None, "modal")], jobs=1,
                                  out=out, timeout=timeout)
        assert rows[0]["timeout_s"] == int(timeout)
        assert int(campaign.read_rows(out)[0]["timeout_s"]) == int(timeout)


def test_timeout_s_schema():
    row = campaign._empty_row(("d", 5, None, "modal"), "timeout")
    assert row["timeout_s"] == 120
    assert "timeout_s" in campaign.CSV_FIELDS
def test_limited_run_returns_a_timeout_row():
    row = campaign.run_one_limited(("threeOf9", 1, None, "modal"), timeout=0.001)
    assert row["status"] == "timeout"
    assert row["timeout"] is True


def test_run_one_schema_on_smallest_real_bundle():
    row = campaign.run_one(("threeOf9", 1, None, "modal"))
    assert set(row) == set(campaign.CSV_FIELDS)
    assert row["status"] == "completed"
    assert row["dataset"] == "threeOf9"
    assert row["n_bits"] == 1
    assert int(row["counts_sum"]) == 2


def test_build_models_uses_only_full_grid_rows():
    models = (("a", 40), ("b", 40), ("c", 40))
    rows = [
        {"dataset": "a", "trees": 40, "phase": "grid", "status": "completed"},
        {"dataset": "b", "trees": 40, "phase": "grid", "status": "timeout"},
        {"dataset": "c", "trees": 5, "phase": "build", "status": "completed"},
    ]
    assert campaign.build_models(rows, models=models) == (("b", 40), ("c", 40))


def test_build_pairs_are_three_budgets_times_five_selectors():
    models = (("a", 40), ("b", 40), ("c", 40))
    pairs = campaign.build_pairs(models)
    assert len(pairs) == len(models) * 3 * 5
    assert {pair[2] for pair in pairs} == set(config.BUILD_BUDGETS)
    assert {pair[3] for pair in pairs} == set(config.BUILD_SELECTORS)


def test_build_search_reaches_below_five():
    calls = []

    def measure(spec):
        calls.append(spec[1])
        return {"status": "completed" if spec[1] <= 4 else "timeout"}

    rows = campaign.build_search(("d", 100, 1000, "modal"), measure)
    assert calls == [5, 2, 3, 4]
    assert len(rows) == 4
    assert all(spec < 100 for spec in calls)


def test_build_search_never_runs_full():
    calls = []

    def measure(spec):
        calls.append(spec[1])
        return {"status": "completed" if spec[1] <= 27 else "timeout"}

    rows = campaign.build_search(("d", 98, 1000, "modal"), measure)
    assert calls == [5, 10, 20, 30, 25, 27, 28]
    assert max(spec for spec, row in zip(calls, rows) if row["status"] == "completed") == 27
    assert 98 not in calls


def test_build_search_no_build_stops_at_one():
    calls = []

    def measure(spec):
        calls.append(spec[1])
        return {"status": "timeout"}

    rows = campaign.build_search(("d", 40, 1000, "modal"), measure)
    assert calls == [5, 2, 1]
    assert len(rows) == 3


def test_run_build_local_merges_one_pair_and_resumes(tmp_path, monkeypatch):
    pair = ("d", 5, 1000, "modal")
    row = campaign._empty_row(pair, "completed")

    monkeypatch.setattr(campaign, "build_models", lambda rows: (("d", 5),))
    monkeypatch.setattr(campaign, "build_pairs", lambda models: [pair])
    monkeypatch.setattr(campaign, "build_search", lambda pair, measure: [dict(row)])
    out = tmp_path / "campaign.csv"

    rows = campaign.run_build_local([], out=out, jobs=1)
    assert len(rows) == 1
    assert rows[0]["phase"] == "build"
    assert rows[0]["source"] == "build"
    assert campaign.read_rows(out)[0]["phase"] == "build"

    def fail(*args, **kwargs):
        raise AssertionError("completed pair must be skipped on resume")

    monkeypatch.setattr(campaign, "build_search", fail)
    campaign.run_build_local(rows, out=out, jobs=1)


def test_final_limit_refuses_canonical(monkeypatch):
    import sys

    from bdss import study

    monkeypatch.setattr(sys, "argv", ["bdss", "final", "--limit", "1"])
    with pytest.raises(SystemExit, match="canonical campaign"):
        study.final_main()
