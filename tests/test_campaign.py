"""M5 campaign planning: specifications, resume, cost estimate and row schema."""

import csv
import os
import threading

import pytest

from bdss import campaign, config


def test_budget_soft_and_hard():
    assert config.M5_SOFT_USD == 10.0
    assert config.M5_MAX_USD == 30.0


def test_grid_specs_count_and_uniqueness():
    models = [(f"dataset-{i}", 10) for i in range(54)]
    specs = campaign.grid_specs(models=models)
    assert len(specs) == 54 * len(config.M5_BUDGETS) * len(config.M5_SELECTORS)
    assert len(specs) == len(set(specs))
    assert {budget for _, _, budget, _ in specs} == set(config.M5_BUDGETS)
    assert {selector for _, _, _, selector in specs} == set(config.M5_SELECTORS)


def test_prefix_specs_cap_and_order():
    walls = [(f"dataset-{i}", 60) for i in range(18)]
    specs = campaign.prefix_specs(walls)
    assert len(specs) == config.M5_PREFIX_CAP == 180
    assert len(specs) == len(set(specs))
    assert all(trees < full for _, trees, _, _ in specs for _, full in walls[:1])
    assert {selector for _, _, _, selector in specs} == {"modal", "random"}


def test_bisection_between_completed_and_failed():
    rows = [
        {"dataset": "d", "trees": 5, "budget": "", "selector": "modal", "status": "completed"},
        {"dataset": "d", "trees": 10, "budget": "", "selector": "modal", "status": "timeout"},
    ]
    assert campaign.bisection_specs(rows) == (("d", 7, None, "modal"),)


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


def test_three_timeouts_do_not_grow_threads_or_fds():
    before_threads = threading.active_count()
    before_fds = len(os.listdir("/dev/fd")) if os.path.isdir("/dev/fd") else None
    for _ in range(3):
        row = campaign.run_one_limited(("threeOf9", 1, None, "modal"), timeout=0.001)
        assert row["status"] == "timeout"
    assert threading.active_count() <= before_threads + 1
    if before_fds is not None:
        assert len(os.listdir("/dev/fd")) <= before_fds + 2


def test_post_specs_prefers_100k_modal_and_ignores_failures():
    rows = [
        {"dataset": "d", "trees": 10, "budget": "1000", "selector": "modal",
         "status": "completed"},
        {"dataset": "d", "trees": 10, "budget": "100000", "selector": "modal",
         "status": "completed"},
        {"dataset": "d", "trees": 10, "budget": "100000", "selector": "random",
         "status": "timeout"},
        {"dataset": "e", "trees": 5, "budget": "", "selector": "modal",
         "status": "completed"},
    ]
    assert campaign.post_specs(rows) == (("d", 10, 100_000, "modal", 120, "base"),
                                         ("e", 5, None, "modal", 120, "base"))


def test_post_coverage_reports_build_and_post_status():
    post_rows = [
        {"dataset": name, "trees": 10, "status": "timeout",
         "timeout_s": 120, "source": "base"}
        for name in ("a", "b", "c", "d", "e")
    ] + [
        {"dataset": name, "trees": 10, "status": "completed",
         "timeout_s": 300, "source": "rescue"}
        for name in ("f", "g")
    ]
    models = [(name, 10) for name in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")]
    coverage = campaign.post_coverage(post_rows, models=models)
    assert sum(row["post_status"] == "timeout" for row in coverage) == 5
    assert sum(row["post_status"] == "completed" for row in coverage) == 2
    assert sum(row["build_status"] == "missing" for row in coverage) == 3
    assert coverage[0]["build_status"] == "completed"
    assert coverage[0]["timeout_s"] == 120
    assert coverage[-1]["build_status"] == "missing"
    assert coverage[-1]["post_status"] == "not_run"


def test_post_one_schema_and_per_sample_rows():
    row = campaign.post_one(("threeOf9", 1, None, "modal"))
    assert row["status"] == "completed"
    assert row["feature_order"] != ""
    assert row["labels"] != ""
    assert row["counts_json"] and row["volumes_json"] and row["rankings_json"]
    assert row["_samples"]
    assert set(row["_samples"][0]) == set(campaign.POST_SAMPLE_FIELDS)
    assert row["timeout_s"] == 120 and row["source"] == "base"
    assert int(row["samples"]) == len(row["_samples"])


def test_post_local_resume_keeps_one_row(tmp_path):
    out = tmp_path / "post.csv"
    samples = tmp_path / "samples.csv"
    spec = ("threeOf9", 1, None, "modal")
    campaign.run_post_local([spec], out=out, samples_out=samples, timeout=600)
    first = campaign.read_rows(out)
    campaign.run_post_local([spec], out=out, samples_out=samples, timeout=600, resume=True)
    second = campaign.read_rows(out)
    assert len(first) == len(second) == 1
    assert samples.exists()


def test_post_resume_regenerates_missing_samples(tmp_path):
    out = tmp_path / "post.csv"
    samples = tmp_path / "samples.csv"
    spec = ("threeOf9", 1, None, "modal")
    campaign.run_post_local([spec], out=out, samples_out=samples, timeout=600)
    first_samples = campaign.read_rows(samples)
    # Simulate a crash after the summary but before/while writing samples.
    samples.write_text(",".join(campaign.POST_SAMPLE_FIELDS) + "\n")
    campaign.run_post_local([spec], out=out, samples_out=samples, timeout=600, resume=True)
    summary = campaign.read_rows(out)
    second_samples = campaign.read_rows(samples)
    assert len(summary) == 1
    assert len(second_samples) == len(first_samples)
    assert len({(r["sample"]) for r in second_samples}) == len(second_samples)


def test_run_local_records_90_and_300(tmp_path):
    for timeout in (90.0, 300.0):
        out = tmp_path / f"run_{int(timeout)}.csv"
        rows = campaign.run_local([("threeOf9", 1, None, "modal")], jobs=1,
                                  out=out, timeout=timeout)
        assert rows[0]["timeout_s"] == int(timeout)
        assert int(campaign.read_rows(out)[0]["timeout_s"]) == int(timeout)


def test_timeout_s_schema_and_rescue_specs():
    row = campaign._empty_row(("d", 5, None, "modal"), "timeout")
    assert row["timeout_s"] == 120
    assert "timeout_s" in campaign.CSV_FIELDS
    rows = [
        {"dataset": "a", "trees": 10, "budget": "1000", "selector": "modal",
         "status": "timeout"},
        {"dataset": "a", "trees": 10, "budget": "1000", "selector": "modal",
         "status": "timeout"},
        {"dataset": "b", "trees": 5, "budget": "1000", "selector": "modal",
         "status": "completed"},
        {"dataset": "c", "trees": 7, "budget": "", "selector": "ig",
         "status": "memory"},
    ]
    assert campaign.rescue_specs(rows) == (("a", 10, 1000, "modal"),
                                           ("c", 7, None, "ig"))


def test_merge_prefers_completed_and_keeps_provenance(tmp_path):
    base = tmp_path / "base.csv"
    rescue = tmp_path / "rescue.csv"
    out = tmp_path / "merged.csv"
    base_rows = [
        campaign._empty_row(("a", 10, 1000, "modal"), "completed"),
        campaign._empty_row(("b", 5, 1000, "modal"), "timeout"),
    ]
    rescue_rows = [
        campaign._empty_row(("b", 5, 1000, "modal"), "completed", timeout_s=300),
        campaign._empty_row(("a", 10, 1000, "modal"), "timeout", timeout_s=300),
    ]
    for path, rows in ((base, base_rows), (rescue, rescue_rows)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=campaign.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    merged = campaign.merge_campaigns(base, rescue, out)
    by_key = {(row["dataset"], int(row["trees"])): row for row in merged}
    assert by_key[("a", 10)]["status"] == "completed"
    assert int(by_key[("a", 10)]["timeout_s"]) == 120
    assert by_key[("b", 5)]["status"] == "completed"
    assert int(by_key[("b", 5)]["timeout_s"]) == 300
    assert len(campaign.read_rows(out)) == 2


def test_limited_run_returns_a_timeout_row():
    row = campaign.run_one_limited(("threeOf9", 1, None, "modal"), timeout=0.001)
    assert row["status"] == "timeout"
    assert row["timeout"] is True


def test_completed_keys_excludes_timeouts(tmp_path):
    path = tmp_path / "campaign.csv"
    rows = [
        campaign._empty_row(("d", 5, None, "modal"), "completed"),
        campaign._empty_row(("d", 10, None, "modal"), "timeout"),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=campaign.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    assert campaign.completed_keys(path) == {("d", 5, "", "modal")}


def test_run_one_schema_on_smallest_real_bundle():
    row = campaign.run_one(("threeOf9", 1, None, "modal"))
    assert set(row) == set(campaign.CSV_FIELDS)
    assert row["status"] == "completed"
    assert row["dataset"] == "threeOf9"
    assert row["n_bits"] == 1
    assert int(row["counts_sum"]) == 2
