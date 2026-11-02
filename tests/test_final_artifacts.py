"""Final M5 deliverables: exact coverage, keys and campaign/readout counts."""

import json

import pandas as pd

from bdss import campaign, post


def test_final_coverage_and_keys():
    coverage = pd.read_csv("results/m5_post_coverage.csv")
    samples = pd.read_csv("results/m5_robustness_samples.csv")
    targets = pd.read_csv("results/m5_robustness_targets.csv", low_memory=False)
    campaign_rows = pd.read_csv("results/m5_final_campaign.csv")

    assert len(campaign_rows) == 3_045
    assert campaign_rows["dataset"].nunique() == 54
    assert (campaign_rows["phase"] == "grid").sum() == 810
    assert len(coverage) == 54
    assert set(coverage["readout_status"]) == {"completed"}
    assert set(coverage["robustness_status"]) == {"completed"}
    assert int(coverage["test_samples"].sum()) == 37_950
    assert int(coverage["samples"].sum()) == 37_950
    assert int(coverage["targets"].sum()) == 282_654

    assert len(samples) == 37_950
    assert len(targets) == 282_654
    readouts = pd.read_csv("results/m5_readouts.csv")
    assert len(readouts) == 54
    assert set(readouts["status"]) == {"completed"}
    assert not samples.duplicated(["dataset", "sample"]).any()
    assert not targets.duplicated(["dataset", "sample", "label"]).any()
    per_sample = targets.groupby(["dataset", "sample"]).size()
    assert per_sample.index.is_unique
    assert (per_sample.groupby(level=0).nunique() == 1).all()


def test_readout_counts_match_campaign():
    rows = campaign.read_rows("results/m5_final_campaign.csv")
    readouts = pd.read_csv("results/m5_readouts.csv")
    by_key = {post._readout_key(row): row for _, row in readouts.iterrows()}
    for spec in post.selected_specs(rows):
        key = post._spec_key(spec)
        source = next(
            row for row in rows
            if post._spec_key(row) == key
        )
        counts = json.loads(by_key[key]["counts_json"])
        assert sum(counts.values()) == int(source["counts_sum"])
