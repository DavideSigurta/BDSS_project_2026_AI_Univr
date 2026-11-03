"""Post-build readouts, per-label robustness and resumable chunk persistence."""

from bisect import bisect_right
import json

import pandas as pd
import pytest

from bdss import config, post, robustness


SPEC = ("threeOf9", 1, None, "modal", "final")


def test_selected_specs_prefers_largest_k_then_100k_modal():
    rows = [
        {"dataset": "d", "trees": 50, "k": 50, "budget": "1000", "selector": "random",
         "status": "completed", "peak": 10, "leaves": 2, "source": "grid"},
        {"dataset": "d", "trees": 50, "k": 50, "budget": "100000", "selector": "modal",
         "status": "completed", "peak": 20, "leaves": 3, "source": "prefix"},
        {"dataset": "d", "trees": 30, "k": 30, "budget": "100000", "selector": "modal",
         "status": "completed", "peak": 5, "leaves": 1, "source": "prefix"},
    ]
    assert post.selected_specs(rows) == (("d", 50, 100_000, "modal", "prefix"),)


def test_readout_schema_on_small_model():
    row = post.readout_one(SPEC[:4])
    assert row["status"] == "completed"
    assert row["timeout_s"] == int(config.READOUT_TIMEOUT_S)
    assert row["feature_order"] and row["labels"]
    assert json.loads(row["counts_json"])
    assert json.loads(row["volumes_json"])
    assert json.loads(row["rankings_json"])
    assert row["n_classes"] == len(row["labels"].split(","))


def test_witness_bits_reach_target_and_map_to_destination_cells():
    bundle, forest, disc, build, order = post._load_selected(SPEC, config.FINAL_NODE_CAP)
    taus = [disc.tau_array(tree) for tree in forest.trees]
    row = pd.read_csv(
        bundle.path.parent / "threeOf9_test.csv"
    ).iloc[0, :-1].to_numpy(dtype=float)
    cells = disc.cells_of_row(row)
    code = post._code_of(disc.encode_cells(cells))
    for label in range(forest.n_classes):
        distance, bits = robustness.radius(build, code, label)
        flat = list(disc.code_bits(code))
        for level in bits:
            flat[level] ^= 1
        assert forest.hard_label(disc.decode_bits(tuple(flat)), taus) == label
        wb, wf, wc, dest = post.witness(
            build, order, bundle.feature_names, code, bits
        )
        assert wb == "|".join(map(str, bits))
        if bits:
            assert len(wf.split("|")) == len(wc.split("|")) == len(bits)
        else:
            assert wf == wc == ""
        payload = json.loads(dest)
        affected = {bisect_right(disc.offsets, level) - 1 for level in bits}
        assert len(payload) == len(affected)
        assert distance >= 0


def test_robustness_range_has_one_row_per_sample_and_label():
    result = post.robustness_range(SPEC, start=0, limit_s=60, max_samples=1)
    assert result["next"] == 1
    assert len(result["samples"]) == 1
    assert len(result["targets"]) == result["samples"][0]["n_targets"]
    sample = result["samples"][0]
    assert set(sample) == set(post.SAMPLE_FIELDS)
    assert set(result["targets"][0]) == set(post.TARGET_FIELDS)
    assert sum(row["is_min_rival"] for row in result["targets"]) == 1


def test_chunks_roundtrip_and_combine(tmp_path):
    result = post.robustness_range(SPEC, start=0, limit_s=60, max_samples=1)
    path = post.write_chunk(tmp_path, result)
    assert path.exists()
    chunks = post.read_chunks(tmp_path)
    assert post.next_start(chunks, SPEC) == 1
    assert not post.chunk_done(chunks, SPEC)
    samples, targets, sample_rows, target_rows = post.combine_robustness(
        chunks, tmp_path / "samples.csv", tmp_path / "targets.csv"
    )
    assert samples == len(sample_rows) == 1
    assert targets == len(target_rows) == len(result["targets"])


def test_coverage_marks_complete():
    result = post.robustness_range(SPEC, start=0, limit_s=60)
    readout = post.readout_one(SPEC[:4])
    spec = SPEC[:5]
    coverage = post.coverage([spec], [readout], result["samples"], result["targets"], [{"dataset": spec[0], "trees": spec[1], "budget": spec[2], "selector": spec[3], "source": spec[4], "k": spec[1], "phase": "build", "status": "completed", "cap_suspended": False}])
    assert len(coverage) == 1
    row = coverage[0]
    assert row["readout_status"] == "completed"
    assert row["robustness_status"] == "completed"
    assert row["samples"] == len(result["samples"])
    assert row["targets"] == len(result["targets"])


def test_readout_local_resume_keeps_one_row(tmp_path):
    out = tmp_path / "readouts.csv"
    post.run_readouts_local([SPEC[:4]], out=out, timeout=120)
    first = post.read_rows(out)
    post.run_readouts_local([SPEC[:4]], out=out, timeout=120, resume=True)
    second = post.read_rows(out)
    assert len(first) == len(second) == 1
    assert first[0]["status"] == "completed"


def test_readout_local_timeout_becomes_a_row(tmp_path):
    out = tmp_path / "readouts.csv"
    post.run_readouts_local([SPEC[:4]], out=out, timeout=0.001)
    row = post.read_rows(out)[0]
    assert row["status"] == "timeout"
    assert row["timeout"] == "True"


def test_canonical_guard_refuses_partial_readout(tmp_path, monkeypatch):
    canonical = tmp_path / "readouts.csv"
    monkeypatch.setattr(post, "CANONICAL_READOUTS", canonical)
    with pytest.raises(ValueError, match="54"):
        post.run_readouts_local([SPEC[:4]], out=canonical)
    with pytest.raises(ValueError, match="54"):
        post.run_readouts_local([], out=canonical)
    assert not canonical.exists()


def test_canonical_guard_refuses_partial_robustness(tmp_path, monkeypatch):
    samples = tmp_path / "robustness_samples.csv"
    targets = tmp_path / "robustness_targets.csv"
    monkeypatch.setattr(post, "CANONICAL_SAMPLES", samples)
    monkeypatch.setattr(post, "CANONICAL_TARGETS", targets)
    with pytest.raises(ValueError, match="54"):
        post.run_robustness_local(
            [SPEC[:4]], chunks=tmp_path / "chunks",
            samples_out=samples, targets_out=targets,
        )
    assert not samples.exists()
    assert not targets.exists()


def test_canonical_guard_is_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="54"):
        post._guard_canonical([SPEC], [post.CANONICAL_READOUTS])
    with pytest.raises(ValueError, match="54"):
        post._guard_canonical([SPEC], [post.CANONICAL_COVERAGE])
    variant = post.CANONICAL_READOUTS.with_name("READOUTS.CSV")
    with pytest.raises(ValueError, match="54"):
        post._guard_canonical([SPEC], [variant])


def test_write_csv_is_atomic(tmp_path):
    path = tmp_path / "out.csv"
    path.write_text("old")
    post._write_csv(path, ["a"], [{"a": 1}])
    assert path.read_text().splitlines() == ["a", "1"]
    assert not path.with_name("out.csv.part").exists()
