"""M2: bundle loader, fixed input order, prefix analyses and the predict gate."""

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdss import bdd, compose, config, oracle, study, transitions  # noqa: E402
from bdss.discretize import Disc  # noqa: E402
from bdss.forest import (  # noqa: E402
    Forest,
    Tree,
    load_bundle,
    soft_vote,
    validate_bundle,
)


def _synthetic_bundle(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3)).round(2)
    raw = np.array(["b", "a", "c", "a", "b", "c"] * 10)
    codes, labels = pd.factorize(raw)
    train = np.arange(40)
    test = np.arange(40, 60)
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    rf.fit(x[train], codes[train])

    forest = Forest.from_arrays(
        tuple(Tree.from_sklearn(est.tree_, len(rf.classes_)) for est in rf.estimators_),
        x.shape[1],
        tuple(labels),
    )
    soft = np.array([soft_vote(forest, row) for row in x[test]])
    acc = float((soft == codes[test]).mean())

    frame = pd.DataFrame(x, columns=["f0", "f1", "f2"])
    frame["target"] = raw
    frame.iloc[train].to_csv(tmp_path / "toy_train.csv", index=False)
    frame.iloc[test].to_csv(tmp_path / "toy_test.csv", index=False)
    frame.to_csv(tmp_path / "toy.csv", index=False)
    joblib.dump(
        {
            "model": rf,
            "dataset": "toy",
            "n_features": x.shape[1],
            "n_classes": len(labels),
            "n_rows": len(frame),
            "train_rows": len(train),
            "test_rows": len(test),
            "test_acc": acc,
            "cv_acc": acc,
            "params": {},
            "seed": 0,
        },
        tmp_path / "toy_rf.pkl",
    )
    return tmp_path / "toy_rf.pkl"


def test_bundle_label_map_is_first_appearance(tmp_path):
    path = _synthetic_bundle(tmp_path)
    bundle = load_bundle(path)
    assert bundle.forest.labels == ("b", "a", "c")
    assert bundle.feature_names == ("f0", "f1", "f2")
    row = validate_bundle(bundle)
    assert row["soft_acc"] == pytest.approx(bundle.test_acc, abs=1e-12)
    assert row["test_rows"] == 20


def test_input_order_is_fixed_and_counts_are_invariant(tmp_path):
    bundle = load_bundle(_synthetic_bundle(tmp_path))
    base = bundle.forest
    order = study.shallow_order(base)
    assert sorted(order) == list(range(base.n_features))
    forest, disc = study.permute(base, order)

    build_base = compose.build(base, Disc.from_forest(base))
    build_order = compose.build(forest, disc)
    assert compose_build_counts(build_base) == compose_build_counts(build_order)
    names = transitions.input_names(disc)
    assert [build_order.mgr.level_of_var(name) for name in names] == list(range(disc.n_bits))
    positions = study.bit_positions(disc, order, bundle.feature_names)
    assert len(positions) == disc.n_bits
    assert {name for name, _bit, _level in positions} <= set(bundle.feature_names)


def compose_build_counts(build):
    from bdss import readouts

    return readouts.label_counts(build)


def test_analyze_writes_rank_logp(tmp_path):
    bundle = load_bundle(_synthetic_bundle(tmp_path))
    _row, _counts, ranks = study.analyze(bundle, 5)
    assert ranks
    assert all("rank_logp" in item for item in ranks)
    for label in range(bundle.forest.n_classes):
        values = [item["rank_logp"] for item in ranks if item["label"] == bundle.forest.labels[label]]
        assert sorted(values) == list(range(1, len(values) + 1))


def test_pipeline_never_calls_predict_fast(tmp_path, monkeypatch):
    path = _synthetic_bundle(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("RandomForestClassifier.predict must not be called")

    monkeypatch.setattr(RandomForestClassifier, "predict", boom)
    bundle = load_bundle(path)
    validate_bundle(bundle)
    study.analyze(bundle, 5, with_rankings=False)


@pytest.mark.slow
def test_all_54_bundles_load_and_validate():
    paths = sorted(config.DATASETS_DIR.glob("*/*_rf.pkl"))
    assert len(paths) == 54
    for path in paths:
        bundle = load_bundle(path)
        row = validate_bundle(bundle)
        assert row["dataset"] == bundle.name
        assert row["soft_acc"] == pytest.approx(bundle.test_acc, abs=1e-12)


@pytest.mark.slow
def test_banknote_prefix5_matches_bruteforce():
    bundle = load_bundle(config.DATASETS_DIR / "banknote" / "banknote_rf.pkl")
    row, counts, _ranks = study.analyze(bundle, 5, with_rankings=False)
    assert row["status"] == "completed"
    assert row["wins_disjoint"] is True
    assert row["binary_odd_exactly_one"] is True
    assert row["certified"] == row["samples"] == 275
    assert row["mismatch"] == 0

    base = Forest.from_arrays(
        bundle.forest.trees[:5], bundle.forest.n_features, bundle.forest.labels
    )
    expected = oracle.cell_weight_counts(base, Disc.from_forest(base))
    assert [expected[label] for label in range(2)] == [item["count"] for item in counts]
    assert sum(item["count"] for item in counts) == 1 << row["n_bits"]

    order = study.shallow_order(base)
    forest, disc = study.permute(base, order)
    sizes = [bdd.size(compose.accumulator(forest, disc, upto=t)[1]) for t in range(6)]
    assert max(sizes) == row["peak"] == 999


@pytest.mark.slow
def test_banknote_prefix10_counts_and_samples():
    bundle = load_bundle(config.DATASETS_DIR / "banknote" / "banknote_rf.pkl")
    row, counts, _ranks = study.analyze(bundle, 10, with_rankings=False)
    assert row["wins_disjoint"] is True
    assert row["certified"] == row["samples"] == 275
    assert row["mismatch"] == 0
    assert sum(item["count"] for item in counts) == 64_247_000

    base = Forest.from_arrays(
        bundle.forest.trees[:10], bundle.forest.n_features, bundle.forest.labels
    )
    order = study.shallow_order(base)
    forest, disc = study.permute(base, order)
    sizes = [bdd.size(compose.accumulator(forest, disc, upto=t)[1]) for t in range(11)]
    assert max(sizes) == row["peak"] == 8_333


@pytest.mark.slow
def test_real_prefixes_samples_and_disjoint_wins():
    peaks = {("vehicle", 5): 1_924_750, ("ann-thyroid", 10): 393_635}
    for name, n_trees in (("vehicle", 5), ("ann-thyroid", 10)):
        bundle = load_bundle(config.DATASETS_DIR / name / f"{name}_rf.pkl")
        row, _counts, _ranks = study.analyze(bundle, n_trees, with_rankings=False)
        assert row["status"] == "completed"
        assert row["wins_disjoint"] is True
        assert row["mismatch"] == 0
        assert row["certified"] <= row["samples"]
        assert row["peak"] == peaks[(name, n_trees)]
        gc.collect()


@pytest.mark.slow
def test_pipeline_never_calls_predict(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("RandomForestClassifier.predict must not be called")

    monkeypatch.setattr(RandomForestClassifier, "predict", boom)
    bundle = load_bundle(config.DATASETS_DIR / "banknote" / "banknote_rf.pkl")
    validate_bundle(bundle)
    study.analyze(bundle, 5, with_rankings=False)
