"""The Modal entrypoint is optional: skip when the extra is not installed."""

import pytest

pytest.importorskip("modal")


def test_modal_app_imports_without_starting_a_run():
    import modal

    from bdss import campaign, modal_app

    assert modal_app.APP_NAME == "bdss"
    assert isinstance(modal_app.run_final_spec, modal.Function)
    assert isinstance(modal_app.run_build_spec, modal.Function)
    assert isinstance(modal_app.run_readout, modal.Function)
    assert isinstance(modal_app.run_robustness_worker, modal.Function)


def test_modal_canonical_guards_refuse_partial_specs(tmp_path, monkeypatch):
    from bdss import modal_app, post

    spec = ("threeOf9", 1, None, "modal", "final")
    readouts = tmp_path / "readouts.csv"
    samples = tmp_path / "robustness_samples.csv"
    targets = tmp_path / "robustness_targets.csv"
    monkeypatch.setattr(post, "CANONICAL_READOUTS", readouts)
    monkeypatch.setattr(post, "CANONICAL_SAMPLES", samples)
    monkeypatch.setattr(post, "CANONICAL_TARGETS", targets)
    with pytest.raises(ValueError, match="54"):
        modal_app._run_readouts(None, [spec], readouts, False)
    with pytest.raises(ValueError, match="54"):
        modal_app._run_robustness(None, [spec], tmp_path / "chunks", 1.0, 1.0)
    assert not readouts.exists()
    assert not samples.exists()
    assert not targets.exists()


def test_modal_campaign_rejects_limit():
    from bdss import modal_app

    with pytest.raises(SystemExit, match="not supported"):
        modal_app.main(limit=1, campaign=True)
