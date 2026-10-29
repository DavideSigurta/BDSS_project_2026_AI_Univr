"""The Modal entrypoint is optional: skip when the extra is not installed."""

import pytest

pytest.importorskip("modal")


def test_modal_app_imports_without_starting_a_run():
    import modal

    from bdss import modal_app

    assert modal_app.APP_NAME == "bdss-m5"
    assert callable(modal_app._run_impl)
    assert isinstance(modal_app.run_spec, modal.Function)
    assert isinstance(modal_app.run_rescue_spec, modal.Function)
    assert isinstance(modal_app.run_final_spec, modal.Function)
    assert isinstance(modal_app.run_final_post, modal.Function)
    assert isinstance(modal_app.run_build_spec, modal.Function)
    assert modal_app._containers(True, None) == 16
    assert modal_app._containers(False, None) == 64
    assert modal_app._containers(False, None, build=True) == 16
    assert modal_app._containers(True, 4) == 4
    row = modal_app._row_for(("d", 5, 1000, "modal", 300, "grid"),
                             TimeoutError("timeout"))
    assert row["status"] == "timeout"
    assert row["_samples"] == []
    assert set(row) == set(__import__("bdss.campaign", fromlist=["POST_FIELDS"]).POST_FIELDS) | {"_samples"}
