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
