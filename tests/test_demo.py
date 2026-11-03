"""The local demo is read-only and runs on the shipped campaign."""

import sys

from bdss import study


def test_demo_prints_one_iris_sample(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bdss", "demo", "--dataset", "iris", "--samples", "1"])
    study.demo_main()
    out = capsys.readouterr().out
    assert "model iris" in out
    assert "certified counts" in out
    assert "sample 0: own=Setosa" in out
    assert "witness bits=" in out
