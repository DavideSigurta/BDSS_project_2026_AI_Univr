"""Full composition: certified regions equal the hard-vote regions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.toy_forest import build_toy_forest  # noqa: E402
from bdss import bdd, compose, oracle, readouts, transitions  # noqa: E402
from bdss.discretize import Disc  # noqa: E402


def _eval_win(mgr, win, code, disc):
    bits = disc.code_bits(code)
    names = transitions.input_names(disc)
    return mgr.let({name: bool(bit) for name, bit in zip(names, bits)}, win) == mgr.true


def test_win_matches_hard_vote():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    wins = {label: build.win(label) for label in range(forest.n_classes)}
    wins_by_code = oracle.win_codes(forest, disc)
    for code in disc.all_codes():
        got = [label for label, win in wins.items() if _eval_win(build.mgr, win, code, disc)]
        expected = [label for label, codes in wins_by_code.items() if code in codes]
        assert got == expected, (code, got, expected)
        assert len(got) == 1  # binary, three trees


def test_counts_and_volumes():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    counts = readouts.label_counts(build)
    oracle_counts = {label: len(codes) for label, codes in oracle.win_codes(forest, disc).items()}
    assert counts == oracle_counts
    assert sum(counts.values()) == 1 << disc.n_bits
    volumes = readouts.label_volumes(build)
    assert volumes == {label: count / 16 for label, count in counts.items()}
    disjoint = build.win(0) & build.win(1)
    assert disjoint == build.mgr.false
    assert bdd.size(build.win(0)) == 6
    assert bdd.size(build.leaves[0].A) == 9
    assert build.peak == 19
