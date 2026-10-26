"""Transitions: offset comparator and state after every prefix."""

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.toy_forest import build_toy_forest  # noqa: E402
from bdss import bdd, compose, oracle, transitions  # noqa: E402
from bdss.discretize import Disc  # noqa: E402


def test_offset_comparator_matches_integer_ge():
    width = 4
    mgr = bdd.new_manager()
    mgr.declare(*[f"a{j}" for j in range(width)], *[f"b{j}" for j in range(width)])
    a_bits = [mgr.var(f"a{j}") for j in range(width)]
    b_bits = [mgr.var(f"b{j}") for j in range(width)]
    for r in range(1 << width):
        u = transitions._clinch_once(mgr, a_bits, b_bits, r)
        for a in range(1 << width):
            for b in range(1 << width):
                assignment = {
                    **{f"a{j}": bool((a >> j) & 1) for j in range(width)},
                    **{f"b{j}": bool((b >> j) & 1) for j in range(width)},
                }
                got = mgr.let(assignment, u) == mgr.true
                # The comparator is used only when b + r fits M+1 bits (CONDIZIONI §4);
                # overflowing values are outside its contract.
                if b + r < (1 << width):
                    assert got == (a >= b + r), (a, b, r)


def test_prefix_states_against_simulator():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    n_labels = forest.n_classes
    width = transitions.counter_width(forest.n_trees)
    counter_values = range(1 << width)
    flag_values = (0, 1)
    state_space = [
        (counters, flags)
        for counters in itertools.product(counter_values, repeat=n_labels)
        for flags in itertools.product(flag_values, repeat=n_labels)
    ]

    for upto in range(forest.n_trees + 1):
        mgr, acc, _peak, _width, _seconds = compose.accumulator(forest, disc, upto)
        input_names = transitions.input_names(disc)
        state_names = transitions.current_names(n_labels, width)
        expected_by_code = {}
        for code in disc.all_codes():
            cells = disc.code_cells(code)
            states = oracle.simulate_prefix(forest, disc, cells, upto=upto)
            counters, flags = states[upto]
            if any(flags):
                expected = {(c, flags) for c in itertools.product(counter_values, repeat=n_labels)}
            else:
                expected = {(counters, flags)}
            expected_by_code[code] = expected

        for code in disc.all_codes():
            bits = disc.code_bits(code)
            expected = expected_by_code[code]
            for counters, flags in state_space:
                assignment = {name: bool(bit) for name, bit in zip(input_names, bits)}
                for label in range(n_labels):
                    for j in range(width):
                        assignment[transitions.c_name(label, j)] = bool((counters[label] >> j) & 1)
                    assignment[transitions.w_name(label)] = bool(flags[label])
                got = mgr.let(assignment, acc) == mgr.true
                assert got == ((counters, flags) in expected), (upto, code, counters, flags)
