"""BDD basics: sizes, integer model counts, comparators."""

import pytest

from bdss import bdd


def _manager_with_inputs(n):
    mgr = bdd.new_manager()
    names = [f"x{i}" for i in range(n)]
    mgr.declare(*names)
    return mgr, names


def test_size_counts_unique_nodes():
    mgr, names = _manager_with_inputs(3)
    u = mgr.add_expr("x0 & x1")
    assert bdd.size(u) == 3  # x0, x1, one constant node
    assert bdd.size(mgr.true) == 1
    assert bdd.size(mgr.false) == 1


def test_size_with_negated_edges():
    """A node reached with both polarities must be counted once, not twice."""
    mgr, names = _manager_with_inputs(2)
    u = mgr.add_expr("x0 ^ x1")
    assert u.negated  # the diagram uses complemented edges
    assert bdd.size(u) == len(u) == 3
    assert bdd.size(mgr.add_expr("x0 & ~x1")) == 3


def test_count_models_against_bruteforce():
    mgr, names = _manager_with_inputs(4)
    exprs = ["x0 & x1", "x0 | x1", "x0 ^ x1", "(x0 & x1) | (x2 & x3)", "~x0 | x3"]
    for expr in exprs:
        u = mgr.add_expr(expr)
        expected = 0
        for code in range(1 << 4):
            values = {name: bool((code >> (3 - i)) & 1) for i, name in enumerate(names)}
            if mgr.let(values, u) == mgr.true:
                expected += 1
        assert bdd.count_models(u, 4, mgr) == expected, expr


def test_count_skipped_variables():
    mgr, names = _manager_with_inputs(4)
    u = mgr.add_expr("x1 & x3")  # x0 and x2 are skipped
    assert bdd.count_models(u, 4, mgr) == 4


def test_le_const_against_bruteforce():
    mgr, names = _manager_with_inputs(3)
    bits = [mgr.var("x0"), mgr.var("x1"), mgr.var("x2")]  # LSB first
    for tau in range(8):
        u = bdd.le_const(mgr, bits, tau)
        for code in range(8):
            values = {f"x{i}": bool((code >> i) & 1) for i in range(3)}
            got = mgr.let(values, u) == mgr.true
            assert got == (code <= tau), (tau, code)


def test_eq_const_and_xor():
    mgr, names = _manager_with_inputs(2)
    bits = [mgr.var("x0"), mgr.var("x1")]
    u = bdd.eq_const(mgr, bits, 2)
    assert bdd.count_models(u, 2, mgr) == 1
    x = bdd.xor(mgr, mgr.var("x0"), mgr.var("x1"))
    assert bdd.count_models(x, 2, mgr) == 2
