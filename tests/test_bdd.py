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


def test_count_active_noncontiguous_levels():
    mgr, names = _manager_with_inputs(3)
    u = mgr.add_expr("x0 & x2")
    assert bdd.count_active(u, [0, 2], mgr) == 1
    assert bdd.count_active(u, [0, 1, 2], mgr) == 2  # x1 free
    assert bdd.count_models(u, 3, mgr) == 2


def test_zero_strata_against_bruteforce():
    mgr, names = _manager_with_inputs(3)
    for expr in ["x0 & x1", "x0 ^ x1", "x0", "~x0", "x0 & x2", "x1 | x2"]:
        u = mgr.add_expr(expr)
        expected = [0] * 4
        for code in range(8):
            assignment = {f"x{i}": bool((code >> i) & 1) for i in range(3)}
            if mgr.let(assignment, u) == mgr.true:
                zeros = sum(1 for bit in assignment.values() if not bit)
                expected[zeros] += 1
        got = bdd.zero_strata(u, [0, 1, 2], mgr)
        assert got == expected, (expr, got, expected)
        assert sum(got) == bdd.count_models(u, 3, mgr)


def test_zero_strata_noncontiguous_levels():
    mgr, names = _manager_with_inputs(3)
    assert bdd.zero_strata(mgr.var("x1"), [0, 1, 2], mgr) == [1, 2, 1, 0]
    assert bdd.zero_strata(mgr.var("x2"), [0, 2], mgr) == [1, 1, 0]


def test_dynamic_reordering_is_disabled():
    """CUDD enables dynamic reordering by default; levels must stay the declared order."""
    mgr = bdd.new_manager()
    names = [f"x{i}" for i in range(6)]
    mgr.declare(*names)
    assert mgr.configure()["reordering"] is False
    u = mgr.add_expr("(x0 & x1) | (x2 & x3) | (x4 & x5)")
    assert [mgr.level_of_var(name) for name in names] == list(range(6))
    assert bdd.size(u) > 0
