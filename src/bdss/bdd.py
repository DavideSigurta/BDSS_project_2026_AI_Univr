"""BDD backend and graph-level helpers.

Counting and sizing are done here with our own bottom-up passes on Python integers; the
library `count` (a float on CUDD) is never the source of a reported number.
"""

from bdss.config import BDD_BACKEND


def _import_backend(name):
    if name == "autoref":
        from dd import autoref

        return autoref
    from dd import cudd

    return cudd


if BDD_BACKEND == "autoref":
    _backend = _import_backend("autoref")
    BACKEND = "autoref"
else:
    try:
        _backend = _import_backend("cudd")
        BACKEND = "cudd"
    except Exception:  # CUDD may fail at import time even when the wheel installed
        _backend = _import_backend("autoref")
        BACKEND = "autoref"


def new_manager():
    """Return a fresh BDD manager of the selected backend."""
    return _backend.BDD()


def size(root):
    """Unique BDD nodes reachable from `root`, terminals included.

    `dd` counts each underlying node once even when it is reached with both polarities
    (complemented edges); a manual walk over `Function.low/high` would count it twice.
    """
    return len(root)


def count_models(root, n_vars, mgr):
    """Number of assignments to the first `n_vars` (input) variables that satisfy `root`.

    Levels are 0-based.  Terminals sit at the end of the whole manager order, so their level
    is clamped to `n_vars`: state levels below the input block must not be counted as free
    input variables.
    """
    if root == mgr.false:
        return 0
    if root == mgr.true:
        return 1 << n_vars

    memo = {}

    def count(u):
        if u.var is None:
            return 1 if u == mgr.true else 0
        if u in memo:
            return memo[u]
        level = u.level
        # CUDD (and dd) store complement bits on edges; `.low`/`.high` are the children of
        # the underlying node, so a negated node has both cofactors negated.
        lo = ~u.low if u.negated else u.low
        hi = ~u.high if u.negated else u.high
        lo_level = n_vars if lo.var is None else min(lo.level, n_vars)
        hi_level = n_vars if hi.var is None else min(hi.level, n_vars)
        gap_lo = lo_level - level - 1
        gap_hi = hi_level - level - 1
        total = (1 << gap_lo) * count(lo) + (1 << gap_hi) * count(hi)
        memo[u] = total
        return total

    root_level = n_vars if root.var is None else min(root.level, n_vars)
    return (1 << root_level) * count(root)


def le_const(mgr, bits_lsb, tau):
    """BDD for `X <= tau`, with bits given least-significant first."""
    psi = mgr.true
    for j, bit in enumerate(bits_lsb):
        not_bit = ~bit
        if (tau >> j) & 1:
            psi = psi | not_bit
        else:
            psi = psi & not_bit
    return psi


def eq_const(mgr, bits_lsb, value):
    out = mgr.true
    for j, bit in enumerate(bits_lsb):
        literal = bit if (value >> j) & 1 else ~bit
        out = out & literal
    return out


def equiv(mgr, u, v):
    return (u & v) | (~u & ~v)


def xor(mgr, u, v):
    return (u & ~v) | (~u & v)
