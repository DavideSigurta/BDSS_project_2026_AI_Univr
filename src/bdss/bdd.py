"""BDD backend and graph-level helpers.

Counting and sizing are done here with our own bottom-up passes on Python integers; the
library `count` (a float on CUDD) is never the source of a reported number.
"""

import math

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
    """Return a fresh BDD manager, with the variable order fixed once.

    CUDD enables dynamic reordering by default; a reordered level would no longer be the
    input bit index (PLAN decision 13: no dynamic reordering).
    """
    mgr = _backend.BDD()
    mgr.configure(reordering=False)
    return mgr


def size(root):
    """Unique BDD nodes reachable from `root`, terminals included.

    `dd` counts each underlying node once even when it is reached with both polarities
    (complemented edges); a manual walk over `Function.low/high` would count it twice.
    """
    return len(root)


def _level_table(active_levels):
    active = sorted(set(active_levels))
    if not active:
        return active, [0], 0
    limit = active[-1] + 1
    active_set = set(active)
    prefix = [0] * (limit + 1)
    for k in range(1, limit + 1):
        prefix[k] = prefix[k - 1] + (1 if (k - 1) in active_set else 0)
    return active, prefix, limit


def count_active(root, active_levels, mgr):
    """Number of assignments of the `active_levels` variables that satisfy `root`.

    Levels are 0-based.  Only the listed variables are counted; state levels (or fixed input
    bits) below the input block are ignored.  The terminal level is clamped to the last active
    level, so it does not add free state variables.
    """
    active, prefix, limit = _level_table(active_levels)
    if not active:
        return 1 if root == mgr.true else 0
    active_set = set(active)
    n = len(active)

    def level(u):
        return limit if u.var is None or u.level >= limit else u.level

    def count(u):
        if u.var is None:
            return 1 if u == mgr.true else 0
        if u in memo:
            return memo[u]
        i = u.level
        if i not in active_set:
            raise ValueError(f"node level {i} is not in the active set")
        lo = ~u.low if u.negated else u.low
        hi = ~u.high if u.negated else u.high
        gap_lo = prefix[level(lo)] - prefix[i + 1]
        gap_hi = prefix[level(hi)] - prefix[i + 1]
        total = (1 << gap_lo) * count(lo) + (1 << gap_hi) * count(hi)
        memo[u] = total
        return total

    memo = {}
    if root.var is None:
        return (1 << n) if root == mgr.true else 0
    return (1 << prefix[level(root)]) * count(root)


def count_models(root, n_vars, mgr):
    """Special case of `count_active` over the first `n_vars` variables."""
    return count_active(root, range(n_vars), mgr)


def zero_strata(root, active_levels, mgr):
    """`M[z]` = number of satisfying assignments with exactly z active variables set to 0."""
    active, prefix, limit = _level_table(active_levels)
    n = len(active)
    if n == 0:
        return [1] if root == mgr.true else [0]
    active_set = set(active)

    def level(u):
        return limit if u.var is None or u.level >= limit else u.level

    def mul_free(p, gap):
        # Multiply by (1 + t)^gap: each skipped active variable is free.
        q = list(p)
        for _ in range(gap):
            for k in range(n, 0, -1):
                q[k] += q[k - 1]
        return q

    def shift(p):
        # Low branch: the node variable is 0, so add one zero.
        return [0] + p[:n]

    memo = {}

    def poly(u):
        if u.var is None:
            if u == mgr.true:
                return [1] + [0] * n
            return [0] * (n + 1)
        if u in memo:
            return memo[u]
        if u.negated:
            # Complement over the variables at and below this node's level.
            i = u.level
            m = n - prefix[i]
            p = poly(~u)
            q = [math.comb(m, z) - p[z] for z in range(n + 1)]
            memo[u] = q
            return q
        i = u.level
        if i not in active_set:
            raise ValueError(f"node level {i} is not in the active set")
        lo = ~u.low if u.negated else u.low
        hi = ~u.high if u.negated else u.high
        gap_lo = prefix[level(lo)] - prefix[i + 1]
        gap_hi = prefix[level(hi)] - prefix[i + 1]
        p_lo = mul_free(shift(poly(lo)), gap_lo)
        p_hi = mul_free(poly(hi), gap_hi)
        q = [a + b for a, b in zip(p_lo, p_hi)]
        memo[u] = q
        return q

    p = poly(root)
    return mul_free(p, prefix[level(root)])


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
