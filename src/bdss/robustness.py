"""Robustness of one sample: minimum bit-switch distance to a certified label.

Two distances are reported with distinct names:
- `min_rival_radius`: the nearest rival label (targeted attack);
- `loss_radius`: losing certification of the sample's own label (the complement region).
They coincide for binary forests with an odd number of trees.
"""

import itertools
import math

INF = math.inf


def distance_bits(mgr, root, sample):
    """Return (minimum Hamming distance, one realizing list of flipped levels).

    Skipped input variables cost 0 (they can be set to match the sample).  Complemented
    edges are unfolded with `~child`.
    """
    memo = {}

    def go(u):
        if u.var is None:
            return (0, ()) if u == mgr.true else (INF, ())
        if u in memo:
            return memo[u]
        level = u.level
        lo = ~u.low if u.negated else u.low
        hi = ~u.high if u.negated else u.high
        bit = sample[level]
        d0, b0 = go(lo)
        d1, b1 = go(hi)
        cost0 = 0 if bit == 0 else 1
        cost1 = 0 if bit == 1 else 1
        if cost0 + d0 <= cost1 + d1:
            out = (cost0 + d0, ((level,) if cost0 else ()) + b0)
        else:
            out = (cost1 + d1, ((level,) if cost1 else ()) + b1)
        memo[u] = out
        return out

    return go(root)


def evaluate(mgr, root, bits, names):
    """Evaluate a BDD over all input bits with the given variable names."""
    return mgr.let({name: bool(bit) for name, bit in zip(names, bits)}, root) == mgr.true


def radius_from_win(mgr, win, bits, names, max_depth=3):
    """Targeted Hamming distance by layers; fall back to the exact BDD DP beyond depth.

    Most radii on real models are 0-3, so enumerating flip combinations is much cheaper
    than a full BDD traversal per sample; the DP is used only when the BFS misses.
    """
    if evaluate(mgr, win, bits, names):
        return 0, ()
    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(range(len(bits)), depth):
            sample = list(bits)
            for level in combo:
                sample[level] ^= 1
            assignment = {name: bool(bit) for name, bit in zip(names, sample)}
            if mgr.let(assignment, win) == mgr.true:
                return depth, combo
    return distance_bits(mgr, win, dict(enumerate(bits)))


def radius(build, code, target):
    """Targeted distance to the region certified as `target`."""
    sample = dict(enumerate(build.disc.code_bits(code)))
    best = (INF, ())
    for leaf in build.leaves:
        win = build.leaf_win(leaf, target)
        d, bits = distance_bits(build.mgr, win, sample)
        entry = tuple(level for level, value in leaf.rho.items() if sample[level] != value)
        total = len(entry) + d
        if total < best[0]:
            best = (total, entry + bits)
    return best


def loss_radius_from_win(mgr, win, bits, names, max_depth=3):
    """Distance to the complement of a certified region, by layers then exact DP."""
    complement = ~win
    if evaluate(mgr, complement, bits, names):
        return 0, ()
    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(range(len(bits)), depth):
            sample = list(bits)
            for level in combo:
                sample[level] ^= 1
            assignment = {name: bool(bit) for name, bit in zip(names, sample)}
            if mgr.let(assignment, complement) == mgr.true:
                return depth, combo
    return distance_bits(mgr, complement, dict(enumerate(bits)))


def loss_radius(build, code, own_label):
    """Distance to an input that no longer certifies `own_label`."""
    sample = dict(enumerate(build.disc.code_bits(code)))
    best = (INF, ())
    for leaf in build.leaves:
        win = build.leaf_win(leaf, own_label)
        d, bits = distance_bits(build.mgr, ~win, sample)
        entry = tuple(level for level, value in leaf.rho.items() if sample[level] != value)
        total = len(entry) + d
        if total < best[0]:
            best = (total, entry + bits)
    return best


def min_rival_radius(build, code, own_label):
    """Nearest rival label and its targeted distance."""
    best = (INF, (), None)
    for target in range(build.forest.n_classes):
        if target == own_label:
            continue
        d, bits = radius(build, code, target)
        if d < best[0]:
            best = (d, bits, target)
    return best
