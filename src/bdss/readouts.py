"""Readouts: certified counts/volumes and per-bit rankings.

All quantities come from the BDD by our own integer / bottom-up passes; the library `count`
(float on CUDD) is never used for a reported number.
"""

import math

from bdss import bdd


def _bit(mgr, level):
    return mgr.var(mgr.var_at_level(level))


def info_gain(mgr, root, levels):
    """Binary entropy of the split induced by each active input bit."""
    levels = list(levels)
    n_models = bdd.count_active(root, levels, mgr)
    out = {}
    for level in levels:
        n1 = bdd.count_active(root & _bit(mgr, level), levels, mgr)
        q = n1 / n_models if n_models else 0.0
        if q in (0.0, 1.0):
            out[level] = 0.0
        else:
            out[level] = -q * math.log2(q) - (1 - q) * math.log2(1 - q)
    return out


def shapley(mgr, root, levels):
    """Shapley value of each active input bit in the coalition counting game."""
    levels = list(levels)
    n = len(levels)
    out = {}
    for level in levels:
        g = root & ~_bit(mgr, level)
        strata = bdd.zero_strata(g, levels, mgr)
        out[level] = -sum(strata[z] / z for z in range(1, n + 1))
    return out


def _hypergeom_sf(a, population, successes, draws):
    """Upper tail P[X >= a] for X ~ Hypergeometric(population, successes, draws)."""
    if a <= 0:
        return 1.0
    if a > min(successes, draws):
        return 0.0
    n = population.bit_length() - 1
    if n <= 1023:
        from scipy.stats import hypergeom

        return float(hypergeom.sf(a - 1, population, successes, draws))
    # Beyond the double range: normal approximation of the same table, declared in the
    # report.  With draws = population / 2 the mean is successes / 2 and the variance is
    # approximately successes / 4.
    mean = successes / 2.0
    var = successes / 4.0
    if var <= 0.0:
        return 0.0
    z = (a - 0.5 - mean) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def pvalues(mgr, root, levels):
    """One-sided hypergeometric p-values for both literals of each active bit."""
    levels = list(levels)
    n = len(levels)
    population = 1 << n
    draws = population >> 1
    successes = bdd.count_active(root, levels, mgr)
    out = {}
    for level in levels:
        a1 = bdd.count_active(root & _bit(mgr, level), levels, mgr)
        a0 = successes - a1
        p1 = _hypergeom_sf(a1, population, successes, draws)
        p0 = _hypergeom_sf(a0, population, successes, draws)
        s = min(p1, p0)
        out[level] = {"p1": p1, "p0": p0, "s": s, "bonferroni": min(1.0, s * 2 * n)}
    return out


def rankings(mgr, root, levels):
    """IG, Shapley and p-value for every active bit, plus the three ranks."""
    levels = list(levels)
    ig = info_gain(mgr, root, levels)
    phi = shapley(mgr, root, levels)
    pv = pvalues(mgr, root, levels)
    rank_ig = {level: i + 1 for i, level in enumerate(sorted(levels, key=lambda x: (-ig[x], x)))}
    rank_phi = {level: i + 1 for i, level in enumerate(sorted(levels, key=lambda x: (phi[x], x)))}
    rank_p = {
        level: i + 1
        for i, level in enumerate(sorted(levels, key=lambda x: (pv[x]["s"], x)))
    }
    return {"ig": ig, "shapley": phi, "pvalue": pv,
            "rank_ig": rank_ig, "rank_shapley": rank_phi, "rank_p": rank_p}


def label_counts(build):
    """Certified input count per label; frontier leaves are summed, never unioned."""
    n_bits = build.disc.n_bits
    counts = {}
    for label in range(build.forest.n_classes):
        if len(build.leaves) == 1 and not build.leaves[0].rho:
            counts[label] = bdd.count_models(build.win(label), n_bits, build.mgr)
            continue
        total = 0
        for leaf in build.leaves:
            win = build.leaf_win(leaf, label)
            active = [level for level in range(n_bits) if level not in leaf.rho]
            total += bdd.count_active(win, active, build.mgr)
        counts[label] = total
    return counts


def label_volumes(build):
    total = 1 << build.disc.n_bits
    return {label: count / total for label, count in label_counts(build).items()}


def label_rankings(build):
    """Per-label rankings over all input bits (the union is built for a frontier)."""
    levels = list(range(build.disc.n_bits))
    return {
        label: rankings(build.mgr, build.win(label), levels)
        for label in range(build.forest.n_classes)
    }
