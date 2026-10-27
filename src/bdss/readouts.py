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
    """Upper tail P[X >= a] for X ~ Hypergeometric(population, successes, draws).

    scipy is used only while the parameters fit comfortably in float64 integers (n <= 16,
    i.e. population <= 2**16).  At n = 21 the exact table on the tails already costs more
    than the analysis; the normal approximation is the declared tail for larger n, with the
    continuity correction.
    """
    if a <= 0:
        return 1.0
    if a > min(successes, draws):
        return 0.0
    n = population.bit_length() - 1
    if n <= 16:
        from scipy.stats import hypergeom

        return float(hypergeom.sf(int(a) - 1, int(population), int(successes), int(draws)))
    # Ratios stay finite even when population > 2**1023; do not cast the integers first.
    p = successes / population
    q = draws / population
    mean = draws * p
    if p in (0.0, 1.0) or q in (0.0, 1.0):
        return 1.0 if a <= mean else 0.0
    var = draws * p * (1.0 - p) * ((population - draws) / (population - 1))
    if var <= 0.0:
        return 1.0 if a <= mean else 0.0
    z = (float(a) - 0.5 - mean) / math.sqrt(var)
    return min(1.0, max(0.0, 0.5 * math.erfc(z / math.sqrt(2.0))))


def _normal_logsf(z):
    """log of the normal upper tail, stable when the tail underflows."""
    x = z / math.sqrt(2.0)
    if x <= 0.0:
        return math.log(0.5 * math.erfc(x))
    if x < 26.0:
        tail = 0.5 * math.erfc(x)
        if tail > 0.0:
            return math.log(tail)
    # erfc(x) ~ exp(-x*x) / (x*sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4) - ...)
    series = 1.0 - 1.0 / (2.0 * x * x) + 3.0 / (4.0 * x**4) - 15.0 / (8.0 * x**6)
    return math.log(0.5) - x * x - math.log(x * math.sqrt(math.pi)) + math.log1p(series - 1.0)


def _hypergeom_logsf(a, population, successes, draws):
    """Natural log of `_hypergeom_sf`, with the same branches."""
    if a <= 0:
        return 0.0
    if a > min(successes, draws):
        return -math.inf
    n = population.bit_length() - 1
    if n <= 16:
        p = _hypergeom_sf(a, population, successes, draws)
        return math.log(p) if p > 0.0 else -math.inf
    p = successes / population
    q = draws / population
    mean = draws * p
    if p in (0.0, 1.0) or q in (0.0, 1.0):
        return 0.0 if a <= mean else -math.inf
    var = draws * p * (1.0 - p) * ((population - draws) / (population - 1))
    if var <= 0.0:
        return 0.0 if a <= mean else -math.inf
    z = (float(a) - 0.5 - mean) / math.sqrt(var)
    return _normal_logsf(z)


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
        lp1 = _hypergeom_logsf(a1, population, successes, draws)
        lp0 = _hypergeom_logsf(a0, population, successes, draws)
        out[level] = {"p1": p1, "p0": p0, "s": s,
                      "logp1": lp1, "logp0": lp0, "logp": min(lp1, lp0),
                      "bonferroni": min(1.0, s * 2 * n)}
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
    rank_logp = {
        level: i + 1
        for i, level in enumerate(sorted(levels, key=lambda x: (pv[x]["logp"], x)))
    }
    return {"ig": ig, "shapley": phi, "pvalue": pv,
            "rank_ig": rank_ig, "rank_shapley": rank_phi, "rank_p": rank_p,
            "rank_logp": rank_logp}


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
