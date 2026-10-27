"""Per-bit rankings against brute force on the toy forest."""

import math
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.toy_forest import build_toy_forest  # noqa: E402
from bdss import compose, oracle, readouts  # noqa: E402
from bdss.discretize import Disc  # noqa: E402


def _entropy(q):
    if q in (0.0, 1.0):
        return 0.0
    return -q * math.log2(q) - (1 - q) * math.log2(1 - q)


def _brute(forest, disc, label):
    models = oracle.win_codes(forest, disc)[label]
    n = disc.n_bits
    total = len(models)
    ig, phi, pv = {}, {}, {}
    for level in range(n):
        n1 = sum(1 for code in models if (code >> (n - 1 - level)) & 1)
        ig[level] = _entropy(n1 / total) if total else 0.0
        phi[level] = -sum(
            1 / sum(1 for i in range(n) if not ((code >> (n - 1 - i)) & 1))
            for code in models
            if not ((code >> (n - 1 - level)) & 1)
        )

        def tail(a):
            if a <= 0:
                return 1.0
            draws = 1 << (n - 1)
            return float(
                Fraction(
                    sum(comb(draws, i) * comb(draws, total - i)
                        for i in range(a, min(total, draws) + 1)),
                    comb(1 << n, total),
                )
            )

        a1 = n1
        a0 = total - a1
        s = min(tail(a1), tail(a0))
        pv[level] = {"p1": tail(a1), "p0": tail(a0), "s": s,
                     "bonferroni": min(1.0, s * 2 * n)}
    return {"ig": ig, "shapley": phi, "pvalue": pv, "total": total}


def test_rankings_match_bruteforce():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    levels = list(range(disc.n_bits))
    for label in range(forest.n_classes):
        got = readouts.rankings(build.mgr, build.win(label), levels)
        expected = _brute(forest, disc, label)
        for level in levels:
            assert math.isclose(got["ig"][level], expected["ig"][level], rel_tol=1e-12,
                                abs_tol=1e-12)
            assert math.isclose(got["shapley"][level], expected["shapley"][level],
                                rel_tol=1e-12, abs_tol=1e-12)
            for key in ("p1", "p0", "s", "bonferroni"):
                assert math.isclose(got["pvalue"][level][key], expected["pvalue"][level][key],
                                    rel_tol=1e-12, abs_tol=1e-12), (label, level, key)
        # Efficiency: sum phi = v(V) - #f.
        v_all = 1 if (1 << disc.n_bits) - 1 in oracle.win_codes(forest, disc)[label] else 0
        assert math.isclose(sum(got["shapley"].values()), v_all - expected["total"],
                            rel_tol=1e-12, abs_tol=1e-12)


def test_ranks_are_permutations():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    got = readouts.rankings(build.mgr, build.win(0), list(range(disc.n_bits)))
    for key in ("rank_ig", "rank_shapley", "rank_p"):
        assert sorted(got[key].values()) == list(range(1, disc.n_bits + 1))


def test_hypergeom_normal_branch_matches_normal_sf():
    # The declared normal fallback must agree with an independent normal tail.
    from bdss import readouts
    from scipy.stats import norm

    n = 1024
    population = 1 << n
    draws = 1 << (n - 1)
    successes = 1 << (n - 1)
    a = successes >> 1
    p = readouts._hypergeom_sf(a, population, successes, draws)
    mean = successes / 2.0
    var = successes / 4.0
    z = (a - 0.5 - mean) / math.sqrt(var)
    assert math.isclose(p, norm.sf(z), rel_tol=1e-12, abs_tol=1e-15)


def test_shapley_against_coalition_definition():
    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    build = compose.build(forest, disc)
    n = disc.n_bits
    for label in range(forest.n_classes):
        models = oracle.win_codes(forest, disc)[label]
        value = {}
        for mask in range(1 << n):
            value[mask] = sum(
                1
                for code in models
                if all((code >> (n - 1 - level)) & 1 for level in range(n) if (mask >> level) & 1)
            )
        got = readouts.shapley(build.mgr, build.win(label), list(range(n)))
        for level in range(n):
            expected = 0.0
            for mask in range(1 << n):
                if (mask >> level) & 1:
                    continue
                size = bin(mask).count("1")
                weight = math.factorial(size) * math.factorial(n - 1 - size) / math.factorial(n)
                expected += weight * (value[mask | (1 << level)] - value[mask])
            assert math.isclose(got[level], expected, rel_tol=1e-12, abs_tol=1e-12), (label, level)


def test_hypergeom_normal_branch_general_parameters():
    from bdss import readouts

    n = 17
    population = 1 << n
    draws = 1 << (n - 1)
    successes = (1 << (n - 1)) + (1 << (n - 3))
    a = successes >> 1
    p = readouts._hypergeom_sf(a, population, successes, draws)
    logp = readouts._hypergeom_logsf(a, population, successes, draws)
    rate = successes / population
    mean = draws * rate
    var = draws * rate * (1.0 - rate) * ((population - draws) / (population - 1))
    z = (a - 0.5 - mean) / math.sqrt(var)
    from scipy.stats import norm

    expected = norm.sf(z)
    assert math.isclose(p, expected, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(logp, math.log(expected), rel_tol=1e-12, abs_tol=0.0)


def test_normal_logsf_matches_scipy_log_ndtr():
    from bdss import readouts
    from scipy.special import log_ndtr

    for z in (6.0, 8.485, 20.0, 26.0, 30.0, 505.0):
        assert math.isclose(readouts._normal_logsf(z), log_ndtr(-z), rel_tol=1e-9,
                            abs_tol=1e-9), z


def test_hypergeom_huge_population_does_not_overflow():
    from bdss import readouts

    n = 2000
    population = 1 << n
    draws = population >> 1
    successes = (population >> 1) + (population >> 3)
    a = successes >> 1
    p = readouts._hypergeom_sf(a, population, successes, draws)
    logp = readouts._hypergeom_logsf(a, population, successes, draws)
    assert 0.0 <= p <= 1.0
    assert logp <= 0.0
    # An extreme upper tail stays a valid zero, not an OverflowError.
    assert readouts._hypergeom_sf(successes, population, successes, draws) == 0.0


def test_hypergeom_normal_branch_is_close_to_scipy_at_17_and_20():
    from scipy.stats import hypergeom
    from bdss import readouts

    for n in (17, 20):
        population = 1 << n
        draws = population >> 1
        successes = (population >> 1) + (population >> 3)
        a = successes >> 1
        got = readouts._hypergeom_sf(a, population, successes, draws)
        exact = float(hypergeom.sf(a - 1, population, successes, draws))
        assert abs(got - exact) < 1e-8
