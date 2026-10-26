"""Five split-variable selectors for the budgeted frontier."""

from bdss import readouts

SELECTORS = ("ig", "shapley", "pvalue", "random", "modal")


def _support_levels(mgr, root):
    names = mgr.support(root)
    return sorted({mgr.level_of_var(name) for name in names})


def _node_levels(root, n_bits):
    """Unique nodes of the diagram, as (level) with each node counted once."""
    found = {}
    stack = [root]
    seen = set()
    while stack:
        u = stack.pop()
        if u.var is None:
            continue
        key = (u.var, u.level, u.low, u.high)
        if key in seen:
            continue
        seen.add(key)
        if u.level < n_bits:
            found[u.level] = found.get(u.level, 0) + 1
        stack.append(u.low)
        stack.append(u.high)
    return found


def select(root, mgr, disc, kind, rng):
    """Return the input level to split on, or None when the diagram has no input variable."""
    if kind not in SELECTORS:
        raise ValueError(f"unknown selector {kind!r}")
    levels = _support_levels(mgr, root)
    candidates = [level for level in levels if level < disc.n_bits]
    if not candidates:
        return None

    # The input projection of the accumulator is true (every input reaches some state), so
    # the semantic criteria are evaluated on the accumulator's own solution set, which also
    # contains the state bits; only input bits are eligible for the split.
    if kind == "ig":
        score = readouts.info_gain(mgr, root, levels)
        return max(candidates, key=lambda level: (score[level], -level))
    if kind == "shapley":
        score = readouts.shapley(mgr, root, levels)
        return min(candidates, key=lambda level: (score[level], level))
    if kind == "pvalue":
        score = readouts.pvalues(mgr, root, levels)
        return min(candidates, key=lambda level: (score[level]["s"], level))
    if kind == "modal":
        counts = _node_levels(root, disc.n_bits)
        return max(levels, key=lambda level: (counts.get(level, 0), -level))

    # Random node: uniform over the nodes that test an input bit.
    counts = _node_levels(root, disc.n_bits)
    nodes = [level for level, count in counts.items() for _ in range(count)]
    return rng.choice(nodes)
