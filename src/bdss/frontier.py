"""Five split-variable selectors for the budgeted frontier.

The semantic selectors score the *certified region reached so far*,
`∃c (A ∧ ⋁_ℓ w_ℓ)`, not the accumulator A: the input projection of A is true (every input
reaches a state) and even A's own solution set treats every input bit uniformly.  When no input
is certified yet, the semantic selectors fall back to the structural modal variable.
"""

from bdss import readouts, transitions

SELECTORS = ("ig", "shapley", "pvalue", "random", "modal")


def _support_levels(mgr, root, n_bits):
    names = mgr.support(root)
    return sorted(level for level in (mgr.level_of_var(name) for name in names) if level < n_bits)


def _node_levels(root, n_bits):
    """Unique input-testing nodes of the diagram, counted once per level."""
    counts = {}
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
            counts[u.level] = counts.get(u.level, 0) + 1
        stack.append(u.low)
        stack.append(u.high)
    return counts


def _modal(mgr, root, candidates, n_bits):
    counts = _node_levels(root, n_bits)
    return max(candidates, key=lambda level: (counts.get(level, 0), -level))


def select(root, mgr, disc, kind, rng, n_labels, width, free_levels):
    """Return the input level to split on, or None when no free input is in the diagram."""
    if kind not in SELECTORS:
        raise ValueError(f"unknown selector {kind!r}")
    free_levels = sorted(set(free_levels))
    candidates = [level for level in _support_levels(mgr, root, disc.n_bits)
                  if level in free_levels]
    if not candidates:
        return None

    if kind == "modal":
        return _modal(mgr, root, candidates, disc.n_bits)
    if kind == "random":
        counts = _node_levels(root, disc.n_bits)
        nodes = [level for level in candidates for _ in range(counts.get(level, 0))]
        return rng.choice(nodes)

    state_names = transitions.current_names(n_labels, width)
    won = mgr.false
    for label in range(n_labels):
        won = won | transitions.flag(mgr, label)
    certified = mgr.exist(state_names, root & won)

    if certified == mgr.false:
        return _modal(mgr, root, candidates, disc.n_bits)

    if kind == "ig":
        score = readouts.info_gain(mgr, certified, free_levels)
        return max(candidates, key=lambda level: (score[level], -level))
    if kind == "shapley":
        score = readouts.shapley(mgr, certified, free_levels)
        return min(candidates, key=lambda level: (score[level], level))

    score = readouts.pvalues(mgr, certified, free_levels)
    return min(candidates, key=lambda level: (score[level]["s"], level))
