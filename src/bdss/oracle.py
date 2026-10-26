"""Independent brute force used by tests and examples.

This module must not import the pipeline: it walks the sklearn arrays with its own loop and
simulates the election directly, so a bug in `forest.py` or in the BDD code cannot hide.
"""

import itertools
import numpy as np


def _leaf(tree, cells, tau):
    node = 0
    while tree.left[node] != -1:
        feature = int(tree.feature[node])
        if cells[feature] <= tau[node]:
            node = int(tree.left[node])
        else:
            node = int(tree.right[node])
    return int(node)


def votes(forest, cells, taus):
    counts = np.zeros(forest.n_classes, dtype=int)
    for tree, tau in zip(forest.trees, taus):
        counts[tree.leaf_label(_leaf(tree, cells, tau))] += 1
    return counts


def hard_label(forest, cells, taus):
    counts = votes(forest, cells, taus)
    best = int(counts.max())
    winners = np.flatnonzero(counts == best)
    if len(winners) != 1:
        return None
    return int(winners[0])


def cell_table(forest, disc):
    """All cell combinations, with votes and hard label."""
    taus = [disc.tau_array(tree) for tree in forest.trees]
    rows = []
    for cells in itertools.product(*(range(k) for k in disc.cells)):
        counts = votes(forest, cells, taus)
        rows.append((cells, counts, hard_label(forest, cells, taus)))
    return rows


def code_table(forest, disc):
    """All bit patterns, decoded (clamped) to cells."""
    taus = [disc.tau_array(tree) for tree in forest.trees]
    rows = []
    for code in disc.all_codes():
        bits = disc.code_bits(code)
        cells = disc.decode_bits(bits)
        counts = votes(forest, cells, taus)
        rows.append((code, bits, cells, counts, hard_label(forest, cells, taus)))
    return rows


def win_codes(forest, disc):
    """Map label index -> set of bit codes whose hard vote is that label."""
    out = {label: set() for label in range(forest.n_classes)}
    for code, _bits, _cells, _counts, label in code_table(forest, disc):
        if label is not None:
            out[label].add(code)
    return out


def cell_representatives(disc):
    """One real value per cell, for checking cell walks against real-value walks."""
    out = []
    for thresholds, k in zip(disc.thresholds, disc.cells):
        values = []
        if k == 1:
            values = [0.0]
        else:
            values.append(thresholds[0] - 1.0)
            for lo, hi in zip(thresholds[:-1], thresholds[1:]):
                values.append((lo + hi) / 2.0)
            values.append(thresholds[-1] + 1.0)
        out.append(values)
    return out


def simulate_prefix(forest, disc, cells, upto=None, total=None):
    """Independent state simulator after the first `upto` trees of a `total`-tree forest.

    Returns a list of (counters, flags) from the initial state (index 0) to after `upto`
    trees.  Counters are capped by the Clinch rule; a tie leaves no flag.
    """
    if upto is None:
        upto = forest.n_trees
    if total is None:
        total = forest.n_trees
    taus = [disc.tau_array(tree) for tree in forest.trees]
    n_labels = forest.n_classes
    counters = [0] * n_labels
    flags = [0] * n_labels
    states = [(tuple(counters), tuple(flags))]
    win_at = total // 2 + 1
    for t in range(upto):
        if any(flags):
            states.append((tuple(counters), tuple(flags)))
            continue
        tree = forest.trees[t]
        label = tree.leaf_label(_leaf(tree, cells, taus[t]))
        r = total - 1 - t
        clinches = all(
            counters[label] >= counters[rival] + r
            for rival in range(n_labels)
            if rival != label
        )
        if clinches:
            flags[label] = 1
        else:
            counters[label] += 1
            assert counters[label] < win_at, "counter reached W without clinching"
        states.append((tuple(counters), tuple(flags)))
    return states
