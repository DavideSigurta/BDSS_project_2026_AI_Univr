"""Static feature order and permutation shared by M3/M4/M5.

A feature used near the root of a tree constrains the whole tree, so it goes first; the
order is chosen once and never changed during a build.  Keeping these helpers outside the
study module prevents the pipeline from importing the CLI layer.
"""

from __future__ import annotations

import numpy as np

from bdss.discretize import Disc
from bdss.forest import Forest, Tree


def shallow_order(forest):
    """Features by minimum depth of first appearance, then frequency, then index."""
    n_features = forest.n_features
    depth = [None] * n_features
    frequency = [0] * n_features

    def walk(tree, node, level):
        if tree.left[node] == -1:
            return
        feature = int(tree.feature[node])
        if depth[feature] is None or level < depth[feature]:
            depth[feature] = level
        frequency[feature] += 1
        walk(tree, int(tree.left[node]), level + 1)
        walk(tree, int(tree.right[node]), level + 1)

    for tree in forest.trees:
        walk(tree, 0, 0)
    return tuple(
        sorted(
            range(n_features),
            key=lambda f: (10**9 if depth[f] is None else depth[f], -frequency[f], f),
        )
    )


def permute(forest, order):
    """Return the same model with features renamed to the positions in `order`."""
    disc = Disc.from_forest(forest)
    position = {old: new for new, old in enumerate(order)}
    trees = []
    for tree in forest.trees:
        feature = tree.feature.copy()
        used = feature >= 0
        feature[used] = np.array([position[int(f)] for f in feature[used]], dtype=np.int32)
        trees.append(
            Tree(feature=feature, threshold=tree.threshold, left=tree.left,
                 right=tree.right, value=tree.value)
        )
    new_forest = Forest.from_arrays(trees, forest.n_features, forest.labels)
    new_disc = Disc(
        thresholds=tuple(disc.thresholds[old] for old in order),
        n_features=forest.n_features,
    )
    return new_forest, new_disc


def bit_positions(disc, order, feature_names):
    """(original feature name, bit index, BDD level) for every input bit."""
    rows = []
    level = 0
    for position, width in enumerate(disc.bits):
        name = feature_names[order[position]]
        for bit in range(width - 1, -1, -1):
            rows.append((name, bit, level))
            level += 1
    return rows
