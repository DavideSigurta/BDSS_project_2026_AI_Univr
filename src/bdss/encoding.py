"""Logical encoding of one tree: node tests and path formulas.

A node test `X_f <= tau` is the linear comparator of Definition 4; a leaf path is the
conjunction of its node tests with the polarity of the branch.
"""

from bdss.bdd import le_const


def feature_bits(mgr, disc, feature):
    """BDD variables of one feature, least-significant bit first (the formula order)."""
    return [mgr.var(f"x_{feature}_{j}") for j in range(disc.bits[feature])]


def nt(mgr, disc, feature, tau):
    """BDD for `X_f <= tau`, tau a cell index."""
    return le_const(mgr, feature_bits(mgr, disc, feature), tau)


def path_formulas(mgr, disc, tree):
    """List of (leaf node, path formula) for every leaf of `tree`."""
    taus = disc.tau_array(tree)
    leaves = []

    def walk(node, pi):
        if tree.is_leaf(node):
            leaves.append((node, pi))
            return
        feature = int(tree.feature[node])
        test = nt(mgr, disc, feature, int(taus[node]))
        walk(int(tree.left[node]), (pi & test))
        walk(int(tree.right[node]), (pi & ~test))

    walk(0, mgr.true)
    return leaves
