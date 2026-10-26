"""Forest and tree views.

The pipeline works on cell indices, never on real feature values.  A `Tree` keeps the sklearn
arrays (feature, threshold, children, class counts) and can walk both the real values (for
loader validation) and the cell indices (for the encoding).
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Tree:
    feature: np.ndarray      # int32, -2 at leaves
    threshold: np.ndarray    # float64, unused at leaves
    left: np.ndarray         # int32, -1 at leaves
    right: np.ndarray        # int32, -1 at leaves
    value: np.ndarray        # (n_nodes, n_classes) class counts

    @property
    def n_nodes(self):
        return len(self.feature)

    @property
    def n_classes(self):
        return self.value.shape[1]

    def is_leaf(self, node):
        return self.left[node] == -1

    def leaf_label(self, node):
        return int(np.argmax(self.value[node]))

    def leaf_for_values(self, row):
        node = 0
        while not self.is_leaf(node):
            feature = int(self.feature[node])
            if row[feature] <= self.threshold[node]:
                node = int(self.left[node])
            else:
                node = int(self.right[node])
        return int(node)

    def leaf_for_cells(self, cells, tau):
        """Walk with cell indices; `tau[node]` is the cell index of the node threshold."""
        node = 0
        while not self.is_leaf(node):
            feature = int(self.feature[node])
            if cells[feature] <= tau[node]:
                node = int(self.left[node])
            else:
                node = int(self.right[node])
        return int(node)

    @classmethod
    def from_sklearn(cls, tree, n_classes):
        return cls(
            feature=np.asarray(tree.feature, dtype=np.int32),
            threshold=np.asarray(tree.threshold, dtype=float),
            left=np.asarray(tree.children_left, dtype=np.int32),
            right=np.asarray(tree.children_right, dtype=np.int32),
            value=np.asarray(tree.value, dtype=float).reshape(tree.node_count, n_classes),
        )


@dataclass(frozen=True)
class Forest:
    trees: tuple[Tree, ...]
    n_features: int
    labels: tuple[str, ...]

    @property
    def n_trees(self):
        return len(self.trees)

    @property
    def n_classes(self):
        return len(self.labels)

    def vote_counts(self, cells, taus):
        """One vote per tree, then class counts.  Ties are not broken here."""
        counts = np.zeros(self.n_classes, dtype=int)
        for tree, tau in zip(self.trees, taus):
            leaf = tree.leaf_for_cells(cells, tau)
            counts[tree.leaf_label(leaf)] += 1
        return counts

    def hard_label(self, cells, taus):
        """Majority label, or None on a tie (even number of trees)."""
        counts = self.vote_counts(cells, taus)
        best = int(counts.max())
        winners = np.flatnonzero(counts == best)
        if len(winners) != 1:
            return None
        return int(winners[0])

    @classmethod
    def from_arrays(cls, trees, n_features, labels):
        return cls(trees=tuple(trees), n_features=n_features, labels=tuple(labels))
