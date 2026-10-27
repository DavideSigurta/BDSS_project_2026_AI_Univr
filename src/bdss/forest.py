"""Forest and tree views.

The pipeline works on cell indices, never on real feature values.  A `Tree` keeps the sklearn
arrays (feature, threshold, children, class counts) and can walk both the real values (for
loader validation) and the cell indices (for the encoding).
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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


def label_key(value):
    """CSV cell -> the label string used in `Forest.labels`."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


@dataclass(frozen=True)
class Bundle:
    """One `.pkl` bundle plus the metadata and the raw label names."""

    path: Path
    name: str
    forest: Forest
    meta: dict
    feature_names: tuple[str, ...]

    @property
    def test_acc(self):
        return float(self.meta["test_acc"])


def load_bundle(path):
    """Load a joblib bundle and check its metadata against the CSV files.

    The class index of a model is the code assigned by `pandas.factorize` on the *full*
    target column, so the label map is the order of first appearance in `<name>.csv`, not
    the sorted order of the train labels.  The map is validated against `test_acc` by
    `validate_bundle`.
    """
    path = Path(path).resolve()
    name = path.name.removesuffix("_rf.pkl")
    data = joblib.load(path)
    rf = data["model"]
    folder = path.parent

    full = pd.read_csv(folder / f"{name}.csv")
    target = full.columns[-1]
    feature_names = tuple(str(column) for column in full.columns[:-1])
    labels = tuple(label_key(value) for value in pd.unique(full[target]))
    n_classes = len(rf.classes_)
    if n_classes != len(labels):
        raise ValueError(f"{name}: {n_classes} model classes, {len(labels)} labels")
    if list(rf.classes_) != list(range(n_classes)):
        raise ValueError(f"{name}: classes_ is not 0..{n_classes - 1}")

    trees = tuple(
        Tree.from_sklearn(estimator.tree_, n_classes) for estimator in rf.estimators_
    )
    forest = Forest(trees=trees, n_features=int(rf.n_features_in_), labels=labels)
    meta = {key: value for key, value in data.items() if key != "model"}

    if meta.get("dataset") != name:
        raise ValueError(f"{name}: dataset metadata is {meta.get('dataset')!r}")
    if int(meta["n_features"]) != forest.n_features:
        raise ValueError(f"{name}: n_features mismatch")
    if len(feature_names) != forest.n_features:
        raise ValueError(f"{name}: {len(feature_names)} CSV features")
    if int(meta["n_classes"]) != n_classes:
        raise ValueError(f"{name}: n_classes mismatch")
    if int(meta["train_rows"]) + int(meta["test_rows"]) != int(meta["n_rows"]):
        raise ValueError(f"{name}: row counts do not add up")
    if int(meta["n_rows"]) != len(full):
        raise ValueError(f"{name}: n_rows differs from the full CSV")

    train = pd.read_csv(folder / f"{name}_train.csv")
    test = pd.read_csv(folder / f"{name}_test.csv")
    if list(train.columns) != list(full.columns) or list(test.columns) != list(full.columns):
        raise ValueError(f"{name}: train/test columns differ from the full file")
    if len(train) != int(meta["train_rows"]) or len(test) != int(meta["test_rows"]):
        raise ValueError(f"{name}: train/test rows mismatch")
    known = set(labels)
    for frame in (train, test):
        if not {label_key(value) for value in frame[target].unique()} <= known:
            raise ValueError(f"{name}: a target value is missing from the label map")

    return Bundle(path=path, name=name, forest=forest, meta=meta,
                  feature_names=feature_names)


def _sklearn_row(row):
    """sklearn casts the input to float32 before comparing with the float64 thresholds."""
    return np.asarray(row, dtype=np.float32).astype(np.float64)


def soft_vote(forest, row):
    """Soft vote the way sklearn does it: average the leaf class distributions, then argmax.

    This reproduces `RandomForestClassifier.predict` without calling it; the float32 cast
    reproduces sklearn's input precision.
    """
    row = _sklearn_row(row)
    scores = np.zeros(forest.n_classes, dtype=float)
    for tree in forest.trees:
        leaf = tree.leaf_for_values(row)
        value = tree.value[leaf].astype(float)
        total = value.sum()
        scores += value / total if total else value
    return int(np.argmax(scores))


def hard_vote_row(forest, row):
    """Majority of the per-tree leaf labels for one real-valued row; None on a tie."""
    row = _sklearn_row(row)
    counts = np.zeros(forest.n_classes, dtype=int)
    for tree in forest.trees:
        counts[tree.leaf_label(tree.leaf_for_values(row))] += 1
    best = int(counts.max())
    winners = np.flatnonzero(counts == best)
    return None if len(winners) != 1 else int(winners[0])


def validate_bundle(bundle, test=None):
    """Check the label map against `test_acc` and report soft/hard test accuracy."""
    if test is None:
        test = pd.read_csv(bundle.path.parent / f"{bundle.name}_test.csv")
    features = test.iloc[:, :-1].to_numpy(dtype=float)
    target = test.iloc[:, -1].to_numpy()
    index = {label: i for i, label in enumerate(bundle.forest.labels)}
    truth = np.array([index[label_key(value)] for value in target])

    soft = np.array([soft_vote(bundle.forest, row) for row in features])
    hard = [hard_vote_row(bundle.forest, row) for row in features]
    soft_acc = float((soft == truth).mean())
    hard_ok = np.array([label is not None and label == own for label, own in zip(hard, truth)])
    expected = bundle.test_acc
    if abs(soft_acc - expected) > 1e-12:
        raise ValueError(
            f"{bundle.name}: soft accuracy {soft_acc} != bundle test_acc {expected}; "
            "the label map is wrong"
        )
    return {
        "dataset": bundle.name,
        "n_features": bundle.forest.n_features,
        "n_classes": bundle.forest.n_classes,
        "n_trees": bundle.forest.n_trees,
        "train_rows": int(bundle.meta["train_rows"]),
        "test_rows": int(bundle.meta["test_rows"]),
        "labels": "|".join(bundle.forest.labels),
        "test_acc": expected,
        "soft_acc": soft_acc,
        "hard_acc": float(hard_ok.mean()),
        "ties": int(sum(label is None for label in hard)),
    }
