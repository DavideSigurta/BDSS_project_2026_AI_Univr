"""The toy forest: 3 trees, 2 features (3 and 4 cells), 2 labels.

Feature 0 has thresholds {2, 4} -> cells {0, 1, 2} -> 2 bits.
Feature 1 has thresholds {1, 3, 5} -> cells {0, 1, 2, 3} -> 2 bits.
Every threshold is used by at least one tree; the four input bits are therefore enough for all
tests, and the bit patterns 1100/1101/1110/1111 (cell 3 of feature 0) exercise the clamping of
the last cell.

The example writes three tables:
- results/toy_cells.csv: cell-level hard vote;
- results/toy_counts.csv: certified count and volume per label, plus the per-code Win bits.
"""

from pathlib import Path
import csv
import itertools

import numpy as np

from bdss import compose, readouts, robustness, transitions
from bdss.frontier import SELECTORS
from bdss.forest import Forest, Tree

TOY_N_FEATURES = 2
TOY_LABELS = ("A", "B")


def _make_tree(nodes):
    """Build one Tree from (feature, threshold, left, right, label) records.

    Internal nodes have feature >= 0 and label None; leaves have feature None.
    """
    n = len(nodes)
    feature = np.full(n, -2, dtype=np.int32)
    threshold = np.zeros(n, dtype=float)
    left = np.full(n, -1, dtype=np.int32)
    right = np.full(n, -1, dtype=np.int32)
    value = np.zeros((n, 2), dtype=float)
    for i, (f, t, lo, hi, label) in enumerate(nodes):
        if f is None:
            value[i, label] = 1.0
        else:
            feature[i] = f
            threshold[i] = t
            left[i] = lo
            right[i] = hi
    return Tree(feature=feature, threshold=threshold, left=left, right=right, value=value)


def build_toy_forest():
    # T0: f0 <= 2 ? A : (f1 <= 3 ? B : A)
    t0 = _make_tree([
        (0, 2.0, 1, 2, None),
        (None, 0.0, -1, -1, 0),   # A
        (1, 3.0, 3, 4, None),
        (None, 0.0, -1, -1, 1),   # B
        (None, 0.0, -1, -1, 0),   # A
    ])
    # T1: f1 <= 1 ? B : (f0 <= 4 ? A : B)
    t1 = _make_tree([
        (1, 1.0, 1, 2, None),
        (None, 0.0, -1, -1, 1),   # B
        (0, 4.0, 3, 4, None),
        (None, 0.0, -1, -1, 0),   # A
        (None, 0.0, -1, -1, 1),   # B
    ])
    # T2: f0 <= 4 ? (f1 <= 5 ? A : B) : A
    t2 = _make_tree([
        (0, 4.0, 1, 2, None),
        (1, 5.0, 3, 4, None),
        (None, 0.0, -1, -1, 0),   # A
        (None, 0.0, -1, -1, 0),   # A
        (None, 0.0, -1, -1, 1),   # B
    ])
    return Forest.from_arrays((t0, t1, t2), TOY_N_FEATURES, TOY_LABELS)


def toy_cell_counts():
    return (3, 4)


def main():
    from bdss.discretize import Disc

    forest = build_toy_forest()
    disc = Disc.from_forest(forest)
    taus = [disc.tau_array(tree) for tree in forest.trees]
    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)

    with (results / "toy_cells.csv").open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["cell_f0", "cell_f1", "votes_A", "votes_B", "hard_label"])
        for cells in itertools.product(*(range(k) for k in disc.cells)):
            counts = forest.vote_counts(cells, taus)
            label = forest.hard_label(cells, taus)
            writer.writerow([*cells, int(counts[0]), int(counts[1]),
                             "" if label is None else forest.labels[label]])

    build = compose.build(forest, disc)
    counts = readouts.label_counts(build)
    volumes = readouts.label_volumes(build)
    with (results / "toy_counts.csv").open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["label", "count", "volume", "n_bits"])
        for label in range(forest.n_classes):
            writer.writerow([forest.labels[label], counts[label], volumes[label], disc.n_bits])

    names = transitions.input_names(disc)
    with (results / "toy_codes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["code", "bits", "cell_f0", "cell_f1", "hard_label", "win"])
        for code in disc.all_codes():
            cells = disc.code_cells(code)
            bits = disc.code_bits(code)
            label = forest.hard_label(cells, taus)
            assignment = {name: bool(bit) for name, bit in zip(names, bits)}
            win = [
                forest.labels[lab]
                for lab in range(forest.n_classes)
                if build.mgr.let(assignment, build.win(lab)) == build.mgr.true
            ]
            writer.writerow([code, "".join(map(str, bits)), *cells,
                             "" if label is None else forest.labels[label], "".join(win)])

    def bit_info(level):
        offset = 0
        for feature, width in enumerate(disc.bits):
            if level < offset + width:
                return feature, width - 1 - (level - offset)
            offset += width
        raise ValueError(level)

    rankings = readouts.label_rankings(build)
    with (results / "toy_rankings.csv").open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["label", "level", "feature", "bit", "in_support", "ig", "shapley",
                         "p1", "p0", "p_s", "p_bonferroni",
                         "rank_ig", "rank_shapley", "rank_p"])
        for label in range(forest.n_classes):
            win = build.win(label)
            support = build.mgr.support(win)
            data = rankings[label]
            for level in range(disc.n_bits):
                feature, bit = bit_info(level)
                p = data["pvalue"][level]
                name = f"x_{feature}_{bit}"
                writer.writerow([
                    forest.labels[label], level, feature, bit, name in support,
                    data["ig"][level], data["shapley"][level],
                    p["p1"], p["p0"], p["s"], p["bonferroni"],
                    data["rank_ig"][level], data["rank_shapley"][level], data["rank_p"][level],
                ])

    with (results / "toy_frontier.csv").open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["budget", "selector", "leaves", "peak", "cap_suspended", "seconds",
                         "count_A", "count_B"])
        for budget in (1, 2, 4, 10, 12, 18, 19):
            for selector in SELECTORS:
                b = compose.build(forest, disc, budget=budget, selector=selector)
                fc = readouts.label_counts(b)
                writer.writerow([budget, selector, len(b.leaves), b.peak,
                                 b.cap_suspended, b.seconds, fc[0], fc[1]])

    with (results / "toy_robustness.csv").open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["code", "cell_f0", "cell_f1", "own_label", "kind", "target",
                         "distance", "bit_names"])
        for code in disc.all_codes():
            cells = disc.code_cells(code)
            own = forest.hard_label(cells, taus)
            for target in range(forest.n_classes):
                distance, bits = robustness.radius(build, code, target)
                writer.writerow([code, *cells, forest.labels[own], "targeted",
                                 forest.labels[target], distance,
                                 " ".join(f"x_{f}_{j}" for f, j in map(bit_info, bits))])
            distance, bits = robustness.loss_radius(build, code, own)
            writer.writerow([code, *cells, forest.labels[own], "loss", "", distance,
                             " ".join(f"x_{f}_{j}" for f, j in map(bit_info, bits))])

    print(f"wrote {results / 'toy_cells.csv'}, toy_counts.csv, toy_codes.csv, "
          "toy_rankings.csv, toy_frontier.csv, toy_robustness.csv")


if __name__ == "__main__":
    main()
