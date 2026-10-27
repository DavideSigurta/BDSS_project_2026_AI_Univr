"""Brute-force check of the robustness radii on the toy and on real prefixes <= 12 bits.

Writes `results/m4_bruteforce.csv`.  `banknote`/1 (12 bits) is the smallest prefix among the
three M4 datasets; `threeOf9`/1 (1 bit) is the smallest prefix among all 54.  Both code
spaces fit in the oracle.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.toy_forest import build_toy_forest  # noqa: E402
from bdss import compose, config, oracle, robustness  # noqa: E402
from bdss.discretize import Disc  # noqa: E402
from bdss.forest import Forest, load_bundle  # noqa: E402


def _flipped_label(forest, disc, code, flips):
    sample = list(disc.code_bits(code))
    for level in flips:
        sample[level] ^= 1
    cells = disc.decode_bits(tuple(sample))
    return oracle.hard_label(forest, cells, [disc.tau_array(t) for t in forest.trees])


def check(forest, disc, codes, name, trees):
    build = compose.build(forest, disc)
    targets = 0
    radius_mismatch = 0
    loss_mismatch = 0
    witness_failures = 0
    for code in codes:
        own = oracle.hard_label(
            forest, disc.code_cells(code), [disc.tau_array(t) for t in forest.trees]
        )
        for target in range(forest.n_classes):
            got, bits = robustness.radius(build, code, target)
            expected, _ = oracle.brute_radius(forest, disc, code, target)
            targets += 1
            radius_mismatch += got != expected
            if np.isfinite(got):
                witness_failures += _flipped_label(forest, disc, code, bits) != target
        loss, bits = robustness.loss_radius(build, code, own)
        expected, _ = oracle.brute_loss(forest, disc, code, own)
        loss_mismatch += loss != expected
        if np.isfinite(loss):
            witness_failures += _flipped_label(forest, disc, code, bits) == own
    return {
        "dataset": name, "trees": trees, "n_bits": disc.n_bits,
        "samples": len(codes), "targets": targets,
        "radius_mismatch": radius_mismatch, "loss_mismatch": loss_mismatch,
        "witness_failures": witness_failures,
    }


def main():
    rows = []
    toy = build_toy_forest()
    toy_disc = Disc.from_forest(toy)
    rows.append(check(toy, toy_disc, list(toy_disc.all_codes()), "toy", toy.n_trees))
    bundle = load_bundle(config.DATASETS_DIR / "banknote" / "banknote_rf.pkl")
    base = Forest.from_arrays(bundle.forest.trees[:1], bundle.forest.n_features,
                              bundle.forest.labels)
    disc = Disc.from_forest(base)
    test = np.genfromtxt(bundle.path.parent / "banknote_test.csv", delimiter=",", skip_header=1)
    codes = [sum(bit << (disc.n_bits - 1 - i) for i, bit in enumerate(disc.encode_cells(
        disc.cells_of_row(row[:-1])))) for row in test]
    rows.append(check(base, disc, codes, "banknote/1", 1))

    # The smallest real prefix available in the 54 datasets.
    small = load_bundle(config.DATASETS_DIR / "threeOf9" / "threeOf9_rf.pkl")
    small_base = Forest.from_arrays(small.forest.trees[:1], small.forest.n_features,
                                    small.forest.labels)
    small_disc = Disc.from_forest(small_base)
    small_test = np.genfromtxt(small.path.parent / "threeOf9_test.csv", delimiter=",",
                               skip_header=1)
    small_codes = [
        sum(bit << (small_disc.n_bits - 1 - i)
            for i, bit in enumerate(small_disc.encode_cells(
                small_disc.cells_of_row(row[:-1]))))
        for row in small_test
    ]
    rows.append(check(small_base, small_disc, small_codes, "threeOf9/1", 1))
    path = config.REPO_ROOT / "results" / "m4_bruteforce.csv"
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
