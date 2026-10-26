"""Discretization: forest thresholds -> cells -> binary cell indices."""

from bisect import bisect_left
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Disc:
    thresholds: tuple[np.ndarray, ...]   # sorted distinct thresholds, per feature
    n_features: int

    @classmethod
    def from_forest(cls, forest):
        """Collect every internal-node threshold of every tree, per feature, sorted."""
        per_feature = [[] for _ in range(forest.n_features)]
        for tree in forest.trees:
            for node in range(tree.n_nodes):
                if not tree.is_leaf(node):
                    per_feature[int(tree.feature[node])].append(float(tree.threshold[node]))
        thresholds = tuple(np.unique(np.asarray(values, dtype=float)) for values in per_feature)
        return cls(thresholds=thresholds, n_features=forest.n_features)

    @property
    def cells(self):
        return tuple(len(t) + 1 for t in self.thresholds)

    @property
    def bits(self):
        # CONDIZIONI §3: ceil(log2 K), and 0 when the feature is never tested (K = 1).
        return tuple(0 if k == 1 else int(math.ceil(math.log2(k))) for k in self.cells)

    @property
    def n_bits(self):
        return sum(self.bits)

    @property
    def offsets(self):
        out = []
        offset = 0
        for b in self.bits:
            out.append(offset)
            offset += b
        return tuple(out)

    def cell_of_value(self, feature, value):
        return bisect_left(self.thresholds[feature], float(value))

    def tau(self, feature, threshold):
        """Cell index of a node threshold: the test is `cell <= tau`."""
        return bisect_left(self.thresholds[feature], float(threshold))

    def tau_array(self, tree):
        tau = np.full(tree.n_nodes, -1, dtype=np.int32)
        for node in range(tree.n_nodes):
            if not tree.is_leaf(node):
                tau[node] = self.tau(int(tree.feature[node]), tree.threshold[node])
        return tau

    def encode_cells(self, cells):
        """Bits in variable order: feature by feature, most significant bit first."""
        bits = []
        for feature, k in enumerate(cells):
            width = self.bits[feature]
            cell = min(int(k), self.cells[feature] - 1)
            for j in range(width - 1, -1, -1):
                bits.append((cell >> j) & 1)
        return tuple(bits)

    def decode_bits(self, bits):
        """Bit tuple in variable order -> cell indices, clamped to the last cell."""
        cells = []
        pos = 0
        for feature, width in enumerate(self.bits):
            cell = 0
            for _ in range(width):
                cell = (cell << 1) | int(bits[pos])
                pos += 1
            cells.append(min(cell, self.cells[feature] - 1))
        return tuple(cells)

    def cells_of_row(self, row):
        return tuple(self.cell_of_value(f, row[f]) for f in range(self.n_features))

    def all_codes(self):
        return range(1 << self.n_bits)

    def code_bits(self, code):
        return tuple((code >> (self.n_bits - 1 - i)) & 1 for i in range(self.n_bits))

    def code_cells(self, code):
        return self.decode_bits(self.code_bits(code))
