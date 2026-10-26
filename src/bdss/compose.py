"""Sequential composition into an accumulator, and the per-label certified region."""

from dataclasses import dataclass
from time import monotonic

from bdss import bdd, transitions


@dataclass(frozen=True)
class Leaf:
    rho: dict          # input-bit index -> value (empty for the uncapped build)
    A: object          # accumulator BDD over (inputs, current state)
    t: int             # index of the next tree (D when final)


@dataclass(frozen=True)
class Build:
    leaves: tuple
    peak: int
    status: str
    seconds: float
    disc: object
    forest: object
    mgr: object
    width: int

    def win(self, label):
        """Certified inputs for `label`, over the input bits only (single-leaf build)."""
        if len(self.leaves) != 1:
            raise NotImplementedError("frontier union is added in M1b")
        leaf = self.leaves[0]
        flag = transitions.flag(self.mgr, label)
        return self.mgr.exist(
            transitions.current_names(self.forest.n_classes, self.width),
            leaf.A & flag,
        )


def accumulator(forest, disc, upto=None):
    """Compose the first `upto` trees of `forest`; returns (mgr, acc, peak, width).

    `upto` is only used by tests to inspect intermediate states; the transition constants
    still refer to the full forest (R_t = D - 1 - t).
    """
    if upto is None:
        upto = forest.n_trees
    start = monotonic()
    n_labels = forest.n_classes
    width = transitions.counter_width(forest.n_trees)
    mgr = bdd.new_manager()
    mgr.declare(*transitions.input_names(disc))
    transitions.declare_state(mgr, n_labels, width)

    acc = transitions.init_state(mgr, n_labels, width)
    peak = bdd.size(acc)
    state_names = transitions.current_names(n_labels, width)
    rename = {}
    for label in range(n_labels):
        for j in range(width):
            rename[transitions.cp_name(label, j)] = transitions.c_name(label, j)
        rename[transitions.wp_name(label)] = transitions.w_name(label)

    for t, tree in enumerate(forest.trees[:upto]):
        phi = transitions.phi_tree(mgr, disc, tree, t, forest.n_trees, n_labels, width)
        acc = mgr.exist(state_names, acc & phi)
        acc = mgr.let(rename, acc)
        peak = max(peak, bdd.size(acc))
    return mgr, acc, peak, width, monotonic() - start


def build(forest, disc):
    """Compose all trees (uncapped build)."""
    mgr, acc, peak, width, seconds = accumulator(forest, disc)
    leaf = Leaf(rho={}, A=acc, t=forest.n_trees)
    return Build(
        leaves=(leaf,),
        peak=peak,
        status="completed",
        seconds=seconds,
        disc=disc,
        forest=forest,
        mgr=mgr,
        width=width,
    )
