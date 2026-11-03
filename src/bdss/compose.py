"""Sequential composition and the budgeted frontier."""

from dataclasses import dataclass, field
from time import monotonic
import random

from bdss import bdd, config, frontier, transitions


class SelectorTimeout(RuntimeError):
    """A selector call exceeded its own wall-clock budget."""


@dataclass(frozen=True)
class Leaf:
    rho: dict          # input bit level -> value
    A: object          # accumulator BDD over (free inputs, current state)
    t: int             # next tree; equals D for a final leaf (invariant checked by tests)


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
    # Peak is the maximum `len(acc)` of the accumulator after a composition or a split.
    # The tree relation Φ_t and the conjunction A ∧ Φ_t are intermediate diagrams and are
    # excluded by CONDIZIONI §5; their size is not the peak.
    cap_suspended: bool = False
    selector: str | None = None
    budget: int | None = None
    fallbacks: int = 0
    compose_seconds: float = 0.0
    selector_seconds: float = 0.0
    leaf_cache: dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    win_cache: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def leaf_win(self, leaf, label):
        """Certified inputs of one leaf for `label`, over the leaf's free input bits."""
        key = (id(leaf), label)
        cached = self.leaf_cache.get(key)
        if cached is None:
            flag = transitions.flag(self.mgr, label)
            cached = self.mgr.exist(
                transitions.current_names(self.forest.n_classes, self.width),
                leaf.A & flag,
            )
            self.leaf_cache[key] = cached
        return cached

    def win(self, label):
        """Certified inputs for `label` over all n_x bits (union of the leaves)."""
        cached = self.win_cache.get(label)
        if cached is not None:
            return cached
        if len(self.leaves) == 1 and not self.leaves[0].rho:
            out = self.leaf_win(self.leaves[0], label)
        else:
            names = transitions.input_names(self.disc)
            out = self.mgr.false
            for leaf in self.leaves:
                w = self.leaf_win(leaf, label)
                for level, value in leaf.rho.items():
                    var = self.mgr.var(names[level])
                    w = w & var if value else w & ~var
                out = out | w
        self.win_cache[label] = out
        return out


def accumulator(forest, disc, upto=None):
    """Compose the first `upto` trees of `forest`; returns (mgr, acc, peak, width, seconds).

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


def build(forest, disc, budget=None, selector="ig"):
    """Compose all trees, optionally through the budgeted frontier."""
    if budget is None:
        mgr, acc, peak, width, seconds = accumulator(forest, disc)
        return Build(
            leaves=(Leaf(rho={}, A=acc, t=forest.n_trees),),
            peak=peak,
            status="completed",
            seconds=seconds,
            disc=disc,
            forest=forest,
            mgr=mgr,
            width=width,
            compose_seconds=seconds,
        )
    return _build_frontier(forest, disc, budget, selector)


def _build_frontier(forest, disc, budget, selector):
    start = monotonic()
    n_labels = forest.n_classes
    width = transitions.counter_width(forest.n_trees)
    mgr = bdd.new_manager()
    mgr.declare(*transitions.input_names(disc))
    transitions.declare_state(mgr, n_labels, width)
    input_names = transitions.input_names(disc)
    state_names = transitions.current_names(n_labels, width)
    rename = {}
    for label in range(n_labels):
        for j in range(width):
            rename[transitions.cp_name(label, j)] = transitions.c_name(label, j)
        rename[transitions.wp_name(label)] = transitions.w_name(label)
    rng = random.Random(config.SEED)

    init = transitions.init_state(mgr, n_labels, width)
    # Phi_t does not depend on rho: build each tree relation once and restrict it per leaf.
    relation_start = monotonic()
    relations = [
        transitions.phi_tree(mgr, disc, tree, t, forest.n_trees, n_labels, width)
        for t, tree in enumerate(forest.trees)
    ]
    compose_seconds = monotonic() - relation_start
    selector_seconds = 0.0
    stats = {"fallbacks": 0}
    pieces = [({}, init, 0)]
    final = []
    peak = bdd.size(init)
    cap_suspended = False

    def split(rho, acc, t):
        nonlocal compose_seconds, selector_seconds
        free_levels = [level for level in range(disc.n_bits) if level not in rho]
        selector_start = monotonic()
        bit = frontier.select(acc, mgr, disc, selector, rng, n_labels, width, free_levels,
                              stats)
        elapsed = monotonic() - selector_start
        selector_seconds += elapsed
        if elapsed > config.SELECTOR_TIMEOUT_S:
            raise SelectorTimeout(
                f"selector {selector!r} took {elapsed:.3f}s "
                f"> {config.SELECTOR_TIMEOUT_S:.1f}s"
            )
        if bit is None:
            return None
        split_start = monotonic()
        name = input_names[bit]
        a0 = mgr.let({name: False}, acc)
        a1 = mgr.let({name: True}, acc)
        for value, child in ((0, a0), (1, a1)):
            rho_child = dict(rho)
            rho_child[bit] = value
            pieces.append((rho_child, child, t))
        compose_seconds += monotonic() - split_start
        return max(bdd.size(a0), bdd.size(a1))

    while pieces:
        rho, acc, t = pieces.pop()
        while True:
            peak = max(peak, bdd.size(acc))
            over = bdd.size(acc) > budget
            if t == forest.n_trees:
                if over:
                    grew = split(rho, acc, t)
                    if grew is not None:
                        peak = max(peak, grew)
                        break
                    cap_suspended = True
                final.append(Leaf(rho=dict(rho), A=acc, t=t))
                break
            if over:
                grew = split(rho, acc, t)
                if grew is not None:
                    peak = max(peak, grew)
                    break
                cap_suspended = True

            compose_start = monotonic()
            phi = relations[t]
            if rho:
                phi = mgr.let({input_names[level]: bool(value)
                               for level, value in rho.items()}, phi)
            acc = mgr.exist(state_names, acc & phi)
            acc = mgr.let(rename, acc)
            t += 1
            compose_seconds += monotonic() - compose_start

    return Build(
        leaves=tuple(final),
        peak=peak,
        status="completed",
        seconds=monotonic() - start,
        disc=disc,
        forest=forest,
        mgr=mgr,
        width=width,
        cap_suspended=cap_suspended,
        selector=selector,
        budget=budget,
        fallbacks=stats["fallbacks"],
        compose_seconds=compose_seconds,
        selector_seconds=selector_seconds,
    )
