"""Per-tree transition relation Φ_t: winning threshold, Clinch, Hold/Step.

The Clinch comparator is the offset comparator of the paper: independent carries κ_j, the
inline sum bit δ̂_j, and a most-significant-bit-first comparison χ_j.  No ripple chain and no
separate sum word are built.
"""

import math

from bdss import bdd, encoding


def counter_width(n_trees):
    """M+1 bits, with M = ceil(log2(D/2)) + 1; D=1 needs one bit."""
    if n_trees < 1:
        raise ValueError("n_trees must be positive")
    m = math.ceil(math.log2(n_trees / 2)) + 1
    return m + 1


def win_threshold(n_trees):
    return n_trees // 2 + 1


def input_names(disc):
    names = []
    for feature, width in enumerate(disc.bits):
        for j in range(width - 1, -1, -1):
            names.append(f"x_{feature}_{j}")
    return names


def c_name(label, j):
    return f"c_{label}_{j}"


def cp_name(label, j):
    return f"cp_{label}_{j}"


def w_name(label):
    return f"w_{label}"


def wp_name(label):
    return f"wp_{label}"


def current_names(n_labels, width):
    names = []
    for label in range(n_labels):
        names.extend(c_name(label, j) for j in range(width))
        names.append(w_name(label))
    return names


def next_names(n_labels, width):
    names = []
    for label in range(n_labels):
        names.extend(cp_name(label, j) for j in range(width))
        names.append(wp_name(label))
    return names


def declare_state(mgr, n_labels, width):
    """Declare state bits in the fixed order: per label, MSB counter bit (copy adjacent)."""
    for label in range(n_labels):
        for j in range(width - 1, -1, -1):
            mgr.declare(c_name(label, j), cp_name(label, j))
        mgr.declare(w_name(label), wp_name(label))


def bits_lsb(mgr, label, width, next_=False):
    prefix = cp_name if next_ else c_name
    return [mgr.var(prefix(label, j)) for j in range(width)]


def flag(mgr, label, next_=False):
    return mgr.var(wp_name(label) if next_ else w_name(label))


def init_state(mgr, n_labels, width):
    out = mgr.true
    for label in range(n_labels):
        for bit in bits_lsb(mgr, label, width):
            out = out & ~bit
        out = out & ~flag(mgr, label)
    return out


def _copy_eq(mgr, src, dst):
    out = mgr.true
    for a, b in zip(src, dst):
        out = out & bdd.equiv(mgr, a, b)
    return out


def _inc_eq(mgr, src, dst):
    """dst == src + 1, modulo 2**width (no cap; the cap is never reached)."""
    out = mgr.true
    carry = mgr.true
    for a, b in zip(src, dst):
        sum_bit = bdd.xor(mgr, a, carry)
        out = out & bdd.equiv(mgr, b, sum_bit)
        carry = carry & a
    return out


def _clinch_once(mgr, a_bits, b_bits, r):
    """BDD for `LC^s >= LC^l + r`, with the κ/δ̂/χ comparator."""
    width = len(a_bits)
    # κ_j: carry into position j of b + r; κ_0 = ⊥.
    kappa = [mgr.false]
    for j in range(1, width):
        rem = r % (1 << j)
        if rem == 0:
            kappa.append(mgr.false)
        else:
            const = (1 << j) - rem
            kappa.append(~bdd.le_const(mgr, b_bits[:j], const - 1))
    # δ̂_j = b_j xor r_j xor κ_j.
    delta = []
    for j, bit in enumerate(b_bits):
        x = bdd.xor(mgr, bit, kappa[j])
        delta.append(~x if (r >> j) & 1 else x)
    # χ_j from the least significant position upward; the final χ is the comparison.
    chi = mgr.true
    for j, a_bit in enumerate(a_bits):
        greater = a_bit & ~delta[j]
        equal = bdd.equiv(mgr, a_bit, delta[j]) & chi
        chi = greater | equal
    return chi


def clinch(mgr, n_labels, width, n_trees, t, voted_label):
    """Clinch_s^(t); ⊥ when R_t > W-1 (earliest-certification bound)."""
    r = n_trees - 1 - t
    if r > win_threshold(n_trees) - 1:
        return mgr.false
    out = mgr.true
    a_bits = bits_lsb(mgr, voted_label, width)
    for rival in range(n_labels):
        if rival == voted_label:
            continue
        out = out & _clinch_once(mgr, a_bits, bits_lsb(mgr, rival, width), r)
    return out


def _wins(mgr, n_labels, voted_label):
    out = flag(mgr, voted_label, next_=True)
    for label in range(n_labels):
        if label != voted_label:
            out = out & ~flag(mgr, label, next_=True)
    return out


def _inc(mgr, n_labels, width, voted_label):
    out = mgr.true
    for label in range(n_labels):
        if label == voted_label:
            out = out & _inc_eq(
                mgr, bits_lsb(mgr, label, width), bits_lsb(mgr, label, width, next_=True)
            )
        else:
            out = out & _copy_eq(
                mgr, bits_lsb(mgr, label, width), bits_lsb(mgr, label, width, next_=True)
            )
        out = out & ~flag(mgr, label, next_=True)
    return out


def phi_tree(mgr, disc, tree, t, n_trees, n_labels, width):
    """The relation Φ_t over (input, current state, next state)."""
    won = mgr.false
    for label in range(n_labels):
        won = won | flag(mgr, label)

    hold = mgr.true
    for label in range(n_labels):
        hold = hold & bdd.equiv(mgr, flag(mgr, label), flag(mgr, label, next_=True))

    clinch_by_label = {
        label: clinch(mgr, n_labels, width, n_trees, t, label) for label in range(n_labels)
    }
    wins_by_label = {label: _wins(mgr, n_labels, label) for label in range(n_labels)}
    inc_by_label = {label: _inc(mgr, n_labels, width, label) for label in range(n_labels)}

    step = mgr.false
    for leaf, path in encoding.path_formulas(mgr, disc, tree):
        label = tree.leaf_label(leaf)
        branch = (clinch_by_label[label] & wins_by_label[label]) | (
            ~clinch_by_label[label] & inc_by_label[label]
        )
        step = step | (path & branch)

    return (won & hold) | (~won & step)
