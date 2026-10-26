"""Readouts on a build: certified counts and volumes."""

from bdss import bdd


def label_counts(build):
    return {
        label: bdd.count_models(build.win(label), build.disc.n_bits, build.mgr)
        for label in range(build.forest.n_classes)
    }


def label_volumes(build):
    total = 1 << build.disc.n_bits
    return {label: count / total for label, count in label_counts(build).items()}
