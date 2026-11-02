"""M2-M4 study: bundle analysis, prefix trials and cell robustness.

The final M5 campaign lives in `bdss.campaign`, `bdss.post` and `bdss.modal_app`; this
module retains the smaller study commands and the shared analysis helpers used by tests.
"""

from pathlib import Path
import csv
import gc
import math
import statistics
import sys
import time

import numpy as np
import pandas as pd

from bdss import bdd, compose, config, readouts, robustness, transitions
from bdss.order import bit_positions, permute, shallow_order
from bdss.runtime import run_limited as _run_limited
from bdss.discretize import Disc
from bdss.forest import Forest, Tree, label_key, load_bundle, validate_bundle

RESULTS = config.REPO_ROOT / "results"

# `vehicle` with 10 trees is the measured wall: the accumulator does not complete without a
# budget (M3), so M2 records the status instead of inventing a number.
M2_CELLS = (
    ("banknote", 5),
    ("banknote", 10),
    ("vehicle", 5),
    ("ann-thyroid", 5),
    ("ann-thyroid", 10),
)
M2_WALLS = (("vehicle", 10),)


def _holds(mgr, root, bits, names):
    return mgr.let({name: bool(bit) for name, bit in zip(names, bits)}, root) == mgr.true


def analyze(bundle, n_trees, with_rankings=True):
    """Build one prefix, check the samples, and return plain result rows."""
    start = time.monotonic()
    base = Forest.from_arrays(
        bundle.forest.trees[:n_trees], bundle.forest.n_features, bundle.forest.labels
    )
    order = shallow_order(base)
    forest, disc = permute(base, order)
    build = compose.build(forest, disc)
    mgr = build.mgr
    n_bits = disc.n_bits
    n_labels = forest.n_classes

    wins = {label: build.win(label) for label in range(n_labels)}
    counts = readouts.label_counts(build)
    total_codes = 1 << n_bits
    volumes = {label: count / total_codes for label, count in counts.items()}
    disjoint = all(
        (wins[a] & wins[b]) == mgr.false
        for a in range(n_labels)
        for b in range(a + 1, n_labels)
    )
    binary_odd = n_labels == 2 and n_trees % 2 == 1
    exactly_one = None
    if binary_odd:
        exactly_one = (wins[0] | wins[1]) == mgr.true and (wins[0] & wins[1]) == mgr.false

    test = pd.read_csv(bundle.path.parent / f"{bundle.name}_test.csv")
    features = test.iloc[:, :-1].to_numpy(dtype=float)[:, list(order)]
    target = test.iloc[:, -1].to_numpy()
    index = {label: i for i, label in enumerate(forest.labels)}
    truth = np.array([index[label_key(value)] for value in target])
    taus = [disc.tau_array(tree) for tree in forest.trees]
    names = transitions.input_names(disc)

    certified = 0
    ties = 0
    mismatch = 0
    hard_ok = 0
    for row, own in zip(features, truth):
        cells = disc.cells_of_row(row)
        hard = forest.hard_label(cells, taus)
        bits = disc.encode_cells(cells)
        labels = [
            label for label in range(n_labels)
            if _holds(mgr, wins[label], bits, names)
        ]
        if hard is None:
            ties += 1
        if labels:
            certified += 1
            if len(labels) != 1 or hard != labels[0]:
                mismatch += 1
        if hard == own:
            hard_ok += 1
    if mismatch:
        raise ValueError(f"{bundle.name}/{n_trees}: {mismatch} samples violate the Win check")
    if exactly_one is False:
        raise ValueError(f"{bundle.name}/{n_trees}: binary odd D is not exactly one Win")

    build_row = {
        "dataset": bundle.name,
        "trees": n_trees,
        "n_bits": n_bits,
        "leaves": len(build.leaves),
        "peak": build.peak,
        "seconds": round(build.seconds, 6),
        "total_seconds": 0.0,
        "status": "completed",
        "wins_disjoint": disjoint,
        "binary_odd_exactly_one": exactly_one,
        "samples": len(truth),
        "certified": certified,
        "ties": ties,
        "not_certified": len(truth) - certified,
        "mismatch": mismatch,
        "hard_acc": hard_ok / len(truth),
        "feature_order": "|".join(bundle.feature_names[i] for i in order),
    }
    count_rows = [
        {"dataset": bundle.name, "trees": n_trees, "label": forest.labels[label],
         "count": counts[label], "volume": volumes[label]}
        for label in range(n_labels)
    ]

    rank_rows = []
    if with_rankings:
        positions = bit_positions(disc, order, bundle.feature_names)
        levels = list(range(n_bits))
        for label in range(n_labels):
            rankings = readouts.rankings(mgr, wins[label], levels)
            for feature, bit, level in positions:
                pvalues = rankings["pvalue"][level]
                rank_rows.append({
                    "dataset": bundle.name,
                    "trees": n_trees,
                    "label": forest.labels[label],
                    "feature": feature,
                    "bit": bit,
                    "ig": rankings["ig"][level],
                    "shapley": rankings["shapley"][level],
                    "p1": pvalues["p1"],
                    "p0": pvalues["p0"],
                    "p": pvalues["s"],
                    "logp": pvalues["logp"],
                    "bonferroni": pvalues["bonferroni"],
                    "rank_ig": rankings["rank_ig"][level],
                    "rank_shapley": rankings["rank_shapley"][level],
                    "rank_logp": rankings["rank_logp"][level],
                })

    build_row["total_seconds"] = round(time.monotonic() - start, 6)
    return build_row, count_rows, rank_rows


def _run_cell(name, n_trees, queue):
    """Child process: analyze one cell and put the plain rows in `queue`."""
    try:
        bundle = load_bundle(config.DATASETS_DIR / name / f"{name}_rf.pkl")
        queue.put(analyze(bundle, n_trees))
    except BaseException as error:  # a failed child must not hang the parent
        queue.put(("error", repr(error)))


def run_cell_limited(name, n_trees, limit):
    return _run_limited(_run_cell, (name, n_trees), limit)


def _writer(path, fieldnames):
    handle = path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


M3_WALL_TRIALS = (("vehicle", 10, 10_000, "modal"), ("vehicle", 10, 10_000, "random"))


def m3_trials():
    """Per-cell budgets from the pilot; semantic selectors only where declared."""
    trials = []
    for (name, trees), budgets in config.M3_GRIDS.items():
        if (name, trees) == ("vehicle", 10):
            continue
        for budget in budgets:
            selectors = list(config.M3_CHEAP)
            if budget in config.M3_SEMANTIC[(name, trees)]:
                selectors.extend(config.M3_SEMANTIC_SELECTORS)
            for selector in selectors:
                trials.append((name, trees, budget, selector))
    return tuple(trials)


def _run_trial(name, trees, budget, selector, queue):
    try:
        bundle = load_bundle(config.DATASETS_DIR / name / f"{name}_rf.pkl")
        base = Forest.from_arrays(
            bundle.forest.trees[:trees], bundle.forest.n_features, bundle.forest.labels
        )
        order = shallow_order(base)
        forest, disc = permute(base, order)
        build = compose.build(forest, disc, budget=budget, selector=selector)
        counts = readouts.label_counts(build)
        labels = {forest.labels[label]: count for label, count in counts.items()}

        test = pd.read_csv(bundle.path.parent / f"{bundle.name}_test.csv")
        features = test.iloc[:, :-1].to_numpy(dtype=float)[:, list(order)]
        sample_mismatch = 0
        for row in features:
            cells = disc.cells_of_row(row)
            bits = disc.encode_cells(cells)
            matches = sum(
                1 for leaf in build.leaves
                if all(bits[level] == value for level, value in leaf.rho.items())
            )
            if matches != 1:
                sample_mismatch += 1

        queue.put({
            "status": "completed",
            "peak": build.peak,
            "leaves": len(build.leaves),
            "cap_suspended": build.cap_suspended,
            "cap_violations": sum(
                1 for leaf in build.leaves if bdd.size(leaf.A) > budget
            ),
            "leaves_at_D": all(leaf.t == trees for leaf in build.leaves),
            "samples": len(features),
            "sample_mismatch": sample_mismatch,
            "counts": labels,
            "counts_sum": sum(labels.values()),
            "n_bits": disc.n_bits,
            "seconds": round(build.seconds, 6),
        })
    except BaseException as error:
        queue.put(("error", repr(error)))


def run_trial_limited(name, trees, budget, selector, limit):
    return _run_limited(_run_trial, (name, trees, budget, selector), limit)


def _m3_expected():
    expected = {}
    with (RESULTS / "m2_counts.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], int(row["trees"]))
            expected.setdefault(key, {})[row["label"]] = int(row["count"])
    return expected


def _code_of(bits):
    code = 0
    for bit in bits:
        code = (code << 1) | int(bit)
    return code


def _flipped_bits(bits, flips):
    sample = list(bits)
    for level in flips:
        sample[level] ^= 1
    return tuple(sample)


def _flipped_label(forest, disc, bits, flips, taus):
    return forest.hard_label(disc.decode_bits(_flipped_bits(bits, flips)), taus)


def m4_cell(bundle, trees, budget, selector):
    """Radii for every test sample/label on the M3 leaves and the uncapped build."""
    base = Forest.from_arrays(
        bundle.forest.trees[:trees], bundle.forest.n_features, bundle.forest.labels
    )
    order = shallow_order(base)
    forest, disc = permute(base, order)
    uncapped = compose.build(forest, disc)
    frontier = compose.build(forest, disc, budget=budget, selector=selector)
    names = transitions.input_names(disc)
    front_wins = {label: frontier.win(label) for label in range(forest.n_classes)}
    base_wins = {label: uncapped.win(label) for label in range(forest.n_classes)}
    taus = [disc.tau_array(tree) for tree in forest.trees]
    test = pd.read_csv(bundle.path.parent / f"{bundle.name}_test.csv")
    features = test.iloc[:, :-1].to_numpy(dtype=float)[:, list(order)]
    labels = forest.labels

    samples = []
    targets = []
    for index, row in enumerate(features):
        start = time.monotonic()
        cells = disc.cells_of_row(row)
        bits = disc.encode_cells(cells)
        code = _code_of(bits)
        own = forest.hard_label(cells, taus)
        witness_failures = 0
        equal_targets = 0
        radii = {}
        for target in range(forest.n_classes):
            radius, witness = robustness.radius_from_win(
                frontier.mgr, front_wins[target], bits, names
            )
            base_radius, _ = robustness.radius_from_win(
                uncapped.mgr, base_wins[target], bits, names
            )
            radii[target] = (radius, witness)
            equal = radius == base_radius
            equal_targets += not equal
            witness_ok = None
            if math.isfinite(radius):
                witness_ok = _flipped_label(forest, disc, bits, witness, taus) == target
                witness_failures += not witness_ok
            targets.append({
                "dataset": bundle.name, "trees": trees, "sample": index,
                "target": labels[target],
                "radius": "" if math.isinf(radius) else radius,
                "base_radius": "" if math.isinf(base_radius) else base_radius,
                "equal": equal, "witness_ok": "" if witness_ok is None else witness_ok,
            })

        own_radius = ""
        certified = False
        min_rival = ""
        min_rival_label = ""
        loss = ""
        loss_le_rival = ""
        min_bits = ""
        loss_bits = ""
        target_ids = range(forest.n_classes) if own is None else (
            target for target in range(forest.n_classes) if target != own
        )
        rival_distance, rival_bits, rival = math.inf, (), None
        for target in target_ids:
            distance, witness = radii[target]
            if distance < rival_distance:
                rival_distance, rival_bits, rival = distance, witness, target
        min_rival = "" if math.isinf(rival_distance) else rival_distance
        min_rival_label = labels[rival] if rival is not None else ""
        min_bits = "|".join(map(str, rival_bits))

        if own is not None:
            own_distance, _ = radii[own]
            own_radius = own_distance
            certified = own_distance == 0
            loss_distance, loss_bits_tuple = robustness.loss_radius_from_win(
                frontier.mgr, front_wins[own], bits, names
            )
            loss = "" if math.isinf(loss_distance) else loss_distance
            loss_le_rival = (
                "" if (math.isinf(rival_distance) or math.isinf(loss_distance))
                else loss_distance <= rival_distance
            )
            loss_bits = "|".join(map(str, loss_bits_tuple))
            if math.isfinite(loss_distance):
                flipped_bits = _flipped_bits(bits, loss_bits_tuple)
                witness_failures += robustness.evaluate(
                    frontier.mgr, front_wins[own], flipped_bits, names
                )

        samples.append({
            "dataset": bundle.name, "trees": trees, "sample": index,
            "own_label": labels[own] if own is not None else "",
            "certified": certified,
            "own_radius": own_radius,
            "min_rival": min_rival,
            "min_rival_label": min_rival_label,
            "loss": loss,
            "loss_le_rival": loss_le_rival,
            "min_rival_bits": min_bits,
            "loss_bits": loss_bits,
            "target_mismatches": equal_targets,
            "witness_failures": witness_failures,
            "sample_seconds": round(time.monotonic() - start, 6),
        })

    times = [sample["sample_seconds"] for sample in samples]
    rival_hist = {}
    loss_hist = {}
    for sample in samples:
        if sample["min_rival"] != "":
            rival_hist[sample["min_rival"]] = rival_hist.get(sample["min_rival"], 0) + 1
        if sample["loss"] != "":
            loss_hist[sample["loss"]] = loss_hist.get(sample["loss"], 0) + 1
    summary = {
        "dataset": bundle.name, "trees": trees, "budget": budget, "selector": selector,
        "samples": len(samples),
        "certified": sum(sample["certified"] for sample in samples),
        "uncertified": sum(not sample["certified"] for sample in samples),
        "own_zero": sum(sample["own_radius"] == 0 for sample in samples
                        if sample["own_radius"] != ""),
        "loss_le_rival": sum(sample["loss_le_rival"] is True for sample in samples),
        "target_mismatches": sum(sample["target_mismatches"] for sample in samples),
        "witness_failures": sum(sample["witness_failures"] for sample in samples),
        "build_seconds": frontier.seconds,
        "compose_seconds": frontier.compose_seconds,
        "selector_seconds": frontier.selector_seconds,
        "fallbacks": frontier.fallbacks,
        "readout_seconds": round(sum(times), 6),
        "time_total": round(sum(times), 6),
        "time_mean": round(statistics.mean(times), 6),
        "time_median": round(statistics.median(times), 6),
        "time_max": max(times),
        "min_rival_hist": "|".join(f"{key}:{value}" for key, value in sorted(rival_hist.items())),
        "loss_hist": "|".join(f"{key}:{value}" for key, value in sorted(loss_hist.items())),
    }
    distribution = []
    for name, hist in (("min_rival", rival_hist), ("loss", loss_hist)):
        for radius, count in sorted(hist.items()):
            distribution.append({
                "dataset": bundle.name, "trees": trees, "distance": name,
                "radius": radius, "count": count,
            })
    fragile = sorted(
        (sample for sample in samples if sample["min_rival"] != ""),
        key=lambda sample: (sample["min_rival"],
                            sample["loss"] if sample["loss"] != "" else math.inf),
    )[:config.M4_FRAGILE_TOP]
    return samples, targets, summary, distribution, fragile


def m5_final_main():
    """Definitive M5-final: grid + descending prefixes; post is a separate command."""
    import argparse

    from bdss import campaign

    parser = argparse.ArgumentParser(prog="python -m bdss.study m5final")
    parser.add_argument("--out", default=str(RESULTS / "m5_final.csv"))
    parser.add_argument("--timeout", type=float, default=config.M5_FINAL_TIMEOUT_S)
    parser.add_argument("--node-cap", type=int, default=config.M5_FINAL_NODE_CAP)
    parser.add_argument("--jobs", type=int, default=config.M5_LOCAL_JOBS)
    parser.add_argument("--max-usd", type=float, default=config.M5_FINAL_MAX_USD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(sys.argv[2:])

    grid = list(campaign.final_grid_specs())
    if args.limit is not None:
        grid = grid[:args.limit]
    estimate = campaign.estimate_usd(grid, args.timeout, memory_mb=config.M5_FINAL_MEMORY_MB)
    print(f"final estimate (grid): {estimate}", flush=True)
    if args.max_usd > 0 and estimate["total_usd"] > config.M5_SOFT_USD:
        print(f"warning: grid estimate {estimate['total_usd']} exceeds the 10 USD target",
              flush=True)
    if args.max_usd > 0 and estimate["total_usd"] > args.max_usd:
        raise SystemExit(f"final estimate {estimate['total_usd']} USD exceeds --max-usd")
    rows = campaign.run_final_local(
        out=args.out, jobs=args.jobs, resume=args.resume, limit=args.limit,
        timeout=args.timeout, node_cap=args.node_cap,
    )
    print(f"final rows: {len(rows)} -> {args.out}", flush=True)


def m5_readout_main():
    """Readouts for the selected build of every model; no test set."""
    import argparse

    from bdss import campaign, post

    parser = argparse.ArgumentParser(prog="python -m bdss.study m5readout")
    parser.add_argument("--rows", default=str(RESULTS / "m5_final_campaign.csv"))
    parser.add_argument("--out", default=str(RESULTS / "m5_readouts.csv"))
    parser.add_argument("--timeout", type=float, default=config.M5_READOUT_TIMEOUT_S)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-usd", type=float, default=config.M5_FINAL_MAX_USD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(sys.argv[2:])

    rows = campaign.read_rows(args.rows)
    specs = list(post.selected_specs(rows))
    if args.limit is not None:
        specs = specs[:args.limit]
    estimate = campaign.estimate_usd(specs, args.timeout,
                                     memory_mb=config.M5_READOUT_MEMORY_MB)
    print(f"readout estimate: {estimate}", flush=True)
    if args.max_usd > 0 and estimate["total_usd"] > config.M5_SOFT_USD:
        print(f"warning: readout estimate {estimate['total_usd']} exceeds the 10 USD target",
              flush=True)
    if args.max_usd > 0 and estimate["total_usd"] > args.max_usd:
        raise SystemExit(f"readout estimate {estimate['total_usd']} USD exceeds --max-usd")
    result = post.run_readouts_local(
        specs, out=args.out, resume=args.resume, timeout=args.timeout, jobs=args.jobs,
        node_cap=config.M5_FINAL_NODE_CAP,
    )
    print(f"readouts: {len(result)} -> {args.out}", flush=True)


def m5_robustness_main():
    """Local fallback for full-test robustness; writes resumable chunks and final CSVs."""
    import argparse

    from bdss import campaign, post

    parser = argparse.ArgumentParser(prog="python -m bdss.study m5robustness")
    parser.add_argument("--rows", default=str(RESULTS / "m5_final_campaign.csv"))
    parser.add_argument("--chunks", default=str(RESULTS / "m5_robustness_chunks"))
    parser.add_argument("--samples", default=str(RESULTS / "m5_robustness_samples.csv"))
    parser.add_argument("--targets", default=str(RESULTS / "m5_robustness_targets.csv"))
    parser.add_argument("--coverage", default=str(RESULTS / "m5_post_coverage.csv"))
    parser.add_argument("--readouts", default=str(RESULTS / "m5_readouts.csv"))
    parser.add_argument("--limit-s", type=float, default=config.M5_ROBUSTNESS_WORKER_S)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(sys.argv[2:])

    rows = campaign.read_rows(args.rows)
    specs = list(post.selected_specs(rows))
    if args.limit is not None:
        specs = specs[:args.limit]
    post._guard_canonical(specs, [args.coverage])
    samples, targets, sample_rows, target_rows = post.run_robustness_local(
        specs, chunks=args.chunks, samples_out=args.samples,
        targets_out=args.targets, limit_s=args.limit_s,
    )
    readout_rows = post.read_rows(args.readouts)
    coverage = post.coverage(specs, readout_rows, sample_rows, target_rows)
    post.write_coverage(args.coverage, coverage)
    print(f"robustness: {samples} samples, {targets} targets, "
          f"{len(coverage)} coverage rows", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "m5final":
        m5_final_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "m5readout":
        m5_readout_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "m5robustness":
        m5_robustness_main()
    else:
        raise SystemExit("usage: python -m bdss.study {m5final|m5readout|m5robustness}")
