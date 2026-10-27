"""M2 study: real bundles, fixed input order, prefix analyses and result CSVs.

A prefix is its own model: `W` is recomputed from its tree count, and the input order is
chosen once from the prefix's trees (shallow features first), then kept for the build.
The certified counts do not depend on that order; peak and bit numbering do.
"""

from pathlib import Path
import csv
import gc
import math
import multiprocessing
from queue import Empty
import statistics
import sys
import threading
import time

import numpy as np
import pandas as pd

from bdss import bdd, compose, config, readouts, robustness, transitions
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


def shallow_order(forest):
    """Features by minimum depth of first appearance, then frequency, then index.

    A feature used near the root of a tree constrains the whole tree, so it goes first;
    this is a static order, chosen once and never changed during the build.
    """
    n_features = forest.n_features
    depth = [None] * n_features
    frequency = [0] * n_features

    def walk(tree, node, level):
        if tree.left[node] == -1:
            return
        feature = int(tree.feature[node])
        if depth[feature] is None or level < depth[feature]:
            depth[feature] = level
        frequency[feature] += 1
        walk(tree, int(tree.left[node]), level + 1)
        walk(tree, int(tree.right[node]), level + 1)

    for tree in forest.trees:
        walk(tree, 0, 0)
    return tuple(
        sorted(
            range(n_features),
            key=lambda f: (10**9 if depth[f] is None else depth[f], -frequency[f], f),
        )
    )


def permute(forest, order):
    """Return the same model with features renamed to the positions in `order`."""
    disc = Disc.from_forest(forest)
    position = {old: new for new, old in enumerate(order)}
    trees = []
    for tree in forest.trees:
        feature = tree.feature.copy()
        used = feature >= 0
        feature[used] = np.array([position[int(f)] for f in feature[used]], dtype=np.int32)
        trees.append(
            Tree(feature=feature, threshold=tree.threshold, left=tree.left,
                 right=tree.right, value=tree.value)
        )
    new_forest = Forest.from_arrays(trees, forest.n_features, forest.labels)
    new_disc = Disc(
        thresholds=tuple(disc.thresholds[old] for old in order),
        n_features=forest.n_features,
    )
    return new_forest, new_disc


def bit_positions(disc, order, feature_names):
    """(original feature name, bit index, BDD level) for every input bit."""
    rows = []
    level = 0
    for position, width in enumerate(disc.bits):
        name = feature_names[order[position]]
        for bit in range(width - 1, -1, -1):
            rows.append((name, bit, level))
            level += 1
    return rows


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


def _run_limited(target, args, limit):
    """Run `target(*args)` in a child; return None when `limit` expires.

    A reader thread drains the queue while the parent joins: reading after the join would
    deadlock as soon as the child's payload fills the pipe.
    """
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(*args, queue), daemon=True)
    process.start()
    result = []

    def read():
        try:
            result.append(queue.get())
        except (Empty, EOFError):
            pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    process.join(limit)
    if process.is_alive():
        process.terminate()
        process.join()
        reader.join(1.0)
        return None
    if process.exitcode not in (0, None):
        return ("error", f"child exited with {process.exitcode}")
    reader.join(1.0)
    return result[0] if result else ("error", "child produced no result")


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


def _m3_coverage_note(name, trees, selector):
    semantic = selector in config.M3_SEMANTIC_SELECTORS
    if (name, trees) == ("banknote", 5) and semantic:
        return "B=1/10: semantic not attempted (10k+ leaves)"
    if (name, trees) == ("banknote", 10):
        return "B=10 pilot excluded (256,604 leaves); grid 100/1000"
    if (name, trees) == ("ann-thyroid", 10) and semantic:
        return "B=50k/100k: IG timeout; semantic only at 393k"
    if (name, trees) == ("vehicle", 5) and semantic:
        return "semantic not attempted: IG timeout at B=1M"
    if (name, trees) == ("vehicle", 10):
        return "B=10k timeout at 120 s; wall"
    return ""


def _m3_coverage_rows(rows):
    coverage = []
    by_key = {}
    for row in rows:
        key = (row["dataset"], int(row["trees"]), row["selector"])
        by_key.setdefault(key, []).append(row)
    for (name, trees), budgets in config.M3_GRIDS.items():
        for selector in (*config.M3_CHEAP, *config.M3_SEMANTIC_SELECTORS):
            got = by_key.get((name, trees, selector), [])
            completed = sorted(int(r["budget"]) for r in got if r["status"] == "completed")
            timeouts = sorted(int(r["budget"]) for r in got if r["status"] == "timeout")
            attempted = set(completed) | set(timeouts)
            missing = sorted(set(budgets) - attempted)
            coverage.append({
                "dataset": name,
                "trees": trees,
                "selector": selector,
                "budgets_completed": "|".join(map(str, completed)),
                "budgets_timeout": "|".join(map(str, timeouts)),
                "budgets_not_attempted": "|".join(map(str, missing)),
                "note": _m3_coverage_note(name, trees, selector),
            })
    return coverage


def m3_main():
    RESULTS.mkdir(exist_ok=True)
    expected = _m3_expected()
    fields = [
        "dataset", "trees", "budget", "selector", "status", "peak", "leaves",
        "cap_suspended", "cap_violations", "leaves_at_D", "samples", "sample_mismatch",
        "counts_sum", "counts_expected", "counts_match", "seconds", "wall_seconds", "note",
    ]
    count_fields = ["dataset", "trees", "budget", "selector", "label", "count", "volume"]
    frontier_handle, frontier_writer = _writer(RESULTS / "m3_frontier.csv", fields)
    count_handle, count_writer = _writer(RESULTS / "m3_counts.csv", count_fields)
    rows = []
    try:
        for name, trees, budget, selector in m3_trials():
            print(f"trial {name}/{trees} B={budget} {selector}", flush=True)
            start = time.monotonic()
            result = run_trial_limited(name, trees, budget, selector,
                                       config.M3_TIME_LIMIT_S)
            wall = round(time.monotonic() - start, 3)
            row = {"dataset": name, "trees": trees, "budget": budget,
                   "selector": selector, "wall_seconds": wall}
            if result is None:
                row.update({"status": "timeout", "seconds": config.M3_TIME_LIMIT_S,
                            "note": "time limit"})
            elif isinstance(result, tuple) and result[0] == "error":
                row.update({"status": "error", "note": result[1]})
            else:
                row.update({key: result[key] for key in result if key in fields})
                expected_counts = expected.get((name, trees))
                if expected_counts is None:
                    row.update({"counts_expected": "", "counts_match": ""})
                else:
                    row.update({
                        "counts_expected": sum(expected_counts.values()),
                        "counts_match": result["counts"] == expected_counts,
                    })
                n_bits = result["n_bits"]
                for label, count in result["counts"].items():
                    count_writer.writerow({
                        "dataset": name, "trees": trees, "budget": budget,
                        "selector": selector, "label": label, "count": count,
                        "volume": count / (1 << n_bits),
                    })
            frontier_writer.writerow(row)
            frontier_handle.flush()
            count_handle.flush()
            rows.append(row)
            print(f"  {row['status']} peak={row.get('peak', '')} leaves={row.get('leaves', '')} "
                  f"wall={wall}s", flush=True)
            gc.collect()
        for name, trees, budget, selector in M3_WALL_TRIALS:
            print(f"wall trial {name}/{trees} B={budget} {selector}", flush=True)
            start = time.monotonic()
            result = run_trial_limited(name, trees, budget, selector,
                                       config.M3_TIME_LIMIT_S)
            wall = round(time.monotonic() - start, 3)
            row = {"dataset": name, "trees": trees, "budget": budget,
                   "selector": selector, "wall_seconds": wall}
            if result is None:
                row.update({"status": "timeout", "seconds": config.M3_TIME_LIMIT_S,
                            "note": "B=10k modal/random timeout at 120 s; M2 wall"})
            elif isinstance(result, tuple) and result[0] == "error":
                row.update({"status": "error", "note": result[1]})
            else:
                row.update({key: result[key] for key in result if key in fields})
                row.update({"counts_expected": "", "counts_match": ""})
                n_bits = result["n_bits"]
                for label, count in result["counts"].items():
                    count_writer.writerow({
                        "dataset": name, "trees": trees, "budget": budget,
                        "selector": selector, "label": label, "count": count,
                        "volume": count / (1 << n_bits),
                    })
            frontier_writer.writerow(row)
            frontier_handle.flush()
            count_handle.flush()
            rows.append(row)
            print(f"  {row['status']} wall={wall}s", flush=True)
        frontier_handle.flush()
        coverage_handle, coverage_writer = _writer(
            RESULTS / "m3_coverage.csv",
            ["dataset", "trees", "selector", "budgets_completed", "budgets_timeout",
             "budgets_not_attempted", "note"],
        )
        try:
            coverage_writer.writerows(_m3_coverage_rows(rows))
        finally:
            coverage_handle.close()
    finally:
        frontier_handle.close()
        count_handle.close()


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


def m4_main():
    RESULTS.mkdir(exist_ok=True)
    fields_samples = [
        "dataset", "trees", "sample", "own_label", "certified", "own_radius",
        "min_rival", "min_rival_label", "loss", "loss_le_rival", "min_rival_bits",
        "loss_bits", "target_mismatches", "witness_failures", "sample_seconds",
    ]
    fields_targets = ["dataset", "trees", "sample", "target", "radius", "base_radius",
                      "equal", "witness_ok"]
    fields_summary = [
        "dataset", "trees", "budget", "selector", "samples", "certified", "uncertified",
        "own_zero", "loss_le_rival", "target_mismatches", "witness_failures",
        "time_total", "time_mean", "time_median", "time_max", "min_rival_hist", "loss_hist",
    ]
    fields_distribution = ["dataset", "trees", "distance", "radius", "count"]
    fields_fragile = ["dataset", "trees", "sample", "own_label", "min_rival",
                      "min_rival_label", "loss", "min_rival_bits", "loss_bits"]
    handles = {}
    writers = {}
    for name, fields in (("m4_samples.csv", fields_samples),
                         ("m4_targets.csv", fields_targets),
                         ("m4_summary.csv", fields_summary),
                         ("m4_distribution.csv", fields_distribution),
                         ("m4_fragile.csv", fields_fragile)):
        handles[name], writers[name] = _writer(RESULTS / name, fields)
    try:
        for (name, trees), (budget, selector) in config.M4_BUILDS.items():
            print(f"robustness {name}/{trees} B={budget} {selector}", flush=True)
            bundle = load_bundle(config.DATASETS_DIR / name / f"{name}_rf.pkl")
            samples, targets, summary, distribution, fragile = m4_cell(
                bundle, trees, budget, selector
            )
            writers["m4_samples.csv"].writerows(samples)
            writers["m4_targets.csv"].writerows(targets)
            writers["m4_summary.csv"].writerow(summary)
            writers["m4_distribution.csv"].writerows(distribution)
            writers["m4_fragile.csv"].writerows(
                {key: sample[key] for key in fields_fragile} for sample in fragile
            )
            for handle in handles.values():
                handle.flush()
            print(f"  certified={summary['certified']}/{summary['samples']} "
                  f"mismatches={summary['target_mismatches']} "
                  f"witness_failures={summary['witness_failures']} "
                  f"mean={summary['time_mean']}s", flush=True)
            gc.collect()
    finally:
        for handle in handles.values():
            handle.close()


def main():
    RESULTS.mkdir(exist_ok=True)
    print("validating 54 bundles", flush=True)
    models = []
    bundles = {}
    for path in sorted(config.DATASETS_DIR.glob("*/*_rf.pkl")):
        bundle = load_bundle(path)
        bundles[bundle.name] = bundle
        models.append(validate_bundle(bundle))
        print(f"  {bundle.name}: soft={models[-1]['soft_acc']:.6f} "
              f"hard={models[-1]['hard_acc']:.6f} ties={models[-1]['ties']}", flush=True)

    bundles = {name: bundles[name] for name, _ in M2_CELLS}
    models_path = RESULTS / "m2_models.csv"
    with models_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(models[0]))
        writer.writeheader()
        writer.writerows(models)

    build_fields = [
        "dataset", "trees", "n_bits", "leaves", "peak", "seconds", "total_seconds",
        "status", "wins_disjoint", "binary_odd_exactly_one", "samples", "certified",
        "ties", "not_certified", "mismatch", "hard_acc", "feature_order", "note",
    ]
    count_fields = ["dataset", "trees", "label", "count", "volume"]
    rank_fields = [
        "dataset", "trees", "label", "feature", "bit", "ig", "shapley", "p1", "p0",
        "p", "logp", "bonferroni", "rank_ig", "rank_shapley", "rank_logp",
    ]
    build_handle, build_writer = _writer(RESULTS / "m2_builds.csv", build_fields)
    count_handle, count_writer = _writer(RESULTS / "m2_counts.csv", count_fields)
    rank_handle, rank_writer = _writer(RESULTS / "m2_rankings.csv", rank_fields)
    try:
        for name, n_trees in M2_CELLS:
            print(f"building {name}/{n_trees}", flush=True)
            build_row, count_rows, rank_rows = analyze(bundles[name], n_trees)
            build_writer.writerow(build_row)
            count_writer.writerows(count_rows)
            rank_writer.writerows(rank_rows)
            for handle in (build_handle, count_handle, rank_handle):
                handle.flush()
            print(f"  peak={build_row['peak']} seconds={build_row['seconds']} "
                  f"certified={build_row['certified']}/{build_row['samples']}", flush=True)
            gc.collect()
        for name, n_trees in M2_WALLS:
            print(f"trying wall {name}/{n_trees} for {config.TIME_LIMIT_S:.0f}s", flush=True)
            result = run_cell_limited(name, n_trees, config.TIME_LIMIT_S)
            if result is None:
                build_writer.writerow({
                    "dataset": name, "trees": n_trees, "status": "timeout",
                    "seconds": config.TIME_LIMIT_S,
                    "feature_order": "",
                    "note": "stopped at the time limit; a manual 12 min run also did not finish",
                })
                build_handle.flush()
                print(f"recorded wall {name}/{n_trees}", flush=True)
            elif isinstance(result, tuple) and result[0] == "error":
                build_writer.writerow({
                    "dataset": name, "trees": n_trees, "status": "error",
                    "feature_order": "", "note": result[1],
                })
                build_handle.flush()
                print(f"recorded error {name}/{n_trees}: {result[1]}", flush=True)
            else:
                cell_row, cell_counts, cell_ranks = result
                build_writer.writerow(cell_row)
                count_writer.writerows(cell_counts)
                rank_writer.writerows(cell_ranks)
                for handle in (build_handle, count_handle, rank_handle):
                    handle.flush()
    finally:
        for handle in (build_handle, count_handle, rank_handle):
            handle.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("frontier", "m3"):
        m3_main()
    elif len(sys.argv) > 1 and sys.argv[1] in ("robustness", "m4"):
        m4_main()
    else:
        main()
