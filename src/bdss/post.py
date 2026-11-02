"""Post-build readouts and test-set robustness for the M5-final campaign.

The post phase has two independent products:

- **readouts** for one selected build per model (counts, volumes, per-bit rankings);
- **robustness** for every test sample and every label (targeted radius, nearest rival,
  loss of certification, witness bits with feature and destination cell).

Splitting them keeps the cheap, test-set-independent readout out of the expensive robustness
pass.  Robustness is processed in time slices so a preemption loses at most one slice.
"""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import csv
import json
import math
import os
import time

import pandas as pd

from bdss import compose, config, readouts, robustness
from bdss.forest import Forest, load_bundle
from bdss.runtime import run_limited as _run_limited


CANONICAL_READOUTS = config.REPO_ROOT / "results/m5_readouts.csv"
CANONICAL_SAMPLES = config.REPO_ROOT / "results/m5_robustness_samples.csv"
CANONICAL_TARGETS = config.REPO_ROOT / "results/m5_robustness_targets.csv"
CANONICAL_COVERAGE = config.REPO_ROOT / "results/m5_post_coverage.csv"


def _guard_canonical(specs, paths):
    """Refuse to overwrite a canonical final CSV with a partial spec set."""
    keys = {_spec_key(spec) for spec in specs}
    if len(keys) == 54:
        return
    canonical = (CANONICAL_READOUTS, CANONICAL_SAMPLES,
                 CANONICAL_TARGETS, CANONICAL_COVERAGE)
    for path in paths:
        if path is None:
            continue
        target = Path(path)
        for expected in canonical:
            if target.exists() and expected.exists():
                same = os.path.samefile(target, expected)
            else:
                same = (os.path.normcase(os.path.abspath(target))
                        == os.path.normcase(os.path.abspath(expected)))
            if same:
                raise ValueError(
                    f"refusing to overwrite canonical output {expected}: "
                    f"{len(keys)}/54 selected specs"
                )


READOUT_FIELDS = (
    "dataset", "trees", "budget", "selector", "timeout_s", "source", "status",
    "timeout", "n_bits", "n_classes", "feature_order", "labels", "counts_json",
    "volumes_json", "rankings_json", "seconds", "note",
)

SAMPLE_FIELDS = (
    "dataset", "trees", "budget", "selector", "sample", "timeout_s", "source",
    "own_label", "certified", "own_radius", "min_rival", "min_rival_label", "loss",
    "min_rival_bits", "min_rival_features", "min_rival_cells", "loss_bits",
    "loss_features", "loss_cells", "n_targets", "seconds",
)

TARGET_FIELDS = (
    "dataset", "trees", "budget", "selector", "sample", "timeout_s", "source",
    "label", "own_label", "is_own", "is_min_rival", "radius", "witness_bits",
    "witness_features", "witness_cells", "destination_cells_json", "seconds",
)

COVERAGE_FIELDS = (
    "dataset", "trees", "budget", "selector", "k", "build_status", "readout_status",
    "robustness_status", "test_samples", "samples", "targets", "note",
)


# ---------------------------------------------------------------------------
# Selected build


def selected_specs(rows):
    """One completed build per model: largest k, then 100k/modal, then smaller peak/leaves."""
    by_model = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        budget = row.get("budget", "")
        k = int(row.get("k") or row["trees"])
        spec = (
            row["dataset"], int(row["trees"]),
            None if budget in ("", "none", None) else int(budget),
            row["selector"], row.get("source") or row.get("phase") or "final",
        )
        by_model.setdefault(row["dataset"], []).append((k, spec, row))
    out = []
    for name in sorted(by_model):
        candidates = by_model[name]
        max_k = max(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] == max_k]
        preferred = [item for item in tied
                     if item[1][2] == 100_000 and item[1][3] == "modal"]
        if preferred:
            chosen = preferred[0]
        else:
            chosen = min(
                tied,
                key=lambda item: (
                    float(item[2].get("peak") or math.inf),
                    float(item[2].get("leaves") or math.inf),
                ),
            )
        out.append(chosen[1])
    return tuple(out)


def _load_selected(spec, node_cap):
    """Rebuild the selected prefix with the same order and selector used by the campaign."""
    from bdss.order import permute, shallow_order

    if len(spec) == 4:
        spec = (*spec, "final")
    name, trees, budget, selector, _source = spec
    bundle = load_bundle(config.DATASETS_DIR / name / f"{name}_rf.pkl")
    base = Forest.from_arrays(
        bundle.forest.trees[:trees], bundle.forest.n_features, bundle.forest.labels
    )
    order = shallow_order(base)
    forest, disc = permute(base, order)
    build = compose.build(forest, disc, budget=budget, selector=selector)
    if build.peak > node_cap:
        raise MemoryError(f"peak {build.peak} > node cap {node_cap}")
    return bundle, forest, disc, build, order


# ---------------------------------------------------------------------------
# Readouts


def readout_one(spec, node_cap=config.M5_FINAL_NODE_CAP):
    """Compute counts, volumes and per-bit rankings for one selected build."""
    if len(spec) == 4:
        spec = (*spec, "final")
    start = time.monotonic()
    try:
        bundle, forest, disc, build, order = _load_selected(spec, node_cap)
        counts = readouts.label_counts(build)
        total = 1 << disc.n_bits
        volumes = {label: count / total for label, count in counts.items()}
        rankings = readouts.label_rankings(build)
        row = _empty_readout(spec, "", "")
        row.update({
            "status": "completed",
            "timeout": False,
            "n_bits": disc.n_bits,
            "n_classes": forest.n_classes,
            "feature_order": ",".join(map(str, order)),
            "labels": ",".join(forest.labels),
            "counts_json": json.dumps({str(k): v for k, v in counts.items()},
                                      sort_keys=True),
            "volumes_json": json.dumps({str(k): v for k, v in volumes.items()},
                                       sort_keys=True),
            "rankings_json": json.dumps(
                {str(label): {str(level): value for level, value in ranks.items()}
                 for label, ranks in rankings.items()},
                sort_keys=True,
            ),
            "seconds": round(time.monotonic() - start, 6),
            "note": "",
        })
        return row
    except MemoryError as error:
        return _empty_readout(spec, "memory", repr(error), round(time.monotonic() - start, 6))
    except BaseException as error:
        return _empty_readout(spec, "error", repr(error), round(time.monotonic() - start, 6))


def _empty_readout(spec, status, note, seconds=0.0):
    if len(spec) == 4:
        spec = (*spec, "final")
    name, trees, budget, selector, source = spec
    row = {field: "" for field in READOUT_FIELDS}
    row.update({
        "dataset": name, "trees": trees,
        "budget": "" if budget is None else budget,
        "selector": selector, "timeout_s": int(config.M5_READOUT_TIMEOUT_S),
        "source": source, "status": status, "timeout": status == "timeout",
        "seconds": seconds, "note": note,
    })
    return row


def _readout_queue(spec, node_cap, queue):
    try:
        queue.put(readout_one(spec, node_cap=node_cap))
    except BaseException as error:
        queue.put(("error", repr(error)))


def run_readouts_local(specs, out="results/m5_readouts.csv", resume=False,
                       timeout=config.M5_READOUT_TIMEOUT_S, jobs=1,
                       node_cap=config.M5_FINAL_NODE_CAP):
    """Run readouts in child processes; one row per model, written once at the end."""

    _guard_canonical(specs, [out])
    out = Path(out)
    existing = read_rows(out) if resume else []
    done = {_readout_key(row) for row in existing if row.get("status") == "completed"}
    rows = list(existing)
    pending = [spec for spec in specs if _spec_key(spec) not in done]
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            pool.submit(_run_limited, _readout_queue, (spec, node_cap), timeout): spec
            for spec in pending
        }
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            if result is None:
                row = _empty_readout(spec, "timeout", "time limit")
            elif isinstance(result, tuple) and result[0] == "error":
                row = _empty_readout(spec, "error", result[1])
            else:
                row = result
            rows = [old for old in rows if _readout_key(old) != _readout_key(row)]
            rows.append(row)
            print(f"  readout {_spec_key(spec)}: {row['status']}", flush=True)
    write_readouts(out, rows)
    return rows


def write_readouts(path, rows):
    _write_csv(path, READOUT_FIELDS, rows)


# ---------------------------------------------------------------------------
# Robustness


def witness(build, order, feature_names, code, bits):
    """Map witness bit levels to original feature names and destination cells."""
    disc = build.disc
    bits = tuple(int(bit) for bit in bits)
    sample_bits = list(disc.code_bits(code))
    sample_cells = list(disc.decode_bits(tuple(sample_bits)))
    flipped = list(sample_bits)
    for level in bits:
        flipped[level] ^= 1
    destination = disc.decode_bits(tuple(flipped))
    offsets = disc.offsets
    features = []
    cells = []
    for level in bits:
        feature = bisect_right(offsets, level) - 1
        width = disc.bits[feature]
        offset = offsets[feature]
        shift = width - 1 - (level - offset)
        cell = sample_cells[feature] ^ (1 << shift)
        cell = min(cell, disc.cells[feature] - 1)
        original = order[feature]
        features.append(str(feature_names[original]))
        cells.append(cell)
    affected = sorted({bisect_right(offsets, level) - 1 for level in bits})
    dest = [
        {"feature": str(feature_names[order[feature]]), "cell": int(destination[feature])}
        for feature in affected
    ]
    return (
        "|".join(map(str, bits)),
        "|".join(features),
        "|".join(map(str, cells)),
        json.dumps(dest, separators=(",", ":")),
    )


def robustness_range(spec, start=0, limit_s=config.M5_ROBUSTNESS_WORKER_S,
                     max_samples=None, timeout_s=config.M5_ROBUSTNESS_TIMEOUT_S,
                     node_cap=config.M5_FINAL_NODE_CAP):
    """Process a contiguous sample range; return rows and the next index.

    The time limit is checked after each complete sample, so a sample is never split across
    two chunks.  The caller persists the returned chunk and schedules `next`.
    """
    if len(spec) == 4:
        spec = (*spec, "final")
    name, trees, budget, selector, source = spec
    start_time = time.monotonic()
    bundle, forest, disc, build, order = _load_selected(spec, node_cap)
    feature_names = bundle.feature_names
    test = pd.read_csv(bundle.path.parent / f"{name}_test.csv")
    features = test.iloc[:, :-1].to_numpy(dtype=float)[:, list(order)]
    total = len(features)
    if max_samples is not None:
        features = features[:int(max_samples)]
    limit = len(features)
    start = max(0, int(start))
    if start >= limit:
        return _chunk_result(spec, start, start, total, [], [],
                             round(time.monotonic() - start_time, 6))

    taus = [disc.tau_array(tree) for tree in forest.trees]
    deadline = time.monotonic() + max(1.0, float(limit_s))
    sample_rows = []
    target_rows = []
    index = start
    while index < limit:
        if index > start and time.monotonic() >= deadline:
            break
        sample_start = time.monotonic()
        row = features[index]
        cells = disc.cells_of_row(row)
        code = _code_of(disc.encode_cells(cells))
        own = forest.hard_label(cells, taus)
        targets = []
        for label in range(forest.n_classes):
            distance, bits = robustness.radius(build, code, label)
            wb, wf, wc, dest = witness(build, order, feature_names, code, bits)
            targets.append({
                "label": label, "distance": distance, "bits": bits, "wb": wb,
                "wf": wf, "wc": wc, "dest": dest,
            })
        own_target = targets[own] if own is not None else None
        own_radius = (
            "" if own_target is None or not math.isfinite(own_target["distance"])
            else int(own_target["distance"])
        )
        certified = bool(own_target is not None and own_target["distance"] == 0)
        rivals = [item for item in targets if own is None or item["label"] != own]
        best = min(rivals, key=lambda item: (item["distance"], item["label"]), default=None)
        if best is None:
            min_rival = math.inf
            min_label = ""
            min_wb = min_wf = min_wc = ""
        else:
            min_rival = best["distance"]
            min_label = forest.labels[best["label"]]
            min_wb, min_wf, min_wc = best["wb"], best["wf"], best["wc"]
        if own is None:
            loss = math.inf
            loss_wb = loss_wf = loss_wc = ""
            own_label = ""
        else:
            own_label = forest.labels[own]
            loss, loss_bits = robustness.loss_radius(build, code, own)
            loss_wb, loss_wf, loss_wc, _ = witness(
                build, order, feature_names, code, loss_bits
            )
        sample_rows.append({
            "dataset": name, "trees": trees,
            "budget": "" if budget is None else budget, "selector": selector,
            "sample": index, "timeout_s": int(timeout_s), "source": source,
            "own_label": own_label, "certified": certified,
            "own_radius": own_radius,
            "min_rival": "" if not math.isfinite(min_rival) else int(min_rival),
            "min_rival_label": min_label,
            "loss": "" if not math.isfinite(loss) else int(loss),
            "min_rival_bits": min_wb, "min_rival_features": min_wf,
            "min_rival_cells": min_wc,
            "loss_bits": loss_wb, "loss_features": loss_wf, "loss_cells": loss_wc,
            "n_targets": forest.n_classes,
            "seconds": round(time.monotonic() - sample_start, 6),
        })
        for item in targets:
            label = item["label"]
            is_min = bool(
                best is not None and item["distance"] == best["distance"]
                and (own is None or label != own)
            )
            target_rows.append({
                "dataset": name, "trees": trees,
                "budget": "" if budget is None else budget, "selector": selector,
                "sample": index, "timeout_s": int(timeout_s), "source": source,
                "label": forest.labels[label], "own_label": own_label,
                "is_own": own is not None and label == own,
                "is_min_rival": is_min,
                "radius": int(item["distance"]) if math.isfinite(item["distance"]) else "",
                "witness_bits": item["wb"], "witness_features": item["wf"],
                "witness_cells": item["wc"], "destination_cells_json": item["dest"],
                "seconds": round(time.monotonic() - sample_start, 6),
            })
        index += 1
    return _chunk_result(spec, start, index, total, sample_rows, target_rows,
                         round(time.monotonic() - start_time, 6))


def _chunk_result(spec, start, next_index, total, samples, targets, seconds):
    name, trees, budget, selector, source = spec
    return {
        "spec": [name, trees, budget, selector, source],
        "start": start, "next": next_index, "test_samples": total,
        "done": next_index >= total, "samples": samples, "targets": targets,
        "seconds": seconds,
    }


def _code_of(bits):
    code = 0
    for bit in bits:
        code = (code << 1) | int(bit)
    return code


def _key(name, trees, budget, selector):
    return (
        str(name), int(trees),
        "" if budget in (None, "", "none", "None") else str(budget),
        str(selector),
    )


def _spec_key(spec):
    if isinstance(spec, dict):
        return _key(spec["dataset"], spec["trees"], spec.get("budget"), spec["selector"])
    name, trees, budget, selector = spec[:4]
    return _key(name, trees, budget, selector)


def _readout_key(row):
    return _key(row["dataset"], row["trees"], row.get("budget"), row["selector"])


def _sample_key(row):
    return (str(row["dataset"]), int(row["trees"]), int(row["sample"]))


def _target_key(row):
    return (*_sample_key(row), str(row["label"]))


# ---------------------------------------------------------------------------
# Chunk persistence and final CSVs


def chunk_name(spec, start, end):
    name, trees, budget, selector = spec[:4]
    tag = "none" if budget is None else str(budget)
    return f"{name}__{trees}__{tag}__{selector}__{int(start):06d}__{int(end):06d}.json"


def write_chunk(directory, result):
    spec = result["spec"]
    path = Path(directory) / chunk_name(spec, result["start"], result["next"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(result, separators=(",", ":")))
    tmp.replace(path)
    return path


def read_chunks(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    chunks = []
    for path in sorted(directory.glob("*.json")):
        chunks.append(json.loads(path.read_text()))
    return chunks


def chunk_done(chunks, spec):
    """True only when chunks cover the model contiguously from sample 0."""
    return next_start(chunks, spec) >= _test_rows(spec[0])


def next_start(chunks, spec):
    """First sample not covered by a contiguous chain of chunks for `spec`."""
    key = _spec_key(spec)
    intervals = sorted(
        (int(chunk["start"]), int(chunk["next"]))
        for chunk in chunks
        if chunk.get("spec") and _spec_key(chunk["spec"]) == key
    )
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            break
        cursor = max(cursor, end)
    return cursor


def combine_robustness(chunks, samples_out="results/m5_robustness_samples.csv",
                       targets_out="results/m5_robustness_targets.csv"):
    """Write the two final CSVs from chunk files, de-duplicating overlapping chunks."""
    samples = {}
    targets = {}
    for chunk in chunks:
        for row in chunk.get("samples", []):
            samples[_sample_key(row)] = row
        for row in chunk.get("targets", []):
            targets[_target_key(row)] = row
    sample_rows = [samples[key] for key in sorted(samples)]
    target_rows = [targets[key] for key in sorted(targets)]
    _write_csv(samples_out, SAMPLE_FIELDS, sample_rows)
    _write_csv(targets_out, TARGET_FIELDS, target_rows)
    return len(sample_rows), len(target_rows), sample_rows, target_rows


def run_robustness_local(specs, chunks="results/m5_robustness_chunks",
                         samples_out="results/m5_robustness_samples.csv",
                         targets_out="results/m5_robustness_targets.csv",
                         limit_s=config.M5_ROBUSTNESS_WORKER_S, limit=None,
                         node_cap=config.M5_FINAL_NODE_CAP):
    """Sequential local fallback; writes one chunk per time slice."""
    pending = list(specs)
    if limit is not None:
        pending = pending[:limit]
    _guard_canonical(pending, [samples_out, targets_out])
    for spec in pending:
        while True:
            chunks_now = read_chunks(chunks)
            if chunk_done(chunks_now, spec):
                break
            start = next_start(chunks_now, spec)
            result = robustness_range(spec, start=start, limit_s=limit_s, node_cap=node_cap)
            write_chunk(chunks, result)
            print(f"  robustness {_spec_key(spec)} {result['start']}->{result['next']}",
                  flush=True)
            if result["done"]:
                break
    all_chunks = read_chunks(chunks)
    return combine_robustness(all_chunks, samples_out, targets_out)


# ---------------------------------------------------------------------------
# Coverage


def coverage(specs, readouts_rows, samples, targets):
    """Per-model coverage of readouts and full-test robustness."""
    readout_by = {_readout_key(row): row for row in readouts_rows}
    sample_counts = {}
    target_counts = {}
    for row in samples:
        sample_counts[_spec_key(row)] = sample_counts.get(_spec_key(row), 0) + 1
    for row in targets:
        target_counts[_spec_key(row)] = target_counts.get(_spec_key(row), 0) + 1
    out = []
    for spec in specs:
        key = _spec_key(spec)
        name, trees, budget, selector, _source = spec
        readout = readout_by.get(key)
        readout_status = readout["status"] if readout else "missing"
        test_samples = _test_rows(name)
        n_labels = _n_labels(name, readout)
        n_samples = sample_counts.get(key, 0)
        n_targets = target_counts.get(key, 0)
        if n_samples >= test_samples and n_targets >= test_samples * n_labels:
            status = "completed"
            note = ""
        elif n_samples or n_targets:
            status = "partial"
            note = f"{n_samples}/{test_samples} samples, {n_targets}/{test_samples * n_labels} targets"
        else:
            status = "missing"
            note = "no robustness rows"
        out.append({
            "dataset": name, "trees": trees,
            "budget": "" if budget is None else budget, "selector": selector,
            "k": trees, "build_status": "completed",
            "readout_status": readout_status, "robustness_status": status,
            "test_samples": test_samples, "samples": n_samples, "targets": n_targets,
            "note": note,
        })
    return out


def _test_rows(name):
    path = config.DATASETS_DIR / name / f"{name}_test.csv"
    with path.open() as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _n_labels(name, readout):
    if readout and readout.get("labels"):
        return len(str(readout["labels"]).split(","))
    bundle = load_bundle(config.DATASETS_DIR / name / f"{name}_rf.pkl")
    return bundle.forest.n_classes


def write_coverage(path, rows):
    _write_csv(path, COVERAGE_FIELDS, rows)


# ---------------------------------------------------------------------------
# Small CSV helpers


def read_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    # A readout row carries per-label rankings, so the default 128 KiB field limit is too low.
    csv.field_size_limit(10**9)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(part, path)
