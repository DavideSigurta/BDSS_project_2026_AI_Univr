"""M5 campaign: shared specifications, one-run evaluation, local runner, costs.

This module is deliberately free of any Modal import.  The optional Modal entrypoint
(`bdss.modal_app`) wraps `run_one`; the local CLI (`python -m bdss.study m5`) uses
`run_local`.  Both therefore run exactly the same pipeline and write the same row schema.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import math
from pathlib import Path
import csv
import json
import resource
import sys
import time

from bdss import compose, config, readouts
from bdss.forest import Forest, load_bundle


CSV_FIELDS = (
    "dataset", "trees", "n_bits", "budget", "selector", "timeout_s", "status", "timeout",
    "peak", "leaves", "cap_suspended", "build_seconds", "compose_seconds",
    "selector_seconds", "readout_seconds", "fallbacks", "rss_mb", "counts_sum",
    "counts_json",
    "wall_seconds", "note",
)


@lru_cache(maxsize=None)
def dataset_models():
    """All (dataset, full tree count) pairs, sorted by name."""
    models = []
    for path in sorted(config.DATASETS_DIR.glob("*/*_rf.pkl")):
        bundle = load_bundle(path)
        models.append((bundle.name, bundle.forest.n_trees))
    return tuple(models)


def grid_specs(models=None, budgets=None, selectors=None):
    """Full model × budget × selector grid, in a deterministic order."""
    models = tuple(dataset_models() if models is None else models)
    budgets = tuple(config.M5_BUDGETS if budgets is None else budgets)
    selectors = tuple(config.M5_SELECTORS if selectors is None else selectors)
    return tuple(
        (name, trees, budget, selector)
        for name, trees in models
        for budget in budgets
        for selector in selectors
    )


def prefix_specs(wall_models, cap=config.M5_PREFIX_CAP):
    """Prefix candidates for models that never complete the full grid.

    The full grid already covers the whole forest; the prefix search starts at 5 and stops
    at 50 (the true size is the grid).  Both cheap selectors are tried; the cap is a hard
    upper bound so the total stays close to the 1.530-run plan.
    """
    out = []
    for name, full in wall_models:
        for trees in config.M5_PREFIXES:
            if trees is None or trees >= full:
                continue
            for selector in ("modal", "random"):
                out.append((name, trees, None, selector))
                if len(out) >= cap:
                    return tuple(out)
    return tuple(out)


def bisection_specs(rows, cap=config.M5_BISECTION_CAP):
    """Optional adjacent-prefix bisection from completed/failed prefix rows."""
    by_key = {}
    for row in rows:
        if row.get("status") not in ("completed", "timeout", "memory", "error"):
            continue
        key = (row["dataset"], row["budget"], row["selector"])
        by_key.setdefault(key, []).append(row)
    out = []
    for (name, budget, selector), group in by_key.items():
        good = max((int(r["trees"]) for r in group if r["status"] == "completed"),
                   default=None)
        bad = min((int(r["trees"]) for r in group if r["status"] != "completed"),
                  default=None)
        if good is None or bad is None or good + 1 >= bad:
            continue
        middle = (good + bad) // 2
        out.append((name, middle, None if budget in ('', 'none') else int(budget), selector))
        if len(out) >= cap:
            break
    return tuple(out)


def key_of(spec):
    name, trees, budget, selector = spec
    return (str(name), int(trees), "" if budget is None else str(int(budget)), str(selector))


def _row_key(row):
    budget = row.get("budget", "")
    return (str(row["dataset"]), int(row["trees"]), str(budget), str(row["selector"]))


def completed_keys(path):
    """Keys whose row has status `completed` (timeouts are not completed)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        return {
            _row_key(row)
            for row in csv.DictReader(handle)
            if row.get("status") == "completed"
        }


def _empty_row(spec, status, note="", timeout_s=config.M5_TIMEOUT_S):
    name, trees, budget, selector = spec
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "dataset": name, "trees": trees, "budget": "" if budget is None else budget,
        "selector": selector, "timeout_s": int(timeout_s), "status": status,
        "timeout": status == "timeout", "note": note,
    })
    return row


def _rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def run_one(spec, timeout_s=config.M5_TIMEOUT_S, node_cap=config.NODE_CAP):
    """Build one campaign specification and return one CSV-shaped row.

    The pipeline is the existing one: load the bundle, choose the shallow order, rebuild
    the prefix model, then `compose.build`.  Readouts are the certified counts only; a
    timeout is a status, never a silent missing row.
    """
    from bdss.study import permute, shallow_order  # local import avoids a cycle

    name, trees, budget, selector = spec
    start = time.monotonic()
    try:
        bundle = load_bundle(config.DATASETS_DIR / name / f"{name}_rf.pkl")
        base = Forest.from_arrays(
            bundle.forest.trees[:trees], bundle.forest.n_features, bundle.forest.labels
        )
        order = shallow_order(base)
        forest, disc = permute(base, order)
        build = compose.build(forest, disc, budget=budget, selector=selector)
        # NODE_CAP is a post-build status, not a streaming guard: the build has already
        # consumed the memory.  The container's memory limit is the real guard.
        if build.peak > node_cap:
            row = _empty_row(
                spec, "memory", f"peak {build.peak} > node cap {node_cap}", timeout_s
            )
            row.update({
                "peak": build.peak, "leaves": len(build.leaves),
                "build_seconds": round(build.seconds, 6),
                "compose_seconds": round(build.compose_seconds, 6),
                "selector_seconds": round(build.selector_seconds, 6),
                "fallbacks": build.fallbacks,
                "wall_seconds": round(time.monotonic() - start, 6),
            })
            return row

        readout_start = time.monotonic()
        counts = readouts.label_counts(build)
        readout_seconds = time.monotonic() - readout_start
        return {
            "dataset": name, "trees": trees, "n_bits": disc.n_bits,
            "budget": "" if budget is None else budget, "selector": selector,
            "timeout_s": int(timeout_s), "status": "completed", "timeout": False,
            "peak": build.peak, "leaves": len(build.leaves),
            "cap_suspended": build.cap_suspended,
            "build_seconds": round(build.seconds, 6),
            "compose_seconds": round(build.compose_seconds, 6),
            "selector_seconds": round(build.selector_seconds, 6),
            "readout_seconds": round(readout_seconds, 6),
            "fallbacks": build.fallbacks,
            "rss_mb": round(_rss_mb(), 1),
            "counts_sum": sum(counts.values()),
            "counts_json": json.dumps({str(k): v for k, v in counts.items()}, sort_keys=True),
            "wall_seconds": round(time.monotonic() - start, 6),
            "note": "",
        }
    except MemoryError as error:
        return _empty_row(spec, "memory", repr(error), timeout_s)
    except BaseException as error:
        return _empty_row(spec, "error", repr(error), timeout_s)


def read_rows(path):
    """All rows of a campaign CSV (empty when the file does not exist)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resume_keys(path):
    """Keys already present in a campaign CSV, for `--resume`."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        return {_row_key(row) for row in csv.DictReader(handle)}


def _write_rows(path, rows, resume):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and path.exists() and path.stat().st_size > 0 else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            handle.flush()


def _run_one_queue(spec, node_cap, queue):
    try:
        queue.put(run_one(spec, node_cap=node_cap))
    except BaseException as error:  # a failed child must not hang the parent
        queue.put(("error", repr(error)))


def run_one_limited(spec, timeout=config.M5_TIMEOUT_S, node_cap=config.NODE_CAP):
    """One run in a fresh child process; timeout/error become a row, never a lost run."""
    from bdss.study import _run_limited

    result = _run_limited(_run_one_queue, (spec, node_cap), timeout)
    if result is None:
        return _empty_row(spec, "timeout", "time limit", timeout)
    if isinstance(result, tuple) and result[0] == "error":
        return _empty_row(spec, "error", result[1], timeout)
    result["timeout_s"] = int(timeout)
    return result


def run_local(specs, jobs=config.M5_LOCAL_JOBS, out="results/m5_campaign.csv",
              resume=False, timeout=config.M5_TIMEOUT_S, node_cap=config.NODE_CAP):
    """Run specifications in fresh child processes and append one CSV row per trial."""
    from bdss.study import _run_limited

    specs = tuple(specs)
    if not resume:
        Path(out).unlink(missing_ok=True)
    done = resume_keys(out) if resume else set()
    pending = [spec for spec in specs if key_of(spec[:4]) not in done]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            pool.submit(_run_limited, _run_one_queue, (spec, node_cap), timeout): spec
            for spec in pending
        }
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            if result is None:
                row = _empty_row(spec, "timeout", "time limit", timeout)
            elif isinstance(result, tuple) and result[0] == "error":
                row = _empty_row(spec, "error", result[1], timeout)
            else:
                row = result
                row["timeout_s"] = int(timeout)
            row["timeout_s"] = int(timeout)
            rows.append(row)
            _write_rows(out, [row], resume=True)
            print(f"  {spec}: {row['status']}", flush=True)
    return rows


def estimate_usd(specs, timeout=config.M5_TIMEOUT_S, cpu=config.M5_CPU,
                 memory_mb=config.M5_MEMORY_MB):
    """Conservative Modal CPU + memory estimate for the given run count/list."""
    n_runs = len(specs) if not isinstance(specs, int) else specs
    cpu_hours = n_runs * timeout / 3600.0 * cpu
    memory_gib_hours = n_runs * timeout / 3600.0 * (memory_mb / 1024.0)
    cpu_usd = cpu_hours * config.M5_CPU_USD_PER_CORE_HOUR
    memory_usd = memory_gib_hours * config.M5_MEMORY_USD_PER_GIB_HOUR
    return {
        "runs": n_runs,
        "timeout_s": timeout,
        "cpu_hours": round(cpu_hours, 3),
        "memory_gib_hours": round(memory_gib_hours, 3),
        "cpu_usd": round(cpu_usd, 3),
        "memory_usd": round(memory_usd, 3),
        "total_usd": round(cpu_usd + memory_usd, 3),
    }


# ---------------------------------------------------------------------------
# M5 post: one selected build per model -> counts, volumes, rankings, robustness.

POST_FIELDS = (
    "dataset", "trees", "budget", "selector", "timeout_s", "source", "status",
    "timeout", "n_bits", "samples", "certified", "feature_order", "labels", "counts_json",
    "volumes_json", "rankings_json", "robustness_json", "seconds", "note",
)
POST_SAMPLE_FIELDS = (
    "dataset", "trees", "sample", "timeout_s", "source", "own_label", "certified",
    "own_radius", "min_rival", "min_rival_label", "loss", "min_rival_bits", "loss_bits",
)
POST_COVERAGE_FIELDS = ("dataset", "trees", "build_status", "post_status",
                        "timeout_s", "source", "reason")


def post_specs(rows):
    """One completed build per model; prefer 100k/modal, else the first completed."""
    by_model = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        budget = row.get("budget", "")
        timeout_s = int(row.get("timeout_s") or config.M5_TIMEOUT_S)
        source = row.get("source") or ("rescue" if timeout_s >= 300 else "base")
        spec = (
            row["dataset"], int(row["trees"]),
            None if budget in ("", "none", None) else int(budget), row["selector"],
            timeout_s, source,
        )
        by_model.setdefault(spec[0], []).append(spec)
    out = []
    for name in sorted(by_model):
        candidates = by_model[name]
        preferred = [s for s in candidates if s[2] == 100_000 and s[3] == "modal"]
        out.append(preferred[0] if preferred else candidates[0])
    return tuple(out)


def post_coverage(post_rows, models=None):
    """Final coverage from the post summary plus the models absent from it.

    A row in `m5_post.csv` means a build was available; its status is the post status
    (`completed`, `timeout`, ...).  A model absent from the post summary has no completed
    build and is `missing` / `not_run`.
    """
    models = tuple(dataset_models() if models is None else models)
    by_model = {row["dataset"]: row for row in post_rows}
    coverage = []
    for name, full in models:
        row = by_model.get(name)
        if row is None:
            coverage.append({
                "dataset": name, "trees": full, "build_status": "missing",
                "post_status": "not_run", "timeout_s": "", "source": "",
                "reason": "no completed build",
            })
        else:
            coverage.append({
                "dataset": name, "trees": int(row["trees"]),
                "build_status": "completed", "post_status": row["status"],
                "timeout_s": row.get("timeout_s", ""), "source": row.get("source", ""),
                "reason": "post completed" if row["status"] == "completed" else "post failed",
            })
    return coverage


def post_one(spec, node_cap=config.NODE_CAP):
    """Reconstruct the selected build and compute the M5 post readouts."""
    import pandas as pd
    from bdss import robustness
    from bdss.study import permute, shallow_order

    if len(spec) == 4:
        spec = (*spec, int(config.M5_TIMEOUT_S), "base")
    name, trees, budget, selector, timeout_s, source = spec
    start = time.monotonic()
    try:
        bundle = load_bundle(config.DATASETS_DIR / name / f"{name}_rf.pkl")
        base = Forest.from_arrays(
            bundle.forest.trees[:trees], bundle.forest.n_features, bundle.forest.labels
        )
        order = shallow_order(base)
        forest, disc = permute(base, order)
        build = compose.build(forest, disc, budget=budget, selector=selector)
        if build.peak > node_cap:
            return _post_empty(spec, "memory", f"peak {build.peak} > node cap {node_cap}")
        counts = readouts.label_counts(build)
        volumes = readouts.label_volumes(build)
        rankings = readouts.label_rankings(build)
        taus = [disc.tau_array(tree) for tree in forest.trees]
        test = pd.read_csv(bundle.path.parent / f"{name}_test.csv")
        features = test.iloc[:, :-1].to_numpy(dtype=float)
        features = features[:, list(order)]
        rival_hist = {}
        loss_hist = {}
        sample_rows = []
        certified = 0
        for index, row in enumerate(features):
            cells = disc.cells_of_row(row)
            code = _code_of(disc.encode_cells(cells))
            own = forest.hard_label(cells, taus)
            own_radius = ""
            best = math.inf
            best_label = ""
            best_bits = ()
            lost = math.inf
            loss_bits = ()
            if own is None:
                for target in range(forest.n_classes):
                    distance, bits = robustness.radius(build, code, target)
                    if distance < best:
                        best, best_bits, best_label = distance, bits, forest.labels[target]
                certified_here = False
            else:
                own_radius, _ = robustness.radius(build, code, own)
                certified_here = own_radius == 0
                certified += certified_here
                best, best_bits, target = robustness.min_rival_radius(build, code, own)
                best_label = forest.labels[target] if target is not None else ""
                lost, loss_bits = robustness.loss_radius(build, code, own)
            if math.isfinite(best):
                rival_hist[int(best)] = rival_hist.get(int(best), 0) + 1
            if math.isfinite(lost):
                loss_hist[int(lost)] = loss_hist.get(int(lost), 0) + 1
            sample_rows.append({
                "dataset": name, "trees": trees, "sample": index,
                "timeout_s": int(timeout_s), "source": source,
                "own_label": "" if own is None else forest.labels[own],
                "certified": certified_here,
                "own_radius": "" if own_radius == "" else own_radius,
                "min_rival": "" if not math.isfinite(best) else int(best),
                "min_rival_label": best_label,
                "loss": "" if not math.isfinite(lost) else int(lost),
                "min_rival_bits": "|".join(map(str, best_bits)),
                "loss_bits": "|".join(map(str, loss_bits)),
            })
        robustness_json = {
            "samples": len(sample_rows),
            "certified": certified,
            "rival_hist": rival_hist,
            "loss_hist": loss_hist,
        }
        return {
            "dataset": name, "trees": trees, "budget": "" if budget is None else budget,
            "selector": selector, "timeout_s": int(timeout_s), "source": source,
            "status": "completed", "timeout": False,
            "n_bits": disc.n_bits, "samples": len(sample_rows), "certified": certified,
            "feature_order": ",".join(map(str, order)),
            "labels": ",".join(forest.labels),
            "counts_json": json.dumps({str(k): v for k, v in counts.items()}, sort_keys=True),
            "volumes_json": json.dumps({str(k): v for k, v in volumes.items()}, sort_keys=True),
            "rankings_json": json.dumps(
                {str(label): {str(level): value for level, value in ranks.items()}
                 for label, ranks in rankings.items()},
                sort_keys=True,
            ),
            "robustness_json": json.dumps(robustness_json, sort_keys=True),
            "seconds": round(time.monotonic() - start, 6),
            "note": "",
            "_samples": sample_rows,
        }
    except MemoryError as error:
        return _post_empty(spec, "memory", repr(error))
    except BaseException as error:
        return _post_empty(spec, "error", repr(error))


def _code_of(bits):
    code = 0
    for bit in bits:
        code = (code << 1) | int(bit)
    return code


def _post_empty(spec, status, note=""):
    if len(spec) == 4:
        spec = (*spec, int(config.M5_TIMEOUT_S), "base")
    name, trees, budget, selector, timeout_s, source = spec
    row = {field: "" for field in POST_FIELDS}
    row.update({"dataset": name, "trees": trees, "budget": "" if budget is None else budget,
                "selector": selector, "timeout_s": int(timeout_s), "source": source,
                "status": status, "timeout": status == "timeout", "note": note,
                "_samples": []})
    return row


def _post_queue(spec, node_cap, queue):
    try:
        queue.put(post_one(spec, node_cap=node_cap))
    except BaseException as error:
        queue.put(("error", repr(error)))


def _replace_post_samples(path, dataset, trees, samples):
    """Replace all sample rows of one model; sample rows are written before the summary."""
    path = Path(path)
    existing = read_rows(path)
    kept = [
        row for row in existing
        if not (row["dataset"] == dataset and int(row["trees"]) == int(trees))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POST_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(kept + list(samples))
        handle.flush()


def _replace_post_summary(path, row):
    """Replace the summary row of one model; the summary is the completion marker."""
    path = Path(path)
    key = _row_key(row)
    kept = [existing for existing in read_rows(path) if _row_key(existing) != key]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POST_FIELDS)
        writer.writeheader()
        writer.writerows(kept + [row])
        handle.flush()


def run_post_local(specs, out="results/m5_post.csv", samples_out="results/m5_post_samples.csv",
                   resume=False, timeout=config.M5_POST_TIMEOUT_S,
                   jobs=config.M5_POST_JOBS, node_cap=config.NODE_CAP):
    """Run the post phase in fresh processes; samples first, summary as completion marker."""
    from bdss.study import _run_limited

    specs = tuple(specs)
    if not resume:
        Path(out).unlink(missing_ok=True)
        Path(samples_out).unlink(missing_ok=True)
    sample_keys = {
        (row["dataset"], int(row["trees"])) for row in read_rows(samples_out)
    }
    done = {
        _row_key(row) for row in read_rows(out)
        if (row["dataset"], int(row["trees"])) in sample_keys
    } if resume else set()
    pending = [spec for spec in specs if key_of(spec[:4]) not in done]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            pool.submit(_run_limited, _post_queue, (spec, node_cap), timeout): spec
            for spec in pending
        }
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            if result is None:
                row = _post_empty(spec, "timeout", "time limit")
            elif isinstance(result, tuple) and result[0] == "error":
                row = _post_empty(spec, "error", result[1])
            else:
                row = result
            samples = row.pop("_samples", [])
            rows.append(row)
            # Samples first: if the process dies here, the missing summary makes the
            # model pending again and the replacement avoids duplicates.
            _replace_post_samples(samples_out, row["dataset"], row["trees"], samples)
            _replace_post_summary(out, row)
            print(f"  post {spec}: {row['status']}", flush=True)
    return rows


def rescue_specs(rows):
    """Specs of failed base rows (timeout/memory/error), without prefixes/bisection."""
    seen = set()
    out = []
    for row in rows:
        if row.get("status") not in ("timeout", "memory", "error"):
            continue
        budget = row.get("budget", "")
        spec = (
            row["dataset"], int(row["trees"]),
            None if budget in ("", "none", None) else int(budget), row["selector"],
        )
        if spec not in seen:
            seen.add(spec)
            out.append(spec)
    return tuple(out)


def run_rescue_local(base="results/m5_campaign.csv", out="results/m5_campaign_300.csv",
                     resume=False, timeout=config.M5_RESCUE_TIMEOUT_S,
                     jobs=config.M5_LOCAL_JOBS):
    """Re-run only failed base rows at the rescue timeout."""
    specs = rescue_specs(read_rows(base))
    rows = run_local(specs, jobs=jobs, out=out, resume=resume, timeout=timeout)
    return rows


def merge_campaigns(base, rescue, out="results/m5_campaign_merged.csv"):
    """Merge base and rescue: a completed row wins; otherwise the rescue row remains."""
    merged = {}
    for row in read_rows(base):
        row.setdefault("timeout_s", int(config.M5_TIMEOUT_S))
        merged[_row_key(row)] = row
    for row in read_rows(rescue):
        row.setdefault("timeout_s", int(config.M5_RESCUE_TIMEOUT_S))
        key = _row_key(row)
        current = merged.get(key)
        if current is None:
            merged[key] = row
        elif current.get("status") != "completed":
            merged[key] = row
        # current completed: the base row wins, keeping its timeout_s provenance.
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(merged.values())
    return list(merged.values())


# ---------------------------------------------------------------------------
# M5-final: one CSV, 3 budgets, descending prefixes, full-test post.

FINAL_FIELDS = CSV_FIELDS + ("phase", "k", "source", "node_cap")


def final_grid_specs(models=None, budgets=None, selectors=None):
    return grid_specs(
        models=models,
        budgets=config.M5_FINAL_BUDGETS if budgets is None else budgets,
        selectors=config.M5_FINAL_SELECTORS if selectors is None else selectors,
    )


def final_prefix_specs(rows, cap=config.M5_FINAL_PREFIX_CAP, models=None):
    """Descending k=50,30,20,10,5 for models with no completed grid row."""
    grid_done = {
        (row["dataset"], int(row["trees"])) for row in rows
        if row.get("phase") == "grid" and row.get("status") == "completed"
    }
    out = []
    for name, full in (dataset_models() if models is None else models):
        if (name, full) in grid_done:
            continue
        for k in config.M5_FINAL_PREFIX_KS:
            if k >= full:
                continue
            for budget in config.M5_FINAL_PREFIX_BUDGETS:
                for selector in config.M5_FINAL_PREFIX_SELECTORS:
                    out.append((name, k, budget, selector))
                    if len(out) >= cap:
                        return tuple(out)
    return tuple(out)


def final_post_specs(rows):
    """Largest completed k per model; ties: 100k/modal, then smaller peak/leaves."""
    by_model = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        k = int(row.get("k") or row["trees"])
        budget = row.get("budget", "")
        spec = (
            row["dataset"], int(row["trees"]),
            None if budget in ("", "none", None) else int(budget), row["selector"],
            int(config.M5_FINAL_POST_TIMEOUT_S),
            row.get("source") or row.get("phase") or "final",
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


def _enrich_final(rows, phase, node_cap=None, source=None):
    out = []
    for row in rows:
        row = dict(row)
        row.update({
            "phase": phase, "k": int(row.get("trees")),
            "source": source or phase,
            "node_cap": config.M5_FINAL_NODE_CAP if node_cap is None else node_cap,
        })
        out.append(row)
    return out


def build_models(rows, models=None):
    """Models with no completed full-forest build in the final grid.

    Only the full row is used: the selection must not depend on any prefix outcome.
    """
    models = tuple(dataset_models() if models is None else models)
    full_by_name = dict(models)
    complete = {
        row["dataset"] for row in rows
        if row.get("phase") == "grid" and row.get("status") == "completed"
        and int(row["trees"]) == full_by_name.get(row["dataset"], -1)
    }
    return tuple((name, full) for name, full in models if name not in complete)


def build_pairs(models=None):
    """All (model, budget, selector) pairs of the build-completion phase."""
    models = tuple(dataset_models() if models is None else models)
    return tuple(
        (name, full, budget, selector)
        for name, full in models
        for budget in config.M5_BUILD_BUDGETS
        for selector in config.M5_BUILD_SELECTORS
    )


def build_search(pair, measure):
    """Find the largest completing prefix for one (model, budget, selector) pair.

    The fixed k list is tried in ascending order and stops at the first failure; the
    bisection then narrows the adjacent success/failure pair.  `k_ok=0` is a sentinel,
    not a model, and lets the bisection reach k=1..4.  The full forest is a known failure
    for every pair in this phase, so it is the upper bound and is never measured.
    """
    name, full, budget, selector = pair
    rows = []
    k_ok, k_fail = 0, None
    for k in config.M5_BUILD_KS:
        if k >= full:
            continue
        spec = (name, k, budget, selector)
        row = measure(spec)
        rows.append(row)
        if row["status"] == "completed":
            k_ok = k
        else:
            k_fail = k
            break
    if k_fail is None:
        k_fail = full
    while k_fail - k_ok > 1:
        mid = (k_ok + k_fail) // 2
        spec = (name, mid, budget, selector)
        row = measure(spec)
        rows.append(row)
        if row["status"] == "completed":
            k_ok = mid
        else:
            k_fail = mid
    return tuple(rows)


def final_campaign_rows(final_rows, build_rows):
    """Clean grid + build-completion rows; old prefix rows are superseded.

    A full-build row cannot collide with a prefix row.  The key includes the phase so the
    function is idempotent if called twice.
    """
    by_key = {}
    for row in final_rows:
        if row.get("phase") != "grid":
            continue
        key = ("grid", row["dataset"], int(row["trees"]),
               row.get("budget", ""), row["selector"])
        by_key[key] = row
    for row in build_rows:
        key = ("prefix", row["dataset"], int(row["trees"]),
               row.get("budget", ""), row["selector"])
        by_key[key] = row
    return tuple(by_key[key] for key in sorted(by_key))


def _write_final(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_final_local(out="results/m5_final.csv", jobs=config.M5_LOCAL_JOBS,
                    resume=False, limit=None, skip_post=False,
                    timeout=config.M5_FINAL_TIMEOUT_S,
                    node_cap=config.M5_FINAL_NODE_CAP,
                    post_timeout=config.M5_FINAL_POST_TIMEOUT_S,
                    post_out="results/m5_final_post.csv",
                    samples_out="results/m5_final_post_samples.csv",
                    coverage_out="results/m5_final_post_coverage.csv"):
    """Grid, descending prefixes, and post into a separate final CSV set."""
    out = Path(out)
    existing = read_rows(out) if resume else []
    rows = list(existing)
    grid = list(final_grid_specs())
    if limit is not None:
        grid = grid[:limit]
    done = {
        (row["dataset"], int(row["trees"]), row.get("budget", ""), row["selector"])
        for row in rows if row.get("phase") == "grid"
    }
    pending = [spec for spec in grid if key_of(spec) not in done]
    if pending:
        tmp = out.with_name(out.name + ".part")
        part = run_local(pending, jobs=jobs, out=tmp, timeout=timeout, node_cap=node_cap)
        tmp.unlink(missing_ok=True)
        rows += _enrich_final(part, "grid")
        _write_final(out, rows)
    if limit is not None or skip_post:
        return rows

    prefixes = final_prefix_specs(rows)
    done = {
        (row["dataset"], int(row["trees"]), row.get("budget", ""), row["selector"])
        for row in rows if row.get("phase") == "prefix"
    }
    pending = [spec for spec in prefixes if key_of(spec) not in done]
    if pending:
        tmp = out.with_name(out.name + ".part")
        part = run_local(pending, jobs=jobs, out=tmp, timeout=timeout, node_cap=node_cap)
        tmp.unlink(missing_ok=True)
        rows += _enrich_final(part, "prefix")
        _write_final(out, rows)

    specs = final_post_specs(rows)
    run_post_local(specs, out=post_out, samples_out=samples_out,
                   timeout=post_timeout, jobs=1, node_cap=node_cap)
    post_rows = read_rows(post_out)
    coverage = post_coverage(post_rows)
    path = Path(coverage_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POST_COVERAGE_FIELDS)
        writer.writeheader()
        writer.writerows(coverage)
    return rows
