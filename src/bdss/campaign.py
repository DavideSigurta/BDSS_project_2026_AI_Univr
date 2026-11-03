"""Campaign: shared specifications, one-run evaluation, local runner, costs.

This module is deliberately free of any Modal import.  The optional Modal entrypoint
(`bdss.modal_app`) wraps `run_one`; the local CLI (`python -m bdss.study final`) uses
`run_local`.  Both therefore run exactly the same pipeline and write the same row schema.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import csv
import json
import resource
import sys
import time

from bdss import compose, config, readouts
from bdss.forest import Forest, load_bundle
from bdss.runtime import run_limited as _run_limited


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


def key_of(spec):
    name, trees, budget, selector = spec
    return (str(name), int(trees), "" if budget is None else str(int(budget)), str(selector))


def _row_key(row):
    budget = row.get("budget", "")
    return (str(row["dataset"]), int(row["trees"]), str(budget), str(row["selector"]))


def _empty_row(spec, status, note="", timeout_s=config.RUN_TIMEOUT_S):
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


def run_one(spec, timeout_s=config.RUN_TIMEOUT_S, node_cap=config.NODE_CAP):
    """Build one campaign specification and return one CSV-shaped row.

    The pipeline is the existing one: load the bundle, choose the shallow order, rebuild
    the prefix model, then `compose.build`.  Readouts are the certified counts only; a
    timeout is a status, never a silent missing row.
    """
    from bdss.order import permute, shallow_order

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
    except compose.SelectorTimeout as error:
        return _empty_row(spec, "selector-timeout", repr(error), timeout_s)
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


def run_one_limited(spec, timeout=config.RUN_TIMEOUT_S, node_cap=config.NODE_CAP):
    """One run in a fresh child process; timeout/error become a row, never a lost run."""

    result = _run_limited(_run_one_queue, (spec, node_cap), timeout)
    if result is None:
        return _empty_row(spec, "timeout", "time limit", timeout)
    if isinstance(result, tuple) and result[0] == "error":
        return _empty_row(spec, "error", result[1], timeout)
    result["timeout_s"] = int(timeout)
    return result


def run_local(specs, jobs=config.LOCAL_JOBS, out="results/local_campaign.csv",
              resume=False, timeout=config.RUN_TIMEOUT_S, node_cap=config.NODE_CAP):
    """Run specifications in fresh child processes and append one CSV row per trial."""

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


def estimate_usd(specs, timeout=config.RUN_TIMEOUT_S, cpu=config.JOB_CPU,
                 memory_mb=config.JOB_MEMORY_MB):
    """Conservative Modal CPU + memory estimate for the given run count/list."""
    n_runs = len(specs) if not isinstance(specs, int) else specs
    cpu_hours = n_runs * timeout / 3600.0 * cpu
    memory_gib_hours = n_runs * timeout / 3600.0 * (memory_mb / 1024.0)
    cpu_usd = cpu_hours * config.CPU_USD_PER_CORE_HOUR
    memory_usd = memory_gib_hours * config.MEMORY_USD_PER_GIB_HOUR
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
# Final campaign: one CSV, 3 budgets, build completion, full-test post.

FINAL_FIELDS = CSV_FIELDS + ("phase", "k", "source", "node_cap")


def final_grid_specs(models=None, budgets=None, selectors=None):
    """54 models × three budgets × five selectors, deterministic order."""
    models = tuple(dataset_models() if models is None else models)
    budgets = tuple(config.FINAL_BUDGETS if budgets is None else budgets)
    selectors = tuple(config.FINAL_SELECTORS if selectors is None else selectors)
    return tuple(
        (name, trees, budget, selector)
        for name, trees in models
        for budget in budgets
        for selector in selectors
    )


def _enrich_final(rows, phase, node_cap=None, source=None):
    out = []
    for row in rows:
        row = dict(row)
        row.update({
            "phase": phase, "k": int(row.get("trees")),
            "source": source or phase,
            "node_cap": config.FINAL_NODE_CAP if node_cap is None else node_cap,
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
        for budget in config.BUILD_BUDGETS
        for selector in config.BUILD_SELECTORS
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
    for k in config.BUILD_KS:
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


def _write_final(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_final_atomic(path, rows):
    """Replace the clean CSV only after a complete pair/phase has been written."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    _write_final(tmp, rows)
    tmp.replace(path)


def run_build(rows, measure, out="results/campaign.csv", jobs=config.LOCAL_JOBS,
              node_cap=config.BUILD_NODE_CAP):
    """Complete the non-full models, using `measure(spec)` for each prefix trial.

    A pair is written only after its whole search has returned, so a pair present in
    the CSV is complete; resume skips it and starts from the first missing pair.
    """
    out = Path(out)
    rows = list(rows)
    done = {
        (row["dataset"], int(row["budget"]), row["selector"])
        for row in rows if row.get("source") == "build"
    }
    pairs = [
        pair for pair in build_pairs(build_models(rows))
        if (pair[0], int(pair[2]), pair[3]) not in done
    ]
    if not pairs:
        return rows

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(build_search, pair, measure): pair for pair in pairs}
        for future in as_completed(futures):
            pair = futures[future]
            part = _enrich_final(future.result(), "build", node_cap=node_cap, source="build")
            rows.extend(part)
            _write_final_atomic(out, rows)
            print(f"  build {pair}: {len(part)} trials", flush=True)
    return rows


def run_build_local(rows, out="results/campaign.csv", jobs=config.LOCAL_JOBS,
                    timeout=config.BUILD_TIMEOUT_S, node_cap=config.BUILD_NODE_CAP):
    """Local `run_build`: one child process per prefix trial."""
    def measure(spec):
        return run_one_limited(spec, timeout=timeout, node_cap=node_cap)
    return run_build(rows, measure, out=out, jobs=jobs, node_cap=node_cap)


def run_final_local(out="results/campaign.csv", jobs=config.LOCAL_JOBS,
                    limit=None, timeout=config.FINAL_TIMEOUT_S,
                    node_cap=config.FINAL_NODE_CAP):
    """Grid only; build completion is `run_build_local`, post is `bdss.post`.

    An existing clean CSV is always resumed, so the shipped build-completion rows are
    preserved when this command is re-run.
    """
    out = Path(out)
    rows = list(read_rows(out) if out.exists() else [])
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
    if limit is not None:
        return rows

    return rows
