"""Optional Modal CPU entrypoint for the M5-final campaign and its post-build phases.

This is the only module that imports `modal`.  The app expects two existing volumes in the
selected Modal profile:
  - `bdss-datasets`, mounted at `/data`, containing `datasets/<name>/...`;
  - `bdss-results`, mounted at `/results`, for one JSON file per finished grid/build run.

Grid/build rows are written to the volume; readouts and robustness chunks are returned to the
local entrypoint, which writes the final CSVs under `results/`.  A robustness chunk is the
unit of resume: a preemption loses at most the samples of one chunk.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import threading
import time

import modal

from bdss import bdd, campaign, config, post

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "bdss-m5"

app = modal.App(APP_NAME)
datasets = modal.Volume.from_name("bdss-datasets", create_if_missing=False)
results = modal.Volume.from_name("bdss-results", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src", copy=True)
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), remote_path="/root/pyproject.toml",
                    copy=True)
    .add_local_file(str(REPO_ROOT / "uv.lock"), remote_path="/root/uv.lock", copy=True)
    .add_local_file(str(REPO_ROOT / "README.md"), remote_path="/root/README.md", copy=True)
    .workdir("/root")
    .uv_sync(extras=["modal"])
    .env({"PYTHONPATH": "/root/src", "BDSS_DATASETS": "/data/datasets",
          "BDSS_BDD_BACKEND": "cudd"})
)


def _result_path(spec):
    key = campaign.key_of(spec)
    stem = "_".join((key[0], str(key[1]), key[2] or "none", key[3]))
    return Path("/results") / f"{stem}.json"


def _run_impl(spec, timeout_s, out_path, node_cap=config.NODE_CAP):
    if bdd.BACKEND != "cudd":
        raise RuntimeError("dd.cudd is not available; refusing to run on autoref")
    row = campaign.run_one_limited(spec, timeout_s, node_cap=node_cap)
    path = out_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": list(campaign.key_of(spec)), "row": row}))
    results.commit()
    return row


@app.function(
    image=image,
    volumes={"/data": datasets, "/results": results},
    cpu=int(config.M5_CPU),
    memory=config.M5_FINAL_MEMORY_MB,
    timeout=int(config.M5_FINAL_TIMEOUT_S + 30),
    max_containers=config.M5_FINAL_CONTAINERS,
)
def run_final_spec(spec):
    """M5-final grid/prefix run: 300 s internal, 8M node cap."""
    return _run_impl(spec, config.M5_FINAL_TIMEOUT_S, _result_path,
                     config.M5_FINAL_NODE_CAP)


@app.function(
    image=image,
    volumes={"/data": datasets, "/results": results},
    cpu=int(config.M5_CPU),
    memory=config.M5_BUILD_MEMORY_MB,
    timeout=int(config.M5_BUILD_TIMEOUT_S + 30),
    max_containers=config.M5_BUILD_CONTAINERS,
)
def run_build_spec(spec):
    """One measured prefix trial for the build-completion phase (4M, 8 GB, 300 s)."""
    return _run_impl(spec, config.M5_BUILD_TIMEOUT_S, _result_path,
                     config.M5_BUILD_NODE_CAP)


@app.function(
    image=image,
    volumes={"/data": datasets, "/results": results},
    cpu=int(config.M5_CPU),
    memory=config.M5_READOUT_MEMORY_MB,
    timeout=int(config.M5_READOUT_TIMEOUT_S + 60),
    max_containers=config.M5_READOUT_CONTAINERS,
)
def run_readout(spec):
    """One selected build -> counts, volumes and per-bit rankings; no test set."""
    return post.readout_one(spec, node_cap=config.M5_FINAL_NODE_CAP)


@app.function(
    image=image,
    volumes={"/data": datasets, "/results": results},
    cpu=int(config.M5_CPU),
    memory=config.M5_ROBUSTNESS_MEMORY_MB,
    timeout=int(config.M5_ROBUSTNESS_TIMEOUT_S + 300),
    max_containers=config.M5_ROBUSTNESS_CONTAINERS,
)
def run_robustness_worker(job):
    """Process one time slice of one model; the caller persists the returned chunk."""
    spec, start, stop, limit_s = job
    return post.robustness_range(
        spec, start=start, limit_s=limit_s, max_samples=stop,
        timeout_s=config.M5_ROBUSTNESS_TIMEOUT_S,
        node_cap=config.M5_FINAL_NODE_CAP,
    )


def _row_for(spec, result):
    """Turn a Modal exception into a campaign row (grid/build specs are 4-tuples)."""
    if isinstance(result, BaseException):
        name = type(result).__name__.lower()
        status = "timeout" if isinstance(result, TimeoutError) or "timeout" in name else "error"
        return campaign._empty_row(spec, status, repr(result))
    return result


def _run_map(function, specs):
    """Map specs and preserve one row per spec, including exceptions."""
    specs = list(specs)
    results = list(function.map(specs, return_exceptions=True, order_outputs=True))
    if len(results) != len(specs):
        print(f"warning: {len(specs)} specs, {len(results)} results")
    rows = [_row_for(spec, result) for spec, result in zip(specs, results)]
    for spec in specs[len(rows):]:
        rows.append(campaign._empty_row(spec, "error", "missing Modal result"))
    return rows[:len(specs)]


def _exception_row(spec, error):
    name = type(error).__name__.lower()
    status = "timeout" if isinstance(error, TimeoutError) or "timeout" in name else "error"
    return campaign._empty_row(spec, status, repr(error), config.M5_BUILD_TIMEOUT_S)


def _build_pair(pair, function):
    """Sequential search inside one pair; each trial is a separate Modal run."""
    def measure(spec):
        try:
            return function.remote(spec)
        except BaseException as error:
            return _exception_row(spec, error)
    return campaign.build_search(pair, measure)


def _run_build(function, pairs, out, workers):
    """Run all pairs in parallel; write the CSV after every completed pair."""
    pairs = list(pairs)
    rows = []
    lock = threading.Lock()
    campaign._write_final(out, rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_build_pair, pair, function): pair for pair in pairs}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                pair_rows = future.result()
            except BaseException as error:
                pair_rows = [_exception_row((pair[0], 5, pair[2], pair[3]), error)]
            with lock:
                rows.extend(campaign._enrich_final(
                    pair_rows, "prefix",
                    node_cap=config.M5_BUILD_NODE_CAP, source="build",
                ))
                campaign._write_final(out, rows)
            print(f"  {pair[0]} {pair[2]} {pair[3]}: {len(pair_rows)} trials", flush=True)
    return rows


def _readout_remote(function, spec, attempts=3):
    """Call one readout, retrying only transient client/network failures."""
    for attempt in range(attempts):
        try:
            return function.remote(spec)
        except BaseException as error:
            name = type(error).__name__.lower()
            transient = isinstance(error, ConnectionError) or "connection" in name
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def _run_readouts(function, specs, out, resume):
    """One row per selected build, persisted once after all calls return."""
    post._guard_canonical(specs, [out])
    existing = post.read_rows(out) if resume else []
    done = {post._readout_key(row) for row in existing if row.get("status") == "completed"}
    pending = [spec for spec in specs if post._spec_key(spec) not in done]
    rows = list(existing)
    if not pending:
        print(f"wrote {out} ({len(rows)} readouts)")
        return rows
    with ThreadPoolExecutor(max_workers=min(len(pending), 32)) as pool:
        futures = {pool.submit(_readout_remote, function, spec): spec for spec in pending}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                name = type(error).__name__.lower()
                status = "timeout" if isinstance(error, TimeoutError) or "timeout" in name else "error"
                result = post._empty_readout(spec, status, repr(error))
            rows = [old for old in rows if post._readout_key(old) != post._readout_key(result)]
            rows.append(result)
            print(f"  readout {post._spec_key(spec)}: {result['status']}", flush=True)
    post.write_readouts(out, rows)
    print(f"wrote {out} ({len(rows)} readouts)")
    return rows


def _estimate_worker_usd(seconds):
    """Observed Modal cost of one 16 GB / 1 CPU worker, in USD."""
    hours = seconds / 3600.0
    return hours * (config.M5_CPU_USD_PER_CORE_HOUR
                    + (config.M5_ROBUSTNESS_MEMORY_MB / 1024.0)
                    * config.M5_MEMORY_USD_PER_GIB_HOUR)


def _covered_gaps(chunks, spec, total):
    """Return the uncovered sample intervals of one model, in order."""
    key = post._spec_key(spec)
    intervals = sorted(
        (int(chunk["start"]), int(chunk["next"]))
        for chunk in chunks
        if chunk.get("spec") and post._spec_key(chunk["spec"]) == key
    )
    gaps = []
    cursor = 0
    for start, end in intervals:
        if end <= cursor:
            continue
        if start > cursor:
            gaps.append((cursor, min(start, total)))
        cursor = max(cursor, end)
        if cursor >= total:
            break
    if cursor < total:
        gaps.append((cursor, total))
    return gaps


def _run_robustness(function, specs, chunks, limit_s, max_usd):
    """Process small uncovered segments with bounded in-flight calls."""
    post._guard_canonical(specs, [post.CANONICAL_SAMPLES, post.CANONICAL_TARGETS, post.CANONICAL_COVERAGE])
    chunks = Path(chunks)
    all_chunks = post.read_chunks(chunks)
    pending = [spec for spec in specs if not post.chunk_done(all_chunks, spec)]
    errors = {}
    spent = _estimate_worker_usd(
        sum(float(chunk.get("seconds", 0.0)) for chunk in all_chunks)
    )
    segment = 10
    jobs = []
    for spec in pending:
        total = post._test_rows(spec[0])
        for gap_start, gap_stop in _covered_gaps(all_chunks, spec, total):
            start = gap_start
            while start < gap_stop:
                stop = min(gap_stop, start + segment)
                jobs.append((spec, start, stop))
                start = stop
    if not jobs:
        print("robustness: no uncovered segments", flush=True)

    workers = config.M5_ROBUSTNESS_CONTAINERS
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}

        def submit_next():
            while jobs and len(futures) < workers:
                spec, start, stop = jobs.pop(0)
                future = pool.submit(function.remote, (spec, start, stop, limit_s))
                futures[future] = (spec, start, stop)

        while jobs or futures:
            submit_next()
            if not futures:
                continue
            future = next(as_completed(list(futures)))
            spec, start, stop = futures.pop(future)
            key = post._spec_key(spec)
            try:
                result = future.result()
            except BaseException as exc:
                errors[key] = errors.get(key, 0) + 1
                print(f"  robustness {key} {start}:{stop}: {exc!r}", flush=True)
                if errors[key] < 4:
                    jobs.insert(0, (spec, start, stop))
                continue
            next_index = int(result.get("next", start))
            if next_index <= start:
                errors[key] = errors.get(key, 0) + 1
                print(f"  robustness {key} {start}:{stop}: no progress", flush=True)
                if errors[key] < 4:
                    jobs.insert(0, (spec, start, stop))
                continue
            post.write_chunk(chunks, result)
            all_chunks.append(result)
            spent += _estimate_worker_usd(float(result.get("seconds", 0.0)))
            total = post._test_rows(spec[0])
            print(f"  robustness {key} [{start}:{next_index}/{total}] "
                  f"{'done' if result['done'] else 'continues'}; ~{spent:.2f} USD",
                  flush=True)
            if next_index < stop:
                jobs.insert(0, (spec, next_index, stop))
            if max_usd > 0 and spent > max_usd:
                raise SystemExit(f"robustness spent ~{spent:.2f} USD > --max-usd {max_usd}")

    incomplete = [spec for spec in specs if not post.chunk_done(all_chunks, spec)]
    if incomplete:
        names = ", ".join("_".join(map(str, post._spec_key(spec))) for spec in incomplete)
        raise SystemExit(f"robustness incomplete for {len(incomplete)} models: {names}")
    samples, targets, sample_rows, target_rows = post.combine_robustness(all_chunks)
    readout_rows = post.read_rows("results/m5_readouts.csv")
    coverage = post.coverage(specs, readout_rows, sample_rows, target_rows)
    post.write_coverage("results/m5_post_coverage.csv", coverage)
    print(f"wrote robustness: {samples} samples, {targets} targets, "
          f"{len(coverage)} coverage rows; ~{spent:.2f} USD")
    return sample_rows, target_rows


@app.local_entrypoint()
def main(smoke: bool = False, limit: int | None = None,
         max_containers: int | None = None,
         max_usd: float = config.M5_MAX_USD, resume: bool = False,
         final: bool = False, build: bool = False, build_smoke: bool = False,
         readout: bool = False, robustness: bool = False, post_smoke: bool = False,
         limit_s: float = config.M5_ROBUSTNESS_WORKER_S,
         chunks: str = "results/m5_robustness_chunks"):
    """Run one of the four phases: grid/prefix, build completion, readout, robustness."""
    if smoke:
        spec = ("banknote", 5, 1000, "modal")
        row = run_final_spec.remote(spec)
        print(f"smoke {spec}: {row['status']} peak={row.get('peak', '')}")
        return

    if build_smoke:
        spec = ("banknote", 5, 1_000, "modal")
        row = run_build_spec.remote(spec)
        print(f"build smoke {spec}: {row['status']} peak={row.get('peak', '')}")
        return

    if post_smoke:
        rows = campaign.read_rows("results/m5_final_campaign.csv")
        specs = post.selected_specs(rows)
        spec = next(item for item in specs if item[0] == "banknote")
        read = run_readout.remote(spec)
        chunk = run_robustness_worker.remote((spec, 0, 3, 30))
        print(f"post smoke {spec}: readout={read['status']} "
              f"samples={len(chunk['samples'])} targets={len(chunk['targets'])}")
        return

    if build:
        raw = Path("results/m5_final.csv")
        clean = Path("results/m5_final_campaign.csv")
        source = raw if raw.exists() and raw.stat().st_size > 0 else clean
        rows = campaign.read_rows(source)
        if not rows:
            raise SystemExit(f"{source} is required for --build")
        models = campaign.build_models(rows)
        pairs = campaign.build_pairs(models)
        estimate = campaign.estimate_usd(
            len(pairs) * len(config.M5_BUILD_KS), config.M5_BUILD_TIMEOUT_S,
            memory_mb=config.M5_BUILD_MEMORY_MB,
        )
        print(f"build estimate: {estimate}")
        print(f"build pairs: {len(pairs)}; models: {len(models)}")
        if max_usd > 0 and estimate["total_usd"] > max_usd:
            raise SystemExit(f"build estimate {estimate['total_usd']} USD exceeds --max-usd")
        workers = max_containers or config.M5_BUILD_CONTAINERS
        function = run_build_spec.with_options(max_containers=workers)
        rows = _run_build(function, pairs, Path("results/m5_build.csv"), workers)
        print(f"wrote results/m5_build.csv ({len(rows)} trials)")
        return

    if readout or robustness:
        rows = campaign.read_rows("results/m5_final_campaign.csv")
        if not rows:
            raise SystemExit("results/m5_final_campaign.csv is required for post")
        specs = post.selected_specs(rows)
        if limit is not None:
            specs = specs[:limit]
        if readout:
            existing = post.read_rows("results/m5_readouts.csv") if resume else []
            done = {post._readout_key(row) for row in existing
                    if row.get("status") == "completed"}
            pending = [spec for spec in specs if post._spec_key(spec) not in done]
            estimate = campaign.estimate_usd(
                pending, config.M5_READOUT_TIMEOUT_S,
                memory_mb=config.M5_READOUT_MEMORY_MB,
            )
            print(f"readout estimate: {estimate}; specs: {len(pending)}/{len(specs)} pending")
            if max_usd > 0 and estimate["total_usd"] > max_usd:
                raise SystemExit(f"readout estimate {estimate['total_usd']} exceeds --max-usd")
            function = run_readout.with_options(
                max_containers=max_containers or config.M5_READOUT_CONTAINERS
            )
            _run_readouts(function, specs, Path("results/m5_readouts.csv"), resume)
            return
        function = run_robustness_worker.with_options(
            max_containers=max_containers or config.M5_ROBUSTNESS_CONTAINERS
        )
        _run_robustness(function, specs, chunks, limit_s, max_usd)
        return

    if final:
        out = Path("results/m5_final.csv")
        existing = campaign.read_rows(out) if resume else []
        rows = list(existing)
        grid = list(campaign.final_grid_specs())
        if limit is not None:
            grid = grid[:limit]
        done = {
            (row["dataset"], int(row["trees"]), row.get("budget", ""), row["selector"])
            for row in rows if row.get("phase") == "grid"
        }
        pending = [spec for spec in grid if campaign.key_of(spec) not in done]
        estimate = campaign.estimate_usd(pending, config.M5_FINAL_TIMEOUT_S,
                                         memory_mb=config.M5_FINAL_MEMORY_MB)
        print(f"final estimate: {estimate}")
        if max_usd > 0 and estimate["total_usd"] > config.M5_SOFT_USD:
            print(f"warning: final estimate {estimate['total_usd']} exceeds the 10 USD target")
        if max_usd > 0 and estimate["total_usd"] > max_usd:
            raise SystemExit(f"final estimate {estimate['total_usd']} USD exceeds --max-usd")
        function = run_final_spec.with_options(
            max_containers=max_containers or config.M5_FINAL_CONTAINERS
        )
        if pending:
            rows += campaign._enrich_final(_run_map(function, pending), "grid")
            campaign._write_final(out, rows)
        if limit is None:
            prefixes = campaign.final_prefix_specs(rows)
            done = {
                (row["dataset"], int(row["trees"]), row.get("budget", ""), row["selector"])
                for row in rows if row.get("phase") == "prefix"
            }
            pending = [spec for spec in prefixes if campaign.key_of(spec) not in done]
            if pending:
                rows += campaign._enrich_final(_run_map(function, pending), "prefix")
                campaign._write_final(out, rows)
        print(f"wrote {out} ({len(rows)} rows)")
        return

    raise SystemExit("choose --final, --build, --readout or --robustness")
