"""Optional Modal CPU accelerator for the campaign and post-build phases.

The local pipeline is complete on its own; Modal is an alternative execution surface:
`--campaign` reproduces grid and build completion (prefix search included); `--readout` and
`--robustness` reproduce the post-build phases.  The app expects two existing volumes:
  - `bdss-datasets`, mounted at `/data`, containing `datasets/<name>/...`;
  - `bdss-results`, mounted at `/results`, for one JSON per finished grid/build run.

Grid/build rows are returned to the coordinator, which writes the same clean CSV as the
local runner.  Robustness chunks are local; a chunk is the unit of resume.
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
APP_NAME = "bdss"

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
    cpu=int(config.JOB_CPU),
    memory=config.FINAL_MEMORY_MB,
    timeout=int(config.FINAL_TIMEOUT_S + 30),
    max_containers=config.FINAL_CONTAINERS,
)
def run_final_spec(spec):
    """Grid run: 300 s internal, 8M node cap."""
    return _run_impl(spec, config.FINAL_TIMEOUT_S, _result_path,
                     config.FINAL_NODE_CAP)


@app.function(
    image=image,
    volumes={"/data": datasets, "/results": results},
    cpu=int(config.JOB_CPU),
    memory=config.BUILD_MEMORY_MB,
    timeout=int(config.BUILD_TIMEOUT_S + 30),
    max_containers=config.BUILD_CONTAINERS,
)
def run_build_spec(spec):
    """One measured build-completion trial (4M, 8 GB, 300 s)."""
    return _run_impl(spec, config.BUILD_TIMEOUT_S, _result_path,
                     config.BUILD_NODE_CAP)


def _row_for(spec, result):
    """Turn a Modal exception into a campaign row."""
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
    return campaign._empty_row(spec, status, repr(error), config.BUILD_TIMEOUT_S)


def _run_campaign(max_containers):
    """Grid and build completion, writing the same clean CSV as `final`."""
    out = Path("results/campaign.csv")
    rows = campaign.read_rows(out) if out.exists() else []
    final_function = run_final_spec.with_options(
        max_containers=max_containers or config.FINAL_CONTAINERS
    )
    grid = list(campaign.final_grid_specs())
    done = {
        (row["dataset"], int(row["trees"]), row.get("budget", ""), row["selector"])
        for row in rows if row.get("phase") == "grid"
    }
    pending = [spec for spec in grid if campaign.key_of(spec) not in done]
    if pending:
        rows += campaign._enrich_final(_run_map(final_function, pending), "grid")
        campaign._write_final(out, rows)
        print(f"grid: {len(pending)} runs", flush=True)

    build_function = run_build_spec.with_options(
        max_containers=max_containers or config.BUILD_CONTAINERS
    )

    def measure(spec):
        try:
            return build_function.remote(spec)
        except BaseException as error:
            return _exception_row(spec, error)

    rows = campaign.run_build(rows, measure, out=out,
                              jobs=max_containers or config.BUILD_CONTAINERS)
    print(f"wrote {out} ({len(rows)} rows)", flush=True)
    return rows


@app.function(
    image=image,
    volumes={"/data": datasets, "/results": results},
    cpu=int(config.JOB_CPU),
    memory=config.READOUT_MEMORY_MB,
    timeout=int(config.READOUT_TIMEOUT_S + 60),
    max_containers=config.READOUT_CONTAINERS,
)
def run_readout(spec):
    """One selected build -> counts, volumes and per-bit rankings; no test set."""
    return post.readout_one(spec, node_cap=config.FINAL_NODE_CAP)


@app.function(
    image=image,
    volumes={"/data": datasets, "/results": results},
    cpu=int(config.JOB_CPU),
    memory=config.ROBUSTNESS_MEMORY_MB,
    timeout=int(config.ROBUSTNESS_TIMEOUT_S + 300),
    max_containers=config.ROBUSTNESS_CONTAINERS,
)
def run_robustness_worker(job):
    """Process one time slice of one model; the caller persists the returned chunk."""
    spec, start, stop, limit_s = job
    return post.robustness_range(
        spec, start=start, limit_s=limit_s, max_samples=stop,
        timeout_s=config.ROBUSTNESS_TIMEOUT_S,
        node_cap=config.FINAL_NODE_CAP,
    )


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
    return hours * (config.CPU_USD_PER_CORE_HOUR
                    + (config.ROBUSTNESS_MEMORY_MB / 1024.0)
                    * config.MEMORY_USD_PER_GIB_HOUR)


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

    workers = config.ROBUSTNESS_CONTAINERS
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
    readout_rows = post.read_rows("results/readouts.csv")
    coverage = post.coverage(specs, readout_rows, sample_rows, target_rows, campaign.read_rows("results/campaign.csv"))
    post.write_coverage("results/post_coverage.csv", coverage)
    print(f"wrote robustness: {samples} samples, {targets} targets, "
          f"{len(coverage)} coverage rows; ~{spent:.2f} USD")
    return sample_rows, target_rows


@app.local_entrypoint()
def main(limit: int | None = None, max_containers: int | None = None,
         max_usd: float = config.MAX_USD, resume: bool = False,
         campaign: bool = False, readout: bool = False, robustness: bool = False,
         demo: bool = False, limit_s: float = config.ROBUSTNESS_WORKER_S,
         chunks: str = "results/robustness_chunks"):
    """Run the campaign or a post-build phase on Modal."""
    from bdss import campaign as campaign_module

    if campaign:
        if limit is not None:
            raise SystemExit("--limit is not supported with --campaign")
        _run_campaign(max_containers)
        return

    if demo:
        rows = campaign_module.read_rows("results/campaign.csv")
        specs = post.selected_specs(rows)
        spec = next(item for item in specs if item[0] == "banknote")
        read = run_readout.remote(spec)
        chunk = run_robustness_worker.remote((spec, 0, 3, 30))
        print(f"demo {spec}: readout={read['status']} "
              f"samples={len(chunk['samples'])} targets={len(chunk['targets'])}")
        return

    if readout or robustness:
        rows = campaign_module.read_rows("results/campaign.csv")
        if not rows:
            raise SystemExit("results/campaign.csv is required for post")
        specs = post.selected_specs(rows)
        if limit is not None:
            specs = specs[:limit]
        if readout:
            existing = post.read_rows("results/readouts.csv") if resume else []
            done = {post._readout_key(row) for row in existing
                    if row.get("status") == "completed"}
            pending = [spec for spec in specs if post._spec_key(spec) not in done]
            estimate = campaign_module.estimate_usd(
                pending, config.READOUT_TIMEOUT_S,
                memory_mb=config.READOUT_MEMORY_MB,
            )
            print(f"readout estimate: {estimate}; specs: {len(pending)}/{len(specs)} pending")
            if max_usd > 0 and estimate["total_usd"] > max_usd:
                raise SystemExit(f"readout estimate {estimate['total_usd']} exceeds --max-usd")
            function = run_readout.with_options(
                max_containers=max_containers or config.READOUT_CONTAINERS
            )
            _run_readouts(function, specs, Path("results/readouts.csv"), resume)
            return
        function = run_robustness_worker.with_options(
            max_containers=max_containers or config.ROBUSTNESS_CONTAINERS
        )
        _run_robustness(function, specs, chunks, limit_s, max_usd)
        return

    raise SystemExit("choose --campaign, --demo, --readout or --robustness")
