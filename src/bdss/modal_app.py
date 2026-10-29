"""Optional Modal CPU entrypoint for the M5 campaign.

This is the only module that imports `modal`.  The local CLI and `bdss.campaign` never do:
`uv sync` without the optional extra has no Modal dependency and the local test suite does
not import this file.

The app expects two existing volumes in the selected Modal profile:
  - `bdss-datasets`, mounted at `/data`, containing `datasets/<name>/...`;
  - `bdss-results`, mounted at `/results`, for one JSON file per finished run.
Create/upload them in the profile chosen with `--profile` or `MODAL_PROFILE`; the app does
not create or populate them.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import threading

import modal

from bdss import bdd, campaign, config

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "bdss-m5"

app = modal.App(APP_NAME)
datasets = modal.Volume.from_name("bdss-datasets", create_if_missing=False)
results = modal.Volume.from_name("bdss-results", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Build inputs are copied into the image before `uv_sync`; local sources added after a
    # build step are rejected by Modal.  PYTHONPATH also exposes the package explicitly.
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src", copy=True)
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), remote_path="/root/pyproject.toml",
                    copy=True)
    .add_local_file(str(REPO_ROOT / "uv.lock"), remote_path="/root/uv.lock", copy=True)
    .add_local_file(str(REPO_ROOT / "README.md"), remote_path="/root/README.md", copy=True)
    .workdir("/root")
    .uv_sync(extras=["modal"])
    .env({"PYTHONPATH": "/root/src", "BDSS_DATASETS": "/data/datasets", "BDSS_BDD_BACKEND": "cudd"})
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
    memory=config.M5_MEMORY_MB,
    timeout=int(config.M5_TIMEOUT_S + 30),
    max_containers=config.M5_MAX_CONTAINERS,
)
def run_spec(spec):
    """Base run: internal 120 s, function 150 s."""
    return _run_impl(spec, config.M5_TIMEOUT_S, _result_path, config.NODE_CAP)


@app.function(
    image=image,
    volumes={"/data": datasets, "/results": results},
    cpu=int(config.M5_CPU),
    memory=config.M5_MEMORY_MB,
    timeout=int(config.M5_RESCUE_TIMEOUT_S + 30),
    max_containers=config.M5_MAX_CONTAINERS,
)
def run_rescue_spec(spec):
    """Rescue run: internal 300 s, function 330 s."""
    return _run_impl(spec, config.M5_RESCUE_TIMEOUT_S, _result_path, config.NODE_CAP)


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
    memory=config.M5_FINAL_POST_MEMORY_MB,
    timeout=int(config.M5_FINAL_POST_TIMEOUT_S + 30),
    max_containers=config.M5_FINAL_CONTAINERS,
)
def run_final_post(spec):
    """M5-final post: one build, full test set, 1800 s internal."""
    return campaign.post_one(spec, node_cap=config.M5_FINAL_NODE_CAP)


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


def _row_for(spec, result):
    """Turn a Modal exception into a row; post specs are 6-tuples."""
    if isinstance(result, BaseException):
        name = type(result).__name__.lower()
        status = "timeout" if isinstance(result, TimeoutError) or "timeout" in name else "error"
        if len(spec) == 6:
            return campaign._post_empty(spec, status, repr(result))
        return campaign._empty_row(spec, status, repr(result))
    return result


def _containers(final, requested, build=False):
    """Final and build default to 16 containers; the base path keeps 64."""
    if requested is not None:
        return requested
    if build:
        return config.M5_BUILD_CONTAINERS
    return config.M5_FINAL_CONTAINERS if final else config.M5_MAX_CONTAINERS


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


def _key(row):
    budget = row.get("budget", "")
    return campaign.key_of((
        row["dataset"], int(row["trees"]),
        None if budget in ("", None) else int(budget), row["selector"],
    ))


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


def _write_post(post_rows, stem="m5_final_post"):
    """Write the three post CSVs; samples are removed from the summary rows."""
    import csv
    samples = []
    for row in post_rows:
        samples.extend(row.pop("_samples", []))
    post_out = Path(f"results/{stem}.csv")
    post_out.parent.mkdir(parents=True, exist_ok=True)
    with post_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=campaign.POST_FIELDS)
        writer.writeheader(); writer.writerows(post_rows)
    samples_out = Path(f"results/{stem}_samples.csv")
    with samples_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=campaign.POST_SAMPLE_FIELDS)
        writer.writeheader(); writer.writerows(samples)
    coverage_out = Path(f"results/{stem}_coverage.csv")
    with coverage_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=campaign.POST_COVERAGE_FIELDS)
        writer.writeheader(); writer.writerows(campaign.post_coverage(post_rows))


@app.local_entrypoint()
def main(smoke: bool = False, limit: int | None = None,
         max_containers: int | None = None,
         max_usd: float = config.M5_MAX_USD, resume: bool = False,
         rescue: bool = False, final: bool = False, final_smoke: bool = False,
         build: bool = False, build_smoke: bool = False,
         final_post: bool = False):
    """Run a smoke, limited, or full campaign; the caller decides which mode."""
    if smoke:
        spec = ("banknote", 5, None, "modal")
        try:
            row = run_spec.remote(spec)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        print(f"smoke {spec}: {row['status']} peak={row.get('peak', '')}")
        return

    max_containers = _containers(final, max_containers, build=build)

    if build_smoke:
        spec = ("banknote", 5, 1_000, "modal")
        row = run_build_spec.remote(spec)
        print(f"build smoke {spec}: {row['status']} peak={row.get('peak', '')}")
        return

    if build:
        final_rows = campaign.read_rows(Path("results/m5_final.csv"))
        if not final_rows:
            raise SystemExit("results/m5_final.csv is required for --build")
        models = campaign.build_models(final_rows)
        pairs = campaign.build_pairs(models)
        estimate = campaign.estimate_usd(
            len(pairs) * len(config.M5_BUILD_KS), config.M5_BUILD_TIMEOUT_S,
            memory_mb=config.M5_BUILD_MEMORY_MB,
        )
        print(f"build estimate: {estimate}")
        print(f"build pairs: {len(pairs)}; models: {len(models)}")
        if max_usd > 0 and estimate["total_usd"] > config.M5_SOFT_USD:
            print(f"warning: build estimate {estimate['total_usd']} exceeds the 10 USD target")
        if max_usd > 0 and estimate["total_usd"] > max_usd:
            raise SystemExit(f"build estimate {estimate['total_usd']} USD exceeds --max-usd")
        function = run_build_spec.with_options(max_containers=max_containers)
        rows = _run_build(function, pairs, Path("results/m5_build.csv"), max_containers)
        print(f"wrote results/m5_build.csv ({len(rows)} trials)")
        return

    if final_post:
        rows = campaign.read_rows(Path("results/m5_final_campaign.csv"))
        if not rows:
            raise SystemExit("results/m5_final_campaign.csv is required for --final-post")
        specs = campaign.final_post_specs(rows)
        estimate = campaign.estimate_usd(
            specs, config.M5_FINAL_POST_TIMEOUT_S,
            memory_mb=config.M5_FINAL_POST_MEMORY_MB,
        )
        print(f"final post estimate: {estimate}; specs: {len(specs)}")
        if max_usd > 0 and estimate["total_usd"] > max_usd:
            raise SystemExit(f"final post estimate {estimate['total_usd']} USD exceeds --max-usd")
        containers = (config.M5_FINAL_CONTAINERS
                      if max_containers == config.M5_MAX_CONTAINERS else max_containers)
        function = run_final_post.with_options(max_containers=containers)
        post_rows = _run_map(function, specs)
        _write_post(post_rows)
        print(f"wrote final post for {len(post_rows)} builds")
        return

    if final_smoke:
        spec = ("banknote", 5, 1000, "modal")
        row = run_final_spec.remote(spec)
        post = run_final_post.remote((*spec, 300, "grid"))
        print(f"final smoke {spec}: {row['status']} peak={row.get('peak', '')}; "
              f"post={post['status']}")
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
        function = run_final_spec.with_options(max_containers=max_containers)
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
            specs = campaign.final_post_specs(rows)
            post_function = run_final_post.with_options(max_containers=max_containers)
            _write_post(_run_map(post_function, specs))
        print(f"wrote {out} ({len(rows)} rows)")
        return

    if rescue:
        specs = list(campaign.rescue_specs(campaign.read_rows("results/m5_campaign.csv")))
        if limit is not None:
            specs = specs[:limit]
        estimate = campaign.estimate_usd(specs, config.M5_RESCUE_TIMEOUT_S)
        print(f"rescue estimate: {estimate}")
        if max_usd > 0 and estimate["total_usd"] > config.M5_SOFT_USD:
            print(f"warning: rescue estimate {estimate['total_usd']} exceeds the 10 USD target")
        if max_usd > 0 and estimate["total_usd"] > max_usd:
            raise SystemExit(f"rescue estimate {estimate['total_usd']} USD exceeds --max-usd")
        function = run_rescue_spec.with_options(max_containers=max_containers)
        rows = _run_map(function, specs)
        out = Path("results/m5_campaign_300.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=campaign.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out} ({len(rows)} rescue results)")
        return

    out = Path("results/m5_campaign.csv")
    existing = campaign.read_rows(out) if resume else []
    by_key = {_key(row): row for row in existing}
    grid_specs = list(campaign.grid_specs())
    todo = grid_specs[:limit] if limit is not None else grid_specs
    pending = [spec for spec in todo if campaign.key_of(spec) not in by_key]
    estimate = campaign.estimate_usd(pending)
    print(f"estimate: {estimate}")
    if max_usd > 0 and estimate["total_usd"] > config.M5_SOFT_USD:
        print(f"warning: estimate {estimate['total_usd']} exceeds the 10 USD target")
    if max_usd > 0 and estimate["total_usd"] > max_usd:
        raise SystemExit(f"estimate {estimate['total_usd']} USD exceeds --max-usd")

    function = run_spec.with_options(max_containers=max_containers)
    for row in _run_map(function, pending):
        by_key[_key(row)] = row

    if limit is None:
        grid_keys = {campaign.key_of(spec) for spec in grid_specs}
        completed = {
            (row["dataset"], int(row["trees"])) for key, row in by_key.items()
            if key in grid_keys and row["status"] == "completed"
        }
        models = {(spec[0], spec[1]) for spec in grid_specs}
        walls = sorted(models - completed)
        prefixes = [spec for spec in campaign.prefix_specs(walls)
                    if campaign.key_of(spec) not in by_key]
        if prefixes:
            total = campaign.estimate_usd(pending + prefixes)
            print(f"estimate with prefixes: {total}")
            if max_usd > 0 and total["total_usd"] > config.M5_SOFT_USD:
                print(f"warning: prefix estimate {total['total_usd']} exceeds the 10 USD target")
            if max_usd > 0 and total["total_usd"] > max_usd:
                print("prefix phase skipped: max_usd would be exceeded")
            else:
                for row in _run_map(function, prefixes):
                    by_key[_key(row)] = row
        all_rows = list(by_key.values())
        bisection = [spec for spec in campaign.bisection_specs(all_rows)
                     if campaign.key_of(spec) not in by_key]
        if bisection:
            total = campaign.estimate_usd(list(pending) + prefixes + bisection)
            print(f"estimate with bisection: {total}")
            if max_usd > 0 and total["total_usd"] > config.M5_SOFT_USD:
                print(f"warning: bisection estimate {total['total_usd']} exceeds the 10 USD target")
            if max_usd > 0 and total["total_usd"] > max_usd:
                print("bisection phase skipped: max_usd would be exceeded")
            else:
                for row in _run_map(function, bisection):
                    by_key[_key(row)] = row

    rows = list(by_key.values()) if resume else [
        row for key, row in by_key.items()
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=campaign.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} results)")
