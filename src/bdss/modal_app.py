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

from pathlib import Path
import json

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


def _run_impl(spec, timeout_s, out_path):
    if bdd.BACKEND != "cudd":
        raise RuntimeError("dd.cudd is not available; refusing to run on autoref")
    row = campaign.run_one_limited(spec, timeout_s)
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
    return _run_impl(spec, config.M5_TIMEOUT_S, _result_path)


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
    return _run_impl(spec, config.M5_RESCUE_TIMEOUT_S, _result_path)


def _row_for(spec, result):
    """Turn a Modal exception into a row instead of losing the specification."""
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


def _key(row):
    budget = row.get("budget", "")
    return campaign.key_of((
        row["dataset"], int(row["trees"]),
        None if budget in ("", None) else int(budget), row["selector"],
    ))


@app.local_entrypoint()
def main(smoke: bool = False, limit: int | None = None,
         max_containers: int = config.M5_MAX_CONTAINERS,
         max_usd: float = config.M5_MAX_USD, resume: bool = False,
         rescue: bool = False):
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
