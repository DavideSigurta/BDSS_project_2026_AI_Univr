"""Local CLI: campaign, readout, robustness and demo.

The pipeline itself lives in `bdss.campaign`, `bdss.post` and the core modules; this file
only parses the four local commands.
"""

from pathlib import Path
import sys

from bdss import config

RESULTS = config.REPO_ROOT / "results"


def final_main():
    """Grid plus build completion; post is a separate command."""
    import argparse

    from bdss import campaign

    parser = argparse.ArgumentParser(prog="python -m bdss.study final")
    parser.add_argument("--out", default=str(RESULTS / "campaign.csv"))
    parser.add_argument("--timeout", type=float, default=config.FINAL_TIMEOUT_S)
    parser.add_argument("--node-cap", type=int, default=config.FINAL_NODE_CAP)
    parser.add_argument("--jobs", type=int, default=config.LOCAL_JOBS)
    parser.add_argument("--max-usd", type=float, default=config.FINAL_MAX_USD)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(sys.argv[2:])

    out = Path(args.out)
    if args.limit is not None and out.resolve() == (RESULTS / "campaign.csv").resolve():
        raise SystemExit("refusing to overwrite the canonical campaign with --limit; use --out")

    grid = list(campaign.final_grid_specs())
    if args.limit is not None:
        grid = grid[:args.limit]
    estimate = campaign.estimate_usd(grid, args.timeout, memory_mb=config.FINAL_MEMORY_MB)
    print(f"final estimate (grid): {estimate}", flush=True)
    if args.max_usd > 0 and estimate["total_usd"] > config.SOFT_USD:
        print(f"warning: grid estimate {estimate['total_usd']} exceeds the 10 USD target",
              flush=True)
    if args.max_usd > 0 and estimate["total_usd"] > args.max_usd:
        raise SystemExit(f"final estimate {estimate['total_usd']} USD exceeds --max-usd")
    rows = campaign.run_final_local(
        out=args.out, jobs=args.jobs, limit=args.limit,
        timeout=args.timeout, node_cap=args.node_cap,
    )
    if args.limit is None:
        rows = campaign.run_build_local(
            rows, out=args.out, jobs=args.jobs,
            timeout=config.BUILD_TIMEOUT_S, node_cap=config.BUILD_NODE_CAP,
        )
    print(f"final rows: {len(rows)} -> {args.out}", flush=True)


def demo_main():
    """Small local end-to-end demo: one selected build, readouts and a few samples."""
    import argparse
    import json

    from bdss import campaign, post

    parser = argparse.ArgumentParser(prog="python -m bdss.study demo")
    parser.add_argument("--dataset", default="iris")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(sys.argv[2:])

    rows = campaign.read_rows(RESULTS / "campaign.csv")
    specs = post.selected_specs(rows)
    spec = next((item for item in specs if item[0] == args.dataset), None)
    if spec is None:
        raise SystemExit(f"no selected build for {args.dataset!r}")

    readout = post.readout_one(spec, config.FINAL_NODE_CAP)
    if readout["status"] != "completed":
        raise SystemExit(f"readout unavailable for {args.dataset}: {readout['note']}")
    counts = json.loads(readout["counts_json"])
    print(f"model {spec[0]}: {readout['n_bits']} input bits, {readout['n_classes']} classes")
    print(f"selected build: k={spec[1]} budget={spec[2]} selector={spec[3]} ({spec[4]})")
    print("certified counts: " + ", ".join(f"{label}={counts[str(label)]}"
                                            for label in range(int(readout["n_classes"]))))
    result = post.robustness_range(
        spec, start=0, max_samples=args.samples, limit_s=config.ROBUSTNESS_WORKER_S,
        node_cap=config.FINAL_NODE_CAP,
    )
    print(f"test samples: {result['test_samples']}; showing {len(result['samples'])}")
    for row in result["samples"]:
        own = row["own_label"] if row["own_label"] != "" else "-"
        rival = row["min_rival"] if row["min_rival"] != "" else "-"
        own_radius = row["own_radius"] if row["own_radius"] != "" else "-"
        loss = row["loss"] if row["loss"] != "" else "-"
        rival_label = row["min_rival_label"] if row["min_rival_label"] != "" else "-"
        print(
            f"sample {row['sample']}: own={own} certified={row['certified']} "
            f"own_radius={own_radius} min_rival={rival} "
            f"rival={rival_label} loss={loss}"
        )
        if row["min_rival_bits"]:
            print(f"  witness bits={row['min_rival_bits']} "
                  f"features={row['min_rival_features']} cells={row['min_rival_cells']}")


def readout_main():
    """Readouts for the selected build of every model; no test set."""
    import argparse

    from bdss import campaign, post

    parser = argparse.ArgumentParser(prog="python -m bdss.study readout")
    parser.add_argument("--rows", default=str(RESULTS / "campaign.csv"))
    parser.add_argument("--out", default=str(RESULTS / "readouts.csv"))
    parser.add_argument("--timeout", type=float, default=config.READOUT_TIMEOUT_S)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-usd", type=float, default=config.FINAL_MAX_USD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(sys.argv[2:])

    rows = campaign.read_rows(args.rows)
    specs = list(post.selected_specs(rows))
    if args.limit is not None:
        specs = specs[:args.limit]
    estimate = campaign.estimate_usd(specs, args.timeout,
                                     memory_mb=config.READOUT_MEMORY_MB)
    print(f"readout estimate: {estimate}", flush=True)
    if args.max_usd > 0 and estimate["total_usd"] > config.SOFT_USD:
        print(f"warning: readout estimate {estimate['total_usd']} exceeds the 10 USD target",
              flush=True)
    if args.max_usd > 0 and estimate["total_usd"] > args.max_usd:
        raise SystemExit(f"readout estimate {estimate['total_usd']} USD exceeds --max-usd")
    result = post.run_readouts_local(
        specs, out=args.out, resume=args.resume, timeout=args.timeout, jobs=args.jobs,
        node_cap=config.FINAL_NODE_CAP,
    )
    print(f"readouts: {len(result)} -> {args.out}", flush=True)


def robustness_main():
    """Local fallback for full-test robustness; writes resumable chunks and final CSVs."""
    import argparse

    from bdss import campaign, post

    parser = argparse.ArgumentParser(prog="python -m bdss.study robustness")
    parser.add_argument("--rows", default=str(RESULTS / "campaign.csv"))
    parser.add_argument("--chunks", default=str(RESULTS / "robustness_chunks"))
    parser.add_argument("--samples", default=str(RESULTS / "robustness_samples.csv"))
    parser.add_argument("--targets", default=str(RESULTS / "robustness_targets.csv"))
    parser.add_argument("--coverage", default=str(RESULTS / "post_coverage.csv"))
    parser.add_argument("--readouts", default=str(RESULTS / "readouts.csv"))
    parser.add_argument("--limit-s", type=float, default=config.ROBUSTNESS_WORKER_S)
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
    coverage = post.coverage(specs, readout_rows, sample_rows, target_rows, rows)
    post.write_coverage(args.coverage, coverage)
    print(f"robustness: {samples} samples, {targets} targets, "
          f"{len(coverage)} coverage rows", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        final_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "readout":
        readout_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "robustness":
        robustness_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_main()
    else:
        raise SystemExit("usage: python -m bdss.study {final|readout|robustness|demo}")