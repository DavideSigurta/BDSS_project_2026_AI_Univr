# Modal execution

The final CSVs are already in `results/`; the Modal app reproduces the post-build phases on
the same data and code.

## Image and volumes

`bdss.modal_app` uses the `bdss-m5` app, the repository image and two Volumes:

- `bdss-datasets` mounted at `/data/datasets`;
- `bdss-results` mounted at `/results`.

Set `BDSS_DATASETS=/data/datasets` and `BDSS_BDD_BACKEND=cudd` inside the image, as the app
does. All commands below run from `repo/` with the `modal` extra:

```bash
uv run --extra modal modal run -m bdss.modal_app --help
```

## Smoke test

`--post-smoke` runs one 8 GB readout plus a 30 s robustness slice on `banknote/44`. It checks
the image, the datasets Volume and the chunk return path without writing canonical CSVs.

```bash
uv run --extra modal modal run -m bdss.modal_app --post-smoke
```

## Readouts

One selected build per model, no test set. Resource profile: 8 GB memory, 14,400 s function
timeout, 32 containers.

```bash
uv run --extra modal modal run -m bdss.modal_app --readout \
  --resume --max-usd 30
```

The canonical output is `results/m5_readouts.csv`. The app writes it once, after all 54
selected builds have returned; an interrupted run leaves the previous file intact. Results
are resumed from completed rows.

## Robustness

For every selected build, every test sample and every label: targeted radius, nearest rival,
loss of certification and witness bit → feature → destination cell. Resource profile: 16 GB
memory, 600 s worker slice (1,500 s function timeout), 32 containers.

```bash
uv run --extra modal modal run -m bdss.modal_app --robustness \
  --limit-s 600 --max-usd 60
```

Chunks are written under `results/m5_robustness_chunks/` and resumed by interval. The final
files are generated only after every model is contiguous-complete:

```text
results/m5_robustness_samples.csv
results/m5_robustness_targets.csv
results/m5_post_coverage.csv
```

The canonical guard refuses partial selections: `--limit`, a missing campaign or an
incomplete chunk set cannot overwrite the delivered CSVs.

## Canonical files

The delivered files are the five CSVs listed in the README. Raw campaign intermediates,
robustness chunks and old post-build files are not part of the repository deliverable.
