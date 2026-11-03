# Modal execution

Local and Modal are two execution surfaces for the same pipeline. Local:
`python -m bdss.study final|readout|robustness|demo`. Modal: `--campaign` for grid,
and build completion (prefix search included), then `--readout`/`--robustness` for the post-build phases.
The final CSVs are already in `results/`.

## Image and volumes

`bdss.modal_app` uses the `bdss` app, the repository image and two Volumes:

- `bdss-datasets` mounted at `/data/datasets`;
- `bdss-results` mounted at `/results`.

Set `BDSS_DATASETS=/data/datasets` and `BDSS_BDD_BACKEND=cudd` inside the image, as the app
does. All commands below run from `repo/` with the `modal` extra:

```bash
uv run --extra modal modal run -m bdss.modal_app --help
```

## Campaign

`--campaign` is the Modal counterpart of local `final`: it fills the grid, runs build
completion for the models whose full forest does not close, and writes
`results/campaign.csv`. It resumes an existing file and never writes with `--limit`.

```bash
uv run --extra modal modal run -m bdss.modal_app --campaign
```

## Demo

`--demo` runs one 8 GB readout plus a 30 s robustness slice on `banknote/44`. It checks the
image, the datasets Volume and the chunk return path without writing canonical CSVs.

```bash
uv run --extra modal modal run -m bdss.modal_app --demo
```

## Readouts

One selected build per model, no test set. Resource profile: 8 GB memory, 14,400 s function
timeout, 32 containers.

```bash
uv run --extra modal modal run -m bdss.modal_app --readout \
  --resume --max-usd 30
```

The canonical output is `results/readouts.csv`. Each completed row is written immediately;
an interrupted run leaves the previous rows and can be resumed from completed rows.

## Robustness

For every selected build, every test sample and every label: targeted radius, nearest rival,
loss of certification and witness bit → feature → destination cell. Resource profile: 16 GB
memory, 600 s worker slice (1,500 s function timeout), 32 containers.

```bash
uv run --extra modal modal run -m bdss.modal_app --robustness \
  --limit-s 600 --max-usd 60
```

Chunks are written under `results/robustness_chunks/` and resumed by interval. The final
files are generated only after every model is contiguous-complete:

```text
results/robustness_samples.csv
results/robustness_targets.csv
results/post_coverage.csv
```

The canonical guard refuses partial selections: `--limit`, a missing campaign or an
incomplete chunk set cannot overwrite the delivered CSVs.

## Canonical files

The delivered files are the five CSVs listed in the README. Raw campaign intermediates,
robustness chunks and old post-build files are not part of the repository deliverable.
