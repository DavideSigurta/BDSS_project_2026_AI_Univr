# BDSS Project 2026 — AI Univr

Post-hoc logical analysis of 54 pre-trained random forests, for the first half of the BDSS
course. The project does not train or validate a model: it reads the trees, concentrates each
feature into cells, encodes the cell index in binary, composes the per-tree vote counters as
OBDDs, and answers exact queries (certified region size, bit rankings, budgeted decomposition,
robustness radius).

The numbers live in `results/` (CSV and figures). The PDF report is a separate presentation
and is not part of this repository.

## Setup

Python 3.12 with `uv`:

```bash
uv sync
```

The `.pkl` bundles were saved with scikit-learn 1.4.2; the pinned dependencies are in
`pyproject.toml`.

### Fast BDD backend (CUDD)

`dd==0.6.0` ships CUDD only in its Linux wheels. On macOS the plain `uv sync` installs the
pure-Python `dd.autoref`; to build the CUDD extension as well:

```bash
DD_CUDD=1 DD_FETCH=1 uv sync --no-binary-package dd --no-cache --reinstall-package dd
```

`bdss.bdd` selects `dd.cudd` when it imports and falls back to `dd.autoref` otherwise, so the
pipeline runs either way.

## Datasets

The 54 datasets are expected at `../datasets/<name>/<name>_{rf.pkl,train.csv,test.csv}`.
Override the location with `BDSS_DATASETS=/path/to/datasets`.

## Smoke test

```bash
uv run pytest            # fast tests
uv run pytest -m slow    # real .pkl, hard vote, dd.cudd availability
```

## Status

M1.1: repository, environment, and the real-forest smoke test.
