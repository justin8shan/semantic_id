# Semantic ID

RQ-VAE-based semantic IDs for the sports-merchandise catalog, per
[`doc/semantic_id_design.md`](doc/semantic_id_design.md) and
[`doc/hyper_param_tuning.md`](doc/hyper_param_tuning.md). This repo implements
training, evaluation, and batch inference (design-doc phases 1–2); ranker
integration (phase 3) is out of scope here.

## Environment

Plain Python — no Spark, no JVM. One env needs `torch` + `polars`
(+ `pyyaml` + `fsspec` + `s3fs` for real S3 runs; Polars has its own native
parquet engine, so no pandas/pyarrow needed). See `requirements.txt` /
`requirements-dev.txt` for exact versions verified together.

```bash
pip install -r requirements-dev.txt   # runtime deps + pytest
# or, without test tooling:
pip install -r requirements.txt
```

### Running the tests

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python -m pytest tests/
```

(`pyproject.toml` adds the repo root to `pythonpath` so `semantic_id` imports
resolve without an install step.)

### Local smoke run (synthetic data, no S3 needed)

```bash
PYTHON_BIN=/path/to/your/python scripts/run_local_smoke.sh
```

Generates a synthetic catalog matching the real schema (with a College-style
skew and a slice of `is_hot_market` rows), then runs `train_rqvae.py ->
batch_assign.py -> run_eval.py` exactly as below, and prints the §11 report.
This is the fastest way to confirm the pipeline runs end to end in a new
environment.

## Config

Everything lives in `configs/default.yaml` (paths, `W`/`L`/`d`, training
hyperparameters, versioning). Every CLI accepts `--set key.path=value` to
override without editing the file — this is how the `hyperparameter_tuning.md`
sweep is driven (e.g. `--set quantizer.W=1024 --set quantizer.L=4`).

Key things to set before a real run:
- `data.input_path` — the source parquet path (read as-is, no version filtering
  — see design-doc §8 note in `configs/default.yaml`)
- `data.output_path` — where the artifact and published assignment table land
- `versioning.semantic_id_version` — tags this quantizer's artifact and output

## Running against the real S3 path

These are the exact CLIs to run in an environment with AWS access (this dev
sandbox doesn't have it — see the note in the original design conversation):

```bash
python semantic_id/cli/train_rqvae.py --config configs/default.yaml

python semantic_id/cli/batch_assign.py --config configs/default.yaml

python semantic_id/cli/run_eval.py --config configs/default.yaml \
    --report-path s3://fanatics.prod.internal.confidential/xshan/semantic_id/output/reports/v1
```

`train_rqvae.py` trains the RQ-VAE and writes the frozen artifact to
`<output_path>/semantic_id_version=<v>/artifact/model.pt`. `batch_assign.py`
loads that artifact, assigns `(c1..cL, dedup)` to every row, and publishes
tokens-only to `<output_path>/semantic_id_version=<v>/assignments/`.
`run_eval.py` joins that published table back to the source table's taxonomy
columns and computes the §11.1-§11.3 report (utilization, collision,
taxonomy-prefix alignment).

## Hyperparameter sweep

Follow `hyper_param_tuning.md` §4: grid `{256,512,1024} x {3,4}` at `d=32`,
then one `d=64` run on the winner. Each run is just:

```bash
python semantic_id/cli/train_rqvae.py --config configs/default.yaml \
    --set quantizer.W=1024 --set quantizer.L=4 \
    --set versioning.semantic_id_version=sweep-W1024-L4

python semantic_id/cli/batch_assign.py --config configs/default.yaml \
    --set quantizer.W=1024 --set quantizer.L=4 \
    --set versioning.semantic_id_version=sweep-W1024-L4

python semantic_id/cli/run_eval.py --config configs/default.yaml \
    --set quantizer.W=1024 --set quantizer.L=4 \
    --set versioning.semantic_id_version=sweep-W1024-L4 \
    --report-path /tmp/reports/sweep-W1024-L4
```

Each `report.json`'s `run_config` records the `(W, L, d)` it was produced
under, so results are directly comparable across the sweep. Pick the smallest
configuration whose hot-bucket utilization and collision diagnostics clear
the §11 thresholds (`passes: true`) — see the "Selection rule" in
`hyperparameter_tuning.md`.

## What's not built here

- §11.4 (cold-start placement) — deliberately dropped; the ID construction is
  content-only and identical for every product regardless of launch age, so
  this was a slice-level check, not a load-bearing one for this build.
- §11.5 (ranker NDCG/Recall lift) — needs a ranker training run, out of scope.
- Ranker feature integration (design-doc §12, phase 3).
- The full dual-write/migration runbook for retraining (design-doc §9) — only
  a `training.warm_start_from` hook exists to warm-start codebooks from a
  prior artifact.
