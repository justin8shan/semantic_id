#!/usr/bin/env bash
# End-to-end local smoke run: generates a synthetic parquet catalog matching
# the real schema, then runs train_rqvae -> batch_assign -> run_eval as plain
# python scripts (no Spark) against it.
#
# Needs one python env with `torch` + `polars` (+ pyyaml/fsspec) installed.
# Set PYTHON_BIN to override the default.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${PYTHON_BIN:=/Users/xshan/miniconda3/envs/torchrec/bin/python}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

WORKDIR="$(mktemp -d)"
INPUT_PATH="${WORKDIR}/input.parquet"
OUTPUT_PATH="${WORKDIR}/output"
CONFIG_PATH="${WORKDIR}/smoke_config.yaml"

echo "Generating synthetic catalog at ${INPUT_PATH} ..."
"${PYTHON_BIN}" - <<PYEOF
import sys
sys.path.insert(0, "tests")
from conftest import make_synthetic_catalog

catalog = make_synthetic_catalog(n_products=20000, hot_teams=30, other_leagues=4, teams_per_other_league=6)
catalog.write_parquet("${INPUT_PATH}")
print("Wrote", len(catalog), "rows")
PYEOF

cat > "${CONFIG_PATH}" <<YAMLEOF
data:
  input_path: "${INPUT_PATH}"
  output_path: "${OUTPUT_PATH}"
  embedding_dim: 128
  embedding_set_version: "smoke-local-embset-v1"
preprocess:
  val_fraction: 0.10
  split_seed: 42
  strata_columns: ["league", "team"]
weighting:
  hot_bucket_top_k: 5
  hot_bucket_weight: 3.0
  hot_market_weight: 3.0
quantizer:
  W: 64
  L: 3
  d: 16
  beta: 0.25
  gamma: 0.95
  kmeans_init: true
  dead_code_reset: true
  dead_code_reset_threshold: 0.02
  dead_code_reset_every: 2
training:
  lr: 1.0e-3
  batch_size: 512
  max_epochs: 15
  patience: 4
  min_delta: 1.0e-5
  seed: 42
  device: "cpu"
  warm_start_from: null
versioning:
  semantic_id_version: "smoke-local"
eval:
  hot_bucket_top_k: 5
YAMLEOF

echo "=== train_rqvae ==="
"${PYTHON_BIN}" semantic_id/cli/train_rqvae.py --config "${CONFIG_PATH}"

echo "=== batch_assign ==="
"${PYTHON_BIN}" semantic_id/cli/batch_assign.py --config "${CONFIG_PATH}"

echo "=== run_eval ==="
"${PYTHON_BIN}" semantic_id/cli/run_eval.py --config "${CONFIG_PATH}" --report-path "${WORKDIR}/report"

echo
echo "Smoke run artifacts under: ${WORKDIR}"
