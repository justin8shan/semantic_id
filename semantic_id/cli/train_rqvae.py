"""Plain-python entrypoint: train the RQ-VAE quantizer and publish the artifact.

Usage:
    python semantic_id/cli/train_rqvae.py --config configs/default.yaml \
        [--set quantizer.W=1024 --set quantizer.L=4 ...]
"""
from __future__ import annotations

import argparse

from semantic_id.config import load_config
from semantic_id.train.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    artifact_path, history = run_training(config)

    print(f"Trained {len(history)} epochs. Artifact: {artifact_path}")
    last = history[-1]
    print(
        f"Final: val_reconstruction={last.val_reconstruction:.6f} "
        f"val_dead_code_rate={last.val_dead_code_rate:.3f} "
        f"val_normalized_utilization={last.val_normalized_utilization:.3f}"
    )


if __name__ == "__main__":
    main()
