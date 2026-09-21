"""Plain-python entrypoint: batch-assign (c1..cL,dedup) to every row and publish.

Usage:
    python semantic_id/cli/batch_assign.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse

from semantic_id.config import load_config
from semantic_id.data.io import read_product_embeddings
from semantic_id.data.preprocess import drop_invalid_embeddings, l2_normalize
from semantic_id.inference.assign import assign_semantic_ids
from semantic_id.inference.dedup import add_dedup_index, max_dedup
from semantic_id.inference.publish import publish_assignment
from semantic_id.schema import level_columns
from semantic_id.train.artifact import artifact_dir, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--artifact-path", default=None, help="Override the resolved artifact path")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    data_cfg, v_cfg = config["data"], config["versioning"]

    artifact_path = args.artifact_path
    if artifact_path is None:
        base_dir = artifact_dir(data_cfg["output_path"], v_cfg["semantic_id_version"])
        artifact_path = f"{base_dir}/model.pt"

    model, bundle = load_model(artifact_path)
    num_levels = model.config.L

    lf = read_product_embeddings(data_cfg["input_path"])
    lf = drop_invalid_embeddings(lf, data_cfg["embedding_dim"])
    lf = l2_normalize(lf)
    df = lf.collect()

    assigned = assign_semantic_ids(df, artifact_path)
    assigned = add_dedup_index(assigned, level_columns(num_levels))

    worst_bucket = max_dedup(assigned) + 1
    print(f"Worst-case bucket size (n_dedup needed): {worst_bucket}")

    path = publish_assignment(
        assigned,
        output_path=data_cfg["output_path"],
        semantic_id_version=v_cfg["semantic_id_version"],
        embedding_set_version=bundle["embedding_set_version"],
        num_levels=num_levels,
    )
    print(f"Published assignment table: {path}")


if __name__ == "__main__":
    main()
