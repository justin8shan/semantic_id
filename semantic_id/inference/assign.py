"""Design-doc §8 steps 4-6: batch-assign (c1..cL) to every row via the frozen
encoder+quantizer, run in fixed-size batches to bound memory rather than as
one giant forward pass.

Expects `df` to already have gone through the *same* preprocess pipeline
used in training (drop_invalid_embeddings + l2_normalize) — assign.py does
not re-normalize, so training/inference apply the transform identically by
construction rather than by convention.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch

from semantic_id.schema import CONTENT_EMBEDDING, PRODUCT_ID, level_columns
from semantic_id.train.artifact import load_model


def assign_semantic_ids(df: pl.DataFrame, artifact_path: str, batch_size: int = 200_000) -> pl.DataFrame:
    model, _bundle = load_model(artifact_path)
    level_cols = level_columns(model.config.L)

    embeddings = df[CONTENT_EMBEDDING].list.to_array(model.config.d_in).to_numpy()
    product_ids = df[PRODUCT_ID].to_numpy()
    n = len(df)

    code_arrays = {col: np.empty(n, dtype=np.int64) for col in level_cols}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(embeddings[start:end].astype(np.float32))
            output = model(x, training=False)
            for i, col in enumerate(level_cols):
                code_arrays[col][start:end] = output.code_indices_per_level[i].numpy()

    return pl.DataFrame({PRODUCT_ID: product_ids, **code_arrays})
