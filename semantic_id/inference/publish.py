"""Design-doc §8 step 7: publish product_id -> (c1..cL, dedup) + versions.
Tokens only — the float embedding never leaves this stage."""
from __future__ import annotations

import polars as pl

from semantic_id.io_utils import ensure_local_dir
from semantic_id.schema import output_columns


def publish_assignment(
    df: pl.DataFrame,
    output_path: str,
    semantic_id_version: str,
    embedding_set_version: str,
    num_levels: int,
) -> str:
    out_df = df.with_columns(pl.lit(semantic_id_version).alias("semantic_id_version"))
    if "embedding_set_version" not in out_df.columns:
        out_df = out_df.with_columns(pl.lit(embedding_set_version).alias("embedding_set_version"))
    out_df = out_df.select(output_columns(num_levels))

    path = f"{output_path.rstrip('/')}/semantic_id_version={semantic_id_version}/assignments"
    ensure_local_dir(path)
    out_df.write_parquet(f"{path}/part-0.parquet")
    return path
