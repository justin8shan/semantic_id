"""Design-doc §8 steps 5-6: deterministic collision-breaker.

Colliding items are ordered by ascending product_id — never row/run order —
so a product keeps its dedup slot across re-runs of the same assignment.
"""
from __future__ import annotations

import polars as pl

from semantic_id.schema import PRODUCT_ID


def add_dedup_index(df: pl.DataFrame, level_cols: list[str]) -> pl.DataFrame:
    # Sort by product_id first: cum_count() numbers rows in the order they
    # appear in the frame, so a global ascending-product_id sort makes each
    # group's cum_count equivalent to row_number() ordered by product_id.
    return df.sort(PRODUCT_ID).with_columns(
        (pl.col(PRODUCT_ID).cum_count().over(level_cols) - 1).alias("dedup")
    )


def max_dedup(df: pl.DataFrame) -> int:
    return int(df["dedup"].max()) if len(df) else 0
