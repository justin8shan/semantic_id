"""Design-doc §11.2: collision rate, dedup fraction, worst-case bucket, percentiles.

Computed on the pre-dedup tuples t(i) = (c1, c2, c3); "collisions are broken by
the dedup token regardless" so this gauges ID resolution quality, not correctness.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from semantic_id.schema import level_columns


@dataclass
class CollisionDiagnostics:
    n: int
    collision_rate: float
    dedup_fraction: float
    worst_case_bucket: int
    p50: float
    p90: float
    p99: float


def compute_collision_diagnostics(df: pl.DataFrame, num_levels: int = 3) -> CollisionDiagnostics:
    level_cols = level_columns(num_levels)
    n = len(df)
    bucket_counts = df.group_by(level_cols).agg(pl.len().alias("count"))

    # Aggregate once per distinct bucket (not per row) — collided_rows/dedup_rows
    # are sums over buckets t, not over items i (design-doc §11.2's "Σ_t").
    collided = bucket_counts.filter(pl.col("count") > 1)["count"]
    collided_rows = int(collided.sum())
    dedup_rows = int((collided - 1).sum())
    worst_case_bucket = int(bucket_counts["count"].max()) if len(bucket_counts) else 0

    # Percentiles, by contrast, are over the per-item distribution of
    # bucket[t(i)] — so join each row back to its own bucket's size here.
    per_row = df.join(bucket_counts, on=level_cols, how="left")
    p50, p90, p99 = per_row.select(
        pl.col("count").quantile(0.5).alias("p50"),
        pl.col("count").quantile(0.9).alias("p90"),
        pl.col("count").quantile(0.99).alias("p99"),
    ).row(0)

    return CollisionDiagnostics(
        n=n,
        collision_rate=collided_rows / n,
        dedup_fraction=dedup_rows / n,
        worst_case_bucket=worst_case_bucket,
        p50=float(p50),
        p90=float(p90),
        p99=float(p99),
    )
