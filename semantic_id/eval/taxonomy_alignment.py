"""Design-doc §11.3: taxonomy-prefix alignment, ground-truthed on league/team.

SharePrefix(G, l): chance two random products in the same taxon g share their
length-l code prefix. Reported against a chance baseline (ignoring taxonomy) as
a lift; lift >> 1 confirms the coarse codes absorb taxonomy structure.

Uses long/sparse group_by aggregations throughout (never a dense pivot of
taxonomy-value x prefix) — that cross product can be huge (thousands of teams
x thousands of distinct prefixes) even though the number of *actually observed*
(taxonomy, prefix) pairs is bounded by N rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from semantic_id.schema import level_columns


@dataclass
class PrefixAlignment:
    taxonomy_field: str
    prefix_length: int
    share_prefix: float
    baseline: float
    lift: float
    prefix_purity: float  # optional reverse view: same prefix => same taxon value


def _with_prefix(df: pl.DataFrame, level_cols: list[str]) -> pl.DataFrame:
    return df.with_columns(pl.concat_str(level_cols, separator="_").alias("_prefix"))


def _share_prefix_and_baseline(df: pl.DataFrame, taxonomy_col: str, level_cols: list[str], n: int) -> tuple[float, float]:
    df = _with_prefix(df, level_cols)

    # Counts are squared below (n*(n-1)) — cast to Int64 up front, since a
    # single prefix bucket can hold up to N (~4M) rows and u32 would overflow.
    n_p = df.group_by("_prefix").agg(pl.len().cast(pl.Int64).alias("n_p"))
    baseline = float(((n_p["n_p"] / n) ** 2).sum())

    df_g = df.filter(pl.col(taxonomy_col).is_not_null())
    n_gp = df_g.group_by([taxonomy_col, "_prefix"]).agg(pl.len().cast(pl.Int64).alias("n_gp"))
    per_g = n_gp.group_by(taxonomy_col).agg(
        (pl.col("n_gp") * (pl.col("n_gp") - 1)).sum().alias("numerator"),
        pl.col("n_gp").sum().alias("n_g"),
    )
    per_g = per_g.with_columns(
        pl.when(pl.col("n_g") > 1)
        .then(pl.col("numerator") / (pl.col("n_g") * (pl.col("n_g") - 1)))
        .otherwise(0.0)
        .alias("share_g")
    )
    share_prefix = float((per_g["n_g"] / n * per_g["share_g"]).sum())

    return share_prefix, baseline


def _prefix_purity(df: pl.DataFrame, taxonomy_col: str, level_cols: list[str], n: int) -> float:
    df = _with_prefix(df, level_cols).filter(pl.col(taxonomy_col).is_not_null())
    counts = df.group_by(["_prefix", taxonomy_col]).agg(pl.len().alias("count"))
    modal_count = counts.group_by("_prefix").agg(pl.col("count").max().alias("modal_count"))
    return float(modal_count["modal_count"].sum()) / n


def compute_prefix_alignment(
    df: pl.DataFrame, num_levels: int = 3, taxonomy_fields: list[str] = ("league", "team")
) -> list[PrefixAlignment]:
    """Reports SharePrefix at every prefix length from 1 to num_levels (design-doc
    §11.3 examines l=1,2,3 for its default L=3 config; this generalizes to L).
    """
    n = len(df)
    results = []
    all_level_cols = level_columns(num_levels)
    for field in taxonomy_fields:
        for l in range(1, num_levels + 1):
            level_cols = all_level_cols[:l]
            share_prefix, baseline = _share_prefix_and_baseline(df, field, level_cols, n)
            purity = _prefix_purity(df, field, level_cols, n)
            lift = share_prefix / baseline if baseline > 0 else float("inf")
            results.append(
                PrefixAlignment(
                    taxonomy_field=field,
                    prefix_length=l,
                    share_prefix=share_prefix,
                    baseline=baseline,
                    lift=lift,
                    prefix_purity=purity,
                )
            )
    return results
