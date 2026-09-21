"""Column-name constants for the product_emb table and a helper for its
string-typed boolean-ish flags (is_hot_market etc. are ``string``, not
``boolean``, in the real schema — a naive ``.astype(bool)`` on non-empty
strings like "false" would silently give the wrong answer, so parse
explicitly).
"""
from __future__ import annotations

import polars as pl

PRODUCT_ID = "product_id"
CONTENT_EMBEDDING = "content_embedding"

LEAGUE = "league"
TEAM = "team"
IS_HOT_MARKET = "is_hot_market"
EMBEDDING_SET_VERSION = "embedding_set_version"  # not a source column — see configs/default.yaml's data.embedding_set_version

# Columns projected out of the source table: quantization input + everything
# needed for training-time hot-bucket weighting (league/team/is_hot_market).
# The source table also has brand, color, product_name, price_band, and
# launch_age_bucket — not selected because nothing in this pipeline uses them.
INPUT_COLUMNS = [
    PRODUCT_ID,
    CONTENT_EMBEDDING,
    LEAGUE,
    TEAM,
    IS_HOT_MARKET,
]


def level_columns(num_levels: int) -> list[str]:
    return [f"c{i + 1}" for i in range(num_levels)]


def output_columns(num_levels: int) -> list[str]:
    """Semantic-ID output columns published to the feature store / S3 — tokens
    only, never the float embedding (design-doc §8 step 7)."""
    return [PRODUCT_ID, *level_columns(num_levels), "dedup", EMBEDDING_SET_VERSION, "semantic_id_version"]


def parse_bool_string(column: str) -> pl.Expr:
    """Parse a string-typed boolean-ish column ("true"/"false", any case) to
    a nullable boolean. Unrecognized/missing values map to null (not False),
    so callers can distinguish "known false" from "missing" if they need to.
    """
    lowered = pl.col(column).str.strip_chars().str.to_lowercase()
    return (
        pl.when(lowered == "true").then(True)
        .when(lowered == "false").then(False)
        .otherwise(None)
    )
