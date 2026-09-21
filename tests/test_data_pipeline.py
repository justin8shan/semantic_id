"""Unit tests for the Polars data layer — data/io.py, preprocess.py,
weights.py, hot_buckets.py. None of these need torch, unlike the rest of the
pipeline, so they're kept in their own file to run in a torch-free env.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from semantic_id.data.hot_buckets import flag_hot_bucket_membership, top_k_hot_buckets
from semantic_id.data.io import read_product_embeddings
from semantic_id.data.preprocess import drop_invalid_embeddings, l2_normalize, stratified_train_val_split
from semantic_id.data.weights import add_sample_weight
from semantic_id.schema import CONTENT_EMBEDDING


def _synthetic_rows(n=20, dim=4):
    rows = []
    for i in range(n):
        rows.append(
            {
                "product_id": i,
                "content_embedding": [float(i), 1.0, 1.0, 1.0][:dim],
                "league": "A" if i < n // 2 else "B",
                "team": f"team_{i % 4}",
                "is_hot_market": "true" if i % 5 == 0 else "false",
            }
        )
    return pl.DataFrame(rows)


def test_read_product_embeddings_projects_expected_columns(tmp_path):
    df = _synthetic_rows()
    # Columns the source table has but the pipeline shouldn't select.
    df = df.with_columns(pl.lit("some brand").alias("brand"), pl.lit("some product").alias("product_name"))
    path = str(tmp_path / "catalog.parquet")
    df.write_parquet(path)

    result = read_product_embeddings(path).collect()

    assert "brand" not in result.columns and "product_name" not in result.columns
    assert set(result.columns) == {"product_id", "content_embedding", "league", "team", "is_hot_market"}
    assert len(result) == len(df)


def test_drop_invalid_embeddings_filters_null_and_wrong_length():
    df = pl.DataFrame(
        {
            "product_id": [0, 1, 2, 3, 4],
            "content_embedding": [
                [1.0, 1.0, 1.0, 1.0],
                None,
                [1.0, 2.0],  # wrong length
                [1.0, 1.0, 1.0, 1.0],
                [1.0, None, 1.0, 1.0],  # null element
            ],
        }
    )

    result = drop_invalid_embeddings(df.lazy(), embedding_dim=4).collect()

    assert set(result["product_id"].to_list()) == {0, 3}


def test_l2_normalize_produces_unit_vectors():
    df = pl.DataFrame(
        {
            "product_id": [0, 1],
            CONTENT_EMBEDDING: [[3.0, 4.0], [0.0, 0.0]],
        }
    )
    result = l2_normalize(df.lazy()).collect()

    vec0 = np.array(result[CONTENT_EMBEDDING][0])
    assert math.isclose(np.linalg.norm(vec0), 1.0, rel_tol=1e-6)
    # Zero vector stays zero (guarded div-by-zero), doesn't become NaN.
    assert np.allclose(result[CONTENT_EMBEDDING][1], [0.0, 0.0])


def test_stratified_split_every_group_appears_in_both_and_ratio_holds():
    rows = [{"product_id": i, "league": "A" if i < 100 else "B", "team": "t1"} for i in range(200)]
    df = pl.DataFrame(rows)

    train_df, val_df = stratified_train_val_split(df, val_fraction=0.1, seed=42, strata_columns=["league", "team"])

    assert len(train_df) + len(val_df) == len(df)
    assert set(train_df["product_id"].to_list()).isdisjoint(set(val_df["product_id"].to_list()))
    for league in ("A", "B"):
        val_count = val_df.filter(pl.col("league") == league).height
        assert val_count == 10  # 10% of each 100-row group, exactly


def test_stratified_split_is_deterministic():
    df = pl.DataFrame({"product_id": list(range(50)), "league": ["A"] * 25 + ["B"] * 25, "team": ["t"] * 50})
    train1, val1 = stratified_train_val_split(df, 0.2, seed=7, strata_columns=["league", "team"])
    train2, val2 = stratified_train_val_split(df, 0.2, seed=7, strata_columns=["league", "team"])

    assert sorted(val1["product_id"].to_list()) == sorted(val2["product_id"].to_list())


def test_top_k_hot_buckets_and_flag_membership():
    df = pl.DataFrame(
        {
            "product_id": list(range(10)),
            "league": ["A"] * 6 + ["B"] * 4,
            "team": ["t1"] * 6 + ["t2"] * 4,
        }
    )
    hot = top_k_hot_buckets(df, top_k=1)
    assert len(hot) == 1
    assert hot["league"][0] == "A" and hot["bucket_count"][0] == 6

    flagged = flag_hot_bucket_membership(df, hot)
    assert flagged.filter(pl.col("league") == "A")["is_hot_bucket"].all()
    assert not flagged.filter(pl.col("league") == "B")["is_hot_bucket"].any()


def test_add_sample_weight_combines_multiplicatively():
    df = pl.DataFrame(
        {
            "product_id": list(range(10)),
            "league": ["A"] * 6 + ["B"] * 4,
            "team": ["t1"] * 6 + ["t2"] * 4,
            "is_hot_market": ["true", "false"] * 5,
        }
    )
    weighted = add_sample_weight(df, hot_bucket_top_k=1, hot_bucket_weight=3.0, hot_market_weight=2.0)

    # league A + is_hot_market -> both multipliers apply
    row = weighted.filter((pl.col("league") == "A") & (pl.col("is_hot_market") == "true")).row(0, named=True)
    assert math.isclose(row["sample_weight"], 6.0)

    # league B + not hot_market -> neither multiplier applies
    row = weighted.filter((pl.col("league") == "B") & (pl.col("is_hot_market") == "false")).row(0, named=True)
    assert math.isclose(row["sample_weight"], 1.0)
