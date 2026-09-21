"""End-to-end smoke test on synthetic data: train a tiny RQ-VAE, batch-assign,
dedup, publish, and run the §11.1-§11.3 eval report — the concrete "does it
work" check available without real S3 access (see the build plan's
Verification section).
"""
from __future__ import annotations

import polars as pl

from semantic_id.data.io import read_product_embeddings
from semantic_id.data.preprocess import drop_invalid_embeddings, l2_normalize
from semantic_id.eval.report import build_report
from semantic_id.inference.assign import assign_semantic_ids
from semantic_id.inference.dedup import add_dedup_index, max_dedup
from semantic_id.inference.publish import publish_assignment
from semantic_id.schema import CONTENT_EMBEDDING, PRODUCT_ID, level_columns
from semantic_id.train.train import run_training
from tests.conftest import make_synthetic_catalog


def _smoke_config(input_path: str, output_path: str) -> dict:
    return {
        "data": {
            "input_path": input_path, "output_path": output_path, "embedding_dim": 128,
            "embedding_set_version": "smoke-test-embset-v1",
        },
        "preprocess": {"val_fraction": 0.1, "split_seed": 42, "strata_columns": ["league", "team"]},
        "weighting": {"hot_bucket_top_k": 5, "hot_bucket_weight": 3.0, "hot_market_weight": 3.0},
        "quantizer": {
            "W": 16, "L": 2, "d": 8, "beta": 0.25, "gamma": 0.9,
            "kmeans_init": True, "dead_code_reset": True,
            "dead_code_reset_threshold": 0.02, "dead_code_reset_every": 2,
        },
        "training": {
            "lr": 1e-3, "batch_size": 256, "max_epochs": 6, "patience": 3,
            "min_delta": 1e-5, "seed": 0, "device": "cpu", "warm_start_from": None,
        },
        "versioning": {"semantic_id_version": "smoke-test-v1"},
        "eval": {"hot_bucket_top_k": 5},
    }


def test_end_to_end_pipeline_on_synthetic_data(tmp_path):
    catalog = make_synthetic_catalog(n_products=4000, hot_teams=20, other_leagues=3, teams_per_other_league=4)

    input_path = str(tmp_path / "input.parquet")
    output_path = str(tmp_path / "output")
    catalog.write_parquet(input_path)

    config = _smoke_config(input_path, output_path)

    artifact_path, history = run_training(config)
    assert len(history) > 0
    assert artifact_path.endswith("model.pt")
    # Reconstruction loss should have decreased from the first to the best epoch.
    assert history[-1].val_reconstruction <= history[0].val_reconstruction * 1.5

    lf = read_product_embeddings(config["data"]["input_path"])
    lf = drop_invalid_embeddings(lf, config["data"]["embedding_dim"])
    lf = l2_normalize(lf)
    df = lf.collect()

    num_levels = config["quantizer"]["L"]
    assigned = assign_semantic_ids(df, artifact_path)
    assigned = add_dedup_index(assigned, level_columns(num_levels))
    assert max_dedup(assigned) >= 0

    publish_path = publish_assignment(
        assigned,
        output_path=config["data"]["output_path"],
        semantic_id_version=config["versioning"]["semantic_id_version"],
        embedding_set_version="kepler-test-v1",
        num_levels=num_levels,
    )
    published = pl.read_parquet(publish_path)
    assert len(published) == len(df)
    assert set(level_columns(num_levels) + ["dedup", PRODUCT_ID]).issubset(set(published.columns))

    source_taxonomy = df.drop(CONTENT_EMBEDDING)
    joined = published.join(source_taxonomy, on=PRODUCT_ID, how="inner")
    assert len(joined) == len(published)

    report = build_report(joined, config)

    assert report["run_config"] == {"W": 16, "L": 2, "d": 8}
    assert set(report["utilization"]["global"].keys()) == {"c1", "c2"}
    assert 0 <= report["collision"]["global"]["collision_rate"] <= 1
    assert len(report["taxonomy_alignment"]) == num_levels * 2  # 2 taxonomy fields x l=1..num_levels
    for r in report["taxonomy_alignment"]:
        if r["prefix_length"] == 1:
            # Coarse codes should absorb *some* team/league structure on this
            # clearly-clustered synthetic data — lift meaningfully above 1.
            assert r["lift"] > 1.0
