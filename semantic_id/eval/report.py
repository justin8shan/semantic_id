"""Orchestrates design-doc §11.1-§11.3 into one report, tagged with the run's
(W, L, d) so results are comparable across the hyperparameter_tuning.md sweep.

§11.4 (cold-start placement) and §11.5 (ranker NDCG/Recall lift) are
intentionally not computed here — not needed for this build.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

import polars as pl

from semantic_id.data.hot_buckets import top_k_hot_buckets
from semantic_id.eval.collision import compute_collision_diagnostics
from semantic_id.eval.taxonomy_alignment import compute_prefix_alignment
from semantic_id.eval.utilization import usage_counts_global, utilization_from_counts
from semantic_id.schema import level_columns


def build_report(assignment_df: pl.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """`assignment_df` must carry both the assigned tuple (c1..cL,dedup) and the
    taxonomy columns (league, team) — i.e. the post-assign, pre-publish
    dataframe, or the published table re-joined to the source table on product_id.
    """
    q = config["quantizer"]
    W, num_levels = q["W"], q["L"]
    top_k = config["eval"]["hot_bucket_top_k"]

    report: dict[str, Any] = {
        "run_config": {"W": W, "L": num_levels, "d": q["d"]},
        "utilization": _utilization_section(assignment_df, W, num_levels, top_k),
        "collision": _collision_section(assignment_df, num_levels, top_k),
        "taxonomy_alignment": [
            dataclasses.asdict(r) for r in compute_prefix_alignment(assignment_df, num_levels)
        ],
    }
    return report


def _utilization_section(df: pl.DataFrame, W: int, num_levels: int, top_k: int) -> dict[str, Any]:
    level_cols = level_columns(num_levels)
    global_section = {
        level: dataclasses.asdict(utilization_from_counts(usage_counts_global(df, level, W), W))
        for level in level_cols
    }

    hot_buckets = top_k_hot_buckets(df, top_k).to_dicts()
    per_bucket = []
    for bucket in hot_buckets:
        bucket_df = df.filter((pl.col("league") == bucket["league"]) & (pl.col("team") == bucket["team"]))
        bucket_result = {
            level: dataclasses.asdict(utilization_from_counts(usage_counts_global(bucket_df, level, W), W))
            for level in level_cols
        }
        per_bucket.append({"league": bucket["league"], "team": bucket["team"], "bucket_count": bucket["bucket_count"], **bucket_result})

    return {"global": global_section, "hot_buckets": per_bucket, "level_cols": level_cols}


def _collision_section(df: pl.DataFrame, num_levels: int, top_k: int) -> dict[str, Any]:
    global_result = dataclasses.asdict(compute_collision_diagnostics(df, num_levels))

    hot_buckets = top_k_hot_buckets(df, top_k).to_dicts()
    per_bucket = []
    for bucket in hot_buckets:
        bucket_df = df.filter((pl.col("league") == bucket["league"]) & (pl.col("team") == bucket["team"]))
        result = dataclasses.asdict(compute_collision_diagnostics(bucket_df, num_levels))
        per_bucket.append({"league": bucket["league"], "team": bucket["team"], **result})

    return {"global": global_result, "hot_buckets": per_bucket}


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)


def to_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Semantic ID eval report — W={report['run_config']['W']}, "
             f"L={report['run_config']['L']}, d={report['run_config']['d']}", ""]

    lines.append("## §11.1 Utilization (global)")
    for level, diag in report["utilization"]["global"].items():
        lines.append(
            f"- **{level}**: U={diag['normalized_utilization']:.3f}, "
            f"dead_code_rate={diag['dead_code_rate']:.3f}, M={diag['max_code_share']:.4f}, "
            f"scope_size={diag['scope_size']}, passes={diag['passes']}"
        )

    lines.append("\n## §11.1 Utilization (hot buckets)")
    for bucket in report["utilization"]["hot_buckets"]:
        lines.append(f"- {bucket['league']}/{bucket['team']} (n={bucket['bucket_count']}):")
        for level in report["utilization"]["level_cols"]:
            diag = bucket[level]
            lines.append(
                f"    - {level}: dead_code_rate={diag['dead_code_rate']:.3f}, "
                f"M={diag['max_code_share']:.4f}, passes={diag['passes']}"
            )

    c = report["collision"]["global"]
    lines.append(
        f"\n## §11.2 Collision (global)\n- collision_rate={c['collision_rate']:.4f}, "
        f"dedup_fraction={c['dedup_fraction']:.4f}, worst_case_bucket={c['worst_case_bucket']}, "
        f"p50/p90/p99={c['p50']:.1f}/{c['p90']:.1f}/{c['p99']:.1f}"
    )

    lines.append("\n## §11.3 Taxonomy-prefix alignment")
    for r in report["taxonomy_alignment"]:
        lines.append(
            f"- {r['taxonomy_field']} @ l={r['prefix_length']}: "
            f"lift={r['lift']:.2f} (share={r['share_prefix']:.4f}, baseline={r['baseline']:.4f}), "
            f"purity={r['prefix_purity']:.3f}"
        )

    return "\n".join(lines)
