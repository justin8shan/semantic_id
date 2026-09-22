"""Join a published (tokens-only) semantic-ID assignment table
(`data/output/semantic_id_version=<v>/assignments/`) against the source
catalog's metadata columns, producing a denormalized mapping parquet in the
same shape as `data/output/product_id_semantic_id_full_mapping.parquet` —
but for *any* semantic_id_version, so a sweep config can be pointed at from
the Inspector app (`SEMANTIC_ID_INSPECTOR_MAPPING`) without re-running the pipeline.

Usage:
    python inspector/build_full_mapping.py --semantic-id-version local-v1-W512-L5
    python inspector/build_full_mapping.py \
        --assignments-path data/output/semantic_id_version=local-v1-W512-L5/assignments/ \
        --output-path data/output/product_id_semantic_id_full_mapping__local-v1-W512-L5.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from inspector.config import InspectorPaths, REPO_ROOT
from semantic_id.schema import PRODUCT_ID

# Metadata columns carried on the source table but not part of the pipeline's
# own INPUT_COLUMNS projection (semantic_id/schema.py deliberately excludes
# them — see its comment) — this is exactly the set the existing
# full_mapping.parquet denormalizes for inspection/debugging purposes.
METADATA_COLUMNS = [
    "league", "team", "brand", "color", "product_name", "price_band",
    "is_hot_market", "launch_age_bucket",
]

_LEVEL_COL_RE = re.compile(r"^c(\d+)$")


def _level_cols(columns: list[str]) -> list[str]:
    levels = sorted(int(m.group(1)) for c in columns if (m := _LEVEL_COL_RE.fullmatch(c)))
    return [f"c{i}" for i in levels]


def build_full_mapping(assignments_path: Path, embedding_source: Path, output_path: Path) -> int:
    assignments_lf = pl.scan_parquet(assignments_path)
    assignment_cols = assignments_lf.collect_schema().names()
    level_cols = _level_cols(assignment_cols)
    if not level_cols:
        raise ValueError(f"No level columns (c1, c2, ...) found in {assignments_path}")

    codes_lf = assignments_lf.select([PRODUCT_ID, *level_cols, "dedup"])
    metadata_lf = pl.scan_parquet(embedding_source).select([PRODUCT_ID, *METADATA_COLUMNS])

    joined = codes_lf.join(metadata_lf, on=PRODUCT_ID, how="left")
    semantic_id = pl.concat_str(
        [pl.col(c).cast(pl.Utf8) for c in [*level_cols, "dedup"]], separator="-"
    )
    joined = joined.with_columns(semantic_id.alias("semantic_id")).select(
        [PRODUCT_ID, "semantic_id", *level_cols, "dedup", *METADATA_COLUMNS]
    )

    df = joined.collect()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df.height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-id-version", default=None, help="e.g. local-v1-W512-L5")
    parser.add_argument("--assignments-path", default=None, help="Override the assignments dir directly")
    parser.add_argument("--embedding-source", default=None, help="Override the source catalog parquet dir/file")
    parser.add_argument("--output-path", default=None, help="Override the output parquet path")
    args = parser.parse_args()

    if not args.assignments_path and not args.semantic_id_version:
        parser.error("Pass either --semantic-id-version or --assignments-path")

    paths = InspectorPaths.default(embedding_source=args.embedding_source)
    if args.assignments_path:
        assignments_path = Path(args.assignments_path)
        if not assignments_path.is_absolute():
            assignments_path = REPO_ROOT / assignments_path
    else:
        assignments_path = (
            paths.mapping_path.parent
            / f"semantic_id_version={args.semantic_id_version}"
            / "assignments"
        )

    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
    else:
        version_tag = args.semantic_id_version or assignments_path.parent.name.split("=", 1)[-1]
        output_path = paths.mapping_path.parent / f"product_id_semantic_id_full_mapping__{version_tag}.parquet"

    n = build_full_mapping(assignments_path, paths.embedding_source, output_path)
    print(f"Wrote {n} rows to {output_path}")


if __name__ == "__main__":
    main()
