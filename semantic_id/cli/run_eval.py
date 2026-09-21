"""Plain-python entrypoint: compute the design-doc §11.1-§11.3 intrinsic eval report.

Joins the published (c1,c2,c3,dedup) assignment table back to the source
table's taxonomy columns on product_id, then computes utilization, collision,
and taxonomy-prefix alignment diagnostics.

Usage:
    python semantic_id/cli/run_eval.py --config configs/default.yaml \
        --report-path /path/to/report
"""
from __future__ import annotations

import argparse

import polars as pl

from semantic_id.config import load_config
from semantic_id.data.io import read_product_embeddings
from semantic_id.eval.report import build_report, to_json, to_markdown
from semantic_id.io_utils import write_text
from semantic_id.schema import CONTENT_EMBEDDING, PRODUCT_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--report-path", default=None, help="Local or s3 path to write report.json/.md")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    data_cfg, v_cfg = config["data"], config["versioning"]

    assignments_path = (
        f"{data_cfg['output_path'].rstrip('/')}/semantic_id_version={v_cfg['semantic_id_version']}/assignments"
    )
    assignments = pl.read_parquet(assignments_path)
    source = read_product_embeddings(data_cfg["input_path"]).drop(CONTENT_EMBEDDING).collect()

    joined = assignments.join(source, on=PRODUCT_ID, how="inner")

    report = build_report(joined, config)
    markdown = to_markdown(report)
    print(markdown)

    if args.report_path:
        write_text(f"{args.report_path}.json", to_json(report))
        write_text(f"{args.report_path}.md", markdown)
        print(f"Wrote report to {args.report_path}.json / .md")


if __name__ == "__main__":
    main()
