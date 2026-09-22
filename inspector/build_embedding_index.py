"""One-time builder: memmap index of (product_id, content_embedding) for fast
cosine-similarity lookups in the Inspector app.

Applies the *same* validity filter + L2 normalization as
`semantic_id/cli/batch_assign.py` (drop_invalid_embeddings + l2_normalize), so
Inspector cosines are computed on exactly the vectors the quantizer saw.

Usage:
    python inspector/build_embedding_index.py
    python inspector/build_embedding_index.py --embedding-source data/input/product_emb/ \
        --index-dir data/inspector_index/ --force
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from inspector.config import InspectorPaths
from semantic_id.data.preprocess import drop_invalid_embeddings, l2_normalize
from semantic_id.schema import CONTENT_EMBEDDING, PRODUCT_ID


def _source_files(embedding_source: Path) -> list[Path]:
    if embedding_source.is_file():
        return [embedding_source]
    files = sorted(embedding_source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {embedding_source}")
    return files


def _valid_row_count(path: Path, embedding_dim: int) -> int:
    lf = pl.scan_parquet(path).select([PRODUCT_ID, CONTENT_EMBEDDING])
    lf = drop_invalid_embeddings(lf, embedding_dim)
    return int(lf.select(pl.len()).collect().item())


def build_index(
    embedding_source: Path,
    index_dir: Path,
    embedding_dim: int = 128,
    force: bool = False,
) -> dict:
    files = _source_files(embedding_source)
    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = index_dir / "embeddings.npy"
    product_ids_path = index_dir / "product_ids.npy"
    meta_path = index_dir / "meta.json"

    if not force and (embeddings_path.exists() or product_ids_path.exists() or meta_path.exists()):
        raise FileExistsError(
            f"Index already exists at {index_dir} — pass --force to overwrite."
        )

    print(f"Counting valid rows across {len(files)} file(s)...")
    counts = []
    for f in files:
        n = _valid_row_count(f, embedding_dim)
        counts.append(n)
        print(f"  {f.name}: {n} valid rows")
    total = sum(counts)
    print(f"Total valid rows: {total}")

    embeddings = np.lib.format.open_memmap(
        embeddings_path, mode="w+", dtype=np.float32, shape=(total, embedding_dim)
    )
    product_ids = np.empty(total, dtype=np.int64)

    dropped_invalid = 0
    offset = 0
    for f, expected_n in zip(files, counts):
        raw_lf = pl.scan_parquet(f).select([PRODUCT_ID, CONTENT_EMBEDDING])
        raw_n = int(raw_lf.select(pl.len()).collect().item())

        lf = drop_invalid_embeddings(raw_lf, embedding_dim)
        lf = l2_normalize(lf)
        df = lf.collect()

        n = len(df)
        dropped_invalid += raw_n - n
        assert n == expected_n, f"row count mismatch for {f.name}: {n} != {expected_n}"

        product_ids[offset : offset + n] = df[PRODUCT_ID].to_numpy()
        embeddings[offset : offset + n] = df[CONTENT_EMBEDDING].list.to_array(embedding_dim).to_numpy()
        offset += n
        print(f"  wrote {f.name}: rows [{offset - n}, {offset})")

    embeddings.flush()

    source_files_meta = [
        {"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime} for f in files
    ]
    meta = {
        "dim": embedding_dim,
        "n_rows": total,
        "normalized": True,
        "embedding_source": str(embedding_source),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_files": source_files_meta,
        "dropped_invalid": dropped_invalid,
    }
    np.save(product_ids_path, product_ids)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Built index: {total} rows, {dropped_invalid} dropped as invalid.")
    print(f"  {embeddings_path}")
    print(f"  {product_ids_path}")
    print(f"  {meta_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-source", default=None, help="Override the source parquet dir/file")
    parser.add_argument("--index-dir", default=None, help="Override the index output dir")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing index")
    args = parser.parse_args()

    paths = InspectorPaths.default(embedding_source=args.embedding_source, index_dir=args.index_dir)
    build_index(
        embedding_source=paths.embedding_source,
        index_dir=paths.index_dir,
        embedding_dim=args.embedding_dim,
        force=args.force,
    )


if __name__ == "__main__":
    main()
