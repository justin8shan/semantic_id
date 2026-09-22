"""Backing stores for the Inspector app: the semantic-ID/metadata mapping (polars) and
the content-embedding memmap index (numpy).

Both are read-only and built once per process (see `inspector/app.py`'s
`@st.cache_resource` usage).
"""
from __future__ import annotations

import json
import re

import numpy as np
import polars as pl

from inspector.config import InspectorPaths
from semantic_id.schema import PRODUCT_ID, level_columns

DEDUP_COL = "dedup"
SEMANTIC_ID_COL = "semantic_id"

_LEVEL_COL_RE = re.compile(r"^c(\d+)$")


class MappingStore:
    """Wraps the product_id -> (c1..cL, dedup) + metadata mapping table.

    The integer code columns (small: product_id, c1..cL, dedup) are loaded
    eagerly so repeated lookups/filters are in-memory. String metadata columns
    are fetched lazily, only for the rows being displayed, to avoid pulling
    e.g. product_name across all ~3M rows into memory.
    """

    def __init__(self, paths: InspectorPaths):
        self.paths = paths
        lf = pl.scan_parquet(paths.mapping_path)
        columns = lf.collect_schema().names()

        level_matches = [_LEVEL_COL_RE.fullmatch(c) for c in columns]
        found_levels = sorted(int(m.group(1)) for m in level_matches if m)
        if not found_levels:
            raise ValueError(f"No level columns (c1, c2, ...) found in {paths.mapping_path}")
        num_levels = max(found_levels)
        expected = [int(c[1:]) for c in level_columns(num_levels)]
        if found_levels != expected:
            raise ValueError(
                f"Mapping table's level columns {found_levels} are not a contiguous "
                f"c1..c{num_levels} sequence."
            )
        self.num_levels = num_levels
        self.level_cols = level_columns(num_levels)

        if DEDUP_COL not in columns:
            raise ValueError(f"Mapping table is missing the '{DEDUP_COL}' column.")
        if PRODUCT_ID not in columns:
            raise ValueError(f"Mapping table is missing the '{PRODUCT_ID}' column.")

        int_cols = [PRODUCT_ID, *self.level_cols, DEDUP_COL]
        self.codes_df = lf.select(int_cols).collect()
        self.meta_cols = [c for c in columns if c not in int_cols and c != SEMANTIC_ID_COL]
        self.n_rows = len(self.codes_df)

    def get_codes(self, product_id: int) -> dict | None:
        row = self.codes_df.filter(pl.col(PRODUCT_ID) == product_id)
        if row.is_empty():
            return None
        return row.row(0, named=True)

    def bucket_size(self, codes: dict) -> int:
        """Count of products sharing the full c1..cL code tuple with `codes`."""
        conds = [pl.col(c) == codes[c] for c in self.level_cols]
        return int(self.codes_df.filter(pl.all_horizontal(conds)).height)

    def format_semantic_id(self, codes: dict) -> str:
        parts = [str(codes[c]) for c in self.level_cols] + [str(codes[DEDUP_COL])]
        return "-".join(parts)

    def filter_by_tokens(self, tokens: list[int]) -> pl.DataFrame:
        """Filter to rows matching `tokens` against c1..ck (k <= L), or against
        the full c1..cL + dedup when len(tokens) == L + 1 (an exact ID).

        Raises ValueError if `tokens` has more than L + 1 entries — callers
        should validate before calling this if they want a softer error path.
        """
        if len(tokens) > self.num_levels + 1:
            raise ValueError(
                f"Too many tokens ({len(tokens)}); expected at most {self.num_levels + 1} "
                f"(c1..c{self.num_levels} + dedup)."
            )
        conds = [pl.col(self.level_cols[i]) == tokens[i] for i in range(min(len(tokens), self.num_levels))]
        if len(tokens) == self.num_levels + 1:
            conds.append(pl.col(DEDUP_COL) == tokens[self.num_levels])
        if not conds:
            return self.codes_df
        return self.codes_df.filter(pl.all_horizontal(conds))

    def metadata_for(self, product_ids: list[int]) -> pl.DataFrame:
        if not product_ids:
            return pl.DataFrame({PRODUCT_ID: []}, schema={PRODUCT_ID: self.codes_df.schema[PRODUCT_ID]})
        return (
            pl.scan_parquet(self.paths.mapping_path)
            .filter(pl.col(PRODUCT_ID).is_in(product_ids))
            .select([PRODUCT_ID, *self.meta_cols])
            .collect()
        )


class EmbeddingIndex:
    """Memmap-backed content_embedding lookup, built by
    `inspector/build_embedding_index.py`. `available` is False when the index hasn't
    been built yet — callers should degrade gracefully rather than raise.
    """

    def __init__(self, paths: InspectorPaths):
        self.paths = paths
        self.available = paths.index_exists()
        self.meta: dict | None = None
        self.product_ids: np.ndarray | None = None
        self.embeddings: np.memmap | None = None
        self._sorted_ids: np.ndarray | None = None
        self._sort_idx: np.ndarray | None = None

        if not self.available:
            return

        with open(paths.meta_file) as f:
            self.meta = json.load(f)
        self.product_ids = np.load(paths.product_ids_file)
        self.embeddings = np.load(paths.embeddings_file, mmap_mode="r")
        self._sort_idx = np.argsort(self.product_ids)
        self._sorted_ids = self.product_ids[self._sort_idx]

    def get(self, product_id: int) -> np.ndarray | None:
        if not self.available:
            return None
        pos = int(np.searchsorted(self._sorted_ids, product_id))
        if pos >= len(self._sorted_ids) or self._sorted_ids[pos] != product_id:
            return None
        row_idx = int(self._sort_idx[pos])
        return np.asarray(self.embeddings[row_idx], dtype=np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def is_stale(self) -> bool:
        """True if the source parquet files have changed (size/mtime) since
        the index was built, or the index doesn't exist at all."""
        if not self.available:
            return True
        source = self.paths.embedding_source
        files = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
        recorded = {f["name"]: f for f in self.meta.get("source_files", [])}
        if len(files) != len(recorded):
            return True
        for f in files:
            rec = recorded.get(f.name)
            if rec is None:
                return True
            stat = f.stat()
            if rec["size"] != stat.st_size or abs(rec["mtime"] - stat.st_mtime) > 1e-6:
                return True
        return False
