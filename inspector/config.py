"""Path resolution for the Inspector app.

Defaults follow the `configs/local.yaml` layout (local parquet, no S3). Every
path is overridable by env var or by CLI flag on the index builder, so a sweep
version's output can be QA'd without editing code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MAPPING = "data/output/product_id_semantic_id_full_mapping.parquet"
DEFAULT_EMBEDDING_SOURCE = "data/input/product_emb/"
DEFAULT_INDEX_DIR = "data/inspector_index/"

ENV_MAPPING = "SEMANTIC_ID_INSPECTOR_MAPPING"
ENV_EMBEDDINGS = "SEMANTIC_ID_INSPECTOR_EMBEDDINGS"
ENV_INDEX = "SEMANTIC_ID_INSPECTOR_INDEX"

EMBEDDINGS_FILENAME = "embeddings.npy"
PRODUCT_IDS_FILENAME = "product_ids.npy"
META_FILENAME = "meta.json"


def _resolve(value: str | os.PathLike[str]) -> Path:
    """Resolve `value` against the repo root when it isn't already absolute,
    so the app behaves the same from any working directory."""
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass(frozen=True)
class InspectorPaths:
    mapping_path: Path
    embedding_source: Path
    index_dir: Path

    @classmethod
    def default(
        cls,
        mapping_path: str | os.PathLike[str] | None = None,
        embedding_source: str | os.PathLike[str] | None = None,
        index_dir: str | os.PathLike[str] | None = None,
    ) -> "InspectorPaths":
        """Build paths from (in precedence order) explicit arguments, env vars,
        then the local-run defaults."""
        return cls(
            mapping_path=_resolve(
                mapping_path or os.environ.get(ENV_MAPPING) or DEFAULT_MAPPING
            ),
            embedding_source=_resolve(
                embedding_source or os.environ.get(ENV_EMBEDDINGS) or DEFAULT_EMBEDDING_SOURCE
            ),
            index_dir=_resolve(index_dir or os.environ.get(ENV_INDEX) or DEFAULT_INDEX_DIR),
        )

    @property
    def embeddings_file(self) -> Path:
        return self.index_dir / EMBEDDINGS_FILENAME

    @property
    def product_ids_file(self) -> Path:
        return self.index_dir / PRODUCT_IDS_FILENAME

    @property
    def meta_file(self) -> Path:
        return self.index_dir / META_FILENAME

    def index_exists(self) -> bool:
        return all(
            p.exists() for p in (self.embeddings_file, self.product_ids_file, self.meta_file)
        )
