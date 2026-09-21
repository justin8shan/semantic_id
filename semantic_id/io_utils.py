"""Byte-level read/write via fsspec, so the same code works for local paths
and s3:// paths without branching (fsspec picks the backend from the URL
scheme; s3:// needs `s3fs` installed alongside).
"""
from __future__ import annotations

import os

import fsspec


def ensure_local_dir(path: str) -> None:
    """Create `path` if it's a local directory path — a no-op for remote
    schemes (s3:// etc.) which have no real directories to create.

    Needed before writing with Polars' `write_parquet`: unlike
    `fsspec.open(..., "wb")` (which auto-creates parent dirs for local paths),
    Polars' own file I/O does not create missing directories.
    """
    if "://" in path and not path.startswith("file://"):
        return
    local_path = path.removeprefix("file://") if path.startswith("file://") else path
    os.makedirs(local_path, exist_ok=True)


def write_bytes(path: str, data: bytes) -> None:
    with fsspec.open(path, "wb") as f:
        f.write(data)


def write_text(path: str, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def read_bytes(path: str) -> bytes:
    with fsspec.open(path, "rb") as f:
        return f.read()
