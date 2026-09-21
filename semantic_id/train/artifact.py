"""Save/load the frozen RQ-VAE artifact (design-doc §9): encoder+decoder+codebooks,
config, the normalize flag, and both version identifiers. Freezing means: once
saved, this artifact is loaded read-only for batch assignment — training never
mutates a loaded artifact in place.
"""
from __future__ import annotations

import io
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import torch

from semantic_id.io_utils import read_bytes, write_bytes
from semantic_id.model.rqvae import RQVAE, RQVAEConfig

MODEL_FILENAME = "model.pt"


def artifact_dir(output_path: str, semantic_id_version: str) -> str:
    base = output_path.rstrip("/")
    return f"{base}/semantic_id_version={semantic_id_version}/artifact"


def save_artifact(
    model: RQVAE,
    output_path: str,
    semantic_id_version: str,
    embedding_set_version: str,
    normalize: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    bundle = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
        "normalize": normalize,
        "embedding_set_version": embedding_set_version,
        "semantic_id_version": semantic_id_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metadata": extra_metadata or {},
    }
    buffer = io.BytesIO()
    torch.save(bundle, buffer)

    path = f"{artifact_dir(output_path, semantic_id_version)}/{MODEL_FILENAME}"
    write_bytes(path, buffer.getvalue())
    return path


def load_artifact(path: str) -> dict[str, Any]:
    data = read_bytes(path)
    bundle = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return bundle


def load_model(path: str) -> tuple[RQVAE, dict[str, Any]]:
    bundle = load_artifact(path)
    config = RQVAEConfig(**bundle["config"])
    model = RQVAE(config)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, bundle
