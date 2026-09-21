"""RQ-VAE training over the full table (design-doc §7, §6.3).

Polars reads + filters + normalizes the full embedding table lazily (the one
place this stage touches "big data" — the 128-d column across ~4M rows), then
a single `.collect()` materializes it to an in-memory frame for the train/val
split, hot-bucket weighting, and the actual torch training loop — all of
which operate on small per-row data once collected to numpy.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from semantic_id.data.io import read_product_embeddings
from semantic_id.data.preprocess import drop_invalid_embeddings, l2_normalize, stratified_train_val_split
from semantic_id.data.weights import add_sample_weight
from semantic_id.eval.utilization import utilization_from_counts
from semantic_id.model.losses import compute_loss
from semantic_id.model.rqvae import RQVAE, RQVAEConfig
from semantic_id.schema import CONTENT_EMBEDDING
from semantic_id.train.artifact import load_model, save_artifact


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_reconstruction: float
    val_dead_code_rate: float
    val_normalized_utilization: float
    dead_codes_reset: list[int]


def _stack_embeddings(df, embedding_dim: int) -> np.ndarray:
    return df[CONTENT_EMBEDDING].list.to_array(embedding_dim).to_numpy().astype(np.float32)


def _collect_high_error_pool(
    model: RQVAE, x: torch.Tensor, pool_size: int
) -> list[torch.Tensor]:
    """Probe a sample batch to find, per level, the residuals with the largest
    quantization error — the "high-error inputs" dead_code_reset re-seeds from
    (design-doc §6.3).
    """
    model.eval()
    with torch.no_grad():
        residual = model.encoder(x)
        pools = []
        for level in model.quantizer.levels:
            dist = torch.cdist(residual, level.codebook)
            min_dist, idx = dist.min(dim=1)
            k = min(pool_size, residual.shape[0])
            top_idx = torch.topk(min_dist, k=k).indices
            pools.append(residual[top_idx].clone())
            e = level.codebook[idx]
            residual = residual - e
    return pools


def run_training(config: dict[str, Any]) -> tuple[str, list[EpochStats]]:
    data_cfg, pre_cfg, w_cfg, q_cfg, t_cfg, v_cfg = (
        config["data"], config["preprocess"], config["weighting"],
        config["quantizer"], config["training"], config["versioning"],
    )

    torch.manual_seed(t_cfg["seed"])

    lf = read_product_embeddings(data_cfg["input_path"])
    lf = drop_invalid_embeddings(lf, data_cfg["embedding_dim"])
    lf = l2_normalize(lf)
    df = lf.collect()

    # The source table carries no version column of its own — this is a
    # manually-set provenance tag for the input snapshot (design-doc §9).
    embedding_set_version = data_cfg["embedding_set_version"]

    train_df, val_df = stratified_train_val_split(
        df, pre_cfg["val_fraction"], pre_cfg["split_seed"], pre_cfg["strata_columns"]
    )
    train_df = add_sample_weight(
        train_df, w_cfg["hot_bucket_top_k"], w_cfg["hot_bucket_weight"], w_cfg["hot_market_weight"]
    )

    device = torch.device(t_cfg["device"])
    embedding_dim = data_cfg["embedding_dim"]
    train_x = torch.from_numpy(_stack_embeddings(train_df, embedding_dim)).to(device)
    train_w = torch.from_numpy(train_df["sample_weight"].to_numpy().astype(np.float32)).to(device)
    val_x = torch.from_numpy(_stack_embeddings(val_df, embedding_dim)).to(device)

    config_obj = RQVAEConfig(
        d_in=embedding_dim, W=q_cfg["W"], L=q_cfg["L"], d=q_cfg["d"],
        beta=q_cfg["beta"], gamma=q_cfg["gamma"], dead_code_reset_threshold=q_cfg["dead_code_reset_threshold"],
    )
    model = RQVAE(config_obj).to(device)

    if t_cfg.get("warm_start_from"):
        prior_model, _ = load_model(t_cfg["warm_start_from"])
        model.load_state_dict(prior_model.state_dict())
    elif q_cfg["kmeans_init"]:
        probe_size = min(100_000, train_x.shape[0])
        probe_idx = torch.randperm(train_x.shape[0])[:probe_size]
        model.kmeans_init(train_x[probe_idx])

    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"])

    dataset = TensorDataset(train_x, train_w)
    sampler = WeightedRandomSampler(train_w.cpu().numpy(), num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=t_cfg["batch_size"], sampler=sampler)

    history: list[EpochStats] = []
    best_val_recon = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(t_cfg["max_epochs"]):
        model.train()
        model.quantizer.reset_epoch_usage()
        train_losses = []
        for x_batch, _w_batch in loader:
            optimizer.zero_grad()
            output = model(x_batch, training=True)
            loss = compute_loss(x_batch, output, config_obj.beta)
            loss.total.backward()
            optimizer.step()
            train_losses.append(loss.total.item())

        dead_codes_reset = [0] * config_obj.L
        if q_cfg["dead_code_reset"] and (epoch + 1) % q_cfg["dead_code_reset_every"] == 0:
            probe_size = min(20_000, train_x.shape[0])
            probe_idx = torch.randperm(train_x.shape[0])[:probe_size]
            pools = _collect_high_error_pool(model, train_x[probe_idx], pool_size=2000)
            dead_codes_reset = model.quantizer.dead_code_reset(pools)

        model.eval()
        with torch.no_grad():
            val_output = model(val_x, training=False)
            val_loss = compute_loss(val_x, val_output, config_obj.beta)
            val_recon = val_loss.reconstruction.item()

            level1_idx = val_output.code_indices_per_level[0].cpu().numpy()
            counts = np.bincount(level1_idx, minlength=config_obj.W)
            util = utilization_from_counts(counts, config_obj.W)

        stats = EpochStats(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            val_reconstruction=val_recon,
            val_dead_code_rate=util.dead_code_rate,
            val_normalized_utilization=util.normalized_utilization,
            dead_codes_reset=dead_codes_reset,
        )
        history.append(stats)

        if val_recon < best_val_recon - t_cfg["min_delta"]:
            best_val_recon = val_recon
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= t_cfg["patience"]:
                break

    model.load_state_dict(best_state)

    artifact_path = save_artifact(
        model=model,
        output_path=data_cfg["output_path"],
        semantic_id_version=v_cfg["semantic_id_version"],
        embedding_set_version=embedding_set_version,
        normalize=True,
        extra_metadata={"trained_epochs": len(history), "best_val_reconstruction": best_val_recon},
    )
    return artifact_path, history
