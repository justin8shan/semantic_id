"""RQ-VAE over the 128-d content_embedding (design-doc §7).

Encoder 128->256->128->d, residual quantizer of depth L with per-level W x d
EMA codebooks, straight-through estimator, k-means init, dead-code reset.
Decoder d->128->256->128 reconstructs from the summed codewords.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    def __init__(self, d_in: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, d),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, d: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, d_out),
        )

    def forward(self, z_hat: Tensor) -> Tensor:
        return self.net(z_hat)


@dataclass
class LevelOutput:
    quantized_st: Tensor          # straight-through value, used to accumulate z_hat
    residual_next: Tensor         # detached residual passed to the next level
    code_indices: Tensor          # (batch,) int64 codebook index chosen per row
    sg_r_minus_e: Tensor          # scalar loss term ||sg[r] - e||^2
    r_minus_sg_e: Tensor          # scalar loss term ||r - sg[e]||^2 (the real commitment force)


class EMAVectorQuantizerLevel(nn.Module):
    """One residual-quantizer level: a W x d codebook updated by EMA (never by
    gradient), with k-means init and dead-code reset (design-doc §6.3, §7).
    """

    def __init__(self, codebook_size: int, code_dim: int, gamma: float, dead_code_reset_threshold: float):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.gamma = gamma
        self.dead_code_reset_threshold = dead_code_reset_threshold

        self.register_buffer("codebook", torch.randn(codebook_size, code_dim) * 0.01)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", self.codebook.clone())
        # Usage counter reset at the start of each epoch, read by the caller for
        # dead-code diagnostics/reset (design-doc §6.3 dead_code_reset).
        self.register_buffer("epoch_usage_count", torch.zeros(codebook_size))
        self._initialized = False

    @torch.no_grad()
    def kmeans_init(self, data: Tensor, n_iters: int = 20) -> None:
        """Initialize the codebook via Lloyd's-algorithm k-means on `data`
        (design-doc §6.3: "the single biggest lever against dead codes")."""
        n = data.shape[0]
        k = self.codebook_size
        perm = torch.randperm(n, device=data.device)[: min(k, n)]
        centers = data[perm].clone()
        if centers.shape[0] < k:
            pad = data[torch.randint(0, n, (k - centers.shape[0],), device=data.device)]
            centers = torch.cat([centers, pad], dim=0)

        for _ in range(n_iters):
            dist = torch.cdist(data, centers)
            assign = dist.argmin(dim=1)
            new_centers = centers.clone()
            for i in range(k):
                mask = assign == i
                if mask.any():
                    new_centers[i] = data[mask].mean(dim=0)
            centers = new_centers

        self.codebook.copy_(centers)
        self.ema_embed_sum.copy_(centers)
        self.ema_cluster_size.fill_(float(n) / k)
        self._initialized = True

    def forward(self, residual: Tensor, training: bool) -> LevelOutput:
        r = residual
        r_detached = r.detach()

        dist = torch.cdist(r_detached.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)
        code_indices = dist.argmin(dim=1)
        e = self.codebook[code_indices]

        quantized_st = r + (e - r).detach()
        residual_next = (r_detached - e).detach()

        sg_r_minus_e = torch.mean((r_detached - e) ** 2)
        r_minus_sg_e = torch.mean((r - e.detach()) ** 2)

        if training:
            self._ema_update(r_detached, code_indices)
            self.epoch_usage_count.scatter_add_(
                0, code_indices, torch.ones_like(code_indices, dtype=self.epoch_usage_count.dtype)
            )

        return LevelOutput(quantized_st, residual_next, code_indices, sg_r_minus_e, r_minus_sg_e)

    @torch.no_grad()
    def _ema_update(self, r_detached: Tensor, code_indices: Tensor) -> None:
        one_hot = torch.zeros(r_detached.shape[0], self.codebook_size, device=r_detached.device)
        one_hot.scatter_(1, code_indices.unsqueeze(1), 1.0)

        batch_count = one_hot.sum(dim=0)
        batch_sum = one_hot.t() @ r_detached

        self.ema_cluster_size.mul_(self.gamma).add_(batch_count, alpha=1 - self.gamma)
        self.ema_embed_sum.mul_(self.gamma).add_(batch_sum, alpha=1 - self.gamma)

        eps = 1e-5
        n = self.ema_cluster_size.sum()
        smoothed_size = (self.ema_cluster_size + eps) / (n + self.codebook_size * eps) * n
        self.codebook.copy_(self.ema_embed_sum / smoothed_size.unsqueeze(1).clamp_min(eps))

    @torch.no_grad()
    def reset_epoch_usage(self) -> None:
        self.epoch_usage_count.zero_()

    @torch.no_grad()
    def dead_code_reset(self, high_error_residuals: Tensor) -> int:
        """Re-seed codes used less than `dead_code_reset_threshold` of the epoch's
        assignments from `high_error_residuals` (rows with the largest
        quantization error this epoch). Returns the number of codes reset.
        """
        total = self.epoch_usage_count.sum().clamp_min(1.0)
        usage_frac = self.epoch_usage_count / total
        dead_mask = usage_frac < self.dead_code_reset_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0 or high_error_residuals.shape[0] == 0:
            return n_dead

        idx = torch.randint(0, high_error_residuals.shape[0], (n_dead,), device=high_error_residuals.device)
        new_vectors = high_error_residuals[idx]
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        self.codebook[dead_indices] = new_vectors
        self.ema_embed_sum[dead_indices] = new_vectors
        self.ema_cluster_size[dead_indices] = self.ema_cluster_size.mean()
        return n_dead


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int,
        codebook_size: int,
        code_dim: int,
        gamma: float,
        dead_code_reset_threshold: float,
    ):
        super().__init__()
        self.levels = nn.ModuleList(
            [
                EMAVectorQuantizerLevel(codebook_size, code_dim, gamma, dead_code_reset_threshold)
                for _ in range(num_levels)
            ]
        )

    @torch.no_grad()
    def kmeans_init(self, z: Tensor, n_iters: int = 20) -> None:
        residual = z
        for level in self.levels:
            level.kmeans_init(residual, n_iters=n_iters)
            out = level(residual, training=False)
            residual = out.residual_next

    def forward(self, z: Tensor, training: bool) -> tuple[Tensor, list[Tensor], list[LevelOutput]]:
        residual = z
        z_hat = torch.zeros_like(z)
        code_indices_per_level: list[Tensor] = []
        level_outputs: list[LevelOutput] = []
        for level in self.levels:
            out = level(residual, training=training)
            z_hat = z_hat + out.quantized_st
            residual = out.residual_next
            code_indices_per_level.append(out.code_indices)
            level_outputs.append(out)
        return z_hat, code_indices_per_level, level_outputs

    def reset_epoch_usage(self) -> None:
        for level in self.levels:
            level.reset_epoch_usage()

    def dead_code_reset(self, high_error_residuals_per_level: list[Tensor]) -> list[int]:
        return [
            level.dead_code_reset(residuals)
            for level, residuals in zip(self.levels, high_error_residuals_per_level)
        ]


@dataclass
class RQVAEConfig:
    d_in: int = 128
    W: int = 512
    L: int = 3
    d: int = 32
    beta: float = 0.25
    gamma: float = 0.99
    dead_code_reset_threshold: float = 0.01


class RQVAE(nn.Module):
    def __init__(self, config: RQVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.d_in, config.d)
        self.decoder = Decoder(config.d, config.d_in)
        self.quantizer = ResidualQuantizer(
            num_levels=config.L,
            codebook_size=config.W,
            code_dim=config.d,
            gamma=config.gamma,
            dead_code_reset_threshold=config.dead_code_reset_threshold,
        )

    @torch.no_grad()
    def kmeans_init(self, x: Tensor, n_iters: int = 20) -> None:
        z = self.encoder(x)
        self.quantizer.kmeans_init(z, n_iters=n_iters)

    def forward(self, x: Tensor, training: bool) -> "RQVAEOutput":
        z = self.encoder(x)
        z_hat, code_indices_per_level, level_outputs = self.quantizer(z, training=training)
        x_hat = self.decoder(z_hat)
        return RQVAEOutput(x_hat=x_hat, code_indices_per_level=code_indices_per_level, level_outputs=level_outputs)


@dataclass
class RQVAEOutput:
    x_hat: Tensor
    code_indices_per_level: list[Tensor]
    level_outputs: list[LevelOutput] = field(default_factory=list)
