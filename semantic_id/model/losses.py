"""Reconstruction + codebook/commitment loss (design-doc §7)."""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from semantic_id.model.rqvae import RQVAEOutput


@dataclass
class LossBreakdown:
    reconstruction: Tensor
    commitment: Tensor
    total: Tensor


def compute_loss(x: Tensor, output: RQVAEOutput, beta: float) -> LossBreakdown:
    reconstruction = F.mse_loss(output.x_hat, x)

    commitment = x.new_zeros(())
    for level_out in output.level_outputs:
        commitment = commitment + level_out.sg_r_minus_e + beta * level_out.r_minus_sg_e

    total = reconstruction + commitment
    return LossBreakdown(reconstruction=reconstruction, commitment=commitment, total=total)
