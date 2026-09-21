"""Design-doc §11.1: codebook utilization, global and per hot bucket.

Core math (`utilization_from_counts`) operates on a plain usage-count vector
(length W) so it's shared by both the full §11 report and the per-epoch
training diagnostics (semantic_id/train/train.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class UtilizationDiagnostics:
    scope_size: int
    W: int
    dead_code_rate: float
    entropy: float
    ppl: float
    normalized_utilization: float  # U_l = PPL / W; only meaningful if scope_size >= 4W
    max_code_share: float          # M_l
    scope_qualifies: bool          # |S| >= 4W, per the §11.1 scope caveat
    passes: bool                   # per the §11.1 pass rule for this scope's size


def utilization_from_counts(counts: np.ndarray, W: int) -> UtilizationDiagnostics:
    """`counts` must have length W (zero for unused codes)."""
    assert counts.shape[0] == W
    scope_size = int(counts.sum())
    if scope_size == 0:
        return UtilizationDiagnostics(0, W, 1.0, 0.0, 1.0, 0.0, 0.0, False, False)

    p = counts / scope_size
    dead_code_rate = float(np.mean(counts == 0))
    nonzero_p = p[p > 0]
    entropy = float(-np.sum(nonzero_p * np.log(nonzero_p)))
    ppl = float(np.exp(entropy))
    normalized_utilization = ppl / W
    max_code_share = float(p.max())

    scope_qualifies = scope_size >= 4 * W
    if scope_qualifies:
        passes = (
            normalized_utilization >= 0.5
            and dead_code_rate <= 0.10
            and max_code_share <= 10.0 / W
        )
    else:
        # Small-scope rule (§11.1 caveat): judge dead-code rate and M_l only.
        passes = dead_code_rate <= 0.10 and max_code_share <= 10.0 / W

    return UtilizationDiagnostics(
        scope_size=scope_size,
        W=W,
        dead_code_rate=dead_code_rate,
        entropy=entropy,
        ppl=ppl,
        normalized_utilization=normalized_utilization,
        max_code_share=max_code_share,
        scope_qualifies=scope_qualifies,
        passes=passes,
    )


def usage_counts_global(df: pl.DataFrame, level_col: str, W: int) -> np.ndarray:
    counts = df[level_col].value_counts()
    arr = np.zeros(W, dtype=np.int64)
    for row in counts.iter_rows(named=True):
        code = row[level_col]
        if code is not None and 0 <= code < W:
            arr[int(code)] = int(row["count"])
    return arr
