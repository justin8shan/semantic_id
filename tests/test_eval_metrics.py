import math

import numpy as np
import polars as pl

from semantic_id.eval.collision import compute_collision_diagnostics
from semantic_id.eval.taxonomy_alignment import compute_prefix_alignment
from semantic_id.eval.utilization import utilization_from_counts


def test_utilization_uniform_counts_passes():
    counts = np.array([10, 10, 10, 10])
    diag = utilization_from_counts(counts, W=4)

    assert diag.scope_size == 40
    assert diag.dead_code_rate == 0.0
    assert math.isclose(diag.entropy, math.log(4), rel_tol=1e-9)
    assert math.isclose(diag.ppl, 4.0, rel_tol=1e-9)
    assert math.isclose(diag.normalized_utilization, 1.0, rel_tol=1e-9)
    assert math.isclose(diag.max_code_share, 0.25, rel_tol=1e-9)
    # scope_size=40 >= 4*W=16, so the full pass rule applies and is satisfied.
    assert diag.scope_qualifies is True
    assert diag.passes is True


def test_utilization_collapsed_counts_fails_dead_code_rate():
    counts = np.array([100, 0, 0, 0])
    diag = utilization_from_counts(counts, W=4)

    assert diag.dead_code_rate == 0.75
    assert diag.entropy == 0.0
    assert math.isclose(diag.ppl, 1.0, rel_tol=1e-9)
    assert diag.passes is False


def test_utilization_small_scope_uses_reduced_pass_rule():
    # W=512, scope_size=100 < 4*W=2048 -> scope doesn't qualify for the U_l rule.
    W = 512
    counts = np.zeros(W)
    counts[:100] = 1  # 100 items, one per code, spread across 100 distinct codes
    diag = utilization_from_counts(counts, W)

    assert diag.scope_qualifies is False
    # dead_code_rate = (512-100)/512 ~ 0.80 -> fails the small-scope rule too.
    assert diag.dead_code_rate > 0.10
    assert diag.passes is False


def test_collision_diagnostics_hand_computed():
    # Buckets: (0,0,0)x3, (1,1,1)x2, (2,2,2)x1 -> N=6
    rows = [
        (0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0),
        (3, 1, 1, 1), (4, 1, 1, 1),
        (5, 2, 2, 2),
    ]
    df = pl.DataFrame(rows, schema=["product_id", "c1", "c2", "c3"], orient="row")
    diag = compute_collision_diagnostics(df)

    assert diag.n == 6
    # collided rows = 3 (bucket of 3) + 2 (bucket of 2) = 5; rate = 5/6
    assert math.isclose(diag.collision_rate, 5 / 6, rel_tol=1e-9)
    # dedup rows = (3-1) + (2-1) = 3; fraction = 3/6 = 0.5
    assert math.isclose(diag.dedup_fraction, 0.5, rel_tol=1e-9)
    assert diag.worst_case_bucket == 3


def test_taxonomy_alignment_perfect_separation():
    # team_a always gets prefix "0", team_b always gets prefix "1" at l=1 ->
    # share_g(1)=1.0 for both teams, well above the chance baseline.
    rows = []
    for i in range(20):
        rows.append((i, 0, i % 2, 0, "league_x", "team_a"))
    for i in range(20, 40):
        rows.append((i, 1, i % 2, 0, "league_x", "team_b"))
    df = pl.DataFrame(rows, schema=["product_id", "c1", "c2", "c3", "league", "team"], orient="row")

    results = compute_prefix_alignment(df, taxonomy_fields=["team"])
    l1 = next(r for r in results if r.prefix_length == 1)

    assert math.isclose(l1.share_prefix, 1.0, rel_tol=1e-9)
    assert math.isclose(l1.baseline, 0.5, rel_tol=1e-9)  # two equal-size prefix groups
    assert math.isclose(l1.lift, 2.0, rel_tol=1e-9)
    assert math.isclose(l1.prefix_purity, 1.0, rel_tol=1e-9)
