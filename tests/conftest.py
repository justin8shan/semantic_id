"""Synthetic catalog generator matching the real product_emb schema
(product_id, content_embedding, league, team, brand, color, product_name,
price_band, is_hot_market, launch_age_bucket), including a College-style
skew (one league with many more teams than others) and a slice of
is_hot_market rows, for exercising the §11 diagnostics. A plain Polars
DataFrame, written to parquet by the tests that need an on-disk fixture.
"""
from __future__ import annotations

import numpy as np
import polars as pl

EMBEDDING_DIM = 128
LAUNCH_AGE_BUCKETS = ["0-7d", "8-30d", "31-90d", "91+d"]


def _leagues_and_teams(hot_teams: int, other_leagues: int, teams_per_other_league: int) -> dict[str, list[str]]:
    catalog = {"College": [f"college_team_{i}" for i in range(hot_teams)]}
    for l in range(other_leagues):
        league = f"league_{l}"
        catalog[league] = [f"{league}_team_{t}" for t in range(teams_per_other_league)]
    return catalog


def make_synthetic_catalog(
    n_products: int,
    hot_teams: int = 30,
    other_leagues: int = 4,
    teams_per_other_league: int = 5,
    hot_market_fraction: float = 0.05,
    noise_scale: float = 0.15,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    league_teams = _leagues_and_teams(hot_teams, other_leagues, teams_per_other_league)
    bucket_keys = [(league, team) for league, teams in league_teams.items() for team in teams]

    cluster_centers = {key: rng.normal(size=EMBEDDING_DIM) for key in bucket_keys}

    # Skew row counts toward College the way the real catalog is skewed.
    weights = np.array([3.0 if league == "College" else 1.0 for league, _ in bucket_keys])
    weights = weights / weights.sum()
    bucket_choice = rng.choice(len(bucket_keys), size=n_products, p=weights)

    rows = []
    for i in range(n_products):
        league, team = bucket_keys[bucket_choice[i]]
        center = cluster_centers[(league, team)]
        vec = center + rng.normal(scale=noise_scale, size=EMBEDDING_DIM)
        vec = (vec / np.linalg.norm(vec)).astype(np.float32).tolist()

        is_hot_market = "true" if rng.random() < hot_market_fraction else "false"
        launch_age_bucket = LAUNCH_AGE_BUCKETS[0] if is_hot_market == "true" else str(
            rng.choice(LAUNCH_AGE_BUCKETS)
        )

        rows.append(
            {
                "product_id": i,
                "content_embedding": vec,
                "league": league,
                "team": team,
                "brand": f"brand_{i % 10}",
                "color": f"color_{i % 8}",
                "product_name": f"product_{i}",
                "price_band": f"band_{i % 5}",
                "is_hot_market": is_hot_market,
                "launch_age_bucket": launch_age_bucket,
            }
        )

    return pl.DataFrame(rows)
