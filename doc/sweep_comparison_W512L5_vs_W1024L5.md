# Semantic ID sweep comparison: W512,L5,d32 vs W1024,L5,d32

Real local dataset: `data/input/product_emb/` (3,197,960 rows).
Both configs: `d=32`, `beta=0.25`, `gamma=0.99`, default weighting/eval settings from `configs/local.yaml`.

## Metric definitions (§11.1 utilization)

- **entropy** — Shannon entropy of the code-usage distribution: `H = -Σ p[k]·log(p[k])`. Higher = more evenly spread usage across codes.
- **perplexity** (`ppl`) — `exp(entropy)`. The effective number of codes actually in play, smoothed for imbalance.
- **norm_utilization** (`U`) — `perplexity / W`. Perplexity as a fraction of the nominal codebook width, so it's comparable across different `W`. 1.0 = perfectly uniform use of every code.
- **max_code_share** (`M`) — the single busiest code's share of all traffic in scope. Collapse detector; pass rule requires `M ≤ 10×(1/W)`.
- **dead_code_rate** — fraction of the `W` codes never used within scope.

## 0. Training

| Metric | W512,L5 | W1024,L5 |
|---|---|---|
| `val_reconstruction` | 0.002549 | 0.002237 |
| epochs trained | 12 | 12 |

## 1. §11.2 Global collision

| Metric | W512,L5 | W1024,L5 |
|---|---|---|
| `n` | 3,197,960 | 3,197,960 |
| `collision_rate` | 0.2846 | 0.1653 |
| `dedup_fraction` | 0.1974 | 0.1088 |
| `worst_case_bucket` | 631 | 393 |
| `p50` | 1.0 | 1.0 |
| `p90` | 5.0 | 2.0 |
| `p99` | 43.0 | 20.0 |

## 2. §11.1 Global utilization, per level

**W512,L5**
| level | dead_code_rate | entropy | perplexity | norm_utilization | max_code_share | passes |
|---|---|---|---|---|---|---|
| c1 | 0.1074 | 5.202 | 181.70 | 0.355 | 0.0417 | False |
| c2 | 0.0996 | 5.202 | 181.68 | 0.355 | 0.0905 | False |
| c3 | 0.1250 | 5.796 | 328.90 | 0.642 | 0.0109 | False |
| c4 | 0.1289 | 5.811 | 333.94 | 0.652 | 0.0106 | False |
| c5 | 0.1074 | 5.879 | 357.52 | 0.698 | 0.0078 | False |

**W1024,L5**
| level | dead_code_rate | entropy | perplexity | norm_utilization | max_code_share | passes |
|---|---|---|---|---|---|---|
| c1 | 0.2139 | 5.851 | 347.57 | 0.339 | 0.0215 | False |
| c2 | 0.2080 | 6.068 | 431.95 | 0.422 | 0.0689 | False |
| c3 | 0.2188 | 6.404 | 604.18 | 0.590 | 0.0064 | False |
| c4 | 0.2051 | 6.448 | 631.26 | 0.616 | 0.0047 | False |
| c5 | 0.2119 | 6.442 | 627.82 | 0.613 | 0.0059 | False |

`scope_size`=3,197,960 and `scope_qualifies`=True for every level, both configs.

## 3. §11.1 Hot-bucket utilization (top-20 teams, aggregated per level)

| Level | W512,L5 mean/max dead_code_rate | W512,L5 mean/max max_code_share | W1024,L5 mean/max dead_code_rate | W1024,L5 mean/max max_code_share |
|---|---|---|---|---|
| c1 | 0.900 / 0.943 | 0.961 / 0.991 | 0.915 / 0.948 | 0.867 / 0.978 |
| c2 | 0.328 / 0.414 | 0.364 / 0.496 | 0.415 / 0.536 | 0.147 / 0.264 |
| c3 | 0.179 / 0.195 | 0.044 / 0.066 | 0.280 / 0.314 | 0.030 / 0.072 |
| c4 | 0.160 / 0.170 | 0.017 / 0.028 | 0.245 / 0.256 | 0.011 / 0.019 |
| c5 | 0.130 / 0.137 | 0.010 / 0.017 | 0.245 / 0.252 | 0.006 / 0.010 |

## 4. §11.2 Hot-bucket collision, per team (top 20, sorted by W512,L5 worst-case)

| League | Team | W512,L5 collision_rate / worst / p99 | W1024,L5 collision_rate / worst / p99 |
|---|---|---|---|
| MLB | New York Yankees | 0.541 / 623 / 623 | 0.401 / 343 / 257 |
| MLB | Los Angeles Dodgers | 0.447 / 207 / 135 | 0.295 / 74 / 34 |
| NFL | San Francisco 49ers | 0.412 / 72 / 40 | 0.227 / 83 / 29 |
| NFL | Philadelphia Eagles | 0.415 / 48 / 38 | 0.256 / 67 / 30 |
| NFL | Dallas Cowboys | 0.434 / 59 / 48 | 0.247 / 76 / 54 |
| MLB | New York Mets | 0.374 / 95 / 58 | 0.204 / 75 / 28 |
| MLB | Boston Red Sox | 0.376 / 91 / 45 | 0.232 / 97 / 38 |
| NFL | Pittsburgh Steelers | 0.373 / 97 / 39 | 0.212 / 45 / 20 |
| MLB | Atlanta Braves | 0.366 / 69 / 50 | 0.203 / 52 / 22 |
| NFL | Chicago Bears | 0.375 / 68 / 35 | 0.196 / 49 / 20 |
| MLB | Philadelphia Phillies | 0.351 / 63 / 49 | 0.201 / 110 / 48 |
| NFL | New York Giants | 0.319 / 60 / 30 | 0.233 / 71 / 24 |
| MLB | St. Louis Cardinals | 0.353 / 59 / 35 | 0.225 / 35 / 23 |
| NFL | Kansas City Chiefs | 0.332 / 54 / 30 | 0.153 / 22 / 9 |
| MLB | Texas Rangers | 0.299 / 71 / 32 | 0.153 / 23 / 10 |
| NFL | New England Patriots | 0.368 / 45 / 35 | 0.175 / 61 / 17 |
| MLB | Chicago Cubs | 0.365 / 44 / 33 | 0.198 / 54 / 18 |
| NFL | Buffalo Bills | 0.330 / 44 / 38 | 0.159 / 48 / 12 |
| NFL | Green Bay Packers | 0.317 / 90 / 37 | 0.149 / 100 / 15 |
| NFL | Denver Broncos | 0.310 / 31 / 19 | 0.152 / 17 / 9 |

Note: the global `worst_case_bucket` in each config is driven almost entirely by the Yankees bucket (623 of 631 for W512,L5; 343 of 393 for W1024,L5) — not by an outside niche.

## 5. §11.3 Taxonomy-prefix alignment, all levels

**W512,L5**
| field | l | share_prefix | baseline | lift | purity |
|---|---|---|---|---|---|
| league | 1 | 0.02837 | 0.01107 | 2.56 | 0.605 |
| league | 2 | 0.00334 | 0.00053 | 6.28 | 0.795 |
| league | 3 | 0.00036 | 0.00003 | 13.30 | 0.940 |
| league | 4 | 0.00007 | 0.00000 | 17.17 | 0.992 |
| league | 5 | 0.00002 | 0.00000 | 20.48 | 0.999 |
| team | 1 | 0.34050 | 0.01107 | 30.77 | 0.415 |
| team | 2 | 0.06934 | 0.00053 | 130.42 | 0.707 |
| team | 3 | 0.00709 | 0.00003 | 261.48 | 0.915 |
| team | 4 | 0.00140 | 0.00000 | 327.65 | 0.989 |
| team | 5 | 0.00037 | 0.00000 | 355.04 | 0.999 |

**W1024,L5**
| field | l | share_prefix | baseline | lift | purity |
|---|---|---|---|---|---|
| league | 1 | 0.01851 | 0.00575 | 3.22 | 0.670 |
| league | 2 | 0.00133 | 0.00016 | 8.39 | 0.846 |
| league | 3 | 0.00018 | 0.00001 | 12.83 | 0.973 |
| league | 4 | 0.00004 | 0.00000 | 17.39 | 0.998 |
| league | 5 | 0.00001 | 0.00000 | 15.14 | 1.000 |
| team | 1 | 0.29735 | 0.00575 | 51.74 | 0.517 |
| team | 2 | 0.03124 | 0.00016 | 196.91 | 0.784 |
| team | 3 | 0.00407 | 0.00001 | 283.06 | 0.962 |
| team | 4 | 0.00074 | 0.00000 | 347.51 | 0.998 |
| team | 5 | 0.00020 | 0.00000 | 317.87 | 1.000 |

## Summary

`W1024,L5` beats `W512,L5` on every collision-related metric (2-6x better tail behavior) and reaches marginally better taxonomy purity (1.000 vs 0.999 at the deepest level). Neither formally clears the §11.1 `passes` rule at any level. The Yankees remain the single hardest case for both configs. Full context: `d=64` at `W512,L5` was tested and was strictly worse on every metric including reconstruction — `d=32` is settled. Full per-run JSON/markdown reports for both configs live at `data/output/reports/local-v1-W512-L5.{json,md}` and `local-v1-W1024-L5.{json,md}`, and this same comparison doc is also saved there as `comparison_W512L5_vs_W1024L5.md`.
