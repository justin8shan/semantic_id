# Semantic ID — Hyperparameter Tuning & Sweep Plan

**Companion to** `semantic_id_design.md`. That doc defines *what* the parameters are and how each is decided (§6.1–6.3); this doc is the operational procedure for *choosing and sweeping* them. Parameter definitions, the capacity table, and the evaluation metrics are not repeated here — they live in the design doc at §6.1–6.3, §5, and §11 respectively, and are referenced by section.

The whole procedure gates on the intrinsic diagnostics in design-doc §11 (utilization, collision, taxonomy–prefix alignment, cold-start placement). Those are cheap — computable in the Spark assignment job with no model training — so a sweep of ~7 configurations is inexpensive.

---

## 1. Parameter classification

Only a subset of the parameters in design-doc §6 is worth *sweeping*. The rest are fixed by the data, by the architecture, or by a design decision; spending sweep budget on them is wasted.

### Tier 1 — shape the ID (primary sweep)

These change the code space itself, so they come first. `W` and `L` jointly set both capacity (`W^L`) and items-per-code, so sweep them as a **grid, not one-at-a-time**; `d` interacts weakly and is swept after.

| Param | Values to try | Trades | Signal to move it |
|-------|---------------|--------|-------------------|
| `W` — codebook width | 256 / **512** / 1024 | resolution vs items-per-code density | low usage entropy or dead codes → shrink; collisions → widen |
| `L` — learned depth | **3** / 4 | capacity + prefix granularity vs sequence length & per-code sparsity | hot-bucket collisions at depth 3 → 4 |
| `d` — latent / codeword dim | **32** / 64 | reconstruction fidelity vs how compressed/semantic the codes are (must stay below `d_in=128` to remain a bottleneck) | reconstruction plateaus high → raise; codes underused → lower |

### Tier 2 — training dynamics (nudge only if diagnostics are bad)

Not swept for quality; changed only when utilization/collapse diagnostics look wrong.

| Param | Default | Range | When to change |
|-------|---------|-------|----------------|
| `beta` — commitment weight | 0.25 | 0.1–2.0 | codebook usage unstable / commitment loss dominates |
| `gamma` — EMA decay | 0.99 | 0.95–0.999 | codebooks too sluggish (lower) or jittery (raise) |
| `dead_code_reset` threshold + freq | on | — | dead-code count stays high despite k-means init |
| `lr` / `batch_size` / `epochs` | 1e-3 / ~4096 / plateau | standard | rough or non-converging reconstruction curve |

### Tier 3 — data-side (matters for the skew)

Easy to forget; a real lever given the College-scale skew.

| Param | Default | Why tunable here |
|-------|---------|------------------|
| hot-bucket oversampling ratio | oversample | sample composition directly changes whether hot buckets get enough codebook capacity |
| training sample size | few-M | larger improves codebook stability up to a point |
| per-level width taper | uniform 512 (or 512/512/256) | shift capacity to coarse levels if fine residuals are low-variance |

### Fixed — not tunable (do not sweep)

`d_in=128` (given); `source_embedding=content_embedding` (design decision, design-doc §3); straight-through estimator, EMA-vs-gradient codebooks, k-means init (structural, always on); `n_dedup` and the collision-ordering key (data-determined / fixed-deterministic, design-doc §8 step 6); L2 normalization (fixed — lock early, changing it re-derives everything).

---

## 2. Sizing bounds for `W` and `L`

Pick `W` and `L` inside a band bounded from both sides. The capacity arithmetic (the `W × L` table) is in design-doc §5; the two bounds are:

- **Floor (capacity).** Require `W^L ≥ ~10×` the largest expected `(league, team)` item bucket, and comfortably ≥ 4M globally. Capacity alone does *not* rule out 256³ (16.7M clears 4M at ~4×).
- **Ceiling (signal per code).** Require items-per-code `N/W` in roughly the low-thousands so each codeword gets enough examples to learn a stable centroid. This is what rules out 256³ (L1 too coarse at ~15,600 items/code) and very wide `W` (codes go dead).

The starting point that sits inside the band: **`W=512, L=3, d=32`**.

---

## 3. Adjust rules (canonical)

When a trained configuration fails a §11 diagnostic, this is the single source of truth for what to change (Tier-1 tables above point here):

- **Dead codes / low usage entropy** → smaller `W`, or strengthen k-means init / raise `dead_code_reset` frequency.
- **Collisions in a hot bucket** → `L` → 4 (or widen L1).
- **Reconstruction plateaus too high** → raise `d` (32 → 64).
- **Codes underused despite good init** → lower `d`.

---

## 4. Sweep sequence (phase-1 prototype)

1. Fix Tiers 2–3 at their defaults.
2. **Grid over `W × L` at `d=32`** — 6 runs: {256, 512, 1024} × {3, 4}.
3. Read the design-doc §11 diagnostics **inside the hot buckets** (the binding scope).
4. **One added run at `d=64`** on the chosen `W×L` (`d=32` is already in the grid).
5. Lock.

**7 runs total** to settle the ID space.

**Selection rule.** Smallest `(W, L, d)` whose hot-bucket utilization and collision diagnostics clear their §11 thresholds — smaller is better for serving/decoding, so stop at the first configuration that passes rather than maximizing a score.

---

## 5. What is frozen after the sweep

Everything in Tiers 1–3 is chosen once per `semantic_id_version` and then **frozen** (design-doc §9). Only the *downstream* model's code-embedding tables keep training under the ranking loss (design-doc §12). Re-running the sweep is a deliberate, versioned event — never a silent change.