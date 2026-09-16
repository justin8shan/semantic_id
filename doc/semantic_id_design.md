# Semantic ID Construction — Design Document

**Scope:** Build stable, cold-start-robust semantic IDs for a ~4M-SKU sports-merchandise catalog from precomputed 128-d product embeddings, for use (a) as ranker features and (b) as an optional generative-retrieval target.

**Status:** Draft for review (rev 5). Open decisions flagged inline and collected in §13.

**Known catalog shape:** ~4M SKUs · 128-d embeddings · 130 leagues · up to 1,131 teams in a single league (College) · 33 departments.

---

## 1. Purpose

A semantic ID is a short tuple of discrete codes — e.g. `(c1, c2, c3)` — assigned to each product so that products with similar codes are semantically similar, and products sharing a code *prefix* are similar at a coarser grain. Replacing the opaque 4M-way `product_id` categorical with these shared codes buys three things this catalog needs:

- **Cold-start placement.** A newly minted SKU (playoff run, trade, seasonal drop — flagged by `is_hot_market` / early `launch_age_bucket`) gets a usable ID from its content alone, at launch, before it accumulates any interaction history.
- **Statistical strength pooling.** Long-tail SKUs inherit well-trained code embeddings from the head SKUs they share codes with, instead of each getting an undertrained per-item embedding.
- **A compact, shared vocabulary** that downstream models can consume as a few small integers rather than a giant sparse ID table.

The binding design constraint is **not** the 4M total — it is the **skew of the `(league, team)` item distribution** (College alone has 1,131 teams) and the **continuous cold-start pressure** from sports events. Both are addressed explicitly below.

---

## 2. Design decisions at a glance

| # | Decision | Choice | Rationale (short) |
|---|----------|--------|-------------------|
| D1 | Quantization source | **`content_embedding`** (128-d, not `hybrid_embedding`) | Lifecycle-stable IDs; maximal cold-start robustness; avoids ID drift as items warm up. §3 |
| D2 | Where co-click signal goes | Ranker features + evaluation, **not** baked into the ID | Different roles; keeps ID stable. §3.3 |
| D3 | Taxonomy (`league`/`team`/`dept`) | **Fold into the embedding**, keep columns as separate filter metadata | Preserves cross-team style similarity; cardinalities fit comfortably; taxonomy stays available for filtering. §4 |
| D4 | Quantizer | RQ-VAE (residual-quantized VAE) | Learned, balanced, coarse-to-fine codes. §7 |
| D5 | Codebook shape | **Width 512, depth 3**, latent dim 32, + 1 collision-breaker token | ~7,800 items/code at L1; sized by items-per-code, tuned by measured utilization. §5–6 |
| D6 | ID stability | Freeze quantizer across retrains; version via `embedding_set_version`; explicit remap policy | Prevents silent ID reshuffling in production. §9 |

---

## 3. Which embedding do we quantize?

The schema hands us two 128-d vectors per product:

- `content_embedding` — pure content: metadata, description, image.
- `hybrid_embedding` — a score-weighted blend of content and co-click behavior:

```
hybrid_embedding = effective_gate · neighbor_embedding + (1 − effective_gate) · content_embedding
```

where `neighbor_embedding` is the score-weighted aggregate of the product's most co-clicked neighbors, `content_embedding` is the pure-content vector, and `effective_gate ∈ [0, 1]` sets how much behavior is mixed in.

### 3.1 The formula confirms hybrid is a moving target

`effective_gate` scales with how much co-click evidence a product has accumulated. A freshly launched SKU has essentially no neighbors, so `effective_gate ≈ 0` and `hybrid_embedding ≈ content_embedding`. As it accrues clicks the gate opens and the hybrid embedding interpolates toward `neighbor_embedding`.

So the hybrid embedding **moves over a product's lifecycle** — it starts at content and slides toward its behavioral neighborhood. If we quantized it, the product's semantic ID would **change as it warms up**, and it would change most for the hot, high-traffic SKUs whose gate opens fastest — precisely the items whose IDs we most need to hold still. This is a direct consequence of the interpolation above, not a speculative concern.

### 3.2 Recommendation (D1): quantize `content_embedding`

The `content_embedding` is fixed for the life of the SKU and available at full fidelity at launch, so the ID is lifecycle-stable and the cold-start property is free rather than gated. The cost — the ID space itself doesn't encode co-click structure — is acceptable and, here, correct: behavioral signal belongs in components that are *allowed* to move with behavior, not in the identity key. Team is already present in the content signal (it lives in `product_name` / `style_name` / `team`), so the content embedding clusters by team without any behavioral input (§4).

> **Alternative considered:** quantize `hybrid_embedding`. Rejected as default because of lifecycle drift (§3.1). If a future behavior-aware *retrieval* code space is wanted and can tolerate periodic re-assignment, build it as a separate, versioned ID — never a silent swap of the primary ID.

### 3.3 Where the co-click signal goes instead (D2)

1. **Ranker features** — the co-click signal enters the ranker as a projected `hybrid_embedding` beside the semantic-ID token embeddings, so the ranker leans on behavior where it exists and on content-derived codes where it doesn't (feature spec in §12).
2. **Evaluation ground truth** — `hybrid_similar_products` / `site_hybrid_similar_products` validate that ID prefixes are behaviorally coherent (§11).
3. **Optional coarse refinement** — if behavior is ever wanted in the ID, inject only at the slow-moving team/dept grain, never per-item. Out of scope for v1.

---

## 4. Taxonomy: fold in, don't hard-prefix (D3)

The known cardinalities make the folding case concrete rather than hand-wavy.

| Grain | Cardinality | Fits in… |
|-------|-------------|----------|
| League | 130 | one L1 code space (width 512) with room to spare |
| Team (max within a league) | 1,131 (College) | L1×L2 = 262,144 cells — trivially resolvable |
| Department | 33 | absorbed at any level |

**Why fold rather than hard-prefix.** A hard `league→team` prefix would need a symbol space of ≥130 for league and ≥1,131 for team — team alone exceeds a single 512-wide codebook, forcing either width ≥ 2,048 or two whole levels spent on team before any style structure is encoded. That pushes the total ID to ~5–6 slots and, worse, forces every partition boundary to be the team, destroying cross-team style similarity (a Cowboys retro colorway and a Steelers retro colorway in the same design language get torn apart).

Folding avoids all of that. Because team is already in the content embedding, the RQ-VAE's coarse codes will correlate with league/team on their own, while staying free to group by style/theme/colorway where that matters more. Expect the natural coarse→fine to map roughly **league ≈ L1 · team ≈ L1–L2 · dept/style ≈ L2–L3**, though the model allocates codes by variance, not by our labels — a large league may split across several L1 codes and small leagues may share one, which is the balanced behavior we want.

**Keep the taxonomy columns as separate metadata.** Filtering, eligibility, and per-site routing (`site_id` in `site_hybrid_similar_products`) key off the columns directly — they never need to live inside the ID. Similarity is the ID's job; filtering is the columns' job.

> **Alternative:** hybrid ID `[league, team, rq1, rq2, rq3]` with a global RQ-VAE for the residual codes. Only if constrained generation must be scoped by league/team or a human-readable prefix is a hard requirement. Secondary to folding.

---

## 5. Codebook sizing — items-per-code, tuned by utilization

Size against **capacity** (floor) and **training signal per code** (ceiling), then let measured utilization settle it. 256 is the low end, not a magic number.

Let `W` = width, `L` = learned depth. Total cells = `W^L`, which must exceed 4M with headroom so codebooks stay balanced.

| W × L | Total cells | Headroom vs 4M | Items/code @ L1 | Read |
|-------|-------------|----------------|-----------------|------|
| 256³ | 16.7M | ~4× | ~15,600 | Tight; L1 too coarse |
| **512³** | **134M** | **~33×** | **~7,800** | **Recommended start** |
| 1024³ | 1.07B | ~268× | ~3,900 | More resolution; watch sparsity |
| 512⁴ | 68.7B | huge | ~7,800 | Use if depth-3 hot-bucket utilization is poor |

The cardinalities confirm capacity is not the concern at 512×3 (512 L1 codes ≥ 130 leagues; 262k cells ≥ thousands of teams). The **skew** is the concern: average items-per-code is comfortable, but a College-scale region can saturate first. Utilization is therefore measured **within the largest `(league, team)` item buckets**, not just globally (§11). Poor hot-bucket utilization → depth 4 (or wider L1), not a global resize.

> **Terminology.** Throughout this doc a **"hot bucket"** means a **top-K `(league, team)` partition by catalog SKU count** — the skew/saturation concern. This is distinct from **`is_hot_market`**, the event-driven demand flag used for the cold-start path (§10, §11.4). The two are unrelated: a hot bucket is about how many SKUs share a partition; `is_hot_market` is about a specific SKU's demand surge. `K` is a config (start K=20).

---

## 6. Input parameters and how they are decided

This is the full set of knobs for the construction job, grouped, with the decision rule for each. Values marked **[given]** come from the data; **[set]** are recommended starting points; **[tune]** are settled empirically from §11 diagnostics.

### 6.1 Input / data parameters

| Param | Value | How it's decided |
|-------|-------|------------------|
| `source_embedding` | `content_embedding` **[set]** | Stability + cold-start (§3). The single most consequential choice; not a tuning knob. |
| `d_in` (input dim) | 128 **[given]** | Fixed by the embedding producer. Sets encoder input width and reconstruction target. |
| `normalize` | L2-normalize **[set]** | Embeddings are compared by direction, so quantize in cosine-like geometry. Must be **identical** at training and every inference call; store the transform with the artifact. |

### 6.2 Quantizer capacity parameters (the ones that shape the ID)

| Param | Value | How it's decided |
|-------|-------|------------------|
| `W` (codebook width) | 512 **[set/tune]** | Items-per-code band: floor `W^L ≫ 4M`; ceiling `N/W` in a healthy range (~7.8k/code at L1 — enough signal per code, not so wide codes go dead). Cardinalities confirm 512 ≥ 130 leagues. Tune down if utilization is low. |
| `L` (learned depth) | 3, → 4 if needed **[set/tune]** | Capacity + hot-bucket utilization + downstream use. Each level adds capacity but, for generative retrieval, is another decode step; for ranking it's a cheap extra table. Add a level only if hot buckets collide at depth 3. |
| `d` (latent / codeword dim) | 32 **[set/tune]** | Bottleneck **below** `d_in=128` to force semantic compression (the summed codewords `ẑ ∈ R^d` must reconstruct the 128-d input). Raise to 64 if reconstruction plateaus too high; lower if codes are underused. |
| `n_dedup` (collision token) | sized to max bucket occupancy **[tune]** | Set from the observed worst-case `(c1,c2,c3)` bucket after assignment; assigned deterministically (§8 step 6). Per-level width taper is a related lever, kept in §6.5 Tier 3. |

### 6.3 Training parameters (RQ-VAE optimization)

| Param | Value | How it's decided |
|-------|-------|------------------|
| `beta` (commitment weight) | 0.25 **[set]** | Standard VQ default; balances pulling the encoder toward codes vs. codes toward the encoder. Rarely needs changing. |
| codebook update | EMA, decay `gamma`=0.99 **[set]** | EMA is more stable than gradient updates for codebooks and reduces collapse. |
| `kmeans_init` | on **[set]** | Initialize each codebook on a sample of encoder outputs. The single biggest lever against dead codes — do not skip. |
| `dead_code_reset` | on, low-usage threshold **[tune]** | Periodically re-seed codewords that stop being selected, from high-error inputs. Turn up if utilization diagnostics show dead codes. |
| optimizer / `lr` / `batch` | AdamW / ~1e-3 / ~4,096 **[tune]** | Standard; `lr` and batch tuned to a smooth reconstruction curve. |
| `epochs` | until val reconstruction + utilization plateau **[tune]** | Stop on the diagnostics, not a fixed count. |
| `sample_size` + stratification | few-M sample, **oversample hot buckets** **[set]** | Codebooks must see hot-bucket density or they tune to the long tail; stratify on `(league, team)` and `is_hot_market`. |
| straight-through estimator | on **[structural]** | Required to pass gradient through the non-differentiable `argmin`; not optional. |

### 6.4 The sizing decision procedure (how `W`, `L`, `d` are actually chosen)

1. **Floor.** Require `W^L ≥ ~10×` the largest expected `(league, team)` item bucket (and comfortably ≥ 4M globally). Capacity alone does not rule out 256³ (16.7M clears 4M at ~4×).
2. **Ceiling.** Require items-per-code `N/W` in roughly the low-thousands so each codeword gets enough examples to learn a stable centroid. This is what rules out 256³ (L1 too coarse at ~15,600 items/code) and very wide `W` (too sparse).
3. **Pick inside the band** → 512×3, `d`=32.
4. **Train, then measure** (§11): per-level dead-code rate, usage entropy / effective codebook size, collision-bucket sizes — **globally and inside the top hot buckets**.
5. **Adjust — canonical rule (referenced elsewhere):** dead codes / low entropy → smaller `W` or stronger init/reset; collisions in a hot bucket → `L`→4; reconstruction too high → raise `d`.
6. **Freeze** the chosen artifact and version it (§9).

Everything in 6.2–6.3 is chosen once per `semantic_id_version` and then frozen; only the *downstream* model's code-embedding tables keep training (§12).

### 6.5 Tunable parameters & sweep plan

Of the parameters in 6.1–6.3, only a subset is worth *sweeping*. The rest are fixed by the data, by the architecture, or by a design decision, and burning sweep budget on them is wasted. This section separates them and gives the sweep order.

**Tier 1 — shape the ID (primary sweep).** These change the code space itself, so they come first. `W` and `L` jointly set both capacity (`W^L`) and items-per-code, so sweep them as a **grid, not one-at-a-time**; `d` interacts weakly and is swept after.

| Param | Values to try | Trades | Signal to move it |
|-------|---------------|--------|-------------------|
| `W` — codebook width | 256 / **512** / 1024 | resolution vs items-per-code density | low usage entropy or dead codes → shrink; collisions → widen |
| `L` — learned depth | **3** / 4 | capacity + prefix granularity vs sequence length & per-code sparsity | hot-bucket collisions at depth 3 → 4 |
| `d` — latent / codeword dim | **32** / 64 | reconstruction fidelity vs how compressed/semantic the codes are (must stay below `d_in=128` to remain a bottleneck) | reconstruction plateaus high → raise; codes underused → lower |

**Tier 2 — training dynamics (nudge only if diagnostics are bad).** Not swept for quality; changed only when utilization/collapse diagnostics look wrong.

| Param | Default | Range | When to change |
|-------|---------|-------|----------------|
| `beta` — commitment weight | 0.25 | 0.1–2.0 | codebook usage unstable / commitment loss dominates |
| `gamma` — EMA decay | 0.99 | 0.95–0.999 | codebooks too sluggish (lower) or jittery (raise) |
| `dead_code_reset` threshold + freq | on | — | dead-code count stays high despite k-means init |
| `lr` / `batch_size` / `epochs` | 1e-3 / ~4096 / plateau | standard | rough or non-converging reconstruction curve |

**Tier 3 — data-side (matters for the skew).** Easy to forget; a real lever given the College-scale skew.

| Param | Default | Why tunable here |
|-------|---------|------------------|
| hot-bucket oversampling ratio | oversample | sample composition directly changes whether hot buckets get enough codebook capacity |
| training sample size | few-M | larger improves codebook stability up to a point |
| per-level width taper | uniform 512 (or 512/512/256) | shift capacity to coarse levels if fine residuals are low-variance |

**Not tunable (do not sweep):** `d_in=128` (given); `source_embedding=content_embedding` (design decision, §3); straight-through estimator, EMA-vs-gradient, k-means init (structural, always on); `n_dedup` and the collision-ordering key (data-determined / fixed-deterministic, §8 step 6); L2 normalization (fixed — lock early, changing it re-derives everything).

**Sweep sequence (phase-1 prototype).** Fix Tiers 2–3 at defaults → grid over `W × L` at `d=32` (6 runs: {256,512,1024} × {3,4}) → read the §11 diagnostics **inside the hot buckets** → one added run at `d=64` on the chosen `W×L` (`d=32` is already in the grid) → lock. **7 runs total** to settle the ID space. Selection rule: smallest `(W, L, d)` whose hot-bucket utilization and collision diagnostics (§11) clear their thresholds — smaller is better for serving/decoding, so stop at the first configuration that passes rather than maximizing a score.

---

## 7. Model architecture (D4)

RQ-VAE over the 128-d `content_embedding`.

- **Encoder E:** MLP `128 → 256 → 128 → d` (`d`=32). Produces latent `z = r₀`.
- **Residual quantizer (depth L=3):** for `ℓ = 1..3`: `cℓ = argmin_j ‖r_{ℓ-1} − Cℓ[j]‖²`; `eℓ = Cℓ[cℓ]`; `r_ℓ = r_{ℓ-1} − eℓ`. Each `Cℓ` is a separate `512 × 32` learnable codebook.
- **Decoder G:** MLP `d → 128 → 256 → 128`, reconstructs `x̂` from `ẑ = Σ eℓ`.
- **Losses:** reconstruction `‖x − x̂‖²` (trains E, G) + codebook/commitment `‖sg[r] − e‖² + β‖r − sg[e]‖²`; codebooks by EMA; encoder trained through the `argmin` via straight-through.

The `argmin` is non-differentiable, so the **code assignment is frozen** once trained — no downstream loss can change which tuple a product gets. Only the consumer model's code embeddings train end-to-end, which is why §9's stability policy is a separate concern.

---

## 8. Data pipeline

Spark-native (schema is a Spark `printSchema`): train-on-sample → infer-on-all batch.

1. **Source & filter.** Read the embedding table at the latest `embedding_set_version`, with `produced_at`/`valid_from`/`valid_to` bracketing "now". Project `product_id`, `content_embedding`, and the columns needed for stratified eval (`league`, `team`, `merch_class_root`, `is_hot_market`, `launch_age_bucket`).
2. **Sanity/normalize.** Drop null / wrong-length embeddings; L2-normalize; persist the transform.
3. **Train RQ-VAE on a stratified sample** (§6.3), oversampling hot buckets.
4. **Batch-assign** all 4M through the frozen encoder + codebooks → `(c1,c2,c3)`.
5. **Resolve collisions** → group by `(c1,c2,c3)`, assign `dedup`.
6. **Deterministic collision-breaker** — order colliding items by a fixed key (ascending `product_id`) so a product keeps its slot across re-runs. Never by row order or run-hash.
7. **Publish** `product_id → (c1,c2,c3,dedup)` + `embedding_set_version` + `semantic_id_version` to the feature store. Store **only tokens** on the serving path, never the float vector.

---

## 9. Stability & versioning (D6)

- **Freeze the quantizer** per `semantic_id_version`; reuse it for all subsequent daily/weekly assignments. New products flow through the frozen encoder (§10) — no retrain to onboard.
- **Version explicitly** — `embedding_set_version` (input contract) + `semantic_id_version` (quantizer artifact). IDs are comparable only within a fixed `semantic_id_version`.
- **Retrain deliberately, warm-started** — when embeddings shift enough, warm-start codebooks from the prior artifact to minimize churn, publish a new `semantic_id_version`, and **dual-write** old+new IDs through a migration window so the ranker can be backfilled before cutover.
- **Never** couple the quantizer to the ranking loss in v1 — task-aware assignment reintroduces drift on every ranker retrain.

---

## 10. Cold-start / hot-market ingestion

New SKU (`is_hot_market=true`, small `launch_age_bucket`): its 128-d `content_embedding` → **frozen** encoder + codebooks → `(c1,c2,c3)` → `dedup` within its existing bucket → published. **No history, no retrain.** It lands next to content- and team-similar peers immediately, and inherits their trained code embeddings in the ranker on day zero. This is exactly the property that quantizing the drifting `hybrid_embedding` would have broken (§3.1).

---

## 11. Evaluation plan

Four intrinsic metric families (computed from the assignment output alone, plus the schema's behavioral neighbor lists) and one downstream test. For each: the inputs, the computation, and the pass rule. Notation: `N` = number of products; for level `ℓ`, code `cℓ(i)` is product `i`'s code, and `usage[ℓ][k] = |{ i : cℓ(i) = k }|` is the count of products assigned to code `k` at level `ℓ`.

### 11.1 Codebook utilization (per level; global and per hot bucket)

**Inputs:** the assigned codes `cℓ(i)` for all `i`, at each level `ℓ`. Compute on the set of products in scope `S` (all products for global; the members of one `(league, team)` bucket for the per-bucket version).

**Computation, per level `ℓ` over scope `S`** — let `p[k] = usage_S[ℓ][k] / |S|` be the empirical code distribution:

- **Dead-code rate** = `|{ k : usage_S[ℓ][k] = 0 }| / W`. Fraction of the codebook never used within `S`.
- **Usage entropy** `H_ℓ = − Σ_k p[k] · log(p[k])` (natural log; `0·log 0 ≡ 0`).
- **Effective codebook size** `PPL_ℓ = exp(H_ℓ)` — the "perplexity", i.e. the number of codes actually in play. Compare against nominal `W`.
- **Normalized utilization** `U_ℓ = PPL_ℓ / W ∈ (0, 1]` — 1.0 = perfectly uniform usage, near 0 = collapsed onto a few codes.
- **Max single-code mass** `M_ℓ = max_k p[k]` — the share held by the most-used code (collapse detector).

> **Scope caveat (important).** `U_ℓ` is only meaningful when `|S| ≫ W`: at most `min(W, |S|)` codes can be non-empty, so a scope with `|S| < W` is capped at `U_ℓ ≤ |S|/W < 1` no matter how good the assignment is (e.g. a 200-SKU bucket can never exceed `200/512 ≈ 0.39`). Therefore apply the `U_ℓ` pass rule **only to scopes with `|S| ≥ 4W`** (the global set always qualifies; a "hot bucket" qualifies by construction since it is large by definition). For smaller buckets, judge on dead-code rate and `M_ℓ` alone, or report `PPL_ℓ / min(W, |S|)` instead of `PPL_ℓ / W`.

**Pass rule (per level):** for scopes with `|S| ≥ 4W`: `U_ℓ ≥ 0.5` **and** dead-code rate `≤ 0.10` **and** `M_ℓ ≤ 10 × (1/W)` (no code more than ~10× its fair share); for smaller scopes, dead-code rate and `M_ℓ` only. Check **globally and inside the top-K hot buckets** — the skew fails the per-bucket test first, so the per-bucket check is the binding one. Failing → apply the §6.4 step 5 canonical adjust-rule (shrink `W` / strengthen init / raise dead-code-reset frequency).

### 11.2 Collision rate

**Inputs:** the pre-dedup tuples `t(i) = (c1(i), c2(i), c3(i))` for all `i`.

**Computation** — let `bucket[t] = |{ i : t(i) = t }|`:

- **Collision rate** = `|{ i : bucket[t(i)] > 1 }| / N` — fraction of products sharing their tuple with at least one other.
- **Dedup fraction** = `Σ_t max(bucket[t] − 1, 0) / N` — fraction that needs a non-zero `dedup` index.
- **Worst-case bucket** = `max_t bucket[t]` — sets the required `n_dedup` width.
- **Bucket-size distribution** — percentiles (p50/p90/p99/max) of `bucket[t]`; report **globally and per hot bucket**.

**Pass rule:** worst-case bucket small enough that a modest `dedup` token covers it (target `max_t bucket[t] ≲ a few hundred`), and the hot-bucket p99 not materially worse than global p99. Rising hot-bucket collisions → `L`→4 (§6.5). Note: collisions are *broken* by the dedup token regardless, so this metric gauges ID *quality/resolution*, not correctness.

### 11.3 Prefix coherence vs behavior (ground-truthed)

Tests whether content-derived codes recovered *behavioral* structure — the thing we traded away in D1. Uses the schema's precomputed neighbor lists as ground truth.

**Inputs:** for each anchor `i`, its behavioral neighbor list `Neigh(i)` from `hybrid_similar_products` (or the per-site list under `site_hybrid_similar_products`), optionally top-`n` by `score`; and all products' tuples.

**Computation** — define shared-prefix length `spl(i, j)` = the number of leading levels on which `i` and `j` agree (0 if `c1` differs, 1 if only `c1` matches, 2 if `c1,c2` match, 3 if the full learned tuple matches). Then over a sample of anchors `A`:

- **Prefix-match rate at level `ℓ`** `PM_ℓ = mean over i∈A of ( |{ j ∈ Neigh(i) : spl(i,j) ≥ ℓ }| / |Neigh(i)| )` — the average fraction of an item's behavioral neighbors that share at least its first `ℓ` codes, for `ℓ = 1, 2, 3`.
- **Mean shared-prefix length** `MSPL = mean over i∈A, j∈Neigh(i) of spl(i,j)` — a single scalar; higher = behavioral neighbors sit closer in code space.
- **Baseline (required for interpretation):** recompute `PM_ℓ` against **random** neighbor lists of the same length. Report **lift** `PM_ℓ / PM_ℓ^random`. Raw `PM_ℓ` is meaningless without this baseline because large buckets inflate chance agreement.

**Pass rule:** `PM_ℓ` decreasing in `ℓ` (expected — fewer neighbors share longer prefixes) but **lift over random ≫ 1 at every level**, and lift *growing* with `ℓ` (behavioral neighbors concentrate at longer shared prefixes). This is the metric that quantifies the D1 content-vs-hybrid tradeoff; a low lift here is the signal to reconsider injecting coarse behavioral signal (§3.3).

### 11.4 Cold-start placement quality

**Inputs:** a hold-out of recent launches (small `launch_age_bucket` / `is_hot_market=true`); their assigned tuples; taxonomy columns; and, once history exists, their later-materialized `hybrid_similar_products`.

**Computation** — for each held-out cold item `i`, take its tuple-mates `TM(i) = { j : t(j) = t(i), j ≠ i }` (or first-two-level prefix-mates if `TM(i)` is small):

- **Taxonomy purity** `mean over i of ( fraction of TM(i) sharing i's team )` and likewise for `merch_class_leaf`. High purity = the cold item landed among genuinely related products from content alone.
- **Deferred behavioral recall** — after the item accumulates history (e.g. 2–4 weeks), `mean over i of ( |TM(i) ∩ Neigh_later(i)| / |Neigh_later(i)| )` — did its at-launch code-neighbors turn out to be its eventual behavioral neighbors? Compare against the random baseline as in 11.3.

**Pass rule:** taxonomy purity high (cold items are not scattered), and deferred behavioral recall lift ≫ 1 — i.e. the at-launch placement predicted eventual behavior.

### 11.5 Downstream (decisive) test

Ranker offline **NDCG@k / Recall@k** with vs without the semantic-ID features, on the same training/eval split, **sliced by `launch_age_bucket` and `is_hot_market`**. Report per-slice deltas, not just the aggregate. Expected signature: neutral-to-positive on dense head slices, clearly positive on cold/long-tail slices — that slice pattern, not the headline number, is what confirms the ID is doing its intended job.

> All intrinsic metrics (11.1–11.4) are computable in the Spark assignment job directly from the output table plus the neighbor-list columns; none requires model training, so they gate the §6.5 sweep cheaply. 11.5 requires a ranker training run and gates promotion, not the sweep.

---

## 12. Downstream consumption

**Ranker features (primary v1).** Each level is its own categorical with its own table inside the ranker: `emb_L1[c1]`, `emb_L2[c2]`, `emb_L3[c3]` (each `512×d_r`), concatenated — optionally with composite `(c1)`,`(c1,c2)`,`(c1,c2,c3)` for multi-grain learning. These few shared tables train end-to-end under the ranking loss and replace the 4M-row per-item ID embedding. Serve tokens from the feature store; keep a projected `hybrid_embedding` alongside so behavior enters where it exists.

**Generative-retrieval target (optional, later).** The tuples become a generation vocabulary over user histories, decoded under a prefix trie of valid IDs. Highest cold-start payoff, at the autoregressive-latency cost — evaluate separately.

---

## 13. Risks & open decisions

- **[Decide] Depth 3 vs 4** — from hot-bucket utilization (§11.1), not up front.
- **[Decide] Latent `d` = 32 vs 64** — from reconstruction vs utilization tradeoff (§6.4 step 5).
- **[Decide] Multi-site IDs** — v1 builds one global ID; per-site ID *spaces* out of scope, per-site *evaluation* (§11.3) in.
- **[Risk] Hot-bucket saturation** (College-scale) — mitigated by oversampled training (§8 step 3) + hot-bucket utilization gating (§11.1).
- **[Risk] ID churn on retrain** — mitigated by freeze + warm-start + dual-write (§9).
- **[Watch] Customized / drop-ship SKUs** (`is_customized`, `is_drop_ship`) — confirm their content embeddings are meaningful; degenerate vectors distort codebooks.

---

## 14. Phased rollout

1. **Prototype** — 512×3, `d`=32 on a stratified sample; utilization + collision + prefix-coherence reports (§11.1–11.3). Settle depth and `d`. (Cold-start placement §11.4: its taxonomy-purity half runs now; the deferred-behavioral-recall half needs 2–4 weeks of post-launch history and lands in phase 3.)
2. **Full assignment** — batch-assign 4M; publish IDs + versions; stand up frozen-encoder ingestion (§10).
3. **Ranker integration** — add per-level code-embedding tables; measure lift sliced by cold-start/long-tail.
4. **Stability hardening** — dual-write + warm-start retrain runbook.
5. **(Optional) Generative retrieval** — evaluate tuples as a generation target with prefix-trie decoding.

---

*Appendix — schema fields by role.* Quantization input: `content_embedding` (128-d). Behavior (features/eval, not ID): `hybrid_embedding`, `hybrid_similar_products`, `site_hybrid_similar_products` (and the `effective_gate` / `neighbor_embedding` / `content_embedding` that define the score-weighted hybrid blend). Folded into embedding + kept as filter metadata: `league`, `team`, `brand`, `merch_class_root`, `merch_class_leaf`, `dept_names`, `subdept_names`, `classification_levels`, `player_names`, `color`, `color_style`, `price_band`, `age_group`, `gender`, `gender_age_groups`, `is_customized`, `is_drop_ship`, `on_sale`, `coupon_eligible`. Cold-start stratification: `is_hot_market`, `launch_age_bucket`. Versioning/temporal: `embedding_set_id`, `embedding_set_version`, `produced_at`, `valid_from`, `valid_to`. Key: `product_id`. Display/debug: `product_name`, `style_name`.
