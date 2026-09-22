# Semantic ID vs. taxonomy: how the field handles the tension

The conflict between a learned, embedding-derived Semantic ID and a
human-curated taxonomy (category, brand, price tier, etc.) is a known,
actively-discussed problem in generative-recommendation research. Two
broad camps have emerged, plus a cautionary result about trying to make
one ID serve multiple purposes.

## Camp 1 — Don't force alignment; treat Semantic ID as a separate, complementary signal

**[Better Generalization with Semantic IDs: A Case Study in Ranking for
Recommendations](https://arxiv.org/pdf/2306.08121)** (Google/YouTube,
RecSys '24) uses Semantic IDs to **replace random-hashed item-ID
embeddings** — the memorization feature in a ranking model — not to encode
taxonomy. Category and other metadata are fed to the ranking model as
their own separate features. The paper doesn't justify the ID by what it
"means" — it justifies it with an ablation showing improved generalization
on new/long-tail item slices without hurting overall ranking quality. This
is the standard production pattern: prove value through downstream lift,
not through human legibility.

**[Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/pdf/2305.05065)**
(NeurIPS 2023) is the foundational paper: RQ-VAE quantizes a frozen content
embedding into an ordered tuple of codewords. On some single-category
datasets (e.g. Amazon Beauty), the first codeword happened to align with
coarse category as an emergent byproduct — but that's dataset-dependent,
not a guarantee, and won't reliably hold for a large, many-department,
many-brand catalog.

## Camp 2 — Force alignment via supervision, and accept the cost

**[HiD-VAE: Interpretable Generative Recommendation via Hierarchical and
Disentangled Semantic IDs](https://arxiv.org/pdf/2508.04618)** directly
attacks the "flat, uninterpretable ID" problem: it injects category/
taxonomy labels as an auxiliary loss during RQ-VAE training, so coarser
codeword levels are regularized to match human categories while finer
levels capture residual style variation. This buys a prefix that
genuinely means "department" or "category" — at the cost of constraining
the learned embedding space to fit the existing taxonomy, which cuts
against the whole reason for using a learned embedding (discovering
useful similarity the taxonomy doesn't already capture).

## Cautionary data point — one ID can't optimally serve every purpose

**[Semantic IDs for Generative Search and Recommendation](https://research.atspotify.com/2025/9/semantic-ids-for-generative-search-and-recommendation)**
(Spotify Research, Sept 2025) found that a Semantic ID tuned for one task
(search) significantly hurt another task (recommendation), and vice versa.
Their fix was a multi-task bi-encoder trained jointly on both objectives,
landing on a Pareto tradeoff that still underperformed either task-specific
baseline alone. The generalizable lesson: trying to make one Semantic ID
satisfy both "useful to the ML task" and "legible against a human
taxonomy" is likely to produce something mediocre at both, rather than
good at either. Related work on
**[multi-identifier tokenization](https://arxiv.org/pdf/2504.04400)**
reaches a similar conclusion by giving items multiple IDs instead of one
universal code.

## Supporting references

- [Generative Recommendation with Semantic IDs: A Practitioner's Handbook](https://arxiv.org/pdf/2507.22224)
- [SIDInspector: A Mapping-First Diagnostic Resource for Semantic-ID Tokenizers](https://arxiv.org/pdf/2606.10375) — a diagnostic framework built specifically to audit what a Semantic ID tokenizer actually captures vs. what people assume it captures.
- [Decoupled Residual Quantization for Robust Semantic IDs in Recommendation](https://arxiv.org/pdf/2606.01844)

## Sources

- [Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations](https://arxiv.org/pdf/2306.08121)
- [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/pdf/2305.05065)
- [HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs](https://arxiv.org/pdf/2508.04618)
- [Semantic IDs for Generative Search and Recommendation — Spotify Research](https://research.atspotify.com/2025/9/semantic-ids-for-generative-search-and-recommendation)
- [Generative Recommendation with Semantic IDs: A Practitioner's Handbook](https://arxiv.org/pdf/2507.22224)
- [Pre-training Generative Recommender with Multi-Identifier Item Tokenization](https://arxiv.org/pdf/2504.04400)
- [SIDInspector: A Mapping-First Diagnostic Resource for Semantic-ID Tokenizers](https://arxiv.org/pdf/2606.10375)
- [Decoupled Residual Quantization for Robust Semantic IDs in Recommendation](https://arxiv.org/pdf/2606.01844)
