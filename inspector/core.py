"""Pure query functions for the Inspector app's three features:

1. lookup_product        — product_id -> semantic ID + metadata
2. products_by_prefix     — semantic ID / prefix -> matching product_id list
3. compare_products       — two product_ids -> semantic ID, metadata, cosine

No Streamlit imports here — these are plain functions over `inspector.store`
objects, callable from tests, scripts, or the UI alike. Not-found / invalid
input is returned as data (None / an `error` field), never raised.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from inspector.store import EmbeddingIndex, MappingStore
from semantic_id.schema import PRODUCT_ID

# Metadata fields compared in compare_products — the taxonomy/attribute
# columns a human would sanity-check two "similar" products against.
COMPARE_META_FIELDS = ["league", "team", "brand", "color", "price_band", "is_hot_market"]

_TOKEN_SPLIT_RE = re.compile(r"[\s,/\-]+")


def _parse_tokens(query: str) -> tuple[list[int] | None, str | None]:
    """Split a semantic-ID / prefix query into integer tokens.

    Accepts '-', ',', '/', or whitespace as separators (e.g. "402-508",
    "402, 508", "402 508"). Returns (tokens, None) on success or
    (None, error_message) otherwise — never raises.
    """
    stripped = query.strip()
    if not stripped:
        return None, "Query is empty."
    raw_tokens = [t for t in _TOKEN_SPLIT_RE.split(stripped) if t]
    if not raw_tokens:
        return None, "Query is empty."
    tokens = []
    for t in raw_tokens:
        try:
            tokens.append(int(t))
        except ValueError:
            return None, f"'{t}' is not an integer token."
    return tokens, None


def _normalize_bool_str(value: object) -> bool | None:
    """Plain-Python mirror of semantic_id.schema.parse_bool_string, for a
    single scalar value rather than a polars column."""
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


@dataclass
class ProductLookup:
    product_id: int
    semantic_id: str
    codes: dict[str, int]
    dedup: int
    bucket_size: int
    metadata: dict


def lookup_product(store: MappingStore, product_id) -> ProductLookup | None:
    """Feature 1: product_id -> semantic ID, per-level codes, dedup slot,
    bucket size (products sharing the full code tuple), and metadata.
    Returns None if `product_id` isn't in the mapping."""
    try:
        pid = int(product_id)
    except (TypeError, ValueError):
        return None

    codes = store.get_codes(pid)
    if codes is None:
        return None

    semantic_id = store.format_semantic_id(codes)
    bucket_size = store.bucket_size(codes)

    meta_df = store.metadata_for([pid])
    metadata = meta_df.row(0, named=True) if len(meta_df) else {}
    metadata.pop(PRODUCT_ID, None)

    return ProductLookup(
        product_id=pid,
        semantic_id=semantic_id,
        codes={c: codes[c] for c in store.level_cols},
        dedup=codes["dedup"],
        bucket_size=bucket_size,
        metadata=metadata,
    )


@dataclass
class PrefixResult:
    query: str
    matched_levels: int
    is_exact: bool
    total_count: int
    product_ids: list[int]
    truncated: bool
    metadata: dict[int, dict]
    error: str | None = None


def products_by_prefix(store: MappingStore, query: str, limit: int = 1000) -> PrefixResult:
    """Feature 2: a semantic ID or prefix (e.g. "402-508" or the full
    "402-508-366-214-463-0") -> every matching product_id.

    Token count k against L levels:
      k <= L        -> prefix match on c1..ck
      k == L + 1     -> exact match: full c1..cL + dedup
      k > L + 1, a non-integer token, or empty input -> result.error is set.
    """
    empty = PrefixResult(
        query=query, matched_levels=0, is_exact=False, total_count=0,
        product_ids=[], truncated=False, metadata={},
    )

    tokens, err = _parse_tokens(query)
    if err:
        empty.error = err
        return empty

    num_levels = store.num_levels
    if len(tokens) > num_levels + 1:
        empty.error = (
            f"Too many tokens ({len(tokens)}); expected at most {num_levels + 1} "
            f"(c1..c{num_levels} + dedup)."
        )
        return empty

    matches = store.filter_by_tokens(tokens).sort(PRODUCT_ID)
    total_count = matches.height
    is_exact = len(tokens) == num_levels + 1
    matched_levels = min(len(tokens), num_levels)

    truncated = total_count > limit
    limited = matches.head(limit) if truncated else matches
    product_ids = limited[PRODUCT_ID].to_list()

    meta_df = store.metadata_for(product_ids)
    metadata: dict[int, dict] = {}
    if meta_df.height:
        for row in meta_df.iter_rows(named=True):
            row = dict(row)
            pid = row.pop(PRODUCT_ID)
            metadata[pid] = row

    return PrefixResult(
        query=query,
        matched_levels=matched_levels,
        is_exact=is_exact,
        total_count=total_count,
        product_ids=product_ids,
        truncated=truncated,
        metadata=metadata,
        error=None,
    )


@dataclass
class MetadataField:
    field: str
    value_a: object
    value_b: object
    match: bool


@dataclass
class CompareResult:
    product_id_a: int
    product_id_b: int
    lookup_a: ProductLookup | None
    lookup_b: ProductLookup | None
    cosine_similarity: float | None
    cosine_unavailable_reason: str | None
    shared_prefix_depth: int
    same_code_tuple: bool
    same_semantic_id: bool
    metadata_agreement: list[MetadataField] = field(default_factory=list)
    error: str | None = None


def compare_products(
    store: MappingStore, index: EmbeddingIndex | None, product_id_a, product_id_b
) -> CompareResult:
    """Feature 3: semantic ID, metadata, and embedding cosine similarity for
    two product_ids."""
    lookup_a = lookup_product(store, product_id_a)
    lookup_b = lookup_product(store, product_id_b)

    if lookup_a is None or lookup_b is None:
        missing = []
        if lookup_a is None:
            missing.append(str(product_id_a))
        if lookup_b is None:
            missing.append(str(product_id_b))
        return CompareResult(
            product_id_a=product_id_a,
            product_id_b=product_id_b,
            lookup_a=lookup_a,
            lookup_b=lookup_b,
            cosine_similarity=None,
            cosine_unavailable_reason=None,
            shared_prefix_depth=0,
            same_code_tuple=False,
            same_semantic_id=False,
            metadata_agreement=[],
            error=f"product_id(s) not found: {', '.join(missing)}",
        )

    shared_prefix_depth = 0
    for c in store.level_cols:
        if lookup_a.codes[c] == lookup_b.codes[c]:
            shared_prefix_depth += 1
        else:
            break
    same_code_tuple = shared_prefix_depth == store.num_levels
    same_semantic_id = lookup_a.semantic_id == lookup_b.semantic_id

    cosine_similarity: float | None = None
    reason: str | None = None
    if index is None or not index.available:
        reason = "Embedding index not built — run inspector/build_embedding_index.py."
    else:
        vec_a = index.get(lookup_a.product_id)
        vec_b = index.get(lookup_b.product_id)
        if vec_a is None or vec_b is None:
            missing_ids = []
            if vec_a is None:
                missing_ids.append(str(lookup_a.product_id))
            if vec_b is None:
                missing_ids.append(str(lookup_b.product_id))
            reason = (
                f"No embedding in index for product_id(s): {', '.join(missing_ids)} "
                "(dropped as invalid, or index is stale)."
            )
        else:
            cosine_similarity = index.cosine(vec_a, vec_b)

    agreement = []
    for f in COMPARE_META_FIELDS:
        value_a = lookup_a.metadata.get(f)
        value_b = lookup_b.metadata.get(f)
        if f == "is_hot_market":
            match = _normalize_bool_str(value_a) == _normalize_bool_str(value_b)
        else:
            match = value_a == value_b
        agreement.append(MetadataField(field=f, value_a=value_a, value_b=value_b, match=match))

    return CompareResult(
        product_id_a=lookup_a.product_id,
        product_id_b=lookup_b.product_id,
        lookup_a=lookup_a,
        lookup_b=lookup_b,
        cosine_similarity=cosine_similarity,
        cosine_unavailable_reason=reason,
        shared_prefix_depth=shared_prefix_depth,
        same_code_tuple=same_code_tuple,
        same_semantic_id=same_semantic_id,
        metadata_agreement=agreement,
        error=None,
    )
