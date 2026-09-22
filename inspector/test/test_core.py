"""Tests for the Inspector app's three query functions (inspector/core.py) plus the stores
and index builder they sit on.

Embeddings come from `make_synthetic_catalog` (real schema fidelity, exercised
through the actual `build_embedding_index` builder). The semantic-ID codes and
comparison metadata are hand-crafted rather than model-assigned, so every
prefix/collision/agreement case below is exactly known rather than incidental.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from inspector import core
from inspector.config import InspectorPaths
from inspector.build_embedding_index import build_index
from inspector.store import EmbeddingIndex, MappingStore
from semantic_id.inference.dedup import add_dedup_index
from semantic_id.schema import PRODUCT_ID
from tests.conftest import make_synthetic_catalog

L = 3  # number of semantic-ID levels used by every fixture below

# Hand-crafted (c1, c2, c3) codes for products 0..9, chosen to exercise:
#  - a dedup collision pair (0, 1) sharing the full tuple
#  - a partial-prefix match (2) sharing only (c1, c2) with (0, 1)
#  - a triple collision group (7, 8, 9)
#  - product 4's c1=11, specifically to catch a substring-style prefix bug
#    (a query for "1" must match c1==1, never c1==11)
CODES = {
    0: (1, 1, 1),
    1: (1, 1, 1),
    2: (1, 1, 2),
    3: (1, 2, 5),
    4: (11, 5, 5),
    5: (2, 9, 9),
    6: (2, 9, 9),
    7: (3, 3, 3),
    8: (3, 3, 3),
    9: (3, 3, 3),
}

# Hand-crafted metadata, independent of the catalog's own brand/color/etc.
# formula, so agreement checks below are exactly known:
#  - 0 vs 1: identical except is_hot_market differs only in case ("true" vs
#    "TRUE") -> must still report a match once normalized.
#  - 0 vs 2: brand differs -> must report a mismatch.
METADATA = {
    0: {"league": "NFL", "team": "Alpha", "brand": "Nike", "color": "Red",
        "product_name": "P0", "price_band": "lt25", "is_hot_market": "true",
        "launch_age_bucket": "0-7d"},
    1: {"league": "NFL", "team": "Alpha", "brand": "Nike", "color": "Red",
        "product_name": "P1", "price_band": "lt25", "is_hot_market": "TRUE",
        "launch_age_bucket": "0-7d"},
    2: {"league": "NFL", "team": "Alpha", "brand": "Adidas", "color": "Red",
        "product_name": "P2", "price_band": "lt25", "is_hot_market": "false",
        "launch_age_bucket": "8-30d"},
    3: {"league": "NFL", "team": "Beta", "brand": "Adidas", "color": "Blue",
        "product_name": "P3", "price_band": "25to50", "is_hot_market": "false",
        "launch_age_bucket": "8-30d"},
    4: {"league": "College", "team": "Gamma", "brand": "Nike", "color": "Blue",
        "product_name": "P4", "price_band": "25to50", "is_hot_market": "false",
        "launch_age_bucket": "31-90d"},
    5: {"league": "College", "team": "Gamma", "brand": "Nike", "color": "Green",
        "product_name": "P5", "price_band": "lt25", "is_hot_market": "false",
        "launch_age_bucket": "31-90d"},
    6: {"league": "College", "team": "Gamma", "brand": "Nike", "color": "Green",
        "product_name": "P6", "price_band": "lt25", "is_hot_market": "false",
        "launch_age_bucket": "31-90d"},
    7: {"league": "NBA", "team": "Delta", "brand": "Puma", "color": "Black",
        "product_name": "P7", "price_band": "50plus", "is_hot_market": "true",
        "launch_age_bucket": "0-7d"},
    8: {"league": "NBA", "team": "Delta", "brand": "Puma", "color": "Black",
        "product_name": "P8", "price_band": "50plus", "is_hot_market": "true",
        "launch_age_bucket": "0-7d"},
    9: {"league": "NBA", "team": "Delta", "brand": "Puma", "color": "Black",
        "product_name": "P9", "price_band": "50plus", "is_hot_market": "true",
        "launch_age_bucket": "0-7d"},
    # 999 exists in the mapping but deliberately has no row in the embedding
    # source, simulating a product dropped by drop_invalid_embeddings (or an
    # index built before this product was added).
    999: {"league": "NFL", "team": "Omega", "brand": "Reebok", "color": "White",
          "product_name": "P999", "price_band": "lt25", "is_hot_market": "false",
          "launch_age_bucket": "0-7d"},
}


def _build_mapping_df() -> pl.DataFrame:
    codes_rows = [(pid, *codes) for pid, codes in CODES.items()]
    codes_df = pl.DataFrame(codes_rows, schema=[PRODUCT_ID, "c1", "c2", "c3"], orient="row")
    codes_df = add_dedup_index(codes_df, ["c1", "c2", "c3"])

    # product 999 has no collision partner; dedup 0.
    extra = pl.DataFrame(
        {PRODUCT_ID: [999], "c1": [5], "c2": [5], "c3": [5], "dedup": [0]}
    )
    codes_df = pl.concat([codes_df, extra], how="vertical_relaxed")

    meta_rows = [{PRODUCT_ID: pid, **fields} for pid, fields in METADATA.items()]
    meta_df = pl.DataFrame(meta_rows)

    mapping_df = codes_df.join(meta_df, on=PRODUCT_ID, how="inner")
    semantic_id = (
        pl.col("c1").cast(pl.Utf8) + "-" + pl.col("c2").cast(pl.Utf8) + "-"
        + pl.col("c3").cast(pl.Utf8) + "-" + pl.col("dedup").cast(pl.Utf8)
    )
    return mapping_df.with_columns(semantic_id.alias("semantic_id"))


@pytest.fixture(scope="module")
def inspector_env(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("inspector_core")

    # Embedding source: real schema, from products 0..9 only (999 excluded on
    # purpose — see METADATA's comment).
    catalog = make_synthetic_catalog(
        n_products=10, hot_teams=2, other_leagues=1, teams_per_other_league=2, seed=7
    )
    catalog = catalog.with_columns(pl.Series(PRODUCT_ID, list(range(10))))
    input_path = tmp_path / "input.parquet"
    catalog.write_parquet(input_path)

    mapping_path = tmp_path / "mapping.parquet"
    _build_mapping_df().write_parquet(mapping_path)

    index_dir = tmp_path / "inspector_index"
    paths = InspectorPaths.default(
        mapping_path=str(mapping_path), embedding_source=str(input_path), index_dir=str(index_dir)
    )
    build_index(embedding_source=paths.embedding_source, index_dir=paths.index_dir)

    store = MappingStore(paths)
    index = EmbeddingIndex(paths)
    return store, index, catalog


@pytest.fixture(scope="module")
def store(inspector_env):
    return inspector_env[0]


@pytest.fixture(scope="module")
def index(inspector_env):
    return inspector_env[1]


@pytest.fixture(scope="module")
def catalog(inspector_env):
    return inspector_env[2]


# --- lookup_product -----------------------------------------------------


def test_lookup_product_returns_expected_semantic_id(store):
    result = core.lookup_product(store, 0)
    assert result is not None
    assert result.codes == {"c1": 1, "c2": 1, "c3": 1}
    assert result.dedup == 0
    assert result.semantic_id == "1-1-1-0"
    assert result.bucket_size == 2  # products 0 and 1 share the full tuple
    assert result.metadata["brand"] == "Nike"


def test_lookup_product_dedup_slot_is_ascending_by_product_id(store):
    result0 = core.lookup_product(store, 0)
    result1 = core.lookup_product(store, 1)
    assert result0.dedup == 0
    assert result1.dedup == 1


def test_lookup_product_triple_collision_bucket_size(store):
    result = core.lookup_product(store, 7)
    assert result.bucket_size == 3  # products 7, 8, 9


def test_lookup_product_unknown_id_returns_none(store):
    assert core.lookup_product(store, -1) is None
    assert core.lookup_product(store, "not-an-int") is None


# --- products_by_prefix ---------------------------------------------------


def test_prefix_single_token_matches_only_exact_c1(store):
    # Product 4 has c1=11 — a query for "1" must not match it (no substring
    # matching on the semantic-ID string).
    result = core.products_by_prefix(store, "1")
    assert result.error is None
    assert result.matched_levels == 1
    assert result.is_exact is False
    assert sorted(result.product_ids) == [0, 1, 2, 3]


def test_prefix_two_tokens_narrows_further(store):
    result = core.products_by_prefix(store, "1-1")
    assert sorted(result.product_ids) == [0, 1, 2]


def test_prefix_full_id_is_exact_match(store):
    result = core.products_by_prefix(store, "1-1-1-1")
    assert result.is_exact is True
    assert result.matched_levels == 3
    assert result.product_ids == [1]


def test_prefix_accepts_comma_and_space_separators(store):
    assert core.products_by_prefix(store, "1, 1").product_ids == [0, 1, 2]
    assert core.products_by_prefix(store, "1 1").product_ids == [0, 1, 2]


def test_prefix_too_many_tokens_is_an_error(store):
    result = core.products_by_prefix(store, "1-1-1-1-1")
    assert result.error is not None
    assert result.product_ids == []


def test_prefix_non_integer_token_is_an_error(store):
    result = core.products_by_prefix(store, "abc")
    assert result.error is not None


def test_prefix_empty_query_is_an_error(store):
    result = core.products_by_prefix(store, "   ")
    assert result.error is not None


def test_prefix_truncates_and_reports_metadata_for_returned_rows(store):
    result = core.products_by_prefix(store, "1", limit=2)
    assert result.total_count == 4
    assert result.truncated is True
    assert len(result.product_ids) == 2
    for pid in result.product_ids:
        assert pid in result.metadata


# --- compare_products ------------------------------------------------------


def test_compare_self_is_cosine_one_and_full_prefix_match(store, index):
    result = core.compare_products(store, index, 0, 0)
    assert result.error is None
    assert result.cosine_similarity == pytest.approx(1.0, abs=1e-5)
    assert result.shared_prefix_depth == L
    assert result.same_code_tuple is True
    assert result.same_semantic_id is True
    assert all(f.match for f in result.metadata_agreement)


def test_compare_full_collision_pair_shares_full_prefix_but_not_id(store, index):
    result = core.compare_products(store, index, 0, 1)
    assert result.shared_prefix_depth == L
    assert result.same_code_tuple is True
    assert result.same_semantic_id is False  # dedup differs


def test_compare_partial_prefix_depth(store, index):
    result = core.compare_products(store, index, 0, 2)  # (1,1,1) vs (1,1,2)
    assert result.shared_prefix_depth == 2
    assert result.same_code_tuple is False


def test_compare_no_shared_prefix(store, index):
    result = core.compare_products(store, index, 0, 4)  # (1,1,1) vs (11,5,5)
    assert result.shared_prefix_depth == 0


def test_compare_cosine_matches_direct_numpy_computation(store, index, catalog):
    vec_a = np.array(catalog.filter(pl.col(PRODUCT_ID) == 0)["content_embedding"][0], dtype=np.float32)
    vec_b = np.array(catalog.filter(pl.col(PRODUCT_ID) == 2)["content_embedding"][0], dtype=np.float32)
    expected = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

    result = core.compare_products(store, index, 0, 2)
    assert result.cosine_similarity == pytest.approx(expected, abs=1e-4)


def test_compare_metadata_agreement_normalizes_bool_case(store, index):
    # 0 and 1 are identical metadata except is_hot_market's case ("true" vs
    # "TRUE") — must still be reported as a match.
    result = core.compare_products(store, index, 0, 1)
    agreement = {f.field: f.match for f in result.metadata_agreement}
    assert agreement["is_hot_market"] is True
    assert agreement["brand"] is True


def test_compare_metadata_agreement_flags_mismatch(store, index):
    result = core.compare_products(store, index, 0, 2)
    agreement = {f.field: f.match for f in result.metadata_agreement}
    assert agreement["brand"] is False  # Nike vs Adidas


def test_compare_missing_embedding_reports_reason_not_exception(store, index):
    result = core.compare_products(store, index, 999, 0)
    assert result.error is None  # both product_ids exist in the mapping
    assert result.cosine_similarity is None
    assert "999" in result.cosine_unavailable_reason


def test_compare_unknown_product_id_is_an_error(store, index):
    result = core.compare_products(store, index, -1, 0)
    assert result.error is not None
    assert result.cosine_similarity is None
