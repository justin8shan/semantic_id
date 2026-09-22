"""Streamlit Inspector app for the semantic-ID pipeline.

Usage:
    streamlit run inspector/app.py

Three tabs, each a thin UI over `inspector/core.py`:
  1. Product -> semantic ID
  2. Semantic ID / prefix -> product list
  3. Compare two products (semantic ID, metadata, embedding cosine)
"""
from __future__ import annotations

import io

import polars as pl
import streamlit as st

from inspector import core
from inspector.config import InspectorPaths
from inspector.store import EmbeddingIndex, MappingStore

st.set_page_config(page_title="Semantic ID Inspector", layout="wide")


@st.cache_resource
def get_paths() -> InspectorPaths:
    return InspectorPaths.default()


@st.cache_resource
def get_mapping_store(_paths: InspectorPaths) -> MappingStore:
    return MappingStore(_paths)


@st.cache_resource
def get_embedding_index(_paths: InspectorPaths) -> EmbeddingIndex:
    return EmbeddingIndex(_paths)


def render_sidebar(paths: InspectorPaths, store: MappingStore, index: EmbeddingIndex) -> None:
    st.sidebar.header("Inspector status")
    st.sidebar.caption("Mapping table")
    st.sidebar.code(str(paths.mapping_path), language=None)
    st.sidebar.write(f"Rows: **{store.n_rows:,}**")
    st.sidebar.write(f"Semantic ID levels (L): **{store.num_levels}**")

    st.sidebar.caption("Embedding index")
    st.sidebar.code(str(paths.index_dir), language=None)
    if not index.available:
        st.sidebar.error("Index not built.")
        st.sidebar.code(
            f"python inspector/build_embedding_index.py --embedding-source {paths.embedding_source}",
            language="bash",
        )
    else:
        st.sidebar.write(f"Rows: **{index.meta['n_rows']:,}**")
        st.sidebar.write(f"Built at: {index.meta['built_at']}")
        st.sidebar.write(f"Dropped invalid: {index.meta['dropped_invalid']:,}")
        if index.is_stale():
            st.sidebar.warning(
                "Index looks stale — the source parquet files have changed since it "
                "was built. Rebuild with --force."
            )
        else:
            st.sidebar.success("Index is up to date.")


def render_lookup_tab(store: MappingStore) -> None:
    st.subheader("Product ID → Semantic ID")
    product_id = st.text_input("Product ID", key="lookup_product_id")
    if not product_id:
        return

    result = core.lookup_product(store, product_id)
    if result is None:
        st.warning(f"Product ID '{product_id}' not found in the mapping.")
        return

    st.metric("Semantic ID", result.semantic_id)
    col1, col2 = st.columns(2)
    with col1:
        codes_df = pl.DataFrame(
            {"level": list(result.codes.keys()), "code": list(result.codes.values())}
        )
        st.write("Per-level codes")
        st.dataframe(codes_df, hide_index=True, use_container_width=True)
    with col2:
        st.write("Bucket")
        st.write(f"dedup slot: **{result.dedup}**")
        st.write(f"bucket size (products sharing c1..cL): **{result.bucket_size}**")

    st.write("Metadata")
    if result.metadata:
        meta_df = pl.DataFrame(
            {"field": list(result.metadata.keys()), "value": [str(v) for v in result.metadata.values()]}
        )
        st.dataframe(meta_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No metadata columns found.")


def render_prefix_tab(store: MappingStore) -> None:
    st.subheader("Semantic ID / Prefix → Product IDs")
    st.caption(
        f"Enter a full semantic ID (c1..c{store.num_levels}+dedup) or any leading "
        "prefix (c1..ck), separated by '-', ',', '/', or spaces."
    )
    query = st.text_input("Semantic ID or prefix", key="prefix_query")
    limit = st.slider("Max rows to display", min_value=10, max_value=5000, value=500, step=10)
    if not query:
        return

    result = core.products_by_prefix(store, query, limit=limit)
    if result.error:
        st.error(result.error)
        return

    kind = "exact match (full ID)" if result.is_exact else f"prefix match ({result.matched_levels} level(s))"
    st.write(f"**{result.total_count}** product(s) matched — {kind}.")
    if result.truncated:
        st.info(f"Showing first {limit} of {result.total_count} matches.")

    if not result.product_ids:
        st.warning("No products matched.")
        return

    rows = []
    for pid in result.product_ids:
        row = {"product_id": pid}
        row.update(result.metadata.get(pid, {}))
        rows.append(row)
    result_df = pl.DataFrame(rows)
    st.dataframe(result_df, hide_index=True, use_container_width=True)

    buf = io.StringIO()
    result_df.write_csv(buf)
    st.download_button(
        "Download CSV", data=buf.getvalue(), file_name="semantic_id_prefix_matches.csv", mime="text/csv"
    )


def render_compare_tab(store: MappingStore, index: EmbeddingIndex) -> None:
    st.subheader("Compare Two Products")
    col1, col2 = st.columns(2)
    with col1:
        product_id_a = st.text_input("Product ID A", key="compare_a")
    with col2:
        product_id_b = st.text_input("Product ID B", key="compare_b")
    if not product_id_a or not product_id_b:
        return

    result = core.compare_products(store, index, product_id_a, product_id_b)
    if result.error:
        st.error(result.error)
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Product {result.product_id_a}", result.lookup_a.semantic_id)
    with col2:
        st.metric(f"Product {result.product_id_b}", result.lookup_b.semantic_id)

    st.write("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        if result.cosine_similarity is not None:
            st.metric("Embedding cosine similarity", f"{result.cosine_similarity:.4f}")
        else:
            st.metric("Embedding cosine similarity", "N/A")
            st.caption(result.cosine_unavailable_reason or "Unavailable.")
    with m2:
        st.metric("Shared prefix depth", f"{result.shared_prefix_depth} / {store.num_levels}")
    with m3:
        st.metric("Same full code tuple", "Yes" if result.same_code_tuple else "No")

    st.write("Metadata agreement")
    agreement_df = pl.DataFrame(
        {
            "field": [m.field for m in result.metadata_agreement],
            "value_a": [str(m.value_a) for m in result.metadata_agreement],
            "value_b": [str(m.value_b) for m in result.metadata_agreement],
            "match": [m.match for m in result.metadata_agreement],
        }
    )
    st.dataframe(agreement_df, hide_index=True, use_container_width=True)


def main() -> None:
    st.title("Semantic ID Inspector")

    paths = get_paths()
    store = get_mapping_store(paths)
    index = get_embedding_index(paths)

    render_sidebar(paths, store, index)

    tab1, tab2, tab3 = st.tabs(
        ["Product → Semantic ID", "Semantic ID / Prefix → Products", "Compare Two Products"]
    )
    with tab1:
        render_lookup_tab(store)
    with tab2:
        render_prefix_tab(store)
    with tab3:
        render_compare_tab(store, index)


if __name__ == "__main__":
    main()
