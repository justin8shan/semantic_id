import polars as pl

from semantic_id.inference.dedup import add_dedup_index, max_dedup


def _make_df(rows):
    return pl.DataFrame(rows, schema=["product_id", "c1", "c2", "c3"], orient="row")


def test_dedup_index_is_zero_based_and_orders_by_product_id():
    # Two collisions: (0,0,0) shared by ids 5,2,9; (1,1,1) shared by ids 3,1.
    rows = [
        (5, 0, 0, 0), (2, 0, 0, 0), (9, 0, 0, 0),
        (3, 1, 1, 1), (1, 1, 1, 1),
        (7, 2, 2, 2),  # no collision
    ]
    df = _make_df(rows)
    result = add_dedup_index(df, ["c1", "c2", "c3"])
    dedup_by_id = dict(zip(result["product_id"].to_list(), result["dedup"].to_list()))

    assert dedup_by_id[2] == 0 and dedup_by_id[5] == 1 and dedup_by_id[9] == 2  # ascending product_id order
    assert dedup_by_id[1] == 0 and dedup_by_id[3] == 1
    assert dedup_by_id[7] == 0


def test_dedup_index_is_deterministic_across_reruns():
    rows = [(i, i % 3, 0, 0) for i in range(30)]
    df = _make_df(rows)

    run1 = add_dedup_index(df, ["c1", "c2", "c3"]).sort("product_id")
    run2 = add_dedup_index(df, ["c1", "c2", "c3"]).sort("product_id")

    assert run1["dedup"].to_list() == run2["dedup"].to_list()


def test_max_dedup():
    rows = [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 1, 0, 0)]
    df = add_dedup_index(_make_df(rows), ["c1", "c2", "c3"])
    assert max_dedup(df) == 2
