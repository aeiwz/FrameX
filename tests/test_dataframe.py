"""Tests for DataFrame: construction, filter, select, groupby, conversions, lazy mode."""

import pandas as pd
import pyarrow as pa
import pytest

import framex as fx
from framex.core.dataframe import DataFrame


def _scale_partition_value(batch: pa.RecordBatch) -> pa.RecordBatch:
    value_col = batch.column(batch.schema.get_field_index("value"))
    scaled = pa.array([v.as_py() * 10 for v in value_col])
    return pa.record_batch([scaled], names=["value"])


def _shift_partition_value(batch: pa.RecordBatch) -> pa.RecordBatch:
    value_col = batch.column(batch.schema.get_field_index("value"))
    shifted = pa.array([v.as_py() + 1 for v in value_col])
    return pa.record_batch([shifted], names=["value"])


class TestConstruction:
    def test_from_dict(self):
        df = DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        assert df.num_rows == 3
        assert df.num_columns == 2
        assert df.columns == ["a", "b"]

    def test_from_pandas(self):
        pdf = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        df = DataFrame(pdf)
        assert df.num_rows == 2
        assert df.columns == ["x", "y"]

    def test_from_arrow_table(self):
        table = pa.table({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df = DataFrame(table)
        assert df.num_rows == 3
        assert df.shape == (3, 2)

    def test_from_dict_with_series(self):
        from framex.core.series import Series

        s = Series([1, 2, 3], name="a")
        df = DataFrame({"a": s, "b": [10, 20, 30]})
        assert df.num_rows == 3

    def test_empty(self):
        df = DataFrame()
        assert df.num_rows == 0
        assert df.num_columns == 0


class TestColumnAccess:
    def test_getitem_string(self):
        df = DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        s = df["a"]
        assert isinstance(s, fx.Series)
        assert s.to_pylist() == [1, 2, 3]

    def test_getitem_list(self):
        df = DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})
        df2 = df[["a", "c"]]
        assert isinstance(df2, DataFrame)
        assert df2.columns == ["a", "c"]


class TestFilter:
    def test_filter_eq(self):
        df = DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        mask = df["a"] > 2
        result = df.filter(mask)
        assert result.num_rows == 2
        assert result["a"].to_pylist() == [3, 4]

    def test_filter_isin(self):
        df = DataFrame({"name": ["alice", "bob", "charlie"], "val": [1, 2, 3]})
        mask = df["name"].isin(["alice", "charlie"])
        result = df.filter(mask)
        assert result.num_rows == 2
        assert result["name"].to_pylist() == ["alice", "charlie"]


class TestSelect:
    def test_select_columns(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df2 = df.select(["a", "c"])
        assert df2.columns == ["a", "c"]
        assert df2.num_rows == 2


class TestGroupBy:
    def test_groupby_sum(self):
        df = DataFrame({
            "key": ["a", "b", "a", "b"],
            "val": [1, 2, 3, 4],
        })
        result = df.groupby("key").agg({"val": "sum"})
        result_dict = result.to_pydict()
        # Arrow groupby may return in any order; sort for determinism.
        rows = sorted(zip(result_dict["key"], result_dict["val"]))
        assert rows == [("a", 4), ("b", 6)]

    def test_groupby_count(self):
        df = DataFrame({
            "key": ["x", "y", "x", "y", "x"],
            "val": [1, 2, 3, 4, 5],
        })
        result = df.groupby("key").agg({"val": "count"})
        result_dict = result.to_pydict()
        rows = sorted(zip(result_dict["key"], result_dict["val"]))
        assert rows == [("x", 3), ("y", 2)]

    def test_groupby_mean(self):
        df = DataFrame({
            "key": ["a", "a", "b", "b"],
            "val": [10.0, 20.0, 30.0, 40.0],
        })
        result = df.groupby("key").agg({"val": "mean"})
        result_dict = result.to_pydict()
        rows = sorted(zip(result_dict["key"], result_dict["val"]))
        assert rows == [("a", 15.0), ("b", 35.0)]


class TestConversions:
    def test_to_pandas(self):
        df = DataFrame({"a": [1, 2, 3]})
        pdf = df.to_pandas()
        assert isinstance(pdf, pd.DataFrame)
        assert list(pdf["a"]) == [1, 2, 3]

    def test_to_arrow(self):
        df = DataFrame({"a": [1, 2, 3]})
        table = df.to_arrow()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 3

    def test_round_trip_pandas(self):
        pdf = pd.DataFrame({"x": [1.5, 2.5, 3.5], "y": [10, 20, 30]})
        df = DataFrame(pdf)
        pdf2 = df.to_pandas()
        pd.testing.assert_frame_equal(pdf, pdf2)


class TestSort:
    def test_sort_ascending(self):
        df = DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]})
        result = df.sort("a")
        assert result["a"].to_pylist() == [1, 2, 3]
        assert result["b"].to_pylist() == [10, 20, 30]

    def test_sort_descending(self):
        df = DataFrame({"a": [3, 1, 2]})
        result = df.sort("a", ascending=False)
        assert result["a"].to_pylist() == [3, 2, 1]


class TestJoin:
    def test_join_single_key_no_conflict_columns(self):
        left = DataFrame({
            "id": [1, 2, 3, 4],
            "val_left": [10, 20, 30, 40],
        })
        right = DataFrame({
            "id": [2, 4, 5],
            "val_right": [200, 400, 500],
        })

        result = left.join(right, on="id", how="inner")
        data = result.to_pydict()

        assert result.columns == ["id", "val_left", "val_right"]
        rows = sorted(zip(data["id"], data["val_left"], data["val_right"]))
        assert rows == [(2, 20, 200), (4, 40, 400)]

    def test_join_overlap_still_renames_right_columns(self):
        left = DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })
        right = DataFrame({
            "id": [2, 3, 4],
            "value": [200, 300, 400],
        })

        result = left.join(right, on="id", how="inner")
        assert "value" in result.columns
        assert "value_right" in result.columns


class TestHead:
    def test_head(self):
        df = DataFrame({"a": list(range(100))})
        result = df.head(5)
        assert result.num_rows == 5
        assert result["a"].to_pylist() == [0, 1, 2, 3, 4]


class TestLazy:
    def test_lazy_filter_select(self):
        df = DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
        })
        result = (
            df.lazy()
            .filter(lambda d: d["a"] > 2)
            .select(["a", "b"])
            .collect()
        )
        assert result.columns == ["a", "b"]
        assert result["a"].to_pylist() == [3, 4, 5]

    def test_lazy_groupby_agg(self):
        df = DataFrame({
            "key": ["a", "b", "a", "b"],
            "val": [1, 2, 3, 4],
        })
        result = (
            df.lazy()
            .groupby("key")
            .agg({"val": "sum"})
            .collect()
        )
        result_dict = result.to_pydict()
        rows = sorted(zip(result_dict["key"], result_dict["val"]))
        assert rows == [("a", 4), ("b", 6)]

    def test_lazy_sort(self):
        df = DataFrame({"a": [3, 1, 2]})
        result = df.lazy().sort("a").collect()
        assert result["a"].to_pylist() == [1, 2, 3]

    def test_lazy_map_partitions(self):
        df = DataFrame({"value": [1, 2, 3, 4, 5, 6]})
        result = (
            df.lazy()
            .map_partitions(_shift_partition_value, workers=2, backend="threads")
            .collect()
        )
        assert result["value"].to_pylist() == [2, 3, 4, 5, 6, 7]


class TestParallelPartitions:
    def test_map_partitions_threads(self):
        df = DataFrame({"value": list(range(1, 33))})
        result = df.map_partitions(_scale_partition_value, workers=4, backend="threads")
        assert result["value"].to_pylist() == [v * 10 for v in range(1, 33)]

    def test_map_partitions_processes(self):
        df = DataFrame({"value": list(range(1, 17))})
        result = df.map_partitions(_scale_partition_value, workers=2, backend="processes")
        assert result["value"].to_pylist() == [v * 10 for v in range(1, 17)]
