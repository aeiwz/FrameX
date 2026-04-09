"""Tests for I/O: Parquet round-trip, Arrow IPC round-trip, CSV."""

import pandas as pd
import pyarrow as pa
import pytest

import framex as fx
from framex.core.dataframe import DataFrame


class TestParquet:
    def test_round_trip(self, tmp_path):
        df = DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
        path = tmp_path / "test.parquet"
        fx.write_parquet(df, path)
        df2 = fx.read_parquet(path)
        assert df2.num_rows == 3
        assert df2["a"].to_pylist() == [1, 2, 3]
        assert df2["b"].to_pylist() == [10.0, 20.0, 30.0]

    def test_column_selection(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        path = tmp_path / "test.parquet"
        fx.write_parquet(df, path)
        df2 = fx.read_parquet(path, columns=["a", "c"])
        assert df2.columns == ["a", "c"]

    def test_large_round_trip(self, tmp_path):
        """Round-trip with enough rows to trigger partitioning."""
        df = DataFrame({"val": list(range(10_000))})
        path = tmp_path / "large.parquet"
        fx.write_parquet(df, path)
        df2 = fx.read_parquet(path)
        assert df2.num_rows == 10_000
        assert df2["val"].sum() == sum(range(10_000))


class TestArrowIPC:
    def test_round_trip(self, tmp_path):
        df = DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        path = tmp_path / "test.arrow"
        fx.write_ipc(df, str(path))
        df2 = fx.read_ipc(str(path))
        assert df2.num_rows == 3
        assert df2["x"].to_pylist() == [1, 2, 3]
        assert df2["y"].to_pylist() == ["a", "b", "c"]

    def test_empty_round_trip(self, tmp_path):
        schema = pa.schema([("a", pa.int64())])
        table = pa.table({"a": pa.array([], type=pa.int64())})
        df = DataFrame(table)
        path = tmp_path / "empty.arrow"
        fx.write_ipc(df, str(path))
        df2 = fx.read_ipc(str(path))
        assert df2.num_rows == 0


class TestCSV:
    def test_read_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        df = fx.read_csv(str(path))
        assert df.num_rows == 3
        assert df.columns == ["a", "b", "c"]
        assert df["a"].to_pylist() == [1, 4, 7]
