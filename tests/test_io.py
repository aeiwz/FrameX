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


class TestJSON:
    def test_read_json_records(self, tmp_path):
        path = tmp_path / "records.json"
        path.write_text('[{"a":1,"b":"x"},{"a":2,"b":"y"}]', encoding="utf-8")
        df = fx.read_json(path)
        assert df.num_rows == 2
        assert df.columns == ["a", "b"]
        assert df["a"].to_pylist() == [1, 2]

    def test_read_ndjson(self, tmp_path):
        path = tmp_path / "events.ndjson"
        path.write_text('{"a":1,"b":"x"}\n{"a":2,"b":"y"}\n', encoding="utf-8")
        df = fx.read_ndjson(path)
        assert df.num_rows == 2
        assert df.columns == ["a", "b"]
        assert df["b"].to_pylist() == ["x", "y"]


class TestReadFile:
    def test_read_file_auto_csv(self, tmp_path):
        path = tmp_path / "sample.csv"
        path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        df = fx.read_file(path)
        assert df.num_rows == 2
        assert df["a"].to_pylist() == [1, 3]

    def test_read_file_auto_json(self, tmp_path):
        path = tmp_path / "sample.json"
        path.write_text('[{"a":1},{"a":2}]', encoding="utf-8")
        df = fx.read_file(path)
        assert df.num_rows == 2
        assert df["a"].to_pylist() == [1, 2]

    def test_read_file_explicit_format_tsv(self, tmp_path):
        path = tmp_path / "sample.tsv"
        path.write_text("a\tb\n1\tx\n2\ty\n", encoding="utf-8")
        df = fx.read_file(path, format="tsv")
        assert df.num_rows == 2
        assert df.columns == ["a", "b"]


class TestWriteFile:
    def test_write_file_csv_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "out.csv"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]

    def test_write_file_tsv_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "out.tsv"
        fx.write_file(df, path, format="tsv")
        out = fx.read_file(path, format="tsv")
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]

    def test_write_file_json_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "out.json"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]

    def test_write_file_ndjson_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "out.ndjson"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]

    def test_write_file_feather_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "out.feather"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]

    def test_write_file_pickle_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "out.pkl"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]


class TestCompressedIO:
    def test_csv_gzip_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "data.csv.gz"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out["a"].to_pylist() == [1, 2]

    def test_json_bz2_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "data.json.bz2"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out["b"].to_pylist() == ["x", "y"]

    def test_ndjson_xz_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "data.ndjson.xz"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]

    def test_parquet_zip_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "data.parquet.zip"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]

    def test_pickle_gzip_roundtrip(self, tmp_path):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "data.pkl.gz"
        fx.write_file(df, path)
        out = fx.read_file(path)
        assert out.num_rows == 2
        assert out.columns == ["a", "b"]
