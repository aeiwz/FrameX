"""Tests for interchange: __dataframe__ protocol, from_pandas, from_dataframe."""

import pandas as pd
import pyarrow as pa
import pytest

import framex as fx
from framex.core.dataframe import DataFrame
from framex.interchange.dataframe_protocol import from_dataframe, from_pandas
from framex.interchange.numpy_protocols import implements_array_function, implements_array_ufunc


class TestFromPandas:
    def test_basic(self):
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df = from_pandas(pdf)
        assert isinstance(df, DataFrame)
        assert df.num_rows == 3
        assert df.columns == ["a", "b"]

    def test_round_trip(self):
        pdf = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        df = from_pandas(pdf)
        pdf2 = df.to_pandas()
        pd.testing.assert_frame_equal(pdf, pdf2)


class TestFromDataframe:
    def test_from_pandas_direct(self):
        pdf = pd.DataFrame({"a": [10, 20, 30]})
        df = from_dataframe(pdf)
        assert df.num_rows == 3

    def test_from_pandas_interchange_protocol(self):
        """Use the __dataframe__ protocol from a Pandas object."""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        interchange = pdf.__dataframe__()
        df = from_dataframe(interchange)
        assert df.num_rows == 3

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            from_dataframe("not a dataframe")


class TestDataframeProtocol:
    def test_has_dunder_dataframe(self):
        df = DataFrame({"a": [1, 2, 3]})
        assert hasattr(df, "__dataframe__")

    def test_dunder_dataframe_returns_interchange(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        interchange = df.__dataframe__()
        # The interchange object should be consumable by pandas.
        pdf = pd.api.interchange.from_dataframe(interchange)
        assert list(pdf.columns) == ["a", "b"]
        assert len(pdf) == 3

    def test_round_trip_via_protocol(self):
        """FrameX -> __dataframe__ -> pandas -> FrameX."""
        df = DataFrame({"x": [10, 20, 30]})
        interchange = df.__dataframe__()
        pdf = pd.api.interchange.from_dataframe(interchange)
        df2 = from_pandas(pdf)
        assert df2["x"].to_pylist() == [10, 20, 30]


class TestNumpyProtocols:
    def test_array_ufunc_implemented(self):
        assert implements_array_ufunc()

    def test_array_function_implemented(self):
        assert implements_array_function()
