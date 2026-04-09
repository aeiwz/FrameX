"""Tests for ops: filter, groupby+agg, hash join, sort."""

import pyarrow as pa
import pytest

from framex.ops.filter import (
    filter_table,
    predicate_eq,
    predicate_gt,
    predicate_isin,
    predicate_lt,
)
from framex.ops.groupby import groupby_agg
from framex.ops.join import hash_join
from framex.ops.sort import sort_table
from framex.ops.reduction import sum_column, mean_column, count_column, min_column, max_column
from framex.ops.projection import select_columns


class TestFilterOps:
    def test_equality_predicate(self):
        table = pa.table({"a": [1, 2, 3, 2, 1], "b": [10, 20, 30, 40, 50]})
        mask = predicate_eq(table.column("a"), 2)
        result = filter_table(table, mask)
        assert result.num_rows == 2
        assert result.column("a").to_pylist() == [2, 2]

    def test_greater_than(self):
        table = pa.table({"val": [10, 20, 30, 40, 50]})
        mask = predicate_gt(table.column("val"), 25)
        result = filter_table(table, mask)
        assert result.column("val").to_pylist() == [30, 40, 50]

    def test_less_than(self):
        table = pa.table({"val": [10, 20, 30, 40]})
        mask = predicate_lt(table.column("val"), 25)
        result = filter_table(table, mask)
        assert result.column("val").to_pylist() == [10, 20]

    def test_isin(self):
        table = pa.table({"name": ["alice", "bob", "charlie", "dave"]})
        mask = predicate_isin(table.column("name"), ["bob", "dave"])
        result = filter_table(table, mask)
        assert result.column("name").to_pylist() == ["bob", "dave"]


class TestGroupByOps:
    def test_groupby_sum(self):
        table = pa.table({
            "key": ["a", "b", "a", "b", "a"],
            "val": [1, 2, 3, 4, 5],
        })
        result = groupby_agg(table, ["key"], {"val": "sum"})
        result_dict = result.to_pydict()
        rows = sorted(zip(result_dict["key"], result_dict["val_sum"]))
        assert rows == [("a", 9), ("b", 6)]

    def test_groupby_multiple_aggs(self):
        table = pa.table({
            "key": ["x", "y", "x", "y"],
            "val": [10, 20, 30, 40],
        })
        result = groupby_agg(table, ["key"], {"val": ["sum", "count"]})
        assert "val_sum" in result.schema.names
        assert "val_count" in result.schema.names

    def test_groupby_mean(self):
        table = pa.table({
            "g": ["a", "a", "b"],
            "v": [10.0, 20.0, 30.0],
        })
        result = groupby_agg(table, ["g"], {"v": "mean"})
        result_dict = result.to_pydict()
        rows = sorted(zip(result_dict["g"], result_dict["v_mean"]))
        assert rows == [("a", 15.0), ("b", 30.0)]


class TestJoinOps:
    def test_inner_join(self):
        left = pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        right = pa.table({"id": [2, 3, 4], "score": [90, 80, 70]})
        result = hash_join(left, right, on="id", how="inner")
        assert result.num_rows == 2
        assert sorted(result.column("id").to_pylist()) == [2, 3]

    def test_left_join(self):
        left = pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        right = pa.table({"id": [2, 3, 4], "score": [90, 80, 70]})
        result = hash_join(left, right, on="id", how="left")
        assert result.num_rows == 3
        assert sorted(result.column("id").to_pylist()) == [1, 2, 3]

    def test_join_with_overlapping_columns(self):
        left = pa.table({"id": [1, 2], "name": ["a", "b"]})
        right = pa.table({"id": [1, 2], "name": ["x", "y"]})
        result = hash_join(left, right, on="id", how="inner")
        assert "name" in result.schema.names
        assert "name_right" in result.schema.names


class TestSortOps:
    def test_sort_ascending(self):
        table = pa.table({"a": [3, 1, 2], "b": [30, 10, 20]})
        result = sort_table(table, by="a")
        assert result.column("a").to_pylist() == [1, 2, 3]
        assert result.column("b").to_pylist() == [10, 20, 30]

    def test_sort_descending(self):
        table = pa.table({"a": [3, 1, 2]})
        result = sort_table(table, by="a", ascending=False)
        assert result.column("a").to_pylist() == [3, 2, 1]

    def test_sort_multi_column(self):
        table = pa.table({"a": [1, 1, 2, 2], "b": [20, 10, 40, 30]})
        result = sort_table(table, by=["a", "b"])
        assert result.column("b").to_pylist() == [10, 20, 30, 40]


class TestReductionOps:
    def test_sum(self):
        col = pa.chunked_array([[1, 2, 3, 4]])
        assert sum_column(col) == 10

    def test_mean(self):
        col = pa.chunked_array([[10.0, 20.0, 30.0]])
        assert mean_column(col) == 20.0

    def test_count(self):
        col = pa.chunked_array([[1, 2, 3]])
        assert count_column(col) == 3

    def test_min_max(self):
        col = pa.chunked_array([[5, 1, 9, 3]])
        assert min_column(col) == 1
        assert max_column(col) == 9


class TestProjectionOps:
    def test_select_columns(self):
        table = pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = select_columns(table, ["a", "c"])
        assert result.schema.names == ["a", "c"]
        assert result.num_rows == 2
