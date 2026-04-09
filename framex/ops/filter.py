"""Filter / predicate pushdown operations."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from framex.runtime.partition import Partition


def filter_batch(batch: pa.RecordBatch, mask: pa.Array) -> pa.RecordBatch:
    """Filter a RecordBatch by a boolean array mask."""
    return pc.filter(batch, mask)


def filter_table(table: pa.Table, mask: pa.ChunkedArray) -> pa.Table:
    """Filter a Table by a boolean ChunkedArray mask."""
    return table.filter(mask)


def filter_partitions(
    partitions: list[Partition],
    mask: pa.ChunkedArray,
) -> list[Partition]:
    """Filter partitions row-wise by a boolean ChunkedArray.

    The mask must have the same total length as all partitions combined.
    """
    offset = 0
    result: list[Partition] = []
    for pid, p in enumerate(partitions):
        n = p.num_rows
        chunk_mask = mask.slice(offset, n).combine_chunks()
        filtered_batch = filter_batch(p.record_batch, chunk_mask)
        result.append(Partition(record_batch=filtered_batch, partition_id=pid))
        offset += n
    return result


def predicate_eq(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    """Create an equality predicate mask."""
    return pc.equal(column, value)


def predicate_gt(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    return pc.greater(column, value)


def predicate_lt(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    return pc.less(column, value)


def predicate_gte(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    return pc.greater_equal(column, value)


def predicate_lte(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    return pc.less_equal(column, value)


def predicate_isin(column: pa.ChunkedArray, values: list[Any]) -> pa.ChunkedArray:
    """Create an ``isin`` predicate mask."""
    value_set = pa.array(values, type=column.type)
    return pc.is_in(column, value_set=value_set)
