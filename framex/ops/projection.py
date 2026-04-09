"""Select / project columns from tables and partitions."""

from __future__ import annotations

import pyarrow as pa

from framex.runtime.partition import Partition


def select_columns(table: pa.Table, columns: list[str]) -> pa.Table:
    """Project a subset of columns from a Table."""
    return table.select(columns)


def select_columns_batch(batch: pa.RecordBatch, columns: list[str]) -> pa.RecordBatch:
    """Project a subset of columns from a RecordBatch."""
    indices = [batch.schema.get_field_index(c) for c in columns]
    return pa.record_batch(
        [batch.column(i) for i in indices],
        names=columns,
    )


def select_partitions(partitions: list[Partition], columns: list[str]) -> list[Partition]:
    """Project columns across all partitions."""
    return [
        Partition(
            record_batch=select_columns_batch(p.record_batch, columns),
            partition_id=p.partition_id,
        )
        for p in partitions
    ]
