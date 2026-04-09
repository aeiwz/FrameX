"""Sort and shuffle operations."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from framex.runtime.partition import Partition, partition_table


def sort_table(
    table: pa.Table,
    by: str | list[str],
    ascending: bool | list[bool] = True,
) -> pa.Table:
    """Sort a Table by one or more columns."""
    if isinstance(by, str):
        by = [by]
    if isinstance(ascending, bool):
        ascending = [ascending] * len(by)

    sort_keys = [
        (col, "ascending" if asc else "descending")
        for col, asc in zip(by, ascending)
    ]
    indices = pc.sort_indices(table, sort_keys=sort_keys)
    return table.take(indices)


def sort_partitions(
    partitions: list[Partition],
    by: str | list[str],
    ascending: bool | list[bool] = True,
) -> list[Partition]:
    """Sort across all partitions (materialises to a single table then re-partitions)."""
    if not partitions:
        return partitions
    batches = [p.record_batch for p in partitions]
    table = pa.Table.from_batches(batches, schema=partitions[0].schema)
    sorted_table = sort_table(table, by, ascending)
    return partition_table(sorted_table)
