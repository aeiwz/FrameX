"""Partition: the unit of parallelism in FrameX."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from framex.config import get_config


@dataclass
class Partition:
    """A single partition of a DataFrame.

    Wraps a ``pyarrow.RecordBatch`` with metadata.
    """

    record_batch: pa.RecordBatch
    partition_id: int = 0

    @property
    def schema(self) -> pa.Schema:
        return self.record_batch.schema

    @property
    def num_rows(self) -> int:
        return self.record_batch.num_rows

    @property
    def num_columns(self) -> int:
        return self.record_batch.num_columns

    def to_table(self) -> pa.Table:
        return pa.Table.from_batches([self.record_batch])

    def __repr__(self) -> str:
        return f"Partition(id={self.partition_id}, rows={self.num_rows}, cols={self.num_columns})"


def partition_table(table: pa.Table, partition_size: int | None = None) -> list[Partition]:
    """Split a ``pyarrow.Table`` into a list of ``Partition`` objects.

    Uses the configured ``partition_size_rows`` unless overridden.
    """
    if partition_size is None:
        partition_size = get_config().partition_size_rows

    num_rows = table.num_rows
    if num_rows == 0:
        batches = table.to_batches()
        if batches:
            return [Partition(record_batch=batches[0], partition_id=0)]
        # Create an empty batch with the correct schema.
        empty_arrays = [pa.array([], type=field.type) for field in table.schema]
        empty_batch = pa.record_batch(empty_arrays, schema=table.schema)
        return [Partition(record_batch=empty_batch, partition_id=0)]

    cpu_count = os.cpu_count() or 4
    # Target: min(cpu_count * 4, max(1, num_rows // partition_size))
    target_partitions = min(cpu_count * 4, max(1, num_rows // partition_size)) if num_rows >= partition_size else 1
    batch_size = max(1, num_rows // target_partitions)

    batches = table.to_batches(max_chunksize=batch_size)
    return [Partition(record_batch=b, partition_id=i) for i, b in enumerate(batches)]
