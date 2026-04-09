"""Parquet I/O via PyArrow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def read_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Read a Parquet file into a FrameX DataFrame."""
    from framex.core.dataframe import DataFrame

    table = pq.read_table(str(path), columns=columns, **kwargs)
    return DataFrame(table)


def write_parquet(
    df: Any,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Write a FrameX DataFrame to a Parquet file."""
    table = df.to_arrow()
    pq.write_table(table, str(path), **kwargs)
