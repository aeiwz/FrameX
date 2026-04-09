"""CSV reader via PyArrow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pcsv


def read_csv(
    path: str | Path,
    **kwargs: Any,
) -> Any:
    """Read a CSV file into a FrameX DataFrame."""
    from framex.core.dataframe import DataFrame

    table = pcsv.read_csv(str(path), **kwargs)
    return DataFrame(table)


def read_csv_bytes(data: bytes, **kwargs: Any) -> Any:
    """Read CSV bytes into a FrameX DataFrame."""
    from framex.core.dataframe import DataFrame

    table = pcsv.read_csv(pa.BufferReader(data), **kwargs)
    return DataFrame(table)


def write_csv(
    df: Any,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Write a FrameX DataFrame to CSV."""
    table = df.to_arrow()
    pcsv.write_csv(table, str(path), **kwargs)


def write_csv_bytes(df: Any, **kwargs: Any) -> bytes:
    """Serialize a FrameX DataFrame to CSV bytes."""
    table = df.to_arrow()
    sink = pa.BufferOutputStream()
    pcsv.write_csv(table, sink, **kwargs)
    return sink.getvalue().to_pybytes()
