"""Arrow IPC stream I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa


def write_ipc(df: Any, path: str | Path) -> None:
    """Write a FrameX DataFrame to an Arrow IPC file (stream format)."""
    table = df.to_arrow()
    with pa.OSFile(str(path), "wb") as f:
        writer = pa.ipc.new_stream(f, table.schema)
        for batch in table.to_batches():
            writer.write_batch(batch)
        writer.close()


def read_ipc(path: str | Path) -> Any:
    """Read an Arrow IPC stream file into a FrameX DataFrame."""
    from framex.core.dataframe import DataFrame

    with pa.OSFile(str(path), "rb") as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    return DataFrame(table)
