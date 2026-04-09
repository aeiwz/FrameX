"""JSON / NDJSON readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.json as paj


def _table_from_json_document(path: Path) -> pa.Table:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        # Common "records" JSON: [{...}, {...}]
        return pa.Table.from_pylist(payload)
    if isinstance(payload, dict):
        # Column-oriented JSON: {"col": [..], ...}
        return pa.table(payload)
    raise ValueError("JSON root must be an object or array of objects")


def read_json(
    path: str | Path,
    *,
    lines: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Read JSON/NDJSON into a FrameX DataFrame.

    Parameters
    ----------
    path:
        Input file path.
    lines:
        When ``True`` reads newline-delimited JSON via PyArrow's streaming
        JSON reader. When ``False`` reads a standard JSON document. If omitted,
        auto-detects from extension (``.jsonl``/``.ndjson`` => ``True``).
    """
    from framex.core.dataframe import DataFrame

    file_path = Path(path)
    if lines is None:
        lines = file_path.suffix.lower() in {".jsonl", ".ndjson"}

    if lines:
        table = paj.read_json(str(file_path), **kwargs)
    else:
        table = _table_from_json_document(file_path)
    return DataFrame(table)


def read_json_bytes(
    data: bytes,
    *,
    lines: bool = False,
    **kwargs: Any,
) -> Any:
    """Read JSON/NDJSON bytes into a FrameX DataFrame."""
    from framex.core.dataframe import DataFrame

    if lines:
        table = paj.read_json(pa.BufferReader(data), **kwargs)
    else:
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, list):
            table = pa.Table.from_pylist(payload)
        elif isinstance(payload, dict):
            table = pa.table(payload)
        else:
            raise ValueError("JSON root must be an object or array of objects")
    return DataFrame(table)


def read_ndjson(path: str | Path, **kwargs: Any) -> Any:
    """Read newline-delimited JSON (NDJSON/JSONL) into a FrameX DataFrame."""
    return read_json(path, lines=True, **kwargs)


def write_json(
    df: Any,
    path: str | Path,
    *,
    lines: bool = False,
    orient: str = "records",
    indent: int | None = None,
) -> None:
    """Write a FrameX DataFrame as JSON or NDJSON.

    Parameters
    ----------
    lines:
        When True, writes one JSON object per line (NDJSON/JSONL).
    orient:
        For non-lines output:
        - ``"records"`` -> ``[{...}, {...}]`` (default)
        - ``"columns"`` -> ``{"col": [...], ...}``
    indent:
        Optional JSON indentation for non-lines output.
    """
    table = df.to_arrow()
    file_path = Path(path)

    if lines:
        rows = table.to_pylist()
        with file_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str, ensure_ascii=False))
                f.write("\n")
        return

    if orient == "records":
        payload: Any = table.to_pylist()
    elif orient == "columns":
        payload = table.to_pydict()
    else:
        raise ValueError("orient must be 'records' or 'columns'")

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, default=str, ensure_ascii=False)


def write_ndjson(df: Any, path: str | Path) -> None:
    """Write newline-delimited JSON (NDJSON/JSONL)."""
    write_json(df, path, lines=True)


def write_json_bytes(
    df: Any,
    *,
    lines: bool = False,
    orient: str = "records",
    indent: int | None = None,
) -> bytes:
    """Serialize JSON/NDJSON from a FrameX DataFrame to bytes."""
    table = df.to_arrow()
    if lines:
        parts: list[str] = []
        for row in table.to_pylist():
            parts.append(json.dumps(row, default=str, ensure_ascii=False))
        return ("\n".join(parts) + ("\n" if parts else "")).encode("utf-8")

    if orient == "records":
        payload: Any = table.to_pylist()
    elif orient == "columns":
        payload = table.to_pydict()
    else:
        raise ValueError("orient must be 'records' or 'columns'")
    return json.dumps(payload, indent=indent, default=str, ensure_ascii=False).encode("utf-8")
