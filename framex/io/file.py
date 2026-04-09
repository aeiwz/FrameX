"""Unified file read/write helpers with format auto-detection."""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import pickle
import zipfile
from pathlib import Path
from typing import Any

import pyarrow.feather as pfeather
import pyarrow.parquet as pq
import pyarrow as pa

from framex.io.arrow_ipc import read_ipc
from framex.io.csv import read_csv, read_csv_bytes, write_csv, write_csv_bytes
from framex.io.json import read_json, read_json_bytes, read_ndjson, write_json, write_json_bytes, write_ndjson
from framex.io.parquet import read_parquet, write_parquet
from framex.io.arrow_ipc import write_ipc

_COMPRESSION_BY_SUFFIX = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
    ".zip": "zip",
    ".zst": "zstd",
    ".zstd": "zstd",
}


def _normalize_format(path: Path, fmt: str | None) -> str:
    if fmt is not None:
        return fmt.strip().lower()
    suffix = path.suffix.lower()
    if suffix in {".parquet"}:
        return "parquet"
    if suffix in {".arrow", ".ipc"}:
        return "ipc"
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".tsv"}:
        return "tsv"
    if suffix in {".jsonl", ".ndjson"}:
        return "ndjson"
    if suffix in {".json"}:
        return "json"
    if suffix in {".feather"}:
        return "feather"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    raise ValueError(f"Could not infer file format from extension: {path}")


def _infer_compression(path: Path) -> str | None:
    return _COMPRESSION_BY_SUFFIX.get(path.suffix.lower())


def _strip_compression_suffix(path: Path, compression: str | None) -> Path:
    if compression is None:
        return path
    suffix = path.suffix.lower()
    if _COMPRESSION_BY_SUFFIX.get(suffix) == compression:
        return path.with_suffix("")
    return path


def _read_compressed_bytes(path: Path, compression: str) -> bytes:
    if compression == "gzip":
        with gzip.open(path, "rb") as f:
            return f.read()
    if compression == "bz2":
        with bz2.open(path, "rb") as f:
            return f.read()
    if compression == "xz":
        with lzma.open(path, "rb") as f:
            return f.read()
    if compression == "zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                return b""
            return zf.read(names[0])
    if compression == "zstd":
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("zstd compression requires 'zstandard' package") from exc
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(f.read())
    raise ValueError(f"Unsupported compression: {compression!r}")


def _write_compressed_bytes(path: Path, data: bytes, compression: str) -> None:
    if compression == "gzip":
        with gzip.open(path, "wb") as f:
            f.write(data)
        return
    if compression == "bz2":
        with bz2.open(path, "wb") as f:
            f.write(data)
        return
    if compression == "xz":
        with lzma.open(path, "wb") as f:
            f.write(data)
        return
    if compression == "zip":
        inner_name = path.with_suffix("").name or "data"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(inner_name, data)
        return
    if compression == "zstd":
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("zstd compression requires 'zstandard' package") from exc
        cctx = zstd.ZstdCompressor()
        with open(path, "wb") as f:
            f.write(cctx.compress(data))
        return
    raise ValueError(f"Unsupported compression: {compression!r}")


def read_file(
    path: str | Path,
    *,
    format: str | None = None,
    compression: str | None = None,
    **kwargs: Any,
) -> Any:
    """Read a file into a FrameX DataFrame with format inference.

    Supported formats:
    ``parquet``, ``ipc``, ``csv``, ``tsv``, ``json``, ``ndjson``,
    ``feather``, ``pickle``, ``excel``.
    """
    from framex.core.dataframe import DataFrame
    from framex.pandas_engine import get_pandas_module

    file_path = Path(path)
    inferred_compression = _infer_compression(file_path)
    resolved_compression = compression or inferred_compression
    base_path = _strip_compression_suffix(file_path, resolved_compression)
    fmt = _normalize_format(base_path, format)

    if resolved_compression is not None:
        payload = _read_compressed_bytes(file_path, resolved_compression)
        if fmt == "csv":
            return read_csv_bytes(payload, **kwargs)
        if fmt == "tsv":
            import pyarrow.csv as pcsv

            parse_options = kwargs.pop("parse_options", pcsv.ParseOptions(delimiter="\t"))
            return read_csv_bytes(payload, parse_options=parse_options, **kwargs)
        if fmt == "json":
            return read_json_bytes(payload, lines=False, **kwargs)
        if fmt == "ndjson":
            return read_json_bytes(payload, lines=True, **kwargs)
        if fmt == "parquet":
            return DataFrame(pq.read_table(pa.BufferReader(payload), **kwargs))
        if fmt == "ipc":
            reader = pa.ipc.open_stream(pa.BufferReader(payload))
            return DataFrame(reader.read_all())
        if fmt == "feather":
            return DataFrame(pfeather.read_table(pa.BufferReader(payload)))
        if fmt == "pickle":
            obj = pickle.loads(payload)
            if _looks_like_pandas_dataframe(obj):
                return DataFrame(obj)
            return DataFrame(obj)
        if fmt == "excel":
            pd = get_pandas_module()
            return DataFrame(pd.read_excel(io.BytesIO(payload), **kwargs))
        raise ValueError(f"Unsupported format for compressed input: {fmt!r}")

    if fmt == "parquet":
        return read_parquet(file_path, **kwargs)
    if fmt == "ipc":
        return read_ipc(file_path)
    if fmt == "csv":
        return read_csv(file_path, **kwargs)
    if fmt == "tsv":
        import pyarrow.csv as pcsv

        parse_options = kwargs.pop("parse_options", pcsv.ParseOptions(delimiter="\t"))
        return read_csv(file_path, parse_options=parse_options, **kwargs)
    if fmt == "json":
        return read_json(file_path, lines=False, **kwargs)
    if fmt == "ndjson":
        return read_ndjson(file_path, **kwargs)
    if fmt == "feather":
        return DataFrame(pfeather.read_table(file_path))
    if fmt == "pickle":
        pd = get_pandas_module()
        return DataFrame(pd.read_pickle(file_path, **kwargs))
    if fmt == "excel":
        pd = get_pandas_module()
        return DataFrame(pd.read_excel(file_path, **kwargs))

    raise ValueError(
        "Unsupported format. Expected one of: parquet, ipc, csv, tsv, json, ndjson, feather, pickle, excel; "
        f"got {fmt!r}"
    )


def write_file(
    df: Any,
    path: str | Path,
    *,
    format: str | None = None,
    compression: str | None = None,
    **kwargs: Any,
) -> None:
    """Write a FrameX DataFrame with format inference.

    Supported formats:
    ``parquet``, ``ipc``, ``csv``, ``tsv``, ``json``, ``ndjson``,
    ``feather``, ``pickle``, ``excel``.
    """
    from framex.pandas_engine import get_pandas_module

    file_path = Path(path)
    inferred_compression = _infer_compression(file_path)
    resolved_compression = compression or inferred_compression
    base_path = _strip_compression_suffix(file_path, resolved_compression)
    fmt = _normalize_format(base_path, format)

    if resolved_compression is not None:
        if fmt == "csv":
            payload = write_csv_bytes(df, **kwargs)
        elif fmt == "tsv":
            import pyarrow.csv as pcsv

            write_options = kwargs.pop("write_options", pcsv.WriteOptions(delimiter="\t"))
            payload = write_csv_bytes(df, write_options=write_options, **kwargs)
        elif fmt == "json":
            payload = write_json_bytes(df, lines=False, **kwargs)
        elif fmt == "ndjson":
            payload = write_json_bytes(df, lines=True, **kwargs)
        elif fmt == "parquet":
            sink = pa.BufferOutputStream()
            pq.write_table(df.to_arrow(), sink, **kwargs)
            payload = sink.getvalue().to_pybytes()
        elif fmt == "ipc":
            sink = pa.BufferOutputStream()
            table = df.to_arrow()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                for batch in table.to_batches():
                    writer.write_batch(batch)
            payload = sink.getvalue().to_pybytes()
        elif fmt == "feather":
            sink = pa.BufferOutputStream()
            pfeather.write_feather(df.to_arrow(), sink, **kwargs)
            payload = sink.getvalue().to_pybytes()
        elif fmt == "pickle":
            payload = pickle.dumps(df.to_pandas(), protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt == "excel":
            pd = get_pandas_module()
            buf = io.BytesIO()
            pd.DataFrame(df.to_pydict()).to_excel(buf, index=False, **kwargs)
            payload = buf.getvalue()
        else:
            raise ValueError(f"Unsupported format for compressed output: {fmt!r}")

        _write_compressed_bytes(file_path, payload, resolved_compression)
        return

    if fmt == "parquet":
        write_parquet(df, file_path, **kwargs)
        return
    if fmt == "ipc":
        write_ipc(df, file_path)
        return
    if fmt == "csv":
        write_csv(df, file_path, **kwargs)
        return
    if fmt == "tsv":
        import pyarrow.csv as pcsv

        write_options = kwargs.pop("write_options", pcsv.WriteOptions(delimiter="\t"))
        write_csv(df, file_path, write_options=write_options, **kwargs)
        return
    if fmt == "json":
        write_json(df, file_path, lines=False, **kwargs)
        return
    if fmt == "ndjson":
        write_ndjson(df, file_path)
        return
    if fmt == "feather":
        pfeather.write_feather(df.to_arrow(), file_path, **kwargs)
        return
    if fmt == "pickle":
        pd = get_pandas_module()
        pd.DataFrame(df.to_pydict()).to_pickle(file_path, **kwargs)
        return
    if fmt == "excel":
        pd = get_pandas_module()
        pd.DataFrame(df.to_pydict()).to_excel(file_path, index=False, **kwargs)
        return

    raise ValueError(
        "Unsupported format. Expected one of: parquet, ipc, csv, tsv, json, ndjson, feather, pickle, excel; "
        f"got {fmt!r}"
    )


def _looks_like_pandas_dataframe(value: Any) -> bool:
    cls = value.__class__
    return cls.__name__ == "DataFrame" and cls.__module__.startswith(
        ("pandas.", "modin.pandas", "fireducks.pandas")
    )
